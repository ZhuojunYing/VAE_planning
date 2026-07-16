"""Step-batched JAX trainer for the VAE planning model.

This module is a direct JAX implementation of the current TensorFlow training
objective in ``model/``.  The important parity points are:

* the same positional command-line interface and filename convention;
* the same task/path maps for default, bandit, and disjoint trees;
* joint expansion actions ordered as node-observation actions followed by
  terminal path actions; by default observed nodes are masked, with an optional
  revisit mode that keeps them legal;
* an LSTM state, Gaussian posterior, timestep Gaussian prior, decoder, joint
  expansion policy, action-value critic, and reward probe;
* KL scaling uses two explicit weights: ``loss_scale`` for reward/action losses
  and ``memory_lambda`` for the direct paid-KL multiplier;
* expansion return targets support the TensorFlow modes controlled by
  ``EXPANSION_RETURN_TARGET``: ``lambda`` uses sampled-reward counterfactual
  bootstrap targets, ``sampled_lambda`` uses trajectory lambda-returns, and
  ``one_step`` uses observe-one-reward-then-stop targets;
* training batches are fixed numbers of environment steps, and each env's RNN
  state is carried in the runner state across rollout/update batches.

The compiled update step contains rollout collection, PPO-style replay with
fixed actions, loss/gradient computation, optimizer update, and runner-state
advance.  Use ``--backend cpu`` to force CPU compilation, or leave it unset so
JAX can choose CPU/GPU from the installed jaxlib.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


def _append_xla_flag(flag: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if flag not in flags.split():
        os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()


def _append_xla_flag_if_unset(flag: str) -> None:
    flag_name = flag.split("=", 1)[0]
    flags = os.environ.get("XLA_FLAGS", "").split()
    if not any(existing == flag_name or existing.startswith(f"{flag_name}=") for existing in flags):
        _append_xla_flag(flag)


_append_xla_flag_if_unset("--xla_cpu_use_xla_runtime=false")
os.environ.setdefault("JAX_LOG_COMPILES", "0")

# This flag is useful on some newer x86 CPU XLA builds, but older jaxlib
# versions abort on unknown XLA flags. Keep it opt-in.
if os.environ.get("JAX_DISABLE_THUNK_RUNTIME", "").strip().lower() in {"1", "true", "yes"}:
    _append_xla_flag_if_unset("--xla_cpu_use_thunk_runtime=false")

import flax.linen as nn
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd


CATEGORY_VALUES = np.asarray(
    [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
    dtype=np.float32,
)
UNIFORM_REWARDS = np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float32)


@dataclass(frozen=True)
class TaskSpec:
    tree_size: int
    tree_type: str
    tree_name_suffix: str
    path_map: np.ndarray
    reward_values: np.ndarray
    reward_norm: float

    @property
    def num_nodes(self) -> int:
        return int(self.path_map.shape[1])

    @property
    def num_paths(self) -> int:
        return int(self.path_map.shape[0])


@dataclass(frozen=True)
class RunConfig:
    loss_scale: float
    alpha: float
    memory_lambda: float
    model_dir: str
    epochs: int
    input_type: str
    seed: int
    tree_size: int
    train_mode: str
    tree_type: str
    opportunity_cost: float
    expansion_decision_version: str
    model_variant: str
    rnn_units: int
    latent_dim: int
    sim_dir: str
    n_sim_trials: int
    num_envs: int
    num_steps: int
    update_epochs: int
    ppo_minibatches: int
    steps_per_epoch: int
    return_target_rollouts: int
    return_target_mode: str
    sampled_lambda_critic: str
    lambda_return: float
    target_critic_update_interval: int
    target_critic_tau: float
    backend: str | None
    jit_training: bool
    profile_update_components: bool
    profile_update_components_every: int
    enable_reconstruction: bool
    enable_probe: bool
    allow_node_revisit: bool
    max_observations_before_stop: int
    observation_sigma: float
    kl_start_multiplier: float
    kl_annealing_epochs: int
    node_coverage_aux_coef: float
    node_coverage_aux_epochs: int
    critic_huber_delta: float
    advantage_clip: float
    learning_rate: float = 5e-4
    min_learning_rate: float | None = None
    pay_kl_on_stop: bool = False
    choice_at_end_only: bool = False
    save_every_update: bool = False


class ScheduleValues(NamedTuple):
    current_alpha: jax.Array
    current_beta: jax.Array
    current_critic_coef: jax.Array
    expansion_epsilon: jax.Array
    expansion_entropy_coef: jax.Array
    node_coverage_aux_coef: jax.Array
    forced_continue_epsilon: jax.Array
    ppo_clip: jax.Array


class RunnerCarry(NamedTuple):
    rewards: jax.Array
    observed_rewards: jax.Array
    observed: jax.Array
    step_index: jax.Array
    done: jax.Array
    h: jax.Array
    c: jax.Array
    decoded_h: jax.Array
    decoded_c: jax.Array
    lstm_context: jax.Array
    pre_context: jax.Array
    pending_kl: jax.Array
    last_reward_onehot: jax.Array
    trial_id: jax.Array


class StepTransition(NamedTuple):
    rewards: jax.Array
    observed_rewards_before: jax.Array
    observed_rewards_after: jax.Array
    valid: jax.Array
    observed_before: jax.Array
    observed_after: jax.Array
    step_index: jax.Array
    action: jax.Array
    node_index: jax.Array
    terminal_path_index: jax.Array
    is_stop: jax.Array
    is_observe: jax.Array
    log_prob: jax.Array
    entropy: jax.Array
    probs: jax.Array
    legal_mask: jax.Array
    q_values: jax.Array
    q_targets: jax.Array
    selected_q_pred: jax.Array
    selected_q_target: jax.Array
    policy_value_pred: jax.Array
    expansion_input: jax.Array
    pre_decoded_h: jax.Array
    pre_decoded_c: jax.Array
    z_mu: jax.Array
    z_logvar: jax.Array
    z_sample: jax.Array
    prior_mu: jax.Array
    prior_logvar: jax.Array
    paid_kl: jax.Array
    observed_kl: jax.Array
    expanded_reward: jax.Array
    action_output: jax.Array
    terminal_expected_reward: jax.Array
    reconstruction_loss: jax.Array
    probe_loss: jax.Array
    probe_correct: jax.Array
    valid_probe: jax.Array
    reset_rewards: jax.Array
    reset_trial_id: jax.Array


class UpdateMetrics(NamedTuple):
    total_loss: jax.Array
    information_loss: jax.Array
    action_loss: jax.Array
    reconstruction_loss: jax.Array
    expansion_loss: jax.Array
    node_coverage_aux_loss: jax.Array
    critic_loss: jax.Array
    lstm_probe_loss: jax.Array
    lstm_probe_accuracy: jax.Array
    stop_rate: jax.Array
    continue_rate: jax.Array
    decision_mean_unique_nodes: jax.Array
    decision_all_nodes_rate: jax.Array
    decision_mean_unique_paths: jax.Array
    decision_all_paths_rate: jax.Array
    rollout_final_mean_unique_nodes: jax.Array
    rollout_final_all_nodes_rate: jax.Array
    rollout_final_mean_unique_paths: jax.Array
    rollout_final_all_paths_rate: jax.Array
    entropy_coef: jax.Array
    node_coverage_aux_coef: jax.Array
    critic_coef: jax.Array
    current_beta: jax.Array
    learning_rate: jax.Array
    continue_best_sums: jax.Array
    continue_best_counts: jax.Array
    critic_best_sums: jax.Array
    target_best_sums: jax.Array
    advantage_best_sums: jax.Array
    kl_best_sums: jax.Array
    q_stop_best_sums: jax.Array
    q_observe_best_sums: jax.Array
    q_stop_minus_observe_best_sums: jax.Array
    policy_stop_best_sums: jax.Array
    policy_observe_best_sums: jax.Array
    q_argmax_stop_best_sums: jax.Array
    policy_argmax_stop_best_sums: jax.Array


class ProfileReplayInputs(NamedTuple):
    reset_rewards: jax.Array
    replay_keys: jax.Array
    forced_actions: jax.Array
    forced_observations: jax.Array
    old_logp: jax.Array
    fixed_selected_q_target: jax.Array
    fixed_advantage: jax.Array
    cached_expansion_data: CachedExpansionPPOData
    collect_carry: RunnerCarry
    checksum: jax.Array


class CachedExpansionPPOData(NamedTuple):
    expansion_input: jax.Array
    legal_mask: jax.Array
    action: jax.Array
    old_logp: jax.Array
    advantage: jax.Array
    weights: jax.Array
    valid: jax.Array
    entropy_mask: jax.Array
    coverage_target: jax.Array
    coverage_mask: jax.Array


class PlanningTrainState(TrainState):
    target_params: object


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def normalize_return_target_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    one_step_modes = {"one_step", "joint_q", "joint", "counterfactual_one_step"}
    sampled_lambda_modes = {"sampled_lambda", "trajectory_lambda", "lambda_sampled"}
    if key in sampled_lambda_modes:
        return "sampled_lambda"
    if key in one_step_modes or key in {"mc", "monte_carlo", "montecarlo", "sampled"}:
        return "one_step"
    return "lambda"


def normalize_expansion_decision_version(version: str) -> str:
    key = str(version).strip().lower()
    aliases = {
        "1": "decoder",
        "decoder": "decoder",
        "after_decoder": "decoder",
        "2": "lstm",
        "lstm": "lstm",
        "after_lstm": "lstm",
        "3": "pre_lstm",
        "pre_lstm": "pre_lstm",
        "before_lstm": "pre_lstm",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported expansion_decision_version={version!r}.")
    return aliases[key]


def normalize_model_variant(variant: str) -> str:
    key = str(variant).strip().lower()
    aliases = {
        "vae": "vae",
        "autoencoder": "vae",
        "rnn": "rnn",
        "plain_rnn": "rnn",
        "no_autoencoder": "rnn",
        "no_ae": "rnn",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported model_variant={variant!r}.")
    return aliases[key]


def normalize_tree_type(raw_tree_type: str, tree_size: int) -> str:
    key = str(raw_tree_type).strip().lower()
    aliases = {
        "": "legacy",
        "auto": "legacy",
        "default": "legacy",
        "legacy": "legacy",
        "bandit3": "bandit3",
        "3armed": "bandit3",
        "3_arm": "bandit3",
        "3_armed": "bandit3",
        "3-armed": "bandit3",
        "bandit4": "bandit4",
        "4armed": "bandit4",
        "4_arm": "bandit4",
        "4_armed": "bandit4",
        "4-armed": "bandit4",
        "disjoint2x2": "disjoint2x2",
        "disjoint_2x2": "disjoint2x2",
        "2x2": "disjoint2x2",
        "disjoint3x2": "disjoint3x2",
        "disjoint_3x2": "disjoint3x2",
        "3x2": "disjoint3x2",
    }
    if key == "bandit":
        if tree_size in (3, 4):
            return f"bandit{tree_size}"
        raise ValueError("tree_type='bandit' requires tree_size 3 or 4.")
    if key not in aliases:
        raise ValueError(f"Unsupported tree_type={raw_tree_type!r}.")
    normalized = aliases[key]
    if normalized == "legacy" and tree_size == 3:
        return "bandit3"
    return normalized


def build_task(tree_size: int, tree_type: str, input_type: str) -> TaskSpec:
    tree_type = normalize_tree_type(tree_type, tree_size)
    if tree_type == "bandit3":
        if tree_size != 3:
            raise ValueError("bandit3 requires tree_size=3.")
        path_map = np.eye(3, dtype=np.float32)
    elif tree_type == "bandit4":
        if tree_size != 4:
            raise ValueError("bandit4 requires tree_size=4.")
        path_map = np.eye(4, dtype=np.float32)
    elif tree_type == "disjoint2x2":
        if tree_size != 4:
            raise ValueError("disjoint2x2 requires tree_size=4.")
        path_map = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    elif tree_type == "disjoint3x2":
        if tree_size != 6:
            raise ValueError("disjoint3x2 requires tree_size=6.")
        path_map = np.asarray(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
            dtype=np.float32,
        )
    elif tree_type == "legacy" and tree_size == 2:
        path_map = np.eye(2, dtype=np.float32)
    elif tree_type == "legacy" and tree_size == 6:
        path_map = np.asarray(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"JAX task support currently covers default 2/6, bandit3/4, "
            f"disjoint2x2, and disjoint3x2. Got tree_size={tree_size}, tree_type={tree_type!r}."
        )

    reward_values = np.asarray([0.0, 1.0], dtype=np.float32) if input_type == "binary" else UNIFORM_REWARDS
    tree_name_suffix = "" if tree_type == "legacy" else f"_{tree_type}"
    return TaskSpec(
        tree_size=tree_size,
        tree_type=tree_type,
        tree_name_suffix=tree_name_suffix,
        path_map=path_map,
        reward_values=reward_values,
        reward_norm=expected_max_path_reward(path_map, reward_values),
    )


def expected_max_path_reward(path_map: np.ndarray, reward_values: np.ndarray) -> float:
    total_cases = len(reward_values) ** int(path_map.shape[1])
    if total_cases <= 1_000_000:
        total = 0.0
        for rewards in itertools.product(reward_values, repeat=int(path_map.shape[1])):
            total += float(np.max(path_map @ np.asarray(rewards, dtype=np.float32)))
        return total / float(total_cases)
    rng = np.random.default_rng(0)
    samples = rng.choice(reward_values, size=(2_000_000, int(path_map.shape[1])))
    return float(np.mean(np.max(samples @ path_map.T, axis=1)))


def model_variant_label(variant: str) -> str:
    return f"variant_{variant}_"


def architecture_file_label(rnn_units: int, latent_dim: int) -> str:
    return f"rnn_{rnn_units}_latent_{latent_dim}"


def model_name_for(config: RunConfig, task: TaskSpec) -> str:
    tree_label = f"{config.tree_size}n{task.tree_name_suffix}"
    revisit_label = (
        f"_revisit_maxobs_{config.max_observations_before_stop}"
        if config.allow_node_revisit
        else ""
    )
    sigma_label = (
        f"_obs_sigma_{config.observation_sigma:g}"
        if abs(float(config.observation_sigma)) > 1e-12
        else ""
    )
    kl_schedule_label = (
        f"_klstart_{config.kl_start_multiplier:g}_klanneal_{config.kl_annealing_epochs}"
        if (
            abs(float(config.kl_start_multiplier) - 1.0) > 1e-12
            or int(config.kl_annealing_epochs) > 0
        )
        else ""
    )
    node_coverage_aux_label = (
        f"_nodecov_{config.node_coverage_aux_coef:g}_anneal_{config.node_coverage_aux_epochs}"
        if abs(float(config.node_coverage_aux_coef)) > 1e-12
        else ""
    )
    sampled_lambda_critic_label = (
        "_vcritic"
        if (
            normalize_return_target_mode(config.return_target_mode) == "sampled_lambda"
            and str(config.sampled_lambda_critic).strip().lower() in {"value", "v", "scalar_v"}
        )
        else ""
    )
    stop_paid_label = "_stop_paid" if bool(config.pay_kl_on_stop) else ""
    observer_label = "_observer_endchoice" if bool(config.choice_at_end_only) else ""
    visited_lstm_label = "_visitedidx" if use_visited_lstm_input_for_task(task) else ""
    return (
        f"loss_scale_{config.loss_scale}_alpha_{config.alpha}_lambda_{config.memory_lambda}_"
        f"opportunity_{config.opportunity_cost}_expansion_{config.expansion_decision_version}_"
        f"{model_variant_label(config.model_variant)}"
        f"seed_{config.seed}_{tree_label}_{architecture_file_label(config.rnn_units, config.latent_dim)}"
        f"{revisit_label}"
        f"{sigma_label}"
        f"{kl_schedule_label}"
        f"{node_coverage_aux_label}"
        f"{sampled_lambda_critic_label}"
        f"{stop_paid_label}"
        f"{observer_label}"
        f"{visited_lstm_label}"
    )


def legacy_model_name_for(config: RunConfig, task: TaskSpec) -> str:
    """Older planning checkpoints used lambda=loss-scale and beta=memory weight."""
    tree_label = f"{config.tree_size}n{task.tree_name_suffix}"
    revisit_label = (
        f"_revisit_maxobs_{config.max_observations_before_stop}"
        if config.allow_node_revisit
        else ""
    )
    sigma_label = (
        f"_obs_sigma_{config.observation_sigma:g}"
        if abs(float(config.observation_sigma)) > 1e-12
        else ""
    )
    kl_schedule_label = (
        f"_klstart_{config.kl_start_multiplier:g}_klanneal_{config.kl_annealing_epochs}"
        if (
            abs(float(config.kl_start_multiplier) - 1.0) > 1e-12
            or int(config.kl_annealing_epochs) > 0
        )
        else ""
    )
    node_coverage_aux_label = (
        f"_nodecov_{config.node_coverage_aux_coef:g}_anneal_{config.node_coverage_aux_epochs}"
        if abs(float(config.node_coverage_aux_coef)) > 1e-12
        else ""
    )
    sampled_lambda_critic_label = (
        "_vcritic"
        if (
            normalize_return_target_mode(config.return_target_mode) == "sampled_lambda"
            and str(config.sampled_lambda_critic).strip().lower() in {"value", "v", "scalar_v"}
        )
        else ""
    )
    stop_paid_label = "_stop_paid" if bool(config.pay_kl_on_stop) else ""
    observer_label = "_observer_endchoice" if bool(config.choice_at_end_only) else ""
    visited_lstm_label = "_visitedidx" if use_visited_lstm_input_for_task(task) else ""
    return (
        f"lambda_{config.loss_scale}_alpha_{config.alpha}_beta_{config.memory_lambda}_"
        f"opportunity_{config.opportunity_cost}_expansion_{config.expansion_decision_version}_"
        f"{model_variant_label(config.model_variant)}"
        f"seed_{config.seed}_{tree_label}_{architecture_file_label(config.rnn_units, config.latent_dim)}"
        f"{revisit_label}"
        f"{sigma_label}"
        f"{kl_schedule_label}"
        f"{node_coverage_aux_label}"
        f"{sampled_lambda_critic_label}"
        f"{stop_paid_label}"
        f"{observer_label}"
        f"{visited_lstm_label}"
    )


def scalar_to_category_index(values: jax.Array) -> jax.Array:
    indices = jnp.floor(4.0 - values + 0.5).astype(jnp.int32)
    return jnp.clip(indices, 0, 8)


def scalar_to_onehot(values: jax.Array) -> jax.Array:
    return jax.nn.one_hot(scalar_to_category_index(values), 9, dtype=jnp.float32)


def reward_feature_dim_for_sigma(observation_sigma: float) -> int:
    return 1


def use_visited_lstm_input_for_task(task: TaskSpec) -> bool:
    return str(task.tree_type) == "disjoint3x2"


def visited_lstm_feature_dim_for_task(task: TaskSpec) -> int:
    return task.num_nodes if use_visited_lstm_input_for_task(task) else 0


def infer_reward_feature_dim_from_checkpoint(
    checkpoint_path: Path | str,
    time_steps: int,
    visited_feature_dim: int = 0,
) -> int:
    try:
        restored = serialization.msgpack_restore(Path(checkpoint_path).read_bytes())
        kernel = restored.get("lstm_kernel") if isinstance(restored, dict) else None
        if kernel is None and hasattr(restored, "__getitem__"):
            try:
                kernel = restored["lstm_kernel"]
            except Exception:
                kernel = None
        if kernel is None:
            return 0
        input_dim = int(np.asarray(kernel).shape[0])
        reward_dim = input_dim - int(time_steps) - 1 - int(visited_feature_dim)
        return int(reward_dim) if reward_dim > 0 else 0
    except Exception:
        return 0


def sample_reward_matrix(
    rng: jax.Array,
    num_envs: int,
    num_nodes: int,
    reward_values: np.ndarray,
) -> jax.Array:
    values = jnp.asarray(reward_values, dtype=jnp.float32)
    idx = jax.random.randint(rng, (num_envs, num_nodes), 0, len(reward_values))
    return values[idx]


def sample_reward_sequence(
    rng: jax.Array,
    num_steps: int,
    num_envs: int,
    num_nodes: int,
    reward_values: np.ndarray,
) -> jax.Array:
    values = jnp.asarray(reward_values, dtype=jnp.float32)
    idx = jax.random.randint(rng, (num_steps, num_envs, num_nodes), 0, len(reward_values))
    return values[idx]


def initial_carry(
    num_envs: int,
    task: TaskSpec,
    rnn_units: int,
    reward_feature_dim: int = 1,
    visited_feature_dim: int = 0,
) -> RunnerCarry:
    lstm_input_dim = task.num_nodes + 1 + reward_feature_dim + int(visited_feature_dim)
    return RunnerCarry(
        rewards=jnp.zeros((num_envs, task.num_nodes), dtype=jnp.float32),
        observed_rewards=jnp.zeros((num_envs, task.num_nodes), dtype=jnp.float32),
        observed=jnp.zeros((num_envs, task.num_nodes), dtype=jnp.float32),
        step_index=jnp.zeros((num_envs,), dtype=jnp.int32),
        done=jnp.ones((num_envs,), dtype=jnp.bool_),
        h=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        c=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        decoded_h=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        decoded_c=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        lstm_context=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        pre_context=jnp.zeros(
            (num_envs, rnn_units + lstm_input_dim),
            dtype=jnp.float32,
        ),
        pending_kl=jnp.zeros((num_envs,), dtype=jnp.float32),
        last_reward_onehot=jnp.zeros((num_envs, reward_feature_dim), dtype=jnp.float32),
        trial_id=jnp.zeros((num_envs,), dtype=jnp.int32),
    )


def reset_done_envs(carry: RunnerCarry, reset_rewards: jax.Array) -> RunnerCarry:
    done_f = carry.done.astype(jnp.float32)
    done_2 = done_f[:, None]
    zeros_nodes = jnp.zeros_like(carry.observed)
    zeros_h = jnp.zeros_like(carry.h)
    zeros_pre = jnp.zeros_like(carry.pre_context)
    zeros_reward = jnp.zeros_like(carry.last_reward_onehot)
    return RunnerCarry(
        rewards=jnp.where(done_2 > 0, reset_rewards, carry.rewards),
        observed_rewards=jnp.where(done_2 > 0, zeros_nodes, carry.observed_rewards),
        observed=jnp.where(done_2 > 0, zeros_nodes, carry.observed),
        step_index=jnp.where(carry.done, jnp.zeros_like(carry.step_index), carry.step_index),
        done=jnp.zeros_like(carry.done),
        h=jnp.where(done_2 > 0, zeros_h, carry.h),
        c=jnp.where(done_2 > 0, zeros_h, carry.c),
        decoded_h=jnp.where(done_2 > 0, zeros_h, carry.decoded_h),
        decoded_c=jnp.where(done_2 > 0, zeros_h, carry.decoded_c),
        lstm_context=jnp.where(done_2 > 0, zeros_h, carry.lstm_context),
        pre_context=jnp.where(done_2 > 0, zeros_pre, carry.pre_context),
        pending_kl=jnp.where(carry.done, jnp.zeros_like(carry.pending_kl), carry.pending_kl),
        last_reward_onehot=jnp.where(done_2 > 0, zeros_reward, carry.last_reward_onehot),
        trial_id=carry.trial_id + carry.done.astype(jnp.int32),
    )


def reset_all_envs(carry: RunnerCarry, reset_rewards: jax.Array) -> RunnerCarry:
    zeros_nodes = jnp.zeros_like(carry.observed)
    zeros_h = jnp.zeros_like(carry.h)
    zeros_pre = jnp.zeros_like(carry.pre_context)
    zeros_reward = jnp.zeros_like(carry.last_reward_onehot)
    return RunnerCarry(
        rewards=reset_rewards,
        observed_rewards=zeros_nodes,
        observed=zeros_nodes,
        step_index=jnp.zeros_like(carry.step_index),
        done=jnp.zeros_like(carry.done),
        h=zeros_h,
        c=zeros_h,
        decoded_h=zeros_h,
        decoded_c=zeros_h,
        lstm_context=zeros_h,
        pre_context=zeros_pre,
        pending_kl=jnp.zeros_like(carry.pending_kl),
        last_reward_onehot=zeros_reward,
        trial_id=carry.trial_id + jnp.ones_like(carry.trial_id),
    )


class PlanningVAE(nn.Module):
    rnn_units: int
    latent_dim: int
    time_steps: int
    num_paths: int
    path_map: tuple[tuple[float, ...], ...]
    reward_values: tuple[float, ...]
    reward_norm_value: float
    expansion_decision_version: str
    use_autoencoder: bool
    enable_reconstruction: bool
    enable_probe: bool
    allow_node_revisit: bool
    max_observations_before_stop: int
    opportunity_cost: float
    observation_sigma: float
    loss_scale: float
    alpha: float
    memory_lambda: float
    reward_feature_dim_override: int = 0
    include_visited_lstm_input: bool = False
    latent_perturb_mode: str = "none"
    latent_perturb_timestep: int = -1
    latent_perturb_scale: float = 1.0
    latent_ablate_to_prior: bool = False
    latent_keep_dims: tuple[int, ...] = ()
    lstm_context_pca_mean: tuple[float, ...] = ()
    lstm_context_pca_components: tuple[tuple[float, ...], ...] = ()
    pay_kl_on_stop: bool = False
    choice_at_end_only: bool = False

    def reward_feature_dim(self) -> int:
        if int(self.reward_feature_dim_override) > 0:
            return int(self.reward_feature_dim_override)
        return reward_feature_dim_for_sigma(self.observation_sigma)

    def reward_features(self, values: jax.Array) -> jax.Array:
        if self.reward_feature_dim() == 1:
            return values[..., None].astype(jnp.float32)
        return scalar_to_onehot(values)

    def visited_feature_dim(self) -> int:
        return self.time_steps if bool(self.include_visited_lstm_input) else 0

    def lstm_input_dim(self) -> int:
        return self.time_steps + 1 + self.reward_feature_dim() + self.visited_feature_dim()

    def append_visited_features(self, base_input: jax.Array, visited: jax.Array) -> jax.Array:
        if bool(self.include_visited_lstm_input):
            return jnp.concatenate([base_input, visited.astype(jnp.float32)], axis=-1)
        return base_input

    def setup(self):
        input_dim = self.lstm_input_dim()
        self.lstm_kernel = self.param(
            "lstm_kernel",
            nn.initializers.orthogonal(),
            (input_dim, 4 * self.rnn_units),
        )
        self.lstm_recurrent_kernel = self.param(
            "lstm_recurrent_kernel",
            nn.initializers.orthogonal(),
            (self.rnn_units, 4 * self.rnn_units),
        )
        forget_bias = np.zeros((4 * self.rnn_units,), dtype=np.float32)
        forget_bias[self.rnn_units : 2 * self.rnn_units] = 1.0
        self.lstm_bias = self.param(
            "lstm_bias",
            lambda _rng, shape: jnp.asarray(forget_bias),
            (4 * self.rnn_units,),
        )
        self.enc_dense = nn.Dense(self.rnn_units, kernel_init=nn.initializers.glorot_uniform())
        self.enc_ln = nn.LayerNorm()
        self.z_mean = nn.Dense(self.latent_dim, kernel_init=nn.initializers.glorot_uniform())
        self.z_logvar = nn.Dense(self.latent_dim, kernel_init=nn.initializers.glorot_uniform())
        self.dec_dense = nn.Dense(self.rnn_units, kernel_init=nn.initializers.glorot_uniform())
        self.dec_ln = nn.LayerNorm()
        self.dec_out = nn.Dense(2 * self.rnn_units, kernel_init=nn.initializers.glorot_uniform())
        self.reconstruction_head = nn.Dense(
            self.time_steps * 2,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.zeros,
        )
        self.expansion_head = nn.Dense(
            self.time_steps + self.num_paths,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.critic_dense1 = nn.Dense(self.rnn_units, kernel_init=nn.initializers.glorot_uniform())
        self.critic_ln = nn.LayerNorm()
        self.critic_dense2 = nn.Dense(max(self.rnn_units // 2, 16), kernel_init=nn.initializers.glorot_uniform())
        self.critic_out = nn.Dense(self.time_steps + self.num_paths, kernel_init=nn.initializers.glorot_uniform())
        self.value_dense1 = nn.Dense(self.rnn_units, kernel_init=nn.initializers.glorot_uniform())
        self.value_ln = nn.LayerNorm()
        self.value_dense2 = nn.Dense(max(self.rnn_units // 2, 16), kernel_init=nn.initializers.glorot_uniform())
        self.value_out = nn.Dense(1, kernel_init=nn.initializers.glorot_uniform())
        self.probe_head = nn.Dense(9, kernel_init=nn.initializers.glorot_uniform())
        self.prior_mu = self.param("prior_mu", nn.initializers.zeros, (self.time_steps, self.latent_dim))
        self.prior_logvar = self.param("prior_logvar", nn.initializers.zeros, (self.time_steps, self.latent_dim))

    def encode_stats(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = nn.relu(self.enc_ln(self.enc_dense(x)))
        mu = self.z_mean(h)
        logvar = jnp.clip(self.z_logvar(h), -10.0, 10.0)
        return mu, logvar

    def encode(self, x: jax.Array, rng: jax.Array, use_mean: bool = False):
        mu, logvar = self.encode_stats(x)
        eps = jax.random.normal(rng, mu.shape)
        z = mu if use_mean else mu + jnp.exp(0.5 * logvar) * eps
        return mu, logvar, z

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = nn.relu(self.dec_ln(self.dec_dense(z)))
        out = self.dec_out(h)
        return jnp.split(out, 2, axis=-1)

    def lstm_cell(self, x: jax.Array, h: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array]:
        gates = x @ self.lstm_kernel + h @ self.lstm_recurrent_kernel + self.lstm_bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)
        return new_h, new_c

    def critic(self, x: jax.Array) -> jax.Array:
        y = nn.relu(self.critic_ln(self.critic_dense1(x)))
        y = nn.relu(self.critic_dense2(y))
        return self.critic_out(y)

    def critic_values(self, x: jax.Array) -> jax.Array:
        return self.critic(x)

    def value_critic(self, x: jax.Array) -> jax.Array:
        y = nn.relu(self.value_ln(self.value_dense1(x)))
        y = nn.relu(self.value_dense2(y))
        return self.value_out(y)[:, 0]

    def value_critic_values(self, x: jax.Array) -> jax.Array:
        return self.value_critic(x)

    def expansion_logits_from_input(self, x: jax.Array) -> jax.Array:
        return self.expansion_head(x)

    def counterfactual_next_expansion_input(
        self,
        pre_decoded_h: jax.Array,
        pre_decoded_c: jax.Array,
        sampled_rewards: jax.Array,
        next_observed: jax.Array,
        node_idx: int,
    ) -> jax.Array:
        time_steps = self.time_steps
        step_count, batch_size = pre_decoded_h.shape[:2]
        node_token = jax.nn.one_hot(node_idx, time_steps + 1, dtype=jnp.float32)
        node_tokens = jnp.broadcast_to(
            node_token.reshape((1, 1, time_steps + 1)),
            (step_count, batch_size, time_steps + 1),
        )
        reward_tokens = self.reward_features(sampled_rewards)
        lstm_inputs = self.append_visited_features(
            jnp.concatenate([node_tokens, reward_tokens], axis=-1),
            next_observed,
        )

        flat_inputs = lstm_inputs.reshape((-1, self.lstm_input_dim()))
        flat_h = pre_decoded_h.reshape((-1, self.rnn_units))
        flat_c = pre_decoded_c.reshape((-1, self.rnn_units))
        raw_h, raw_c = self.lstm_cell(flat_inputs, flat_h, flat_c)
        raw_h = raw_h.reshape((step_count, batch_size, self.rnn_units))
        raw_c = raw_c.reshape((step_count, batch_size, self.rnn_units))

        if self.use_autoencoder:
            encoder_input = jnp.concatenate([raw_h, raw_c], axis=-1)
            flat_encoder_input = encoder_input.reshape((-1, 2 * self.rnn_units))
            z_mu, _z_logvar = self.encode_stats(flat_encoder_input)
            dec_h, _dec_c = self.decode(z_mu)
            dec_h = dec_h.reshape((step_count, batch_size, self.rnn_units))
        else:
            dec_h = raw_h

        if self.expansion_decision_version == "decoder":
            return dec_h
        if self.expansion_decision_version == "lstm":
            return raw_h
        return jnp.concatenate([pre_decoded_h, lstm_inputs], axis=-1)

    def reconstruct_probs(self, decoded_h: jax.Array) -> jax.Array:
        params = self.reconstruction_head(decoded_h)
        params = params.reshape((params.shape[0], self.time_steps, 2))
        mu = 5.0 * jnp.tanh(params[:, :, 0:1])
        scale = jax.nn.softplus(params[:, :, 1:2]) + 1e-4
        edges = jnp.asarray(
            [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
            dtype=jnp.float32,
        ).reshape((1, 1, 10))
        cdf = jax.nn.sigmoid((edges - mu) / scale)
        raw = cdf[:, :, 1:] - cdf[:, :, :-1]
        raw = jnp.flip(raw, axis=-1) + 1e-6
        return raw / (jnp.sum(raw, axis=-1, keepdims=True) + 1e-8)

    def prior(self, step_index: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        idx = jnp.clip(step_index, 0, self.time_steps - 1)
        mu = self.prior_mu[idx]
        logvar = self.prior_logvar[idx]
        var = jnp.exp(logvar) + 1e-6
        return mu, logvar, var

    def expansion_input(self, carry: RunnerCarry) -> jax.Array:
        if self.expansion_decision_version == "decoder":
            return carry.decoded_h
        if self.expansion_decision_version == "lstm":
            return carry.lstm_context
        return carry.pre_context

    def __call__(
        self,
        carry: RunnerCarry,
        rng: jax.Array,
        schedule: ScheduleValues,
        forced_action: jax.Array | None = None,
        training: bool = True,
        use_posterior_mean: bool = False,
        compute_targets: bool = True,
        forced_observation: jax.Array | None = None,
    ) -> tuple[RunnerCarry, StepTransition]:
        if len(carry.rewards.shape) != 2 or len(carry.h.shape) != 2 or len(carry.decoded_h.shape) != 2:
            raise ValueError(
                "PlanningVAE expects one batch axis in RunnerCarry, e.g. "
                "rewards [num_envs, num_nodes] and hidden states [num_envs, rnn_units]. "
                f"Got rewards {carry.rewards.shape}, h {carry.h.shape}, decoded_h {carry.decoded_h.shape}. "
                "Use vmap outside the rollout step rather than passing an extra leading axis into model.apply."
            )
        path_map = jnp.asarray(self.path_map, dtype=jnp.float32)
        reward_values = jnp.asarray(self.reward_values, dtype=jnp.float32)
        batch_size = carry.rewards.shape[0]
        active = (~carry.done).astype(jnp.float32)
        active_2 = active[:, None]
        expansion_input = self.expansion_input(carry)
        logits = self.expansion_head(expansion_input)
        if compute_targets:
            q_values = self.critic(expansion_input)
        else:
            q_values = jnp.zeros_like(logits)
        observed_count = jnp.sum(carry.observed, axis=-1, keepdims=True)
        min_observations = 1.0 if self.expansion_decision_version == "decoder" else 0.0
        observation_limit_reached = (
            carry.step_index[:, None] >= int(self.max_observations_before_stop)
        )
        base_observe_invalid = (
            jnp.zeros_like(carry.observed) if self.allow_node_revisit else carry.observed
        )
        observe_invalid = jnp.where(
            observation_limit_reached,
            jnp.ones_like(carry.observed),
            base_observe_invalid,
        )
        legal_observe_count = jnp.sum(1.0 - observe_invalid, axis=-1, keepdims=True)
        no_observe_available = legal_observe_count <= 0.0
        observer_end_reached = observation_limit_reached | no_observe_available
        can_stop = (observed_count >= min_observations).astype(jnp.float32)
        if bool(self.choice_at_end_only):
            can_stop = can_stop * observer_end_reached.astype(jnp.float32)
        terminal_invalid = (1.0 - can_stop) * jnp.ones((batch_size, self.num_paths), dtype=jnp.float32)
        decision_mask = jnp.concatenate([observe_invalid, terminal_invalid], axis=-1)
        legal_mask = (1.0 - decision_mask) * active_2
        masked_logits = logits + decision_mask * -1e9
        probs = jax.nn.softmax(masked_logits, axis=-1)
        log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)

        rng_action, rng_eps, rng_force, rng_z, rng_obs = jax.random.split(rng, 5)
        rng_perturb = jax.random.fold_in(rng_z, 31337)
        sampled_action = jax.random.categorical(rng_action, masked_logits, axis=-1)
        if forced_action is not None:
            action = forced_action.astype(jnp.int32)
        elif training:
            explore = jax.random.uniform(rng_eps, (batch_size,)) < schedule.expansion_epsilon
            uniform_logits = decision_mask * -1e9
            uniform_action = jax.random.categorical(rng_eps, uniform_logits, axis=-1)
            base_legal_observe = (
                jnp.ones_like(carry.observed) if self.allow_node_revisit else (1.0 - carry.observed)
            )
            legal_observe = jnp.where(
                observation_limit_reached,
                jnp.zeros_like(base_legal_observe),
                base_legal_observe,
            )
            observe_logits = (1.0 - legal_observe) * -1e9
            observe_action = jax.random.categorical(rng_force, observe_logits, axis=-1)
            has_observe = jnp.sum(legal_observe, axis=-1) > 0
            force_continue = (
                (self.time_steps > 2)
                & has_observe
                & (jax.random.uniform(rng_force, (batch_size,)) < schedule.forced_continue_epsilon)
            )
            action = jnp.where(explore, uniform_action, sampled_action)
            action = jnp.where(force_continue, observe_action, action)
        else:
            action = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)

        action = jnp.clip(action, 0, self.time_steps + self.num_paths - 1)
        log_prob = jnp.take_along_axis(log_probs_all, action[:, None], axis=-1)[:, 0]
        is_stop_raw = action >= self.time_steps
        is_observe_raw = ~is_stop_raw
        is_stop = is_stop_raw & (active > 0.0)
        is_observe = is_observe_raw & (active > 0.0)
        safe_node = jnp.minimum(action, self.time_steps - 1)
        terminal_path = jnp.where(is_stop, action - self.time_steps, -jnp.ones_like(action))
        chosen_true_reward = jnp.take_along_axis(carry.rewards, safe_node[:, None], axis=1)[:, 0]
        if abs(float(self.observation_sigma)) > 1e-12:
            observation_noise = (
                float(self.observation_sigma)
                * jax.random.normal(rng_obs, chosen_true_reward.shape, dtype=jnp.float32)
            )
            sampled_observed_reward = chosen_true_reward + observation_noise
        else:
            sampled_observed_reward = chosen_true_reward
        if forced_observation is not None:
            chosen_observed_reward = jnp.where(
                jnp.isfinite(forced_observation),
                forced_observation,
                sampled_observed_reward,
            )
        else:
            chosen_observed_reward = sampled_observed_reward
        observe_mask = is_observe[:, None].astype(jnp.float32)
        node_obs = jax.nn.one_hot(safe_node, self.time_steps, dtype=jnp.float32) * observe_mask
        observed_after = jnp.minimum(carry.observed + node_obs, 1.0)
        reward_onehot = self.reward_features(chosen_observed_reward) * observe_mask
        node_token = jnp.where(is_observe, safe_node, self.time_steps)
        node_onehot = jax.nn.one_hot(node_token, self.time_steps + 1, dtype=jnp.float32)
        lstm_input = self.append_visited_features(
            jnp.concatenate([node_onehot, reward_onehot], axis=-1),
            observed_after,
        )

        prev_decoded_h = carry.decoded_h
        prev_decoded_c = carry.decoded_c
        pre_context = jnp.concatenate([prev_decoded_h, lstm_input], axis=-1)
        raw_h, raw_c = self.lstm_cell(lstm_input, carry.decoded_h, carry.decoded_c)
        raw_h = raw_h * observe_mask + carry.h * (1.0 - observe_mask)
        raw_c = raw_c * observe_mask + carry.c * (1.0 - observe_mask)
        if len(self.lstm_context_pca_mean) > 0:
            pca_mean = jnp.asarray(self.lstm_context_pca_mean, dtype=jnp.float32)
            centered_h = raw_h - pca_mean[None, :]
            if len(self.lstm_context_pca_components) > 0:
                pca_components = jnp.asarray(
                    self.lstm_context_pca_components,
                    dtype=jnp.float32,
                )
                raw_h = pca_mean[None, :] + (
                    (centered_h @ pca_components.T) @ pca_components
                )
            else:
                raw_h = jnp.broadcast_to(pca_mean[None, :], raw_h.shape)
        encoder_input = jnp.concatenate([raw_h, raw_c], axis=-1)

        if self.use_autoencoder:
            z_mu, z_logvar, z = self.encode(encoder_input, rng_z, use_mean=use_posterior_mean)
            prior_mu, prior_logvar, prior_var = self.prior(carry.step_index)
            perturb_mode = str(self.latent_perturb_mode).strip().lower()
            if perturb_mode in {"prior_noise", "prior-normalized-noise", "prior_normalized_noise"}:
                prior_sigma = jnp.sqrt(prior_var + 1e-6)
                prior_noise_z = (
                    prior_mu
                    + float(self.latent_perturb_scale)
                    * prior_sigma
                    * jax.random.normal(rng_perturb, z.shape, dtype=jnp.float32)
                )
                perturb_mask = (
                    is_observe
                    & ((carry.step_index + 1) == int(self.latent_perturb_timestep))
                )[:, None]
                z = jnp.where(perturb_mask, prior_noise_z, z)
            if bool(self.latent_ablate_to_prior):
                latent_mask = jnp.zeros((self.latent_dim,), dtype=jnp.float32).at[
                    jnp.asarray(self.latent_keep_dims, dtype=jnp.int32)
                ].set(1.0)
                z = prior_mu + latent_mask[None, :] * (z - prior_mu)
            dec_h, dec_c = self.decode(z)
            dec_h = dec_h * observe_mask + prev_decoded_h * (1.0 - observe_mask)
            dec_c = dec_c * observe_mask + prev_decoded_c * (1.0 - observe_mask)
            post_var = jnp.exp(jnp.clip(z_logvar, -10.0, 10.0))
            kl_per_dim = 0.5 * (
                jnp.log(prior_var + 1e-6)
                - jnp.log(post_var + 1e-6)
                + (post_var + jnp.square(z_mu - prior_mu)) / (prior_var + 1e-6)
                - 1.0
            )
            kl_per_sample = jnp.mean(kl_per_dim, axis=-1)
            observed_kl = kl_per_sample * is_observe.astype(jnp.float32)
            if self.expansion_decision_version in ("lstm", "pre_lstm"):
                pay_pending_kl = is_observe
                if bool(self.pay_kl_on_stop):
                    pay_pending_kl = pay_pending_kl | is_stop
                paid_kl = carry.pending_kl * pay_pending_kl.astype(jnp.float32)
                pending_kl = observed_kl
            else:
                paid_kl = observed_kl
                pending_kl = carry.pending_kl
        else:
            dec_h, dec_c = raw_h, raw_c
            z_mu = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            z_logvar = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            z = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            prior_mu = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            prior_logvar = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            observed_kl = jnp.zeros((batch_size,), dtype=jnp.float32)
            paid_kl = jnp.zeros((batch_size,), dtype=jnp.float32)
            pending_kl = carry.pending_kl

        action_input_post = dec_h if self.expansion_decision_version == "decoder" else raw_h
        if self.expansion_decision_version == "pre_lstm":
            action_input_post = pre_context
        terminal_probs_pre = jax.nn.softmax(logits[:, self.time_steps :], axis=-1)
        terminal_probs_post = jax.nn.softmax(self.expansion_head(action_input_post)[:, self.time_steps :], axis=-1)
        action_output = jnp.where(is_stop[:, None], terminal_probs_pre, terminal_probs_post)

        observed_rewards_after = jnp.where(
            node_obs > 0.0,
            chosen_observed_reward[:, None],
            carry.observed_rewards,
        )
        if self.use_autoencoder and self.enable_reconstruction:
            rec_probs = self.reconstruct_probs(dec_h)
            target_onehot = scalar_to_onehot(carry.rewards)
            rec_ce = -jnp.sum(target_onehot * jnp.log(rec_probs + 1e-8), axis=-1) / jnp.log(9.0)
            reconstruction_loss = (
                jnp.sum(rec_ce * observed_after, axis=-1)
                / (jnp.sum(observed_after, axis=-1) + 1e-6)
            )
        else:
            reconstruction_loss = jnp.zeros((batch_size,), dtype=jnp.float32)

        if self.enable_probe:
            probe_logits = self.probe_head(jax.lax.stop_gradient(raw_h))
            reward_idx = scalar_to_category_index(chosen_observed_reward)
            probe_ce = optax.softmax_cross_entropy_with_integer_labels(probe_logits, reward_idx)
            probe_loss = probe_ce * is_observe.astype(jnp.float32)
            probe_correct = (
                (jnp.argmax(probe_logits, axis=-1) == reward_idx).astype(jnp.float32)
                * is_observe.astype(jnp.float32)
            )
            valid_probe = is_observe.astype(jnp.float32)
        else:
            probe_loss = jnp.zeros((batch_size,), dtype=jnp.float32)
            probe_correct = jnp.zeros((batch_size,), dtype=jnp.float32)
            valid_probe = jnp.zeros((batch_size,), dtype=jnp.float32)

        if compute_targets:
            observed_rewards = carry.observed_rewards[:, None, :] * carry.observed[:, None, :]
            reward_prior_mean = jnp.mean(reward_values)
            belief_values = observed_rewards + (1.0 - carry.observed[:, None, :]) * reward_prior_mean
            stop_targets = jnp.einsum("btn,pn->btp", belief_values, path_map)[:, 0, :] / self.reward_norm_value
            observe_cost = self.opportunity_cost + schedule.current_beta * paid_kl

            node_eye = jnp.eye(self.time_steps, dtype=jnp.float32)
            next_values = (
                belief_values[:, 0, :][None, None, :, :]
                * (1.0 - node_eye[:, None, None, :])
            )
            next_values = next_values + (
                reward_values[None, :, None, None] * node_eye[:, None, None, :]
            )
            next_path = jnp.einsum("nrbm,pm->nrbp", next_values, path_map) / self.reward_norm_value
            observe_targets = jnp.mean(jnp.max(next_path, axis=-1), axis=1).T - observe_cost[:, None]
            q_targets = jnp.concatenate([observe_targets, stop_targets], axis=-1)
            q_targets = q_targets * legal_mask

            selected_q_target = jnp.take_along_axis(q_targets, action[:, None], axis=-1)[:, 0]
            selected_q_pred = jnp.take_along_axis(q_values, action[:, None], axis=-1)[:, 0]
            policy_value_pred = jnp.sum(probs * q_values, axis=-1)
        else:
            q_targets = jnp.zeros_like(logits)
            selected_q_target = jnp.zeros((batch_size,), dtype=jnp.float32)
            selected_q_pred = jnp.zeros((batch_size,), dtype=jnp.float32)
            policy_value_pred = jnp.zeros((batch_size,), dtype=jnp.float32)

        path_rewards = carry.rewards @ path_map.T
        terminal_expected_reward = jnp.sum(action_output * path_rewards, axis=-1) / self.reward_norm_value
        episode_done = carry.done | is_stop
        next_step = carry.step_index + active.astype(jnp.int32)
        next_carry = RunnerCarry(
            rewards=carry.rewards,
            observed_rewards=observed_rewards_after,
            observed=observed_after,
            step_index=next_step,
            done=episode_done,
            h=raw_h,
            c=raw_c,
            decoded_h=dec_h,
            decoded_c=dec_c,
            lstm_context=raw_h,
            pre_context=pre_context,
            pending_kl=pending_kl,
            last_reward_onehot=reward_onehot + carry.last_reward_onehot * (1.0 - observe_mask),
            trial_id=carry.trial_id,
        )

        transition = StepTransition(
            rewards=carry.rewards,
            observed_rewards_before=carry.observed_rewards,
            observed_rewards_after=observed_rewards_after,
            valid=active,
            observed_before=carry.observed,
            observed_after=observed_after,
            step_index=carry.step_index,
            action=action,
            node_index=jnp.where(is_observe, safe_node, -jnp.ones_like(safe_node)),
            terminal_path_index=terminal_path,
            is_stop=is_stop.astype(jnp.float32),
            is_observe=is_observe.astype(jnp.float32),
            log_prob=log_prob,
            entropy=entropy,
            probs=probs,
            legal_mask=legal_mask,
            q_values=q_values,
            q_targets=q_targets,
            selected_q_pred=selected_q_pred,
            selected_q_target=selected_q_target,
            policy_value_pred=policy_value_pred,
            expansion_input=expansion_input,
            pre_decoded_h=prev_decoded_h,
            pre_decoded_c=prev_decoded_c,
            z_mu=z_mu,
            z_logvar=z_logvar,
            z_sample=z,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            paid_kl=paid_kl,
            observed_kl=observed_kl,
            expanded_reward=jnp.where(is_observe, chosen_observed_reward, jnp.nan),
            action_output=action_output,
            terminal_expected_reward=terminal_expected_reward,
            reconstruction_loss=reconstruction_loss,
            probe_loss=probe_loss,
            probe_correct=probe_correct,
            valid_probe=valid_probe,
            reset_rewards=carry.rewards,
            reset_trial_id=carry.trial_id,
        )
        return next_carry, transition


def make_schedule(config: RunConfig, update_idx: int, updates_per_epoch: int) -> ScheduleValues:
    # TensorFlow computes schedules once per epoch and reuses that value for
    # every batch in the epoch. Keep the JAX schedule epoch-discrete too.
    epoch = update_idx // max(updates_per_epoch, 1)
    target_beta = config.memory_lambda
    if int(config.kl_annealing_epochs) > 0:
        epoch_value = jnp.asarray(epoch, dtype=jnp.float32)
        kl_progress = jnp.minimum(
            epoch_value / float(max(int(config.kl_annealing_epochs) - 1, 1)),
            1.0,
        )
        kl_multiplier = (
            float(config.kl_start_multiplier)
            + (1.0 - float(config.kl_start_multiplier)) * kl_progress
        )
    else:
        kl_multiplier = 1.0
    current_beta = target_beta * kl_multiplier
    if config.tree_size == 6:
        target_critic_coef = 1.0
    elif config.tree_size == 30:
        target_critic_coef = 0.0
    else:
        target_critic_coef = 0.1
    if epoch < 80:
        critic_coef = target_critic_coef
    elif epoch >= 200:
        critic_coef = 0.0
    else:
        critic_coef = target_critic_coef * (1.0 - ((epoch - 80) / 120.0))
    if config.tree_size > 2:
        entropy_start, entropy_end, entropy_epochs, entropy_hold = 1.5, 0.0, 60.0, 60.0
    else:
        entropy_start, entropy_end, entropy_epochs, entropy_hold = 1.0, 0.0, 50.0, 100.0
    if epoch >= entropy_hold:
        entropy = 0.0
    elif epoch >= entropy_epochs:
        entropy = entropy_end
    else:
        progress = epoch / max(entropy_epochs - 1.0, 1.0)
        entropy = entropy_start + (entropy_end - entropy_start) * progress
    # Keep the coverage auxiliary pressure active throughout training.  The
    # node_coverage_aux_epochs field is retained for filename/CLI compatibility,
    # but no longer anneals this coefficient.
    node_coverage_aux_coef = float(config.node_coverage_aux_coef)
    return ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(current_beta, dtype=jnp.float32),
        current_critic_coef=jnp.asarray(critic_coef, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(entropy, dtype=jnp.float32),
        node_coverage_aux_coef=jnp.asarray(node_coverage_aux_coef, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )


def learning_rate_at(
    step: jax.Array,
    total_steps: int,
    peak_learning_rate: float = 5e-4,
    min_learning_rate: float | None = None,
) -> jax.Array:
    step = jnp.asarray(step)
    progress = jnp.minimum(step.astype(jnp.float32) / float(max(total_steps, 1)), 1.0)
    peak = jnp.asarray(float(peak_learning_rate), dtype=jnp.float32)
    floor_value = float(min_learning_rate) if min_learning_rate is not None else float(peak_learning_rate) * 0.1
    floor = jnp.asarray(floor_value, dtype=jnp.float32)
    return floor + 0.5 * (peak - floor) * (1.0 + jnp.cos(jnp.pi * progress))


def optimizer_steps_per_update(config: RunConfig) -> int:
    if normalize_return_target_mode(config.return_target_mode) == "sampled_lambda":
        return 1 + max(int(config.update_epochs) - 1, 0) * max(int(config.ppo_minibatches), 1)
    return max(int(config.update_epochs), 1)


def use_sampled_lambda_value_critic(config: RunConfig) -> bool:
    return (
        normalize_return_target_mode(config.return_target_mode) == "sampled_lambda"
        and str(config.sampled_lambda_critic).strip().lower() in {"value", "v", "scalar_v"}
    )


def reduced_state_weights(transitions: StepTransition, time_steps: int) -> jax.Array:
    observed_count = jnp.sum(transitions.observed_before, axis=-1).astype(jnp.int32)
    observed_rewards = (
        transitions.observed_rewards_before[:, :, None, :]
        * transitions.observed_before[:, :, None, :]
    )
    # transitions arrays are [num_steps, num_envs, ...]; path calculation is done outside
    return observed_count


def best_path_value_indices(transitions: StepTransition, path_map: jax.Array) -> tuple[jax.Array, jax.Array]:
    observed_rewards = transitions.observed_rewards_before * transitions.observed_before
    path_values = jnp.einsum("sbn,pn->sbp", observed_rewards, path_map)
    path_counts = jnp.einsum("sbn,pn->sbp", transitions.observed_before, path_map)
    masked_path_values = jnp.where(path_counts > 0.0, path_values, -1e9)
    any_path = jnp.any(path_counts > 0.0, axis=-1)
    best_values = jnp.where(any_path, jnp.max(masked_path_values, axis=-1), 0.0)
    return best_values, scalar_to_category_index(best_values)


def compute_policy_weights(transitions: StepTransition, path_map: jax.Array, time_steps: int) -> jax.Array:
    best_values, best_idx = best_path_value_indices(transitions, path_map)
    observed_count = jnp.sum(transitions.observed_before, axis=-1).astype(jnp.int32)
    step_idx = jnp.clip(transitions.step_index, 0, time_steps - 1)
    bins = (step_idx * 9 + best_idx) * (time_steps + 1) + observed_count
    flat_bins = bins.reshape((-1,))
    flat_valid = transitions.valid.reshape((-1,)).astype(jnp.float32)
    num_bins = time_steps * 9 * (time_steps + 1)
    counts = jnp.bincount(flat_bins, weights=flat_valid, length=num_bins).astype(jnp.float32)
    weights = 1.0 / jnp.maximum(counts[bins], 1.0)
    weights = weights * transitions.valid
    mean_weight = jnp.sum(weights) / (jnp.sum(transitions.valid) + 1e-6)
    weights = weights / (mean_weight + 1e-6)
    weights = jnp.clip(weights, 0.01, 100.0)
    weights = weights * transitions.valid
    clipped_mean = jnp.sum(weights) / (jnp.sum(transitions.valid) + 1e-6)
    weights = weights / (clipped_mean + 1e-6)
    return jax.lax.stop_gradient(weights)


def belief_path_values(
    rewards: jax.Array,
    observed_mask: jax.Array,
    path_map: jax.Array,
    reward_prior_mean: jax.Array,
) -> jax.Array:
    belief_node_values = rewards * observed_mask + (1.0 - observed_mask) * reward_prior_mean
    return jnp.einsum("sbn,pn->sbp", belief_node_values, path_map)


def sampled_lambda_selected_targets(
    transitions: StepTransition,
    target_state_values: jax.Array,
    path_map: jax.Array,
    reward_values: jax.Array,
    reward_norm_value: float,
    opportunity_cost: float,
    current_beta: jax.Array,
    lambda_return: float,
) -> jax.Array:
    step_count, batch_size = transitions.valid.shape
    env_idx = jnp.arange(batch_size)
    stop_flags = transitions.is_stop > 0.0
    has_stop = jnp.any(stop_flags, axis=0)
    first_stop = jnp.argmax(stop_flags.astype(jnp.int32), axis=0)
    selected_step = jnp.where(has_stop, first_stop, step_count - 1)

    terminal_action_probs = transitions.action_output[selected_step, env_idx, :]
    terminal_path_indices = transitions.terminal_path_index[selected_step, env_idx]
    terminal_path_is_sampled = terminal_path_indices >= 0
    sampled_terminal_probs = jax.nn.one_hot(
        jnp.maximum(terminal_path_indices, 0),
        path_map.shape[0],
        dtype=jnp.float32,
    )
    terminal_value_probs = jnp.where(
        terminal_path_is_sampled[:, None],
        sampled_terminal_probs,
        terminal_action_probs,
    )

    # For sampled-lambda PPO, use the expected terminal path value under the
    # agent's observed state.  Unobserved node rewards are replaced by the task
    # prior mean, so the expansion target encourages collecting information
    # rather than learning from hidden full path rewards.
    reward_prior_mean = jnp.mean(reward_values)
    expected_path_values = belief_path_values(
        transitions.observed_rewards_before,
        transitions.observed_before,
        path_map,
        reward_prior_mean,
    )
    terminal_path_values = expected_path_values[selected_step, env_idx, :]
    terminal_reward = (
        jnp.sum(terminal_value_probs * terminal_path_values, axis=-1)
        / float(reward_norm_value)
    )

    non_stop = transitions.valid * (1.0 - transitions.is_stop)
    step_costs = non_stop * (
        float(opportunity_cost) + current_beta * transitions.paid_kl
    )
    lambda_return = jnp.clip(jnp.asarray(lambda_return, dtype=jnp.float32), 0.0, 1.0)

    next_lambda_target = terminal_reward
    reversed_targets = []
    for t in reversed(range(step_count)):
        if t < step_count - 1:
            next_bootstrap = (
                (1.0 - lambda_return) * target_state_values[t + 1]
                + lambda_return * next_lambda_target
            )
        else:
            next_bootstrap = terminal_reward
        continue_target = -step_costs[t] + next_bootstrap
        target_t = jnp.where(transitions.is_stop[t] > 0.0, terminal_reward, continue_target)
        target_t = jnp.where(transitions.valid[t] > 0.0, target_t, jnp.zeros_like(target_t))
        reversed_targets.append(target_t)
        next_lambda_target = target_t
    return jnp.stack(list(reversed(reversed_targets)), axis=0)


def counterfactual_bootstrap_q_targets(
    model: PlanningVAE,
    params,
    target_params,
    transitions: StepTransition,
    rng: jax.Array,
    path_map: jax.Array,
    reward_values: jax.Array,
    reward_norm_value: float,
    opportunity_cost: float,
    current_beta: jax.Array,
    min_observations_before_stop: float,
    allow_node_revisit: bool,
    max_observations_before_stop: int,
    choice_at_end_only: bool,
) -> jax.Array:
    step_count, batch_size, time_steps = transitions.observed_before.shape
    reward_prior_mean = jnp.mean(reward_values)
    stop_targets = belief_path_values(
        transitions.observed_rewards_before,
        transitions.observed_before,
        path_map,
        reward_prior_mean,
    ) / float(reward_norm_value)
    observe_cost = float(opportunity_cost) + current_beta * transitions.paid_kl
    step_ids = jnp.arange(step_count, dtype=jnp.int32)[:, None]
    has_next_decision = (step_ids < (time_steps - 1)).astype(jnp.float32)
    if abs(float(model.observation_sigma)) > 1e-12:
        counterfactual_noise = (
            float(model.observation_sigma)
            * jax.random.normal(
                rng,
                (time_steps, step_count, batch_size),
                dtype=jnp.float32,
            )
        )
    else:
        counterfactual_noise = jnp.zeros(
            (time_steps, step_count, batch_size),
            dtype=jnp.float32,
        )

    observe_targets_by_node = []
    for node_idx in range(time_steps):
        node_onehot = jax.nn.one_hot(node_idx, time_steps, dtype=jnp.float32)
        sampled_node_rewards = transitions.rewards[:, :, node_idx]
        sampled_node_observations = sampled_node_rewards + counterfactual_noise[node_idx]
        next_node_values = (
            transitions.observed_rewards_before * transitions.observed_before
            + (1.0 - transitions.observed_before) * reward_prior_mean
        )
        next_node_values = (
            next_node_values * (1.0 - node_onehot.reshape((1, 1, time_steps)))
            + sampled_node_observations[:, :, None] * node_onehot.reshape((1, 1, time_steps))
        )
        terminal_best_stop = jnp.max(
            jnp.einsum("sbn,pn->sbp", next_node_values, path_map)
            / float(reward_norm_value),
            axis=-1,
        )
        next_observed = jnp.minimum(
            transitions.observed_before + node_onehot.reshape((1, 1, time_steps)),
            1.0,
        )
        next_expansion_input = model.apply(
            {"params": params},
            transitions.pre_decoded_h,
            transitions.pre_decoded_c,
            sampled_node_observations,
            next_observed,
            node_idx,
            method=PlanningVAE.counterfactual_next_expansion_input,
        )
        flat_input = next_expansion_input.reshape((-1, next_expansion_input.shape[-1]))
        target_q_values = model.apply(
            {"params": target_params},
            flat_input,
            method=PlanningVAE.critic_values,
        ).reshape((step_count, batch_size, time_steps + path_map.shape[0]))
        next_logits = model.apply(
            {"params": params},
            flat_input,
            method=PlanningVAE.expansion_logits_from_input,
        ).reshape((step_count, batch_size, time_steps + path_map.shape[0]))

        next_observed_count = jnp.sum(next_observed, axis=-1, keepdims=True)
        next_observation_count = transitions.step_index + 1
        next_observation_limit_reached = (
            next_observation_count[:, :, None] >= int(max_observations_before_stop)
        )
        base_next_observe_invalid = (
            jnp.zeros_like(next_observed) if allow_node_revisit else next_observed
        )
        next_observe_invalid = jnp.where(
            next_observation_limit_reached,
            jnp.ones_like(next_observed),
            base_next_observe_invalid,
        )
        next_legal_observe_count = jnp.sum(1.0 - next_observe_invalid, axis=-1, keepdims=True)
        next_no_observe_available = next_legal_observe_count <= 0.0
        next_observer_end_reached = next_observation_limit_reached | next_no_observe_available
        next_can_stop = (next_observed_count >= float(min_observations_before_stop)).astype(jnp.float32)
        if bool(choice_at_end_only):
            next_can_stop = next_can_stop * next_observer_end_reached.astype(jnp.float32)
        next_terminal_invalid = (
            (1.0 - next_can_stop)
            * jnp.ones((step_count, batch_size, path_map.shape[0]), dtype=jnp.float32)
        )
        next_decision_mask = jnp.concatenate([next_observe_invalid, next_terminal_invalid], axis=-1)
        next_active = transitions.valid[:, :, None] * has_next_decision[:, :, None]
        next_legal_mask = (1.0 - next_decision_mask) * next_active
        next_masked_logits = next_logits + (1.0 - next_legal_mask) * -1e9
        next_probs = jax.nn.softmax(next_masked_logits, axis=-1)
        bootstrap_value = jnp.sum(next_probs * target_q_values, axis=-1)
        next_value = jnp.where(
            (next_active[:, :, 0] > 0.0),
            bootstrap_value,
            terminal_best_stop,
        )
        observe_targets_by_node.append(next_value - observe_cost)

    observe_targets = jnp.stack(observe_targets_by_node, axis=-1)
    q_targets = jnp.concatenate([observe_targets, stop_targets], axis=-1)
    return jax.lax.stop_gradient(q_targets * transitions.legal_mask)


def apply_expansion_return_targets(
    model: PlanningVAE,
    params,
    target_params,
    transitions: StepTransition,
    config: RunConfig,
    task: TaskSpec,
    schedule: ScheduleValues,
    rng: jax.Array,
) -> StepTransition:
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)
    reward_values = jnp.asarray(task.reward_values, dtype=jnp.float32)
    if config.return_target_mode == "one_step":
        return transitions

    if config.return_target_mode == "sampled_lambda":
        flat_input = transitions.expansion_input.reshape((-1, transitions.expansion_input.shape[-1]))
        if use_sampled_lambda_value_critic(config):
            target_state_values = model.apply(
                {"params": target_params},
                flat_input,
                method=PlanningVAE.value_critic_values,
            ).reshape(transitions.valid.shape)
        else:
            target_q_values = model.apply(
                {"params": target_params},
                flat_input,
                method=PlanningVAE.critic_values,
            ).reshape(transitions.q_values.shape)
            target_state_values = jnp.sum(transitions.probs * target_q_values, axis=-1)
        target_state_values = jax.lax.stop_gradient(target_state_values)
        selected_q_target = sampled_lambda_selected_targets(
            transitions,
            target_state_values,
            path_map,
            reward_values,
            task.reward_norm,
            config.opportunity_cost,
            schedule.current_beta,
            config.lambda_return,
        )
        action_onehot = jax.nn.one_hot(
            transitions.action,
            transitions.q_values.shape[-1],
            dtype=jnp.float32,
        )
        q_targets = action_onehot * selected_q_target[:, :, None]
    else:
        flat_input = transitions.expansion_input.reshape((-1, transitions.expansion_input.shape[-1]))
        min_observations = 1.0 if config.expansion_decision_version == "decoder" else 0.0
        q_targets = counterfactual_bootstrap_q_targets(
            model,
            params,
            target_params,
            transitions,
            rng,
            path_map,
            reward_values,
            task.reward_norm,
            config.opportunity_cost,
            schedule.current_beta,
            min_observations,
            config.allow_node_revisit,
            config.max_observations_before_stop,
            config.choice_at_end_only,
        )
        selected_q_target = jnp.take_along_axis(
            q_targets,
            transitions.action[:, :, None],
            axis=-1,
        )[:, :, 0]
    return transitions._replace(
        q_targets=jax.lax.stop_gradient(q_targets),
        selected_q_target=jax.lax.stop_gradient(selected_q_target),
    )


def critic_error_loss(error: jax.Array, huber_delta: float) -> jax.Array:
    """Squared critic loss, optionally Huberized for large target errors."""
    if float(huber_delta) <= 0.0:
        return jnp.square(error)
    delta = jnp.asarray(float(huber_delta), dtype=error.dtype)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * jnp.square(quadratic) + delta * linear


def clip_advantages(advantages: jax.Array, clip_value: float) -> jax.Array:
    """Clip PPO advantages when a positive threshold is configured."""
    if float(clip_value) <= 0.0:
        return advantages
    bound = jnp.asarray(float(clip_value), dtype=advantages.dtype)
    return jnp.clip(advantages, -bound, bound)


def fixed_sampled_lambda_targets(
    model: PlanningVAE,
    target_params,
    transitions: StepTransition,
    config: RunConfig,
    task: TaskSpec,
    schedule: ScheduleValues,
) -> tuple[jax.Array, jax.Array]:
    """Compute PPO data for sampled-lambda once from the collected rollout.

    PPO treats returns and advantages as fixed rollout data.  For
    sampled-lambda, use old-policy probabilities from collection with the slow
    target critic to build the bootstrap values and lambda returns once, then
    reuse those targets across PPO epochs.
    """
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)
    reward_values = jnp.asarray(task.reward_values, dtype=jnp.float32)
    flat_input = transitions.expansion_input.reshape((-1, transitions.expansion_input.shape[-1]))
    if use_sampled_lambda_value_critic(config):
        old_target_state_values = model.apply(
            {"params": target_params},
            flat_input,
            method=PlanningVAE.value_critic_values,
        ).reshape(transitions.valid.shape)
    else:
        target_q_values = model.apply(
            {"params": target_params},
            flat_input,
            method=PlanningVAE.critic_values,
        ).reshape(transitions.q_values.shape)
        old_target_state_values = jnp.sum(transitions.probs * target_q_values, axis=-1)
    old_target_state_values = jax.lax.stop_gradient(old_target_state_values)
    selected_q_target = sampled_lambda_selected_targets(
        transitions,
        old_target_state_values,
        path_map,
        reward_values,
        task.reward_norm,
        config.opportunity_cost,
        schedule.current_beta,
        config.lambda_return,
    )
    advantages = (selected_q_target - old_target_state_values) * transitions.valid
    advantages = clip_advantages(advantages, config.advantage_clip) * transitions.valid
    return jax.lax.stop_gradient(selected_q_target), jax.lax.stop_gradient(advantages)


def attach_fixed_sampled_lambda_targets(
    model: PlanningVAE,
    params,
    transitions: StepTransition,
    fixed_selected_q_target: jax.Array,
    use_value_critic: bool = False,
) -> StepTransition:
    flat_input = transitions.expansion_input.reshape((-1, transitions.expansion_input.shape[-1]))
    if use_value_critic:
        policy_value_pred = model.apply(
            {"params": params},
            flat_input,
            method=PlanningVAE.value_critic_values,
        ).reshape(fixed_selected_q_target.shape)
        selected_q_pred = policy_value_pred
        q_values = transitions.q_values
        q_targets = transitions.q_targets
    else:
        q_values = model.apply(
            {"params": params},
            flat_input,
            method=PlanningVAE.critic_values,
        ).reshape(transitions.q_values.shape)
        action_onehot = jax.nn.one_hot(
            transitions.action,
            q_values.shape[-1],
            dtype=jnp.float32,
        )
        q_targets = action_onehot * fixed_selected_q_target[:, :, None]
        selected_q_pred = jnp.take_along_axis(
            q_values,
            transitions.action[:, :, None],
            axis=-1,
        )[:, :, 0]
        policy_value_pred = jnp.sum(transitions.probs * q_values, axis=-1)
    return transitions._replace(
        q_values=q_values,
        q_targets=jax.lax.stop_gradient(q_targets),
        selected_q_pred=selected_q_pred,
        selected_q_target=jax.lax.stop_gradient(fixed_selected_q_target),
        policy_value_pred=policy_value_pred,
    )


def flatten_rollout_env_trajectories(x: jax.Array) -> jax.Array:
    """Convert [rollout, timestep, env, ...] to [rollout * env, timestep, ...]."""
    x = jnp.swapaxes(x, 1, 2)
    return x.reshape((x.shape[0] * x.shape[1], x.shape[2]) + x.shape[3:])


def repeat_carry_for_rollouts(carry: RunnerCarry, n_rollouts: int) -> RunnerCarry:
    """Repeat an env-batched carry into one [rollout * env, ...] batch."""
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None, ...], (n_rollouts,) + x.shape).reshape(
            (n_rollouts * x.shape[0],) + x.shape[1:]
        ),
        carry,
    )


def first_rollout_carry(carry: RunnerCarry, num_envs: int) -> RunnerCarry:
    """Take the first rollout slice from a flattened [rollout * env, ...] carry."""
    return jax.tree_util.tree_map(lambda x: x[:num_envs], carry)


def expand_rewards_for_rollouts(reset_rewards: jax.Array, n_rollouts: int) -> jax.Array:
    """Repeat [env, node] rewards into [rollout * env, node]."""
    return jnp.broadcast_to(
        reset_rewards[None, ...],
        (n_rollouts,) + reset_rewards.shape,
    ).reshape((n_rollouts * reset_rewards.shape[0],) + reset_rewards.shape[1:])


def unflatten_time_batch_rollouts(x: jax.Array, n_rollouts: int, num_envs: int) -> jax.Array:
    """Convert [timestep, rollout * env, ...] to [rollout, timestep, env, ...]."""
    x = x.reshape((x.shape[0], n_rollouts, num_envs) + x.shape[2:])
    return jnp.swapaxes(x, 0, 1)


def flatten_rollout_major_time_batch(x: jax.Array) -> jax.Array:
    """Convert [rollout, timestep, env, ...] to [timestep, rollout * env, ...]."""
    x = jnp.swapaxes(x, 0, 1)
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def node_coverage_auxiliary_targets(
    transitions: StepTransition,
    time_steps: int,
) -> tuple[jax.Array, jax.Array]:
    legal_observe = transitions.legal_mask[..., :time_steps]
    unobserved_legal = (1.0 - transitions.observed_before) * legal_observe
    target_count = jnp.sum(unobserved_legal, axis=-1, keepdims=True)
    target = unobserved_legal / (target_count + 1e-6)
    mask = transitions.valid * (target_count[..., 0] > 0.0).astype(jnp.float32)
    return target, mask


def node_coverage_auxiliary_loss(
    probs: jax.Array,
    coverage_target: jax.Array,
    coverage_mask: jax.Array,
) -> jax.Array:
    observe_probs = probs[..., : coverage_target.shape[-1]]
    coverage_ce = -jnp.sum(
        coverage_target * jnp.log(observe_probs + 1e-8),
        axis=-1,
    )
    return jnp.sum(coverage_ce * coverage_mask) / (jnp.sum(coverage_mask) + 1e-6)


def make_cached_expansion_ppo_data(
    transitions: StepTransition,
    old_logp: jax.Array,
    fixed_advantage: jax.Array,
    policy_weights: jax.Array,
    entropy_mask: jax.Array,
) -> CachedExpansionPPOData:
    return CachedExpansionPPOData(
        expansion_input=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(transitions.expansion_input)
        ),
        legal_mask=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(transitions.legal_mask)
        ),
        action=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(transitions.action)
        ),
        old_logp=jax.lax.stop_gradient(flatten_rollout_env_trajectories(old_logp)),
        advantage=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(fixed_advantage)
        ),
        weights=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(policy_weights)
        ),
        valid=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(transitions.valid)
        ),
        entropy_mask=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(entropy_mask)
        ),
        coverage_target=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(
                node_coverage_auxiliary_targets(
                    transitions,
                    transitions.observed_before.shape[-1],
                )[0]
            )
        ),
        coverage_mask=jax.lax.stop_gradient(
            flatten_rollout_env_trajectories(
                node_coverage_auxiliary_targets(
                    transitions,
                    transitions.observed_before.shape[-1],
                )[1]
            )
        ),
    )


def cached_expansion_ppo_loss(
    model: PlanningVAE,
    params,
    batch: CachedExpansionPPOData,
    schedule: ScheduleValues,
    loss_scale: float,
) -> jax.Array:
    flat_input = batch.expansion_input.reshape((-1, batch.expansion_input.shape[-1]))
    logits = model.apply(
        {"params": params},
        flat_input,
        method=PlanningVAE.expansion_logits_from_input,
    ).reshape(batch.legal_mask.shape)
    masked_logits = logits + (1.0 - batch.legal_mask) * -1e9
    log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    log_prob = jnp.take_along_axis(
        log_probs_all,
        batch.action[:, :, None],
        axis=-1,
    )[:, :, 0]
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
    ratio = jnp.exp(jnp.clip(log_prob - batch.old_logp, -10.0, 10.0))
    clipped_ratio = jnp.clip(ratio, 1.0 - schedule.ppo_clip, 1.0 + schedule.ppo_clip)
    policy_loss = -jnp.minimum(
        ratio * batch.advantage,
        clipped_ratio * batch.advantage,
    ) * batch.weights
    valid_policy_count = jnp.sum(batch.valid) + 1e-6
    entropy_loss = (
        jnp.sum(entropy * batch.entropy_mask)
        / (jnp.sum(batch.entropy_mask) + 1e-6)
    )
    expansion_loss = (
        jnp.sum(policy_loss) / valid_policy_count
        - schedule.expansion_entropy_coef * entropy_loss
    )
    coverage_aux_loss = node_coverage_auxiliary_loss(
        probs,
        batch.coverage_target,
        batch.coverage_mask,
    )
    return (
        expansion_loss * float(loss_scale)
        + schedule.node_coverage_aux_coef * float(loss_scale) * coverage_aux_loss
    )


def expansion_entropy_mask(transitions: StepTransition, expansion_decision_version: str) -> jax.Array:
    if expansion_decision_version == "decoder":
        return transitions.valid
    has_observation = jnp.sum(transitions.observed_before, axis=-1) > 0.0
    return has_observation.astype(jnp.float32) * transitions.valid


def tf_style_terminal_action_loss(
    transitions: StepTransition,
    path_map: jax.Array,
    reward_norm_value: float,
) -> jax.Array:
    """Match TensorFlow's final path-choice loss for one rollout segment.

    TensorFlow gathers the terminal path distribution from the first stop
    decision, or from the final timestep if the trajectory never explicitly
    stops.  In revisit mode the final unique-node observation is not terminal;
    the terminal event is an explicit stop or the final rollout step.
    """
    final_rollout_step = jax.nn.one_hot(
        transitions.is_stop.shape[0] - 1,
        transitions.is_stop.shape[0],
        dtype=jnp.float32,
    )[:, None]
    terminal_flags = (transitions.is_stop > 0.0) | (
        (final_rollout_step > 0.0) & (transitions.valid > 0.0)
    )
    has_terminal = jnp.any(terminal_flags, axis=0)
    first_terminal = jnp.argmax(terminal_flags.astype(jnp.int32), axis=0)
    last_step = transitions.is_stop.shape[0] - 1
    selected_step = jnp.where(has_terminal, first_terminal, last_step)
    env_idx = jnp.arange(transitions.is_stop.shape[1])
    terminal_probs = transitions.action_output[selected_step, env_idx, :]
    rewards = transitions.rewards[selected_step, env_idx, :]
    path_rewards = rewards @ path_map.T
    expected_reward = jnp.sum(terminal_probs * path_rewards, axis=-1) / reward_norm_value
    return 1.0 - jnp.mean(expected_reward)


def aggregate_best_value_metrics(
    transitions: StepTransition,
    path_map: jax.Array,
    time_steps: int,
) -> tuple[jax.Array, ...]:
    _best_values, best_idx = best_path_value_indices(transitions, path_map)
    category = jax.nn.one_hot(best_idx, 9, dtype=jnp.float32)
    step_onehot = jax.nn.one_hot(jnp.clip(transitions.step_index, 0, time_steps - 1), time_steps, dtype=jnp.float32)
    masks = (
        step_onehot[:, :, :, None]
        * category[:, :, None, :]
        * transitions.valid[:, :, None, None]
    )
    counts = jnp.sum(masks, axis=(0, 1))
    continue_sums = jnp.sum(masks * transitions.is_observe[:, :, None, None], axis=(0, 1))
    critic_sums = jnp.sum(masks * transitions.selected_q_pred[:, :, None, None], axis=(0, 1))
    target_sums = jnp.sum(masks * transitions.selected_q_target[:, :, None, None], axis=(0, 1))
    advantage_sums = jnp.sum(
        masks * (transitions.selected_q_target - transitions.selected_q_pred)[:, :, None, None],
        axis=(0, 1),
    )
    kl_sums = jnp.sum(masks * transitions.paid_kl[:, :, None, None], axis=(0, 1))
    legal_observe = transitions.legal_mask[:, :, :time_steps]
    legal_stop = transitions.legal_mask[:, :, time_steps:]
    q_observe = transitions.q_values[:, :, :time_steps]
    q_stop = transitions.q_values[:, :, time_steps:]
    p_observe = transitions.probs[:, :, :time_steps]
    p_stop = transitions.probs[:, :, time_steps:]
    q_observe_max = jnp.max(jnp.where(legal_observe > 0, q_observe, -1e9), axis=-1)
    q_stop_max = jnp.max(jnp.where(legal_stop > 0, q_stop, -1e9), axis=-1)
    q_action = jnp.argmax(jnp.where(transitions.legal_mask > 0, transitions.q_values, -1e9), axis=-1)
    policy_action = jnp.argmax(transitions.probs, axis=-1)
    q_stop_sums = jnp.sum(masks * q_stop_max[:, :, None, None], axis=(0, 1))
    q_observe_sums = jnp.sum(masks * q_observe_max[:, :, None, None], axis=(0, 1))
    q_diff_sums = jnp.sum(masks * (q_stop_max - q_observe_max)[:, :, None, None], axis=(0, 1))
    policy_stop_sums = jnp.sum(masks * jnp.sum(p_stop * legal_stop, axis=-1)[:, :, None, None], axis=(0, 1))
    policy_observe_sums = jnp.sum(masks * jnp.sum(p_observe * legal_observe, axis=-1)[:, :, None, None], axis=(0, 1))
    q_argmax_stop_sums = jnp.sum(masks * (q_action >= time_steps).astype(jnp.float32)[:, :, None, None], axis=(0, 1))
    policy_argmax_stop_sums = jnp.sum(masks * (policy_action >= time_steps).astype(jnp.float32)[:, :, None, None], axis=(0, 1))
    return (
        continue_sums,
        counts,
        critic_sums,
        target_sums,
        advantage_sums,
        kl_sums,
        q_stop_sums,
        q_observe_sums,
        q_diff_sums,
        policy_stop_sums,
        policy_observe_sums,
        q_argmax_stop_sums,
        policy_argmax_stop_sums,
    )


def aggregate_observation_coverage_metrics(
    transitions: StepTransition,
    path_map: jax.Array,
) -> tuple[jax.Array, ...]:
    """Coverage of nodes/paths reached during the training rollout chunk."""
    valid = transitions.valid.astype(jnp.float32)
    valid_count = jnp.sum(valid) + 1e-6
    observed_before = transitions.observed_before.astype(jnp.float32)
    observed_after = transitions.observed_after.astype(jnp.float32)
    num_nodes = observed_before.shape[-1]
    num_paths = path_map.shape[0]

    decision_unique_nodes = jnp.sum(observed_before, axis=-1)
    decision_path_seen = (jnp.einsum("sbn,pn->sbp", observed_before, path_map) > 0.0).astype(jnp.float32)
    decision_unique_paths = jnp.sum(decision_path_seen, axis=-1)
    decision_mean_unique_nodes = jnp.sum(decision_unique_nodes * valid) / valid_count
    decision_all_nodes_rate = (
        jnp.sum((decision_unique_nodes >= float(num_nodes)).astype(jnp.float32) * valid)
        / valid_count
    )
    decision_mean_unique_paths = jnp.sum(decision_unique_paths * valid) / valid_count
    decision_all_paths_rate = (
        jnp.sum((decision_unique_paths >= float(num_paths)).astype(jnp.float32) * valid)
        / valid_count
    )

    # For each environment, take the last valid transition in this rollout
    # chunk. This is a cheap proxy for end-of-chunk coverage during training.
    valid_steps = jnp.sum(valid, axis=0).astype(jnp.int32)
    has_valid = (valid_steps > 0).astype(jnp.float32)
    final_idx = jnp.maximum(valid_steps - 1, 0)
    env_idx = jnp.arange(valid.shape[1])
    final_observed = observed_after[final_idx, env_idx, :]
    final_unique_nodes = jnp.sum(final_observed, axis=-1)
    final_path_seen = (jnp.einsum("bn,pn->bp", final_observed, path_map) > 0.0).astype(jnp.float32)
    final_unique_paths = jnp.sum(final_path_seen, axis=-1)
    final_count = jnp.sum(has_valid) + 1e-6
    rollout_final_mean_unique_nodes = jnp.sum(final_unique_nodes * has_valid) / final_count
    rollout_final_all_nodes_rate = (
        jnp.sum((final_unique_nodes >= float(num_nodes)).astype(jnp.float32) * has_valid)
        / final_count
    )
    rollout_final_mean_unique_paths = jnp.sum(final_unique_paths * has_valid) / final_count
    rollout_final_all_paths_rate = (
        jnp.sum((final_unique_paths >= float(num_paths)).astype(jnp.float32) * has_valid)
        / final_count
    )
    return (
        decision_mean_unique_nodes,
        decision_all_nodes_rate,
        decision_mean_unique_paths,
        decision_all_paths_rate,
        rollout_final_mean_unique_nodes,
        rollout_final_all_nodes_rate,
        rollout_final_mean_unique_paths,
        rollout_final_all_paths_rate,
    )


def build_rollout_fn(model: PlanningVAE, task: TaskSpec, config: RunConfig):
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)

    def rollout(
        params,
        carry: RunnerCarry,
        rng: jax.Array,
        schedule: ScheduleValues,
        forced_actions=None,
        forced_observations=None,
        reset_rewards=None,
        training: bool = True,
        compute_targets: bool = True,
    ):
        if reset_rewards is None:
            rng, reset_rng = jax.random.split(rng)
            reset_rewards = sample_reward_matrix(reset_rng, config.num_envs, task.num_nodes, task.reward_values)
        carry = reset_all_envs(carry, reset_rewards)

        def scan_step(scan_carry, step_i):
            step_carry, step_rng = scan_carry
            step_rng, model_rng = jax.random.split(step_rng)
            forced_action = None if forced_actions is None else forced_actions[step_i]
            forced_observation = None if forced_observations is None else forced_observations[step_i]
            next_carry, transition = model.apply(
                {"params": params},
                step_carry,
                model_rng,
                schedule,
                forced_action,
                training,
                compute_targets=compute_targets,
                forced_observation=forced_observation,
                method=PlanningVAE.__call__,
            )
            transition = transition._replace(reset_rewards=reset_rewards, reset_trial_id=step_carry.trial_id)
            return (next_carry, step_rng), transition

        (new_carry, new_rng), transitions = jax.lax.scan(
            scan_step,
            (carry, rng),
            jnp.arange(config.num_steps),
        )
        return new_carry, new_rng, transitions

    return rollout


def create_train_state(model: PlanningVAE, config: RunConfig, task: TaskSpec, rng: jax.Array, total_updates: int):
    reward_feature_dim = reward_feature_dim_for_sigma(config.observation_sigma)
    dummy_carry = initial_carry(
        config.num_envs,
        task,
        config.rnn_units,
        reward_feature_dim,
        visited_lstm_feature_dim_for_task(task),
    )
    dummy_schedule = ScheduleValues(
        current_alpha=1.0,
        current_beta=config.memory_lambda,
        current_critic_coef=0.1,
        expansion_epsilon=0.0,
        expansion_entropy_coef=1.0,
        node_coverage_aux_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    params = model.init(rng, dummy_carry, rng, dummy_schedule, None, True)["params"]
    if use_sampled_lambda_value_critic(config):
        dummy_expansion_input = model.apply(
            {"params": params},
            dummy_carry,
            method=PlanningVAE.expansion_input,
        )
        value_params = model.init(
            rng,
            dummy_expansion_input,
            method=PlanningVAE.value_critic_values,
        )["params"]
        params = merge_missing_param_subtrees(params, value_params)
    schedule = lambda step: learning_rate_at(
        step,
        total_updates * optimizer_steps_per_update(config),
        config.learning_rate,
        config.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )
    return PlanningTrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params=params,
        tx=tx,
    )


def top_level_param_mask(params, train_keys: set[str]):
    def mark(path, _leaf):
        top_key = getattr(path[0], "key", None) if path else None
        return top_key in train_keys

    return jax.tree_util.tree_map_with_path(mark, params)


def replace_param_subtree(params, key: str, subtree):
    if hasattr(params, "copy"):
        try:
            return params.copy({key: subtree})
        except TypeError:
            pass
    mutable = dict(params)
    mutable[key] = subtree
    return mutable


def merge_missing_param_subtrees(params, extra_params):
    additions = {
        key: extra_params[key]
        for key in extra_params.keys()
        if key not in params
    }
    if not additions:
        return params
    if hasattr(params, "copy"):
        try:
            return params.copy(additions)
        except TypeError:
            pass
    merged = dict(params)
    merged.update(additions)
    return merged


def zeros_like_param_tree(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    if type(zeros) is not type(params) and hasattr(params, "copy"):
        return params.copy({key: zeros[key] for key in params.keys()})
    return zeros


def merge_opt_state_by_param_mask(old_state, new_state, param_mask):
    """Keep optimizer slots for frozen params unchanged, but advance scalars.

    This makes an expansion-head-only update behave like TensorFlow's
    ``optimizer.apply_gradients`` call with only expansion-head variables:
    global optimizer counters advance, expansion slots update, and non-expansion
    variable slots are left untouched.
    """
    if jax.tree_util.tree_structure(old_state) == jax.tree_util.tree_structure(param_mask):
        return jax.tree_util.tree_map(
            lambda keep_new, old, new: jnp.where(keep_new, new, old),
            param_mask,
            old_state,
            new_state,
        )
    if isinstance(old_state, tuple) and hasattr(old_state, "_fields"):
        return type(old_state)(
            *(
                merge_opt_state_by_param_mask(
                    getattr(old_state, field),
                    getattr(new_state, field),
                    param_mask,
                )
                for field in old_state._fields
            )
        )
    if isinstance(old_state, tuple):
        return tuple(
            merge_opt_state_by_param_mask(old_item, new_item, param_mask)
            for old_item, new_item in zip(old_state, new_state)
        )
    if isinstance(old_state, list):
        return [
            merge_opt_state_by_param_mask(old_item, new_item, param_mask)
            for old_item, new_item in zip(old_state, new_state)
        ]
    return new_state


def apply_masked_gradients(train_state: PlanningTrainState, grads, param_mask):
    updates, new_opt_state = train_state.tx.update(
        grads,
        train_state.opt_state,
        train_state.params,
    )
    masked_updates = jax.tree_util.tree_map(
        lambda keep, update: jnp.where(keep, update, jnp.zeros_like(update)),
        param_mask,
        updates,
    )
    new_params = optax.apply_updates(train_state.params, masked_updates)
    new_opt_state = merge_opt_state_by_param_mask(
        train_state.opt_state,
        new_opt_state,
        param_mask,
    )
    return train_state.replace(
        step=train_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )


def soft_update_tree(source, target, tau: float):
    tau_value = jnp.asarray(tau, dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda src, tgt: tau_value * src + (1.0 - tau_value) * tgt,
        source,
        target,
    )


def build_update_fn(model: PlanningVAE, task: TaskSpec, config: RunConfig, total_updates: int):
    rollout = build_rollout_fn(model, task, config)
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)
    backend = config.backend

    def update_step(train_state: PlanningTrainState, carry: RunnerCarry, rng: jax.Array, schedule: ScheduleValues):
        rng, reset_rng, rollout_rng, replay_rng = jax.random.split(rng, 4)
        reset_rewards = sample_reward_matrix(
            reset_rng,
            config.num_envs,
            task.num_nodes,
            task.reward_values,
        )
        rollout_keys = jax.random.split(rollout_rng, config.return_target_rollouts)
        replay_keys = jax.random.split(replay_rng, config.return_target_rollouts)

        if config.return_target_mode == "sampled_lambda":
            flat_carry = repeat_carry_for_rollouts(carry, config.return_target_rollouts)
            flat_reset_rewards = expand_rewards_for_rollouts(
                reset_rewards,
                config.return_target_rollouts,
            )
            flat_collect_carry, _rollout_rng, flat_old_transitions = rollout(
                train_state.params,
                flat_carry,
                rollout_rng,
                schedule,
                forced_actions=None,
                reset_rewards=flat_reset_rewards,
                training=True,
                compute_targets=False,
            )
            collect_carry = first_rollout_carry(flat_collect_carry, config.num_envs)
            old_transitions = jax.tree_util.tree_map(
                lambda x: unflatten_time_batch_rollouts(
                    x,
                    config.return_target_rollouts,
                    config.num_envs,
                ),
                flat_old_transitions,
            )
            old_logp = jax.lax.stop_gradient(old_transitions.log_prob)
            fixed_selected_q_target_flat, fixed_advantage_flat = fixed_sampled_lambda_targets(
                model,
                train_state.target_params,
                flat_old_transitions,
                config,
                task,
                schedule,
            )
            fixed_selected_q_target = jax.lax.stop_gradient(
                unflatten_time_batch_rollouts(
                    fixed_selected_q_target_flat,
                    config.return_target_rollouts,
                    config.num_envs,
                )
            )
            fixed_advantage = jax.lax.stop_gradient(
                unflatten_time_batch_rollouts(
                    fixed_advantage_flat,
                    config.return_target_rollouts,
                    config.num_envs,
                )
            )
            policy_weights = jax.lax.stop_gradient(
                unflatten_time_batch_rollouts(
                    compute_policy_weights(
                        flat_old_transitions,
                        path_map,
                        task.num_nodes,
                    ),
                    config.return_target_rollouts,
                    config.num_envs,
                )
            )
            cached_expansion_data = make_cached_expansion_ppo_data(
                old_transitions,
                old_logp,
                fixed_advantage,
                policy_weights,
                expansion_entropy_mask(
                    old_transitions,
                    config.expansion_decision_version,
                ),
            )
            forced_actions = jax.lax.stop_gradient(flat_old_transitions.action)
            forced_observations = jax.lax.stop_gradient(flat_old_transitions.expanded_reward)
            old_logp_flat = jax.lax.stop_gradient(flat_old_transitions.log_prob)
        else:
            def collect_one(rollout_key):
                rollout_carry, _rollout_rng, transitions = rollout(
                    train_state.params,
                    carry,
                    rollout_key,
                    schedule,
                    forced_actions=None,
                    reset_rewards=reset_rewards,
                    training=True,
                    compute_targets=False,
                )
                return rollout_carry, transitions

            rollout_carries, old_transitions = jax.vmap(collect_one)(rollout_keys)
            collect_carry = jax.tree_util.tree_map(lambda x: x[0], rollout_carries)
            forced_actions = jax.lax.stop_gradient(old_transitions.action)
            forced_observations = jax.lax.stop_gradient(old_transitions.expanded_reward)
            old_logp = jax.lax.stop_gradient(old_transitions.log_prob)
            fixed_selected_q_target = jnp.zeros_like(old_logp)
            fixed_advantage = jnp.zeros_like(old_logp)

        def loss_from_transitions(
            transitions: StepTransition,
            rollout_old_logp: jax.Array,
            advantages: jax.Array,
        ):
            advantages = clip_advantages(advantages, config.advantage_clip) * transitions.valid
            weights = compute_policy_weights(transitions, path_map, task.num_nodes)
            ratio = jnp.exp(jnp.clip(transitions.log_prob - rollout_old_logp, -10.0, 10.0))
            clipped_ratio = jnp.clip(ratio, 1.0 - schedule.ppo_clip, 1.0 + schedule.ppo_clip)
            policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages) * weights
            entropy_mask = expansion_entropy_mask(
                transitions,
                config.expansion_decision_version,
            )
            entropy_loss = (
                jnp.sum(transitions.entropy * entropy_mask)
                / (jnp.sum(entropy_mask) + 1e-6)
            )
            valid_policy_count = jnp.sum(transitions.valid) + 1e-6
            expansion_loss = (
                jnp.sum(policy_loss) / valid_policy_count
                - schedule.expansion_entropy_coef * entropy_loss
            )
            if config.return_target_mode == "sampled_lambda":
                if use_sampled_lambda_value_critic(config):
                    critic_err = (
                        critic_error_loss(
                            transitions.policy_value_pred - transitions.selected_q_target,
                            config.critic_huber_delta,
                        )
                        * transitions.valid
                    )
                    critic_loss = jnp.sum(critic_err) / (jnp.sum(transitions.valid) + 1e-6)
                else:
                    critic_mask = (
                        jax.nn.one_hot(
                            transitions.action,
                            transitions.q_values.shape[-1],
                            dtype=jnp.float32,
                        )
                        * transitions.valid[:, :, None]
                    )
                    critic_err = (
                        critic_error_loss(
                            transitions.q_values - transitions.q_targets,
                            config.critic_huber_delta,
                        )
                        * critic_mask
                    )
                    critic_loss = jnp.sum(critic_err) / (jnp.sum(critic_mask) + 1e-6)
            else:
                critic_mask = transitions.legal_mask
                critic_err = (
                    critic_error_loss(
                        transitions.q_values - transitions.q_targets,
                        config.critic_huber_delta,
                    )
                    * critic_mask
                )
                critic_loss = jnp.sum(critic_err) / (jnp.sum(critic_mask) + 1e-6)
            information_loss = jnp.mean(transitions.paid_kl)
            reconstruction_loss = jnp.mean(transitions.reconstruction_loss)
            action_loss = tf_style_terminal_action_loss(
                transitions,
                path_map,
                float(task.reward_norm),
            )
            probe_loss = jnp.sum(transitions.probe_loss) / (jnp.sum(transitions.valid_probe) + 1e-6)
            coverage_target, coverage_mask = node_coverage_auxiliary_targets(
                transitions,
                task.num_nodes,
            )
            node_coverage_aux_loss = node_coverage_auxiliary_loss(
                transitions.probs,
                coverage_target,
                coverage_mask,
            )
            total_loss = (
                information_loss * schedule.current_beta
                + action_loss * config.loss_scale
                + expansion_loss * config.loss_scale
                + critic_loss * config.loss_scale * schedule.current_critic_coef
                + reconstruction_loss * config.alpha
                + probe_loss
            )
            expansion_head_loss = (
                expansion_loss * config.loss_scale
                + critic_loss * config.loss_scale * schedule.current_critic_coef
                + action_loss * config.loss_scale
                + schedule.node_coverage_aux_coef * config.loss_scale * node_coverage_aux_loss
            )
            metrics_parts = aggregate_best_value_metrics(transitions, path_map, task.num_nodes)
            coverage_metrics = aggregate_observation_coverage_metrics(transitions, path_map)
            probe_acc = jnp.sum(transitions.probe_correct) / (jnp.sum(transitions.valid_probe) + 1e-6)
            valid_count = jnp.sum(transitions.valid) + 1e-6
            metrics = UpdateMetrics(
                total_loss=total_loss,
                information_loss=information_loss,
                action_loss=action_loss,
                reconstruction_loss=reconstruction_loss,
                expansion_loss=expansion_loss,
                node_coverage_aux_loss=node_coverage_aux_loss,
                critic_loss=critic_loss,
                lstm_probe_loss=probe_loss,
                lstm_probe_accuracy=probe_acc,
                stop_rate=jnp.sum(transitions.is_stop) / valid_count,
                continue_rate=jnp.sum(transitions.is_observe) / valid_count,
                decision_mean_unique_nodes=coverage_metrics[0],
                decision_all_nodes_rate=coverage_metrics[1],
                decision_mean_unique_paths=coverage_metrics[2],
                decision_all_paths_rate=coverage_metrics[3],
                rollout_final_mean_unique_nodes=coverage_metrics[4],
                rollout_final_all_nodes_rate=coverage_metrics[5],
                rollout_final_mean_unique_paths=coverage_metrics[6],
                rollout_final_all_paths_rate=coverage_metrics[7],
                entropy_coef=jnp.asarray(schedule.expansion_entropy_coef),
                node_coverage_aux_coef=jnp.asarray(schedule.node_coverage_aux_coef),
                critic_coef=jnp.asarray(schedule.current_critic_coef),
                current_beta=jnp.asarray(schedule.current_beta),
                learning_rate=learning_rate_at(
                    train_state.step,
                    total_updates * optimizer_steps_per_update(config),
                    config.learning_rate,
                    config.min_learning_rate,
                ),
                continue_best_sums=metrics_parts[0],
                continue_best_counts=metrics_parts[1],
                critic_best_sums=metrics_parts[2],
                target_best_sums=metrics_parts[3],
                advantage_best_sums=metrics_parts[4],
                kl_best_sums=metrics_parts[5],
                q_stop_best_sums=metrics_parts[6],
                q_observe_best_sums=metrics_parts[7],
                q_stop_minus_observe_best_sums=metrics_parts[8],
                policy_stop_best_sums=metrics_parts[9],
                policy_observe_best_sums=metrics_parts[10],
                q_argmax_stop_best_sums=metrics_parts[11],
                policy_argmax_stop_best_sums=metrics_parts[12],
            )
            return total_loss, expansion_head_loss, metrics

        def loss_for_rollout(
            params,
            replay_key,
            rollout_actions,
            rollout_observations,
            rollout_old_logp,
            rollout_fixed_selected_q_target,
            rollout_fixed_advantage,
        ):
            replay_rollout_key, target_key = jax.random.split(replay_key)
            _, _, transitions = rollout(
                params,
                carry,
                replay_rollout_key,
                schedule,
                forced_actions=rollout_actions,
                forced_observations=rollout_observations,
                reset_rewards=reset_rewards,
                training=True,
                compute_targets=(config.return_target_mode != "sampled_lambda"),
            )
            if config.return_target_mode == "sampled_lambda":
                transitions = attach_fixed_sampled_lambda_targets(
                    model,
                    params,
                    transitions,
                    rollout_fixed_selected_q_target,
                    use_value_critic=use_sampled_lambda_value_critic(config),
                )
                advantages = jax.lax.stop_gradient(rollout_fixed_advantage)
            else:
                transitions = apply_expansion_return_targets(
                    model,
                    params,
                    train_state.target_params,
                    transitions,
                    config,
                    task,
                    schedule,
                    target_key,
                )
                advantages = jax.lax.stop_gradient(
                    transitions.selected_q_target - transitions.policy_value_pred
                )
            return loss_from_transitions(transitions, rollout_old_logp, advantages)

        def loss_for_flat_sampled_rollout(params):
            _, _, transitions = rollout(
                params,
                flat_carry,
                replay_rng,
                schedule,
                forced_actions=forced_actions,
                forced_observations=forced_observations,
                reset_rewards=flat_reset_rewards,
                training=True,
                compute_targets=False,
            )
            transitions = attach_fixed_sampled_lambda_targets(
                model,
                params,
                transitions,
                fixed_selected_q_target_flat,
                use_value_critic=use_sampled_lambda_value_critic(config),
            )
            advantages = jax.lax.stop_gradient(fixed_advantage_flat)
            total_loss, _expansion_head_loss, metrics = loss_from_transitions(
                transitions,
                old_logp_flat,
                advantages,
            )
            return total_loss, metrics

        def loss_fn(params):
            if config.return_target_mode == "sampled_lambda":
                return loss_for_flat_sampled_rollout(params)
            losses, _expansion_head_losses, metrics_tree = jax.vmap(
                lambda replay_key, rollout_actions, rollout_observations, rollout_old_logp, rollout_fixed_selected_q_target, rollout_fixed_advantage: loss_for_rollout(
                    params,
                    replay_key,
                    rollout_actions,
                    rollout_observations,
                    rollout_old_logp,
                    rollout_fixed_selected_q_target,
                    rollout_fixed_advantage,
                )
            )(
                replay_keys,
                forced_actions,
                forced_observations,
                old_logp,
                fixed_selected_q_target,
                fixed_advantage,
            )
            first_metrics = jax.tree_util.tree_map(lambda x: x[0], metrics_tree)
            metrics = first_metrics._replace(
                total_loss=jnp.mean(metrics_tree.total_loss),
                information_loss=jnp.mean(metrics_tree.information_loss),
                action_loss=jnp.mean(metrics_tree.action_loss),
                reconstruction_loss=jnp.mean(metrics_tree.reconstruction_loss),
                expansion_loss=jnp.mean(metrics_tree.expansion_loss),
                node_coverage_aux_loss=jnp.mean(metrics_tree.node_coverage_aux_loss),
                critic_loss=jnp.mean(metrics_tree.critic_loss),
                lstm_probe_loss=jnp.mean(metrics_tree.lstm_probe_loss),
                lstm_probe_accuracy=jnp.mean(metrics_tree.lstm_probe_accuracy),
                stop_rate=jnp.mean(metrics_tree.stop_rate),
                continue_rate=jnp.mean(metrics_tree.continue_rate),
                decision_mean_unique_nodes=jnp.mean(metrics_tree.decision_mean_unique_nodes),
                decision_all_nodes_rate=jnp.mean(metrics_tree.decision_all_nodes_rate),
                decision_mean_unique_paths=jnp.mean(metrics_tree.decision_mean_unique_paths),
                decision_all_paths_rate=jnp.mean(metrics_tree.decision_all_paths_rate),
                rollout_final_mean_unique_nodes=jnp.mean(metrics_tree.rollout_final_mean_unique_nodes),
                rollout_final_all_nodes_rate=jnp.mean(metrics_tree.rollout_final_all_nodes_rate),
                rollout_final_mean_unique_paths=jnp.mean(metrics_tree.rollout_final_mean_unique_paths),
                rollout_final_all_paths_rate=jnp.mean(metrics_tree.rollout_final_all_paths_rate),
            )
            return jnp.mean(losses), metrics

        def expansion_head_loss_fn(params):
            _losses, expansion_head_losses, _metrics_tree = jax.vmap(
                lambda replay_key, rollout_actions, rollout_observations, rollout_old_logp, rollout_fixed_selected_q_target, rollout_fixed_advantage: loss_for_rollout(
                    params,
                    replay_key,
                    rollout_actions,
                    rollout_observations,
                    rollout_old_logp,
                    rollout_fixed_selected_q_target,
                    rollout_fixed_advantage,
                )
            )(
                replay_keys,
                forced_actions,
                forced_observations,
                old_logp,
                fixed_selected_q_target,
                fixed_advantage,
            )
            return jnp.mean(expansion_head_losses)

        def expansion_head_loss_from_subtree(expansion_head_params, params):
            return expansion_head_loss_fn(
                replace_param_subtree(params, "expansion_head", expansion_head_params)
            )

        (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)
        expansion_head_mask = top_level_param_mask(new_train_state.params, {"expansion_head"})

        if config.return_target_mode == "sampled_lambda":
            n_trajectories = config.return_target_rollouts * config.num_envs
            num_minibatches = max(int(config.ppo_minibatches), 1)
            minibatch_size = n_trajectories // num_minibatches

            def cached_expansion_head_loss_from_subtree(
                expansion_head_params,
                params,
                minibatch: CachedExpansionPPOData,
            ):
                return cached_expansion_ppo_loss(
                    model,
                    replace_param_subtree(params, "expansion_head", expansion_head_params),
                    minibatch,
                    schedule,
                    config.loss_scale,
                )

            def ppo_expansion_only_epoch(state_rng, _):
                state, epoch_rng = state_rng
                epoch_rng, perm_rng = jax.random.split(epoch_rng)
                permutation = jax.random.permutation(perm_rng, n_trajectories)
                minibatches = permutation.reshape((num_minibatches, minibatch_size))

                def ppo_minibatch_step(minibatch_state, minibatch_idx):
                    minibatch = jax.tree_util.tree_map(
                        lambda x: x[minibatch_idx],
                        cached_expansion_data,
                    )
                    expansion_head_grads = jax.grad(cached_expansion_head_loss_from_subtree)(
                        minibatch_state.params["expansion_head"],
                        minibatch_state.params,
                        minibatch,
                    )
                    grads = zeros_like_param_tree(minibatch_state.params)
                    grads = replace_param_subtree(grads, "expansion_head", expansion_head_grads)
                    minibatch_state = apply_masked_gradients(
                        minibatch_state,
                        grads,
                        expansion_head_mask,
                    )
                    return minibatch_state, None

                state, _ = jax.lax.scan(
                    ppo_minibatch_step,
                    state,
                    minibatches,
                )
                return (state, epoch_rng), None

            (new_train_state, rng), _ = jax.lax.scan(
                ppo_expansion_only_epoch,
                (new_train_state, rng),
                xs=None,
                length=max(config.update_epochs - 1, 0),
            )
        else:
            def ppo_expansion_only_step(state, _):
                expansion_head_grads = jax.grad(expansion_head_loss_from_subtree)(
                    state.params["expansion_head"],
                    state.params,
                )
                grads = zeros_like_param_tree(state.params)
                grads = replace_param_subtree(grads, "expansion_head", expansion_head_grads)
                state = apply_masked_gradients(state, grads, expansion_head_mask)
                return state, None

            new_train_state, _ = jax.lax.scan(
                ppo_expansion_only_step,
                new_train_state,
                xs=None,
                length=max(config.update_epochs - 1, 0),
            )
        if (
            config.return_target_mode in {"lambda", "sampled_lambda"}
            and config.target_critic_update_interval > 0
        ):
            should_update_target = (
                (new_train_state.step % config.target_critic_update_interval) == 0
            )
            updated_target_params = soft_update_tree(
                new_train_state.params,
                new_train_state.target_params,
                config.target_critic_tau,
            )
            target_params = jax.tree_util.tree_map(
                lambda old, new: jnp.where(should_update_target, new, old),
                new_train_state.target_params,
                updated_target_params,
            )
        else:
            target_params = new_train_state.target_params
        new_train_state = new_train_state.replace(target_params=target_params)
        return new_train_state, collect_carry, rng, metrics

    if not config.jit_training:
        return update_step
    if backend:
        return jax.jit(update_step, backend=backend)
    return jax.jit(update_step)


def block_until_ready_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def time_jax_profile_call(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    block_until_ready_tree(out)
    return out, time.perf_counter() - start


def build_update_component_profile_fns(
    model: PlanningVAE,
    task: TaskSpec,
    config: RunConfig,
):
    rollout = build_rollout_fn(model, task, config)
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)
    backend = config.backend

    def collect_profile_inputs(
        params,
        target_params,
        carry: RunnerCarry,
        rng: jax.Array,
        schedule: ScheduleValues,
    ) -> ProfileReplayInputs:
        rng, reset_rng, rollout_rng, replay_rng = jax.random.split(rng, 4)
        reset_rewards = sample_reward_matrix(
            reset_rng,
            config.num_envs,
            task.num_nodes,
            task.reward_values,
        )
        rollout_keys = jax.random.split(rollout_rng, config.return_target_rollouts)
        replay_keys = jax.random.split(replay_rng, config.return_target_rollouts)

        if config.return_target_mode == "sampled_lambda":
            flat_carry = repeat_carry_for_rollouts(carry, config.return_target_rollouts)
            flat_reset_rewards = expand_rewards_for_rollouts(
                reset_rewards,
                config.return_target_rollouts,
            )
            flat_collect_carry, _rollout_rng, flat_old_transitions = rollout(
                params,
                flat_carry,
                rollout_rng,
                schedule,
                forced_actions=None,
                reset_rewards=flat_reset_rewards,
                training=True,
                compute_targets=False,
            )
            collect_carry = first_rollout_carry(flat_collect_carry, config.num_envs)
            old_transitions = jax.tree_util.tree_map(
                lambda x: unflatten_time_batch_rollouts(
                    x,
                    config.return_target_rollouts,
                    config.num_envs,
                ),
                flat_old_transitions,
            )
            fixed_selected_q_target_flat, fixed_advantage_flat = fixed_sampled_lambda_targets(
                model,
                target_params,
                flat_old_transitions,
                config,
                task,
                schedule,
            )
            fixed_selected_q_target_for_cache = jax.lax.stop_gradient(
                unflatten_time_batch_rollouts(
                    fixed_selected_q_target_flat,
                    config.return_target_rollouts,
                    config.num_envs,
                )
            )
            fixed_advantage_for_cache = jax.lax.stop_gradient(
                unflatten_time_batch_rollouts(
                    fixed_advantage_flat,
                    config.return_target_rollouts,
                    config.num_envs,
                )
            )
            policy_weights = jax.lax.stop_gradient(
                unflatten_time_batch_rollouts(
                    compute_policy_weights(
                        flat_old_transitions,
                        path_map,
                        task.num_nodes,
                    ),
                    config.return_target_rollouts,
                    config.num_envs,
                )
            )
            old_logp_for_cache = jax.lax.stop_gradient(old_transitions.log_prob)
            cached_expansion_data = make_cached_expansion_ppo_data(
                old_transitions,
                old_logp_for_cache,
                fixed_advantage_for_cache,
                policy_weights,
                expansion_entropy_mask(
                    old_transitions,
                    config.expansion_decision_version,
                ),
            )
            forced_actions = jax.lax.stop_gradient(flat_old_transitions.action)
            forced_observations = jax.lax.stop_gradient(flat_old_transitions.expanded_reward)
            old_logp = jax.lax.stop_gradient(flat_old_transitions.log_prob)
            fixed_selected_q_target = jax.lax.stop_gradient(fixed_selected_q_target_flat)
            fixed_advantage = jax.lax.stop_gradient(fixed_advantage_flat)
        else:
            def collect_one(rollout_key):
                rollout_carry, _rollout_rng, transitions = rollout(
                    params,
                    carry,
                    rollout_key,
                    schedule,
                    forced_actions=None,
                    reset_rewards=reset_rewards,
                    training=True,
                    compute_targets=False,
                )
                return rollout_carry, transitions

            rollout_carries, old_transitions = jax.vmap(collect_one)(rollout_keys)
            collect_carry = jax.tree_util.tree_map(lambda x: x[0], rollout_carries)
            forced_actions = jax.lax.stop_gradient(old_transitions.action)
            forced_observations = jax.lax.stop_gradient(old_transitions.expanded_reward)
            old_logp = jax.lax.stop_gradient(old_transitions.log_prob)
            fixed_selected_q_target = jnp.zeros_like(old_logp)
            fixed_advantage = jnp.zeros_like(old_logp)
            cached_expansion_data = make_cached_expansion_ppo_data(
                old_transitions,
                old_logp,
                fixed_advantage,
                jax.vmap(
                    lambda transitions: compute_policy_weights(
                        transitions,
                        path_map,
                        task.num_nodes,
                    )
                )(old_transitions),
                expansion_entropy_mask(
                    old_transitions,
                    config.expansion_decision_version,
                ),
            )
        checksum = (
            jnp.sum(old_transitions.valid)
            + 1e-6 * jnp.sum(old_transitions.log_prob)
            + 1e-9 * jnp.sum(collect_carry.h)
            + 1e-12 * jnp.sum(fixed_selected_q_target)
        )
        return ProfileReplayInputs(
            reset_rewards=reset_rewards,
            replay_keys=replay_keys,
            forced_actions=forced_actions,
            forced_observations=forced_observations,
            old_logp=old_logp,
            fixed_selected_q_target=jax.lax.stop_gradient(fixed_selected_q_target),
            fixed_advantage=jax.lax.stop_gradient(fixed_advantage),
            cached_expansion_data=cached_expansion_data,
            collect_carry=collect_carry,
            checksum=checksum,
        )

    def profiled_loss_for_rollout(
        params,
        target_params,
        carry: RunnerCarry,
        replay_key,
        schedule: ScheduleValues,
        reset_rewards,
        rollout_actions,
        rollout_observations,
        rollout_old_logp,
        rollout_fixed_selected_q_target,
        rollout_fixed_advantage,
    ):
        replay_rollout_key, target_key = jax.random.split(replay_key)
        _, _, transitions = rollout(
            params,
            carry,
            replay_rollout_key,
            schedule,
            forced_actions=rollout_actions,
            forced_observations=rollout_observations,
            reset_rewards=reset_rewards,
            training=True,
            compute_targets=(config.return_target_mode != "sampled_lambda"),
        )
        if config.return_target_mode == "sampled_lambda":
            transitions = attach_fixed_sampled_lambda_targets(
                model,
                params,
                transitions,
                rollout_fixed_selected_q_target,
                use_value_critic=use_sampled_lambda_value_critic(config),
            )
            advantages = jax.lax.stop_gradient(rollout_fixed_advantage)
        else:
            transitions = apply_expansion_return_targets(
                model,
                params,
                target_params,
                transitions,
                config,
                task,
                schedule,
                target_key,
            )
            advantages = jax.lax.stop_gradient(transitions.selected_q_target - transitions.policy_value_pred)
        advantages = clip_advantages(advantages, config.advantage_clip) * transitions.valid
        weights = compute_policy_weights(transitions, path_map, task.num_nodes)
        ratio = jnp.exp(jnp.clip(transitions.log_prob - rollout_old_logp, -10.0, 10.0))
        clipped_ratio = jnp.clip(ratio, 1.0 - schedule.ppo_clip, 1.0 + schedule.ppo_clip)
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages) * weights
        entropy_mask = expansion_entropy_mask(
            transitions,
            config.expansion_decision_version,
        )
        entropy_loss = (
            jnp.sum(transitions.entropy * entropy_mask)
            / (jnp.sum(entropy_mask) + 1e-6)
        )
        valid_policy_count = jnp.sum(transitions.valid) + 1e-6
        expansion_loss = (
            jnp.sum(policy_loss) / valid_policy_count
            - schedule.expansion_entropy_coef * entropy_loss
        )
        if config.return_target_mode == "sampled_lambda":
            if use_sampled_lambda_value_critic(config):
                critic_err = (
                    critic_error_loss(
                        transitions.policy_value_pred - transitions.selected_q_target,
                        config.critic_huber_delta,
                    )
                    * transitions.valid
                )
                critic_loss = jnp.sum(critic_err) / (jnp.sum(transitions.valid) + 1e-6)
            else:
                critic_mask = (
                    jax.nn.one_hot(
                        transitions.action,
                        transitions.q_values.shape[-1],
                        dtype=jnp.float32,
                    )
                    * transitions.valid[:, :, None]
                )
                critic_err = (
                    critic_error_loss(
                        transitions.q_values - transitions.q_targets,
                        config.critic_huber_delta,
                    )
                    * critic_mask
                )
                critic_loss = jnp.sum(critic_err) / (jnp.sum(critic_mask) + 1e-6)
        else:
            critic_mask = transitions.legal_mask
            critic_err = (
                critic_error_loss(
                    transitions.q_values - transitions.q_targets,
                    config.critic_huber_delta,
                )
                * critic_mask
            )
            critic_loss = jnp.sum(critic_err) / (jnp.sum(critic_mask) + 1e-6)
        information_loss = jnp.mean(transitions.paid_kl)
        reconstruction_loss = jnp.mean(transitions.reconstruction_loss)
        action_loss = tf_style_terminal_action_loss(
            transitions,
            path_map,
            float(task.reward_norm),
        )
        probe_loss = jnp.sum(transitions.probe_loss) / (jnp.sum(transitions.valid_probe) + 1e-6)
        coverage_target, coverage_mask = node_coverage_auxiliary_targets(
            transitions,
            task.num_nodes,
        )
        node_coverage_aux_loss = node_coverage_auxiliary_loss(
            transitions.probs,
            coverage_target,
            coverage_mask,
        )
        total_loss = (
            information_loss * schedule.current_beta
            + action_loss * config.loss_scale
            + expansion_loss * config.loss_scale
            + critic_loss * config.loss_scale * schedule.current_critic_coef
            + reconstruction_loss * config.alpha
            + probe_loss
        )
        expansion_head_loss = (
            expansion_loss * config.loss_scale
            + critic_loss * config.loss_scale * schedule.current_critic_coef
            + action_loss * config.loss_scale
            + schedule.node_coverage_aux_coef * config.loss_scale * node_coverage_aux_loss
        )
        return total_loss, expansion_head_loss

    def full_loss_grad_profile(
        params,
        target_params,
        carry: RunnerCarry,
        schedule: ScheduleValues,
        replay_inputs: ProfileReplayInputs,
    ):
        def loss_fn(loss_params):
            if config.return_target_mode == "sampled_lambda":
                loss, _expansion_loss = profiled_loss_for_rollout(
                    loss_params,
                    target_params,
                    repeat_carry_for_rollouts(carry, config.return_target_rollouts),
                    replay_inputs.replay_keys[0],
                    schedule,
                    expand_rewards_for_rollouts(
                        replay_inputs.reset_rewards,
                        config.return_target_rollouts,
                    ),
                    replay_inputs.forced_actions,
                    replay_inputs.forced_observations,
                    replay_inputs.old_logp,
                    replay_inputs.fixed_selected_q_target,
                    replay_inputs.fixed_advantage,
                )
                return loss
            losses, _expansion_losses = jax.vmap(
                lambda replay_key, rollout_actions, rollout_observations, rollout_old_logp, rollout_fixed_selected_q_target, rollout_fixed_advantage: profiled_loss_for_rollout(
                    loss_params,
                    target_params,
                    carry,
                    replay_key,
                    schedule,
                    replay_inputs.reset_rewards,
                    rollout_actions,
                    rollout_observations,
                    rollout_old_logp,
                    rollout_fixed_selected_q_target,
                    rollout_fixed_advantage,
                )
            )(
                replay_inputs.replay_keys,
                replay_inputs.forced_actions,
                replay_inputs.forced_observations,
                replay_inputs.old_logp,
                replay_inputs.fixed_selected_q_target,
                replay_inputs.fixed_advantage,
            )
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, optax.global_norm(grads)

    def ppo_head_grad_profile(
        params,
        target_params,
        carry: RunnerCarry,
        schedule: ScheduleValues,
        replay_inputs: ProfileReplayInputs,
    ):
        if config.return_target_mode == "sampled_lambda":
            n_trajectories = config.return_target_rollouts * config.num_envs
            num_minibatches = max(int(config.ppo_minibatches), 1)
            minibatch_size = n_trajectories // num_minibatches
            minibatch = jax.tree_util.tree_map(
                lambda x: x[:minibatch_size],
                replay_inputs.cached_expansion_data,
            )

            def cached_expansion_head_loss_from_subtree(expansion_head_params, all_params):
                return cached_expansion_ppo_loss(
                    model,
                    replace_param_subtree(all_params, "expansion_head", expansion_head_params),
                    minibatch,
                    schedule,
                    config.loss_scale,
                )

            expansion_head_grads = jax.grad(cached_expansion_head_loss_from_subtree)(
                params["expansion_head"],
                params,
            )
            return optax.global_norm(expansion_head_grads)

        def expansion_head_loss_fn(loss_params):
            _losses, expansion_head_losses = jax.vmap(
                lambda replay_key, rollout_actions, rollout_observations, rollout_old_logp, rollout_fixed_selected_q_target, rollout_fixed_advantage: profiled_loss_for_rollout(
                    loss_params,
                    target_params,
                    carry,
                    replay_key,
                    schedule,
                    replay_inputs.reset_rewards,
                    rollout_actions,
                    rollout_observations,
                    rollout_old_logp,
                    rollout_fixed_selected_q_target,
                    rollout_fixed_advantage,
                )
            )(
                replay_inputs.replay_keys,
                replay_inputs.forced_actions,
                replay_inputs.forced_observations,
                replay_inputs.old_logp,
                replay_inputs.fixed_selected_q_target,
                replay_inputs.fixed_advantage,
            )
            return jnp.mean(expansion_head_losses)

        def expansion_head_loss_from_subtree(expansion_head_params, all_params):
            return expansion_head_loss_fn(
                replace_param_subtree(all_params, "expansion_head", expansion_head_params)
            )

        expansion_head_grads = jax.grad(expansion_head_loss_from_subtree)(
            params["expansion_head"],
            params,
        )
        return optax.global_norm(expansion_head_grads)

    fns = {
        "collect": collect_profile_inputs,
        "full_loss_grad": full_loss_grad_profile,
        "ppo_head_grad": ppo_head_grad_profile,
    }
    if not config.jit_training:
        return fns
    if backend:
        return {
            name: jax.jit(fn, backend=backend)
            for name, fn in fns.items()
        }
    return {
        name: jax.jit(fn)
        for name, fn in fns.items()
    }


def empty_metric_accumulator(time_steps: int) -> dict[str, np.ndarray | float]:
    shape = (time_steps, 9)
    return {
        "total_loss": 0.0,
        "information_loss": 0.0,
        "action_loss": 0.0,
        "reconstruction_loss": 0.0,
        "expansion_loss": 0.0,
        "node_coverage_aux_loss": 0.0,
        "critic_loss": 0.0,
        "lstm_probe_loss": 0.0,
        "lstm_probe_accuracy": 0.0,
        "stop_rate": 0.0,
        "continue_rate": 0.0,
        "decision_mean_unique_nodes": 0.0,
        "decision_all_nodes_rate": 0.0,
        "decision_mean_unique_paths": 0.0,
        "decision_all_paths_rate": 0.0,
        "rollout_final_mean_unique_nodes": 0.0,
        "rollout_final_all_nodes_rate": 0.0,
        "rollout_final_mean_unique_paths": 0.0,
        "rollout_final_all_paths_rate": 0.0,
        "continue_best_sums": np.zeros(shape, dtype=float),
        "continue_best_counts": np.zeros(shape, dtype=float),
        "critic_best_sums": np.zeros(shape, dtype=float),
        "target_best_sums": np.zeros(shape, dtype=float),
        "advantage_best_sums": np.zeros(shape, dtype=float),
        "kl_best_sums": np.zeros(shape, dtype=float),
        "q_stop_best_sums": np.zeros(shape, dtype=float),
        "q_observe_best_sums": np.zeros(shape, dtype=float),
        "q_stop_minus_observe_best_sums": np.zeros(shape, dtype=float),
        "policy_stop_best_sums": np.zeros(shape, dtype=float),
        "policy_observe_best_sums": np.zeros(shape, dtype=float),
        "q_argmax_stop_best_sums": np.zeros(shape, dtype=float),
        "policy_argmax_stop_best_sums": np.zeros(shape, dtype=float),
    }


def add_metrics(acc: dict, metrics: UpdateMetrics):
    scalar_names = [
        "total_loss",
        "information_loss",
        "action_loss",
        "reconstruction_loss",
        "expansion_loss",
        "node_coverage_aux_loss",
        "critic_loss",
        "lstm_probe_loss",
        "lstm_probe_accuracy",
        "stop_rate",
        "continue_rate",
        "decision_mean_unique_nodes",
        "decision_all_nodes_rate",
        "decision_mean_unique_paths",
        "decision_all_paths_rate",
        "rollout_final_mean_unique_nodes",
        "rollout_final_all_nodes_rate",
        "rollout_final_mean_unique_paths",
        "rollout_final_all_paths_rate",
    ]
    for name in scalar_names:
        acc[name] += float(np.asarray(getattr(metrics, name)))
    array_names = [name for name in acc if name.endswith("_sums") or name.endswith("_counts")]
    for name in array_names:
        acc[name] += np.asarray(getattr(metrics, name))


def reward_label(value: float) -> str:
    if value > 0:
        return f"p{int(value)}"
    if value < 0:
        return f"m{abs(int(value))}"
    return "z0"


def finalize_epoch_row(
    epoch: int,
    acc: dict,
    n_updates: int,
    metrics: UpdateMetrics,
    config: RunConfig,
    task: TaskSpec,
    updates_per_epoch: int,
) -> dict[str, float | int | str]:
    row = {
        "epoch": epoch,
        "learning_rate": float(np.asarray(metrics.learning_rate)),
        "expansion_epsilon": 0.0,
        "return_target_rollouts": config.return_target_rollouts,
        "ppo_minibatches": config.ppo_minibatches,
        "steps_per_epoch": config.steps_per_epoch,
        "steps_per_batch": config.num_envs * config.num_steps,
        "updates_per_epoch": updates_per_epoch,
        "rollout_steps": config.num_steps,
        "expansion_return_target_mode": config.return_target_mode,
        "sampled_lambda_critic": config.sampled_lambda_critic,
        "expansion_lambda_return": (
            config.lambda_return
            if config.return_target_mode in {"lambda", "sampled_lambda"}
            else float("nan")
        ),
        "target_critic_update_interval": (
            config.target_critic_update_interval
            if config.return_target_mode in {"lambda", "sampled_lambda"}
            else 0
        ),
        "target_critic_tau": (
            config.target_critic_tau
            if config.return_target_mode in {"lambda", "sampled_lambda"}
            else float("nan")
        ),
        "loss_scale": config.loss_scale,
        "memory_lambda": config.memory_lambda,
        "critic_huber_delta": config.critic_huber_delta,
        "advantage_clip": config.advantage_clip,
        "forced_continue_epsilon": 0.0,
        "expansion_entropy_coef": float(np.asarray(metrics.entropy_coef)),
        "node_coverage_aux_coef": float(np.asarray(metrics.node_coverage_aux_coef)),
        "critic_coef": float(np.asarray(metrics.critic_coef)),
        "current_memory_lambda": float(np.asarray(metrics.current_beta)),
        "current_beta": float(np.asarray(metrics.current_beta)),
    }
    for name in [
        "total_loss",
        "information_loss",
        "action_loss",
        "reconstruction_loss",
        "expansion_loss",
        "node_coverage_aux_loss",
        "critic_loss",
        "lstm_probe_loss",
        "lstm_probe_accuracy",
        "stop_rate",
        "continue_rate",
        "decision_mean_unique_nodes",
        "decision_all_nodes_rate",
        "decision_mean_unique_paths",
        "decision_all_paths_rate",
        "rollout_final_mean_unique_nodes",
        "rollout_final_all_nodes_rate",
        "rollout_final_mean_unique_paths",
        "rollout_final_all_paths_rate",
    ]:
        out_name = {
            "information_loss": "kl_loss",
            "stop_rate": "expansion_stop_rate",
            "continue_rate": "expansion_continue_rate",
        }.get(name, name)
        row[out_name] = acc[name] / max(n_updates, 1)

    reward_values = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    for t in range(task.num_nodes):
        for idx, value in enumerate(reward_values):
            label = reward_label(value)
            count = acc["continue_best_counts"][t, idx]
            denom = count if count > 0 else np.nan
            prefix = f"t{t + 1}_after_best_path_value_{label}"
            row[f"exp_continue_{prefix}"] = acc["continue_best_sums"][t, idx] / denom
            row[f"exp_continue_n_{prefix}"] = count
            row[f"exp_critic_{prefix}"] = acc["critic_best_sums"][t, idx] / denom
            row[f"exp_return_target_{prefix}"] = acc["target_best_sums"][t, idx] / denom
            row[f"exp_advantage_{prefix}"] = acc["advantage_best_sums"][t, idx] / denom
            row[f"exp_kl_d_{prefix}"] = acc["kl_best_sums"][t, idx] / denom
            row[f"exp_kl_d_n_{prefix}"] = count
            row[f"exp_q_stop_max_{prefix}"] = acc["q_stop_best_sums"][t, idx] / denom
            row[f"exp_q_observe_max_{prefix}"] = acc["q_observe_best_sums"][t, idx] / denom
            row[f"exp_q_stop_minus_observe_{prefix}"] = acc["q_stop_minus_observe_best_sums"][t, idx] / denom
            row[f"exp_policy_stop_prob_{prefix}"] = acc["policy_stop_best_sums"][t, idx] / denom
            row[f"exp_policy_observe_prob_{prefix}"] = acc["policy_observe_best_sums"][t, idx] / denom
            row[f"exp_q_argmax_stop_{prefix}"] = acc["q_argmax_stop_best_sums"][t, idx] / denom
            row[f"exp_policy_argmax_stop_{prefix}"] = acc["policy_argmax_stop_best_sums"][t, idx] / denom
    return row


def train(config: RunConfig, task: TaskSpec) -> tuple[object, PlanningTrainState]:
    setup_start = time.perf_counter()
    rng = jax.random.PRNGKey(config.seed)
    path_tuple = tuple(tuple(float(v) for v in row) for row in task.path_map)
    reward_tuple = tuple(float(v) for v in task.reward_values)
    model = PlanningVAE(
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        time_steps=task.num_nodes,
        num_paths=task.num_paths,
        path_map=path_tuple,
        reward_values=reward_tuple,
        reward_norm_value=float(task.reward_norm),
        expansion_decision_version=config.expansion_decision_version,
        use_autoencoder=(config.model_variant == "vae"),
        enable_reconstruction=config.enable_reconstruction,
        enable_probe=config.enable_probe,
        allow_node_revisit=config.allow_node_revisit,
        max_observations_before_stop=config.max_observations_before_stop,
        opportunity_cost=config.opportunity_cost,
        observation_sigma=config.observation_sigma,
        loss_scale=config.loss_scale,
        alpha=config.alpha,
        memory_lambda=config.memory_lambda,
        include_visited_lstm_input=use_visited_lstm_input_for_task(task),
        pay_kl_on_stop=config.pay_kl_on_stop,
        choice_at_end_only=config.choice_at_end_only,
    )
    updates_per_epoch = max(1, math.ceil(config.steps_per_epoch / (config.num_envs * config.num_steps)))
    total_updates = config.epochs * updates_per_epoch
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, config, task, init_rng, total_updates)
    reward_feature_dim = reward_feature_dim_for_sigma(config.observation_sigma)
    carry = initial_carry(
        config.num_envs,
        task,
        config.rnn_units,
        reward_feature_dim,
        visited_lstm_feature_dim_for_task(task),
    )
    update_fn = build_update_fn(model, task, config, total_updates)
    profile_fns = (
        build_update_component_profile_fns(model, task, config)
        if config.profile_update_components
        else None
    )
    setup_sec = time.perf_counter() - setup_start
    print(
        "JAX setup timing: "
        f"setup={setup_sec:.3f}s | "
        f"updates_per_epoch={updates_per_epoch} | "
        f"total_updates={total_updates} | "
        f"jit_training={config.jit_training} | "
        f"backend={config.backend or 'default'} | "
        f"loss_scale={config.loss_scale} | "
        f"memory_lambda={config.memory_lambda} | "
        f"return_target={config.return_target_mode} | "
        f"sampled_lambda_critic={config.sampled_lambda_critic} | "
        f"lambda_return={config.lambda_return} | "
        f"ppo_minibatches={config.ppo_minibatches} | "
        f"target_interval={config.target_critic_update_interval} | "
        f"reconstruction={config.enable_reconstruction} | "
        f"probe={config.enable_probe} | "
        f"allow_node_revisit={config.allow_node_revisit} | "
        f"max_observations_before_stop={config.max_observations_before_stop} | "
        f"observation_sigma={config.observation_sigma} | "
        f"visited_lstm_input={use_visited_lstm_input_for_task(task)} | "
        f"kl_start_multiplier={config.kl_start_multiplier} | "
        f"kl_annealing_epochs={config.kl_annealing_epochs} | "
        f"node_coverage_aux_coef={config.node_coverage_aux_coef} | "
        f"node_coverage_aux_epochs={config.node_coverage_aux_epochs} | "
        f"critic_huber_delta={config.critic_huber_delta} | "
        f"advantage_clip={config.advantage_clip} | "
        f"pay_kl_on_stop={config.pay_kl_on_stop} | "
        f"choice_at_end_only={config.choice_at_end_only}",
        flush=True,
    )
    rows = []
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        schedule_sec = 0.0
        update_dispatch_sec = 0.0
        update_sync_sec = 0.0
        metrics_sync_sec = 0.0
        acc = empty_metric_accumulator(task.num_nodes)
        last_metrics = None
        for update_in_epoch in range(updates_per_epoch):
            update_idx = epoch * updates_per_epoch + update_in_epoch
            schedule_start = time.perf_counter()
            schedule = make_schedule(config, update_idx, updates_per_epoch)
            schedule_sec += time.perf_counter() - schedule_start
            dispatch_start = time.perf_counter()
            state, carry, rng, metrics = update_fn(state, carry, rng, schedule)
            update_dispatch_sec += time.perf_counter() - dispatch_start
            sync_start = time.perf_counter()
            jax.block_until_ready(metrics.total_loss)
            update_sync_sec += time.perf_counter() - sync_start
            last_metrics = metrics
            metrics_start = time.perf_counter()
            add_metrics(acc, metrics)
            metrics_sync_sec += time.perf_counter() - metrics_start
        finalize_start = time.perf_counter()
        row = finalize_epoch_row(epoch + 1, acc, updates_per_epoch, last_metrics, config, task, updates_per_epoch)
        finalize_sec = time.perf_counter() - finalize_start
        epoch_sec = time.perf_counter() - epoch_start
        profile_collect_sec = np.nan
        profile_full_loss_grad_sec = np.nan
        profile_ppo_head_grad_epoch_sec = np.nan
        profile_estimated_update_sec = np.nan
        profile_unaccounted_update_sec = np.nan
        profile_full_loss_value = np.nan
        profile_full_grad_norm = np.nan
        profile_ppo_head_grad_norm = np.nan
        should_profile = (
            profile_fns is not None
            and ((epoch + 1) % max(config.profile_update_components_every, 1) == 0)
        )
        if should_profile:
            profile_rng = jax.random.fold_in(rng, epoch + 1)
            profile_inputs, profile_collect_sec = time_jax_profile_call(
                profile_fns["collect"],
                state.params,
                state.target_params,
                carry,
                profile_rng,
                schedule,
            )
            full_profile, profile_full_loss_grad_sec = time_jax_profile_call(
                profile_fns["full_loss_grad"],
                state.params,
                state.target_params,
                carry,
                schedule,
                profile_inputs,
            )
            head_grad_norm, profile_ppo_head_grad_epoch_sec = time_jax_profile_call(
                profile_fns["ppo_head_grad"],
                state.params,
                state.target_params,
                carry,
                schedule,
                profile_inputs,
            )
            profile_full_loss_value = float(np.asarray(full_profile[0]))
            profile_full_grad_norm = float(np.asarray(full_profile[1]))
            profile_ppo_head_grad_norm = float(np.asarray(head_grad_norm))
            profile_head_grad_repeats = max(config.update_epochs - 1, 0)
            if config.return_target_mode == "sampled_lambda":
                profile_head_grad_repeats *= max(config.ppo_minibatches, 1)
            profile_estimated_update_sec = (
                profile_collect_sec
                + profile_full_loss_grad_sec
                + profile_head_grad_repeats * profile_ppo_head_grad_epoch_sec
            )
            observed_per_update = update_dispatch_sec / max(updates_per_epoch, 1)
            profile_unaccounted_update_sec = observed_per_update - profile_estimated_update_sec
        row.update(
            {
                "timing_setup_sec": setup_sec if epoch == 0 else 0.0,
                "timing_epoch_sec": epoch_sec,
                "timing_schedule_sec": schedule_sec,
                "timing_update_dispatch_sec": update_dispatch_sec,
                "timing_update_sync_sec": update_sync_sec,
                "timing_metrics_sync_sec": metrics_sync_sec,
                "timing_finalize_sec": finalize_sec,
                "timing_update_dispatch_per_update_sec": update_dispatch_sec / max(updates_per_epoch, 1),
                "timing_update_sync_per_update_sec": update_sync_sec / max(updates_per_epoch, 1),
                "timing_metrics_sync_per_update_sec": metrics_sync_sec / max(updates_per_epoch, 1),
                "timing_profile_collect_sec": profile_collect_sec,
                "timing_profile_full_loss_grad_sec": profile_full_loss_grad_sec,
                "timing_profile_ppo_head_grad_epoch_sec": profile_ppo_head_grad_epoch_sec,
                "timing_profile_estimated_update_sec": profile_estimated_update_sec,
                "timing_profile_unaccounted_update_sec": profile_unaccounted_update_sec,
                "timing_profile_full_loss_value": profile_full_loss_value,
                "timing_profile_full_grad_norm": profile_full_grad_norm,
                "timing_profile_ppo_head_grad_norm": profile_ppo_head_grad_norm,
                "enable_reconstruction": bool(config.enable_reconstruction),
                "enable_probe": bool(config.enable_probe),
                "allow_node_revisit": bool(config.allow_node_revisit),
                "max_observations_before_stop": int(config.max_observations_before_stop),
                "observation_sigma": float(config.observation_sigma),
                "visited_lstm_input": bool(use_visited_lstm_input_for_task(task)),
                "kl_start_multiplier": float(config.kl_start_multiplier),
                "kl_annealing_epochs": int(config.kl_annealing_epochs),
                "node_coverage_aux_start_coef": float(config.node_coverage_aux_coef),
                "node_coverage_aux_epochs": int(config.node_coverage_aux_epochs),
                "pay_kl_on_stop": bool(config.pay_kl_on_stop),
                "choice_at_end_only": bool(config.choice_at_end_only),
            }
        )
        rows.append(row)
        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"Loss = {row['total_loss']:.4f} | KL = {row['kl_loss']:.4f} | "
            f"Stop = {row['expansion_stop_rate']:.4f} | Continue = {row['expansion_continue_rate']:.4f} | "
            f"Final unique nodes = {row['rollout_final_mean_unique_nodes']:.2f}/{task.num_nodes} | "
            f"Final all paths = {row['rollout_final_all_paths_rate']:.3f} | "
            f"Timing epoch = {epoch_sec:.2f}s | "
            f"update dispatch = {update_dispatch_sec:.2f}s | "
            f"update sync = {update_sync_sec:.2f}s | "
            f"metrics sync = {metrics_sync_sec:.2f}s | "
            f"finalize = {finalize_sec:.2f}s",
            flush=True,
        )
        if should_profile:
            print(
                f"Profile epoch {epoch + 1}: "
                f"collect = {profile_collect_sec:.3f}s | "
                f"full loss+grad = {profile_full_loss_grad_sec:.3f}s | "
                f"one PPO head grad unit = {profile_ppo_head_grad_epoch_sec:.3f}s | "
                f"estimated update = {profile_estimated_update_sec:.3f}s | "
                f"observed update dispatch = {row['timing_update_dispatch_per_update_sec']:.3f}s | "
                f"unaccounted = {profile_unaccounted_update_sec:.3f}s | "
                f"full grad norm = {profile_full_grad_norm:.3f} | "
                f"head grad norm = {profile_ppo_head_grad_norm:.3f}",
                flush=True,
            )
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name_for(config, task)
    log_path = model_dir / f"{model_name}_training_logs.csv"
    save_start = time.perf_counter()
    pd.DataFrame(rows).to_csv(log_path, index=False)
    log_save_sec = time.perf_counter() - save_start
    weights_path = model_dir / f"{model_name}.msgpack"
    weights_start = time.perf_counter()
    weights_path.write_bytes(serialization.to_bytes(state.params))
    weights_save_sec = time.perf_counter() - weights_start
    print(f"Saved JAX training logs to: {log_path}")
    print(f"Saved JAX parameters to: {weights_path}")
    print(
        "JAX save timing: "
        f"logs={log_save_sec:.3f}s | "
        f"weights={weights_save_sec:.3f}s",
        flush=True,
    )
    return model, state


def load_state_for_sim(config: RunConfig, task: TaskSpec) -> tuple[PlanningVAE, object]:
    path_tuple = tuple(tuple(float(v) for v in row) for row in task.path_map)
    reward_tuple = tuple(float(v) for v in task.reward_values)
    weights_path = Path(config.model_dir) / f"{model_name_for(config, task)}.msgpack"
    if not weights_path.exists():
        legacy_weights_path = Path(config.model_dir) / f"{legacy_model_name_for(config, task)}.msgpack"
        if legacy_weights_path.exists():
            weights_path = legacy_weights_path
    visited_feature_dim = visited_lstm_feature_dim_for_task(task)
    checkpoint_reward_dim = (
        infer_reward_feature_dim_from_checkpoint(
            weights_path,
            task.num_nodes,
            visited_feature_dim,
        )
        if weights_path.exists()
        else 0
    )
    model = PlanningVAE(
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        time_steps=task.num_nodes,
        num_paths=task.num_paths,
        path_map=path_tuple,
        reward_values=reward_tuple,
        reward_norm_value=float(task.reward_norm),
        expansion_decision_version=config.expansion_decision_version,
        use_autoencoder=(config.model_variant == "vae"),
        enable_reconstruction=config.enable_reconstruction,
        enable_probe=config.enable_probe,
        allow_node_revisit=config.allow_node_revisit,
        max_observations_before_stop=config.max_observations_before_stop,
        opportunity_cost=config.opportunity_cost,
        observation_sigma=config.observation_sigma,
        loss_scale=config.loss_scale,
        alpha=config.alpha,
        memory_lambda=config.memory_lambda,
        reward_feature_dim_override=int(checkpoint_reward_dim),
        include_visited_lstm_input=use_visited_lstm_input_for_task(task),
        pay_kl_on_stop=config.pay_kl_on_stop,
        choice_at_end_only=config.choice_at_end_only,
    )
    rng = jax.random.PRNGKey(config.seed)
    reward_feature_dim = (
        int(checkpoint_reward_dim)
        if int(checkpoint_reward_dim) > 0
        else reward_feature_dim_for_sigma(config.observation_sigma)
    )
    dummy = initial_carry(
        1,
        task,
        config.rnn_units,
        reward_feature_dim,
        visited_feature_dim,
    )
    sched = ScheduleValues(
        current_alpha=1.0,
        current_beta=config.memory_lambda,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        node_coverage_aux_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    params = model.init(rng, dummy, rng, sched, None, False)["params"]
    if use_sampled_lambda_value_critic(config):
        dummy_expansion_input = model.apply(
            {"params": params},
            dummy,
            method=PlanningVAE.expansion_input,
        )
        value_params = model.init(
            rng,
            dummy_expansion_input,
            method=PlanningVAE.value_critic_values,
        )["params"]
        params = merge_missing_param_subtrees(params, value_params)
    if weights_path.exists():
        params = serialization.from_bytes(params, weights_path.read_bytes())
    else:
        print(f"Warning: {weights_path} not found; simulating initialized JAX model.")
    return model, params


def sample_categorical_index(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.asarray(probabilities, dtype=float)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = float(np.sum(probs))
    if total <= 0.0:
        probs = np.ones_like(probs, dtype=float) / max(len(probs), 1)
    else:
        probs = probs / total
    return int(rng.choice(len(probs), p=probs))


def simulate(config: RunConfig, task: TaskSpec, model: PlanningVAE | None = None, params=None):
    sim_start = time.perf_counter()
    if model is None or params is None:
        model, params = load_state_for_sim(config, task)
    load_sec = time.perf_counter() - sim_start
    rng = jax.random.PRNGKey(config.seed + 100_000)
    np_rng = np.random.default_rng(config.seed + 200_000)
    num_trials = int(config.n_sim_trials)
    reward_feature_dim = reward_feature_dim_for_sigma(config.observation_sigma)
    carry = initial_carry(
        num_trials,
        task,
        config.rnn_units,
        reward_feature_dim,
        visited_lstm_feature_dim_for_task(task),
    )
    rng, reset_rng = jax.random.split(rng)
    reset_rewards = sample_reward_matrix(reset_rng, num_trials, task.num_nodes, task.reward_values)
    carry = reset_done_envs(carry, reset_rewards)
    sched = ScheduleValues(
        current_alpha=1.0,
        current_beta=config.memory_lambda,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        node_coverage_aux_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    transitions = []
    rollout_start = time.perf_counter()
    for _ in range(config.num_steps):
        rng, step_rng = jax.random.split(rng)
        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            sched,
            forced_action=None,
            training=True,
            use_posterior_mean=False,
            compute_targets=False,
            method=PlanningVAE.__call__,
        )
        transitions.append(jax.device_get(trans))
    rollout_sec = time.perf_counter() - rollout_start
    rows_start = time.perf_counter()
    rows = []
    path_map = np.asarray(task.path_map, dtype=float)
    all_rewards = np.asarray(reset_rewards)
    for trial in range(num_trials):
        rewards = all_rewards[trial]
        path_rewards = path_map @ rewards
        chosen_path = None
        chosen_value = None
        terminal_probs_last = None
        for t, trans in enumerate(transitions):
            terminal_probs_last = np.asarray(trans.action_output[trial])
            if bool(trans.is_stop[trial]) and chosen_path is None:
                chosen_path = int(trans.terminal_path_index[trial])
                chosen_value = float(path_rewards[chosen_path])
        if chosen_path is None:
            chosen_path = sample_categorical_index(terminal_probs_last, np_rng)
            chosen_value = float(path_rewards[chosen_path])
        trial_row = {
            "graph": trial,
            "chosen_path": chosen_path,
            "V": chosen_value,
            "MI": float(np.nansum([np.asarray(t.paid_kl)[trial] for t in transitions])),
            "opportunity_cost": config.opportunity_cost,
            "observation_sigma": float(config.observation_sigma),
            "expansion_decision_version": config.expansion_decision_version,
            "allow_node_revisit": bool(config.allow_node_revisit),
            "max_observations_before_stop": int(config.max_observations_before_stop),
            "visited_lstm_input": bool(use_visited_lstm_input_for_task(task)),
            "choice_at_end_only": bool(config.choice_at_end_only),
        }
        stopped_before_timestep = False
        for t, trans in enumerate(transitions, start=1):
            node_idx = int(np.asarray(trans.node_index)[trial])
            stopped_now = bool(np.asarray(trans.is_stop)[trial] > 0)
            trial_row[f"expanded_node_t{t}"] = np.nan if node_idx < 0 else node_idx + 1
            trial_row[f"expanded_reward_t{t}"] = np.asarray(trans.expanded_reward)[trial]
            trial_row[f"stop_t{t}"] = stopped_now
            trial_row[f"kl_d_t{t}"] = float(np.asarray(trans.paid_kl)[trial])
            trial_row[f"kl_d_obs_t{t}"] = float(np.asarray(trans.observed_kl)[trial])
            trial_row[f"action_policy_entropy_t{t}"] = (
                np.nan if stopped_before_timestep else float(np.asarray(trans.entropy)[trial])
            )
            terminal_choice_probs = np.asarray(trans.action_output[trial], dtype=float)
            for path_idx, path_prob in enumerate(terminal_choice_probs, start=1):
                trial_row[f"action_output_path{path_idx}_t{t}"] = float(path_prob)
                trial_row[f"terminal_choice_prob_path{path_idx}_t{t}"] = float(path_prob)
            stopped_before_timestep = stopped_before_timestep or stopped_now
        for node in range(task.num_nodes):
            row = dict(trial_row)
            row["node"] = node + 1
            row["actual_reward"] = float(rewards[node])
            for t in range(1, config.num_steps + 1):
                row[f"estimated_reward_t{t}"] = np.nan
            rows.append(row)
    rows_sec = time.perf_counter() - rows_start
    out_dir = Path(config.sim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name_for(config, task)}_{config.input_type}.csv"
    write_start = time.perf_counter()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    write_sec = time.perf_counter() - write_start
    total_sec = time.perf_counter() - sim_start
    print(f"Saved JAX simulation results to: {out_path}")
    print(
        "JAX simulation timing: "
        f"load={load_sec:.3f}s | "
        f"rollout={rollout_sec:.3f}s | "
        f"rows={rows_sec:.3f}s | "
        f"write={write_sec:.3f}s | "
        f"total={total_sec:.3f}s",
        flush=True,
    )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("loss_scale_string")
    parser.add_argument("alpha_string")
    parser.add_argument("memory_lambda_string")
    parser.add_argument("model_dir")
    parser.add_argument("epochs", type=int)
    parser.add_argument("input_type")
    parser.add_argument("seed", type=int)
    parser.add_argument("tree_size", type=int)
    parser.add_argument("train_mode")
    parser.add_argument("tree_type")
    parser.add_argument("opportunity_cost_string", nargs="?", default="0.0")
    parser.add_argument("expansion_decision_version", nargs="?", default="decoder")
    parser.add_argument("model_variant", nargs="?", default="vae")
    parser.add_argument("rnn_units", nargs="?", type=int, default=64)
    parser.add_argument("latent_dim", nargs="?", type=int, default=32)
    parser.add_argument("--sim-dir", default=None)
    parser.add_argument("--n-sim-trials", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument(
        "--ppo-minibatches",
        type=int,
        default=int(os.environ.get("PPO_MINIBATCHES", "1")),
        help=(
            "Number of trajectory minibatches for cached sampled-lambda expansion-head PPO epochs. "
            "Uses whole rollout/env trajectories, not individual timesteps. Default: 1."
        ),
    )
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--return-target-rollouts",
        type=int,
        default=int(os.environ.get("RETURN_TARGET_ROLLOUTS", "8")),
    )
    parser.add_argument(
        "--return-target-mode",
        default=os.environ.get("EXPANSION_RETURN_TARGET", "lambda"),
        help=(
            "Expansion return target mode. Matches TensorFlow EXPANSION_RETURN_TARGET: "
            "lambda, sampled_lambda, or one_step aliases."
        ),
    )
    parser.add_argument(
        "--lambda-return",
        type=float,
        default=float(os.environ.get("EXPANSION_LAMBDA_RETURN", "0.95")),
    )
    parser.add_argument(
        "--sampled-lambda-critic",
        choices=["value", "q"],
        default=os.environ.get("SAMPLED_LAMBDA_CRITIC", "value").strip().lower(),
        help=(
            "Critic used by EXPANSION_RETURN_TARGET=sampled_lambda. "
            "'value' uses a scalar V(h_t) head; 'q' keeps the older "
            "policy-weighted action-Q baseline. Default: value."
        ),
    )
    parser.add_argument(
        "--target-critic-update-interval",
        type=int,
        default=int(os.environ.get("TARGET_CRITIC_UPDATE_INTERVAL", "100")),
    )
    parser.add_argument(
        "--target-critic-tau",
        type=float,
        default=float(os.environ.get("TARGET_CRITIC_TAU", "1.0")),
    )
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT compilation for tiny smoke/debug runs. Full training remains faster with JIT.",
    )
    parser.add_argument(
        "--profile-update-components",
        action="store_true",
        help=(
            "Run extra timed JAX calls for rollout collection, full loss+grad replay, "
            "and one expansion-head PPO grad epoch. This adds overhead and is for runtime diagnosis."
        ),
    )
    parser.add_argument(
        "--profile-update-components-every",
        type=int,
        default=10,
        help="Profile update components every N epochs when --profile-update-components is set.",
    )
    parser.add_argument(
        "--enable-reconstruction",
        action="store_true",
        help="Compute and train the reconstruction diagnostic head. Disabled by default for speed.",
    )
    parser.add_argument(
        "--enable-probe",
        action="store_true",
        help="Compute and train the LSTM reward probe head. Disabled by default for speed.",
    )
    parser.add_argument(
        "--allow-node-revisit",
        action="store_true",
        default=os.environ.get("ALLOW_NODE_REVISIT", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Keep previously observed nodes legal as observe actions. Disabled by default.",
    )
    parser.add_argument(
        "--max-observations-before-stop",
        type=int,
        default=int(os.environ.get("MAX_OBSERVATIONS_BEFORE_STOP", "10")),
        help=(
            "Maximum number of observe decisions before terminal path actions are forced. "
            "Used most often with --allow-node-revisit. Default: 10."
        ),
    )
    parser.add_argument(
        "--observation-sigma",
        "--sigma",
        dest="observation_sigma",
        type=float,
        default=float(os.environ.get("OBSERVATION_SIGMA", "0.0")),
        help=(
            "Standard deviation of Gaussian observation noise. Observed rewards "
            "are sampled as Normal(true_reward, sigma^2). Default: 0."
        ),
    )
    parser.add_argument(
        "--kl-start-multiplier",
        type=float,
        default=float(os.environ.get("KL_START_MULTIPLIER", "1.0")),
        help=(
            "Initial multiplier on the direct memory KL weight lambda. "
            "Use 0 with --kl-annealing-epochs for classic KL warm-up, or >1 "
            "to start with stronger KL pressure and anneal back to target. Default: 1."
        ),
    )
    parser.add_argument(
        "--kl-annealing-epochs",
        type=int,
        default=int(os.environ.get("KL_ANNEALING_EPOCHS", "0")),
        help=(
            "Number of schedule epochs over which KL multiplier moves from "
            "--kl-start-multiplier to 1. Default: 0, no KL annealing."
        ),
    )
    parser.add_argument(
        "--node-coverage-aux-coef",
        type=float,
        default=float(os.environ.get("NODE_COVERAGE_AUX_COEF", "0.0")),
        help=(
            "Optional auxiliary loss coefficient that encourages the expansion "
            "policy to distribute probability over currently unobserved nodes. "
            "The term is independent of the memory lambda and defaults to 0."
        ),
    )
    parser.add_argument(
        "--node-coverage-aux-epochs",
        type=int,
        default=int(os.environ.get("NODE_COVERAGE_AUX_EPOCHS", "0")),
        help=(
            "Linearly anneal --node-coverage-aux-coef to 0 over this many "
            "schedule epochs. If 0, a nonzero coefficient is held constant."
        ),
    )
    parser.add_argument(
        "--critic-huber-delta",
        type=float,
        default=float(os.environ.get("CRITIC_HUBER_DELTA", "10.0")),
        help=(
            "Use Huber loss for expansion critic errors with this delta. Set "
            "<= 0 for legacy squared-error critic loss. Default: 10."
        ),
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=float(os.environ.get("ADVANTAGE_CLIP", "10.0")),
        help=(
            "Clip PPO advantages to +/- this value. Set <= 0 to disable "
            "advantage clipping. Default: 10."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        "--peak-learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=float(os.environ.get("JAX_LEARNING_RATE", "5e-4")),
        help="Peak AdamW learning rate for cosine decay. Default: 5e-4.",
    )
    parser.add_argument(
        "--min-learning-rate",
        "--learning-rate-floor",
        "--lr-floor",
        dest="min_learning_rate",
        type=float,
        default=None if "JAX_MIN_LEARNING_RATE" not in os.environ else float(os.environ["JAX_MIN_LEARNING_RATE"]),
        help="Cosine-decay floor. Default: 0.1 * --learning-rate.",
    )
    parser.add_argument(
        "--pay-kl-on-stop",
        "--pay-memory-cost-on-stop",
        action="store_true",
        default=os.environ.get("PAY_KL_ON_STOP", "").strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "For LSTM/pre-LSTM expansion runs, pay the pending memory KL when the next "
            "decision is a terminal stop action. By default the last observation before "
            "stop remains free, matching older checkpoints. Enabled runs add _stop_paid "
            "to model and simulation filenames."
        ),
    )
    parser.add_argument(
        "--choice-at-end-only",
        "--observer-only",
        "--observer-end-choice",
        action="store_true",
        default=os.environ.get("CHOICE_AT_END_ONLY", "").strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "Train/evaluate as an observer: terminal path choices are masked until "
            "--max-observations-before-stop observe decisions have been made. In "
            "non-revisit tasks terminal choices are also allowed once no observe "
            "actions remain. Enabled runs add _observer_endchoice to filenames."
        ),
    )
    args = parser.parse_args()

    loss_scale_values = parse_float_list(args.loss_scale_string)
    alpha_values = parse_float_list(args.alpha_string)
    memory_lambda_values = parse_float_list(args.memory_lambda_string)
    opportunity_values = parse_float_list(args.opportunity_cost_string)
    if not (
        len(loss_scale_values)
        == len(alpha_values)
        == len(memory_lambda_values)
        == len(opportunity_values)
        == 1
    ):
        raise ValueError(
            "model_jax/planning.py expects one loss_scale/alpha/lambda/opportunity per process."
        )
    tree_size = int(args.tree_size)
    normalized_tree_type = normalize_tree_type(args.tree_type, tree_size)
    max_observations_before_stop = max(int(args.max_observations_before_stop), 1)
    if args.choice_at_end_only:
        default_num_steps = min(max_observations_before_stop, tree_size) + 1
        if args.allow_node_revisit:
            default_num_steps = max_observations_before_stop + 1
    else:
        default_num_steps = (max_observations_before_stop + 1) if args.allow_node_revisit else tree_size
    num_steps = int(args.num_steps or default_num_steps)
    if args.allow_node_revisit and num_steps <= max_observations_before_stop:
        raise ValueError(
            "Revisit mode needs one terminal decision after the observation cap. "
            f"Got --num-steps={num_steps} and --max-observations-before-stop={max_observations_before_stop}."
        )
    if args.choice_at_end_only and num_steps < default_num_steps:
        raise ValueError(
            "Observer/end-choice mode needs one terminal decision after the final allowed observation. "
            f"Got --num-steps={num_steps}; expected at least {default_num_steps}."
        )
    return_target_rollouts = max(int(args.return_target_rollouts), 1)
    num_envs = int(args.num_envs)
    ppo_minibatches = max(int(args.ppo_minibatches), 1)
    if return_target_rollouts * num_envs % ppo_minibatches != 0:
        raise ValueError(
            "--ppo-minibatches must evenly divide return_target_rollouts * num_envs. "
            f"Got return_target_rollouts={return_target_rollouts}, num_envs={num_envs}, "
            f"ppo_minibatches={ppo_minibatches}."
        )
    steps_per_epoch = int(args.steps_per_epoch or (200 * 200 * num_steps))
    sim_dir = args.sim_dir or "outputs/jax_simulations"
    kl_start_multiplier = max(float(args.kl_start_multiplier), 0.0)
    kl_annealing_epochs = max(int(args.kl_annealing_epochs), 0)
    cli_has_kl_start = any(
        arg == "--kl-start-multiplier" or str(arg).startswith("--kl-start-multiplier=")
        for arg in sys.argv[1:]
    )
    cli_has_kl_epochs = any(
        arg == "--kl-annealing-epochs" or str(arg).startswith("--kl-annealing-epochs=")
        for arg in sys.argv[1:]
    )
    if normalized_tree_type == "disjoint3x2":
        if not cli_has_kl_start and "KL_START_MULTIPLIER" not in os.environ:
            kl_start_multiplier = float(os.environ.get("DISJOINT3X2_KL_START_MULTIPLIER", "5.0"))
        if not cli_has_kl_epochs and "KL_ANNEALING_EPOCHS" not in os.environ:
            kl_annealing_epochs = int(os.environ.get("DISJOINT3X2_KL_ANNEALING_EPOCHS", "60"))
    return RunConfig(
        loss_scale=loss_scale_values[0],
        alpha=alpha_values[0],
        memory_lambda=memory_lambda_values[0],
        model_dir=args.model_dir,
        epochs=int(args.epochs),
        input_type=str(args.input_type),
        seed=int(args.seed),
        tree_size=tree_size,
        train_mode=str(args.train_mode),
        tree_type=normalized_tree_type,
        opportunity_cost=opportunity_values[0],
        expansion_decision_version=normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=normalize_model_variant(args.model_variant),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir=sim_dir,
        n_sim_trials=int(args.n_sim_trials),
        num_envs=num_envs,
        num_steps=num_steps,
        update_epochs=int(args.update_epochs),
        ppo_minibatches=ppo_minibatches,
        steps_per_epoch=steps_per_epoch,
        return_target_rollouts=return_target_rollouts,
        return_target_mode=normalize_return_target_mode(args.return_target_mode),
        sampled_lambda_critic=str(args.sampled_lambda_critic),
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=max(int(args.target_critic_update_interval), 0),
        target_critic_tau=float(args.target_critic_tau),
        backend=args.backend,
        jit_training=not bool(args.no_jit),
        profile_update_components=bool(args.profile_update_components),
        profile_update_components_every=max(int(args.profile_update_components_every), 1),
        enable_reconstruction=bool(args.enable_reconstruction),
        enable_probe=bool(args.enable_probe),
        allow_node_revisit=bool(args.allow_node_revisit),
        max_observations_before_stop=max_observations_before_stop,
        observation_sigma=max(float(args.observation_sigma), 0.0),
        kl_start_multiplier=max(float(kl_start_multiplier), 0.0),
        kl_annealing_epochs=max(int(kl_annealing_epochs), 0),
        node_coverage_aux_coef=max(float(args.node_coverage_aux_coef), 0.0),
        node_coverage_aux_epochs=max(int(args.node_coverage_aux_epochs), 0),
        critic_huber_delta=max(float(args.critic_huber_delta), 0.0),
        advantage_clip=max(float(args.advantage_clip), 0.0),
        learning_rate=max(float(args.learning_rate), 0.0),
        min_learning_rate=(
            None if args.min_learning_rate is None else max(float(args.min_learning_rate), 0.0)
        ),
        pay_kl_on_stop=bool(args.pay_kl_on_stop),
        choice_at_end_only=bool(args.choice_at_end_only),
    )


def main():
    config = parse_args()
    task = build_task(config.tree_size, config.tree_type, config.input_type)
    print(
        f"JAX task: {task.tree_type} | nodes: {task.num_nodes} | paths: {task.num_paths} | "
        f"reward_norm: {task.reward_norm:.6g}"
    )
    model = None
    state = None
    if config.train_mode.strip().lower() == "train":
        model, state = train(config, task)
    if model is not None and state is not None:
        simulate(config, task, model, state.params)
    else:
        simulate(config, task)


if __name__ == "__main__":
    main()
