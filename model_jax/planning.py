"""Step-batched JAX trainer for the VAE planning model.

This module is a direct JAX implementation of the current TensorFlow training
objective in ``model/``.  The important parity points are:

* the same positional command-line interface and filename convention;
* the same task/path maps for default, bandit, and disjoint trees;
* joint expansion actions ordered as unobserved-node actions followed by
  terminal path actions;
* an LSTM state, Gaussian posterior, timestep Gaussian prior, decoder, joint
  expansion policy, action-value critic, and reward probe;
* KL scaling semantics match TensorFlow: the command-line beta is an inverse
  KL scale and the effective KL multiplier is ``1 / beta``;
* observe-action Q targets are ``observe one reward, then stop`` targets from
  the current belief state, matching ``build_joint_action_q_targets`` after the
  TensorFlow revert;
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
    lambda_: float
    alpha: float
    beta: float
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
    steps_per_epoch: int
    return_target_rollouts: int
    backend: str | None
    jit_training: bool
    save_every_update: bool = False


class ScheduleValues(NamedTuple):
    current_alpha: jax.Array
    current_beta: jax.Array
    current_critic_coef: jax.Array
    expansion_epsilon: jax.Array
    expansion_entropy_coef: jax.Array
    forced_continue_epsilon: jax.Array
    ppo_clip: jax.Array


class RunnerCarry(NamedTuple):
    rewards: jax.Array
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
    critic_loss: jax.Array
    lstm_probe_loss: jax.Array
    lstm_probe_accuracy: jax.Array
    stop_rate: jax.Array
    continue_rate: jax.Array
    entropy_coef: jax.Array
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


class PlanningTrainState(TrainState):
    target_params: object


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


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
    return (
        f"lambda_{config.lambda_}_alpha_{config.alpha}_beta_{config.beta}_"
        f"opportunity_{config.opportunity_cost}_expansion_{config.expansion_decision_version}_"
        f"{model_variant_label(config.model_variant)}"
        f"seed_{config.seed}_{tree_label}_{architecture_file_label(config.rnn_units, config.latent_dim)}"
    )


def scalar_to_category_index(values: jax.Array) -> jax.Array:
    indices = jnp.floor(4.0 - values + 0.5).astype(jnp.int32)
    return jnp.clip(indices, 0, 8)


def scalar_to_onehot(values: jax.Array) -> jax.Array:
    return jax.nn.one_hot(scalar_to_category_index(values), 9, dtype=jnp.float32)


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


def initial_carry(num_envs: int, task: TaskSpec, rnn_units: int) -> RunnerCarry:
    return RunnerCarry(
        rewards=jnp.zeros((num_envs, task.num_nodes), dtype=jnp.float32),
        observed=jnp.zeros((num_envs, task.num_nodes), dtype=jnp.float32),
        step_index=jnp.zeros((num_envs,), dtype=jnp.int32),
        done=jnp.ones((num_envs,), dtype=jnp.bool_),
        h=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        c=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        decoded_h=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        decoded_c=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        lstm_context=jnp.zeros((num_envs, rnn_units), dtype=jnp.float32),
        pre_context=jnp.zeros((num_envs, rnn_units + task.num_nodes + 1 + 9), dtype=jnp.float32),
        pending_kl=jnp.zeros((num_envs,), dtype=jnp.float32),
        last_reward_onehot=jnp.zeros((num_envs, 9), dtype=jnp.float32),
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
    opportunity_cost: float
    lambda_: float
    alpha: float
    beta: float

    def setup(self):
        input_dim = self.time_steps + 1 + 9
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
        self.probe_head = nn.Dense(9, kernel_init=nn.initializers.glorot_uniform())
        self.prior_mu = self.param("prior_mu", nn.initializers.zeros, (self.time_steps, self.latent_dim))
        self.prior_logvar = self.param("prior_logvar", nn.initializers.zeros, (self.time_steps, self.latent_dim))

    def encode(self, x: jax.Array, rng: jax.Array, use_mean: bool = False):
        h = nn.relu(self.enc_ln(self.enc_dense(x)))
        mu = self.z_mean(h)
        logvar = jnp.clip(self.z_logvar(h), -10.0, 10.0)
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
    ) -> tuple[RunnerCarry, StepTransition]:
        path_map = jnp.asarray(self.path_map, dtype=jnp.float32)
        reward_values = jnp.asarray(self.reward_values, dtype=jnp.float32)
        batch_size = carry.rewards.shape[0]
        expansion_input = self.expansion_input(carry)
        q_values = self.critic(expansion_input)
        logits = self.expansion_head(expansion_input)
        observed_count = jnp.sum(carry.observed, axis=-1, keepdims=True)
        min_observations = 1.0 if self.expansion_decision_version == "decoder" else 0.0
        can_stop = (observed_count >= min_observations).astype(jnp.float32)
        terminal_invalid = (1.0 - can_stop) * jnp.ones((batch_size, self.num_paths), dtype=jnp.float32)
        decision_mask = jnp.concatenate([carry.observed, terminal_invalid], axis=-1)
        legal_mask = 1.0 - decision_mask
        masked_logits = logits + decision_mask * -1e9
        probs = jax.nn.softmax(masked_logits, axis=-1)
        log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)

        rng_action, rng_eps, rng_force, rng_z = jax.random.split(rng, 4)
        sampled_action = jax.random.categorical(rng_action, masked_logits, axis=-1)
        if forced_action is not None:
            action = forced_action.astype(jnp.int32)
        elif training:
            explore = jax.random.uniform(rng_eps, (batch_size,)) < schedule.expansion_epsilon
            uniform_logits = decision_mask * -1e9
            uniform_action = jax.random.categorical(rng_eps, uniform_logits, axis=-1)
            legal_observe = 1.0 - carry.observed
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
        is_stop = action >= self.time_steps
        is_observe = ~is_stop
        safe_node = jnp.minimum(action, self.time_steps - 1)
        terminal_path = jnp.where(is_stop, action - self.time_steps, -jnp.ones_like(action))
        chosen_reward = jnp.take_along_axis(carry.rewards, safe_node[:, None], axis=1)[:, 0]
        reward_onehot = scalar_to_onehot(chosen_reward) * is_observe[:, None].astype(jnp.float32)
        node_token = jnp.where(is_observe, safe_node, self.time_steps)
        node_onehot = jax.nn.one_hot(node_token, self.time_steps + 1, dtype=jnp.float32)
        lstm_input = jnp.concatenate([node_onehot, reward_onehot], axis=-1)

        prev_decoded_h = carry.decoded_h
        prev_decoded_c = carry.decoded_c
        pre_context = jnp.concatenate([prev_decoded_h, lstm_input], axis=-1)
        raw_h, raw_c = self.lstm_cell(lstm_input, carry.decoded_h, carry.decoded_c)
        observe_mask = is_observe[:, None].astype(jnp.float32)
        raw_h = raw_h * observe_mask + carry.h * (1.0 - observe_mask)
        raw_c = raw_c * observe_mask + carry.c * (1.0 - observe_mask)
        encoder_input = jnp.concatenate([raw_h, raw_c], axis=-1)

        if self.use_autoencoder:
            z_mu, z_logvar, z = self.encode(encoder_input, rng_z, use_mean=use_posterior_mean)
            dec_h, dec_c = self.decode(z)
            dec_h = dec_h * observe_mask + prev_decoded_h * (1.0 - observe_mask)
            dec_c = dec_c * observe_mask + prev_decoded_c * (1.0 - observe_mask)
            prior_mu, _prior_logvar, prior_var = self.prior(carry.step_index)
            post_var = jnp.exp(jnp.clip(z_logvar, -10.0, 10.0))
            kl_per_dim = 0.5 * (
                jnp.log(prior_var + 1e-6)
                - jnp.log(post_var + 1e-6)
                + (post_var + jnp.square(z_mu - prior_mu)) / (prior_var + 1e-6)
                - 1.0
            )
            kl_per_sample = jnp.sum(kl_per_dim, axis=-1)
            observed_kl = kl_per_sample * is_observe.astype(jnp.float32)
            if self.expansion_decision_version in ("lstm", "pre_lstm"):
                paid_kl = carry.pending_kl * is_observe.astype(jnp.float32)
                pending_kl = observed_kl
            else:
                paid_kl = observed_kl
                pending_kl = carry.pending_kl
        else:
            dec_h, dec_c = raw_h, raw_c
            z_mu = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            observed_kl = jnp.zeros((batch_size,), dtype=jnp.float32)
            paid_kl = jnp.zeros((batch_size,), dtype=jnp.float32)
            pending_kl = carry.pending_kl

        action_input_post = dec_h if self.expansion_decision_version == "decoder" else raw_h
        if self.expansion_decision_version == "pre_lstm":
            action_input_post = pre_context
        terminal_probs_pre = jax.nn.softmax(logits[:, self.time_steps :], axis=-1)
        terminal_probs_post = jax.nn.softmax(self.expansion_head(action_input_post)[:, self.time_steps :], axis=-1)
        action_output = jnp.where(is_stop[:, None], terminal_probs_pre, terminal_probs_post)

        node_obs = jax.nn.one_hot(safe_node, self.time_steps, dtype=jnp.float32) * observe_mask
        observed_after = jnp.minimum(carry.observed + node_obs, 1.0)
        rec_probs = self.reconstruct_probs(dec_h) if self.use_autoencoder else (
            jnp.ones((batch_size, self.time_steps, 9), dtype=jnp.float32) / 9.0
        )
        target_onehot = scalar_to_onehot(carry.rewards)
        rec_ce = -jnp.sum(target_onehot * jnp.log(rec_probs + 1e-8), axis=-1) / jnp.log(9.0)
        reconstruction_loss = jnp.sum(rec_ce * observed_after, axis=-1) / (jnp.sum(observed_after, axis=-1) + 1e-6)

        probe_logits = self.probe_head(jax.lax.stop_gradient(raw_h))
        reward_idx = scalar_to_category_index(chosen_reward)
        probe_ce = optax.softmax_cross_entropy_with_integer_labels(probe_logits, reward_idx)
        probe_loss = probe_ce * is_observe.astype(jnp.float32)
        probe_correct = (jnp.argmax(probe_logits, axis=-1) == reward_idx).astype(jnp.float32) * is_observe.astype(jnp.float32)

        observed_rewards = carry.rewards[:, None, :] * carry.observed[:, None, :]
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

        path_rewards = carry.rewards @ path_map.T
        terminal_expected_reward = jnp.sum(action_output * path_rewards, axis=-1) / self.reward_norm_value
        episode_done = is_stop | ((carry.step_index >= (self.time_steps - 1)) & is_observe)
        next_step = carry.step_index + 1
        next_carry = RunnerCarry(
            rewards=carry.rewards,
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
            paid_kl=paid_kl,
            observed_kl=observed_kl,
            expanded_reward=jnp.where(is_observe, chosen_reward, jnp.nan),
            action_output=action_output,
            terminal_expected_reward=terminal_expected_reward,
            reconstruction_loss=reconstruction_loss,
            probe_loss=probe_loss,
            probe_correct=probe_correct,
            valid_probe=is_observe.astype(jnp.float32),
            reset_rewards=carry.rewards,
            reset_trial_id=carry.trial_id,
        )
        return next_carry, transition


def make_schedule(config: RunConfig, update_idx: int, updates_per_epoch: int) -> ScheduleValues:
    # TensorFlow computes schedules once per epoch and reuses that value for
    # every batch in the epoch. Keep the JAX schedule epoch-discrete too.
    epoch = update_idx // max(updates_per_epoch, 1)
    target_beta = 1.0 / config.beta
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
        entropy_start, entropy_end, entropy_epochs, entropy_hold = 1.5, 0.01, 80.0, 90.0
    else:
        entropy_start, entropy_end, entropy_epochs, entropy_hold = 1.0, 0.0, 50.0, 100.0
    if epoch >= entropy_hold:
        entropy = 0.0
    elif epoch >= entropy_epochs:
        entropy = entropy_end
    else:
        progress = epoch / max(entropy_epochs - 1.0, 1.0)
        entropy = entropy_start + (entropy_end - entropy_start) * progress
    return ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(target_beta, dtype=jnp.float32),
        current_critic_coef=jnp.asarray(critic_coef, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(entropy, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.2, dtype=jnp.float32),
    )


def learning_rate_at(step: jax.Array, total_steps: int) -> jax.Array:
    progress = jnp.minimum(step.astype(jnp.float32) / float(max(total_steps, 1)), 1.0)
    peak = jnp.asarray(3e-4, dtype=jnp.float32)
    floor = peak * 0.1
    return floor + 0.5 * (peak - floor) * (1.0 + jnp.cos(jnp.pi * progress))


def reduced_state_weights(transitions: StepTransition, time_steps: int) -> jax.Array:
    observed_count = jnp.sum(transitions.observed_before, axis=-1).astype(jnp.int32)
    observed_rewards = transitions.rewards[:, :, None, :] * transitions.observed_before[:, :, None, :]
    # transitions arrays are [num_steps, num_envs, ...]; path calculation is done outside
    return observed_count


def best_path_value_indices(transitions: StepTransition, path_map: jax.Array) -> tuple[jax.Array, jax.Array]:
    observed_rewards = transitions.rewards * transitions.observed_before
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
    num_bins = time_steps * 9 * (time_steps + 1)
    counts = jnp.bincount(flat_bins, length=num_bins).astype(jnp.float32)
    weights = 1.0 / jnp.maximum(counts[bins], 1.0)
    weights = weights / (jnp.mean(weights) + 1e-6)
    weights = jnp.clip(weights, 0.01, 100.0)
    weights = weights / (jnp.mean(weights) + 1e-6)
    return jax.lax.stop_gradient(weights)


def aggregate_best_value_metrics(
    transitions: StepTransition,
    path_map: jax.Array,
    time_steps: int,
) -> tuple[jax.Array, ...]:
    _best_values, best_idx = best_path_value_indices(transitions, path_map)
    category = jax.nn.one_hot(best_idx, 9, dtype=jnp.float32)
    step_onehot = jax.nn.one_hot(jnp.clip(transitions.step_index, 0, time_steps - 1), time_steps, dtype=jnp.float32)
    masks = step_onehot[:, :, :, None] * category[:, :, None, :]
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


def build_rollout_fn(model: PlanningVAE, task: TaskSpec, config: RunConfig):
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)

    def rollout(
        params,
        carry: RunnerCarry,
        rng: jax.Array,
        schedule: ScheduleValues,
        forced_actions=None,
        reset_rewards_seq=None,
        training: bool = True,
    ):
        def scan_step(scan_carry, step_i):
            step_carry, step_rng = scan_carry
            step_rng, reset_rng, model_rng = jax.random.split(step_rng, 3)
            if reset_rewards_seq is None:
                reset_rewards = sample_reward_matrix(reset_rng, config.num_envs, task.num_nodes, task.reward_values)
            else:
                reset_rewards = reset_rewards_seq[step_i]
            step_carry = reset_done_envs(step_carry, reset_rewards)
            forced_action = None if forced_actions is None else forced_actions[step_i]
            next_carry, transition = model.apply(
                {"params": params},
                step_carry,
                model_rng,
                schedule,
                forced_action,
                training,
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
    dummy_carry = initial_carry(config.num_envs, task, config.rnn_units)
    dummy_schedule = ScheduleValues(1.0, 1.0 / config.beta, 0.1, 0.0, 1.0, 0.0, 0.2)
    params = model.init(rng, dummy_carry, rng, dummy_schedule, None, True)["params"]
    schedule = lambda step: learning_rate_at(step, total_updates * max(config.update_epochs, 1))
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


def build_update_fn(model: PlanningVAE, task: TaskSpec, config: RunConfig, total_updates: int):
    rollout = build_rollout_fn(model, task, config)
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)
    backend = config.backend

    def update_step(train_state: PlanningTrainState, carry: RunnerCarry, rng: jax.Array, schedule: ScheduleValues):
        rng, reset_rng, rollout_rng, replay_rng = jax.random.split(rng, 4)
        reset_rewards_seq = sample_reward_sequence(
            reset_rng,
            config.num_steps,
            config.num_envs,
            task.num_nodes,
            task.reward_values,
        )
        rollout_keys = jax.random.split(rollout_rng, config.return_target_rollouts)
        replay_keys = jax.random.split(replay_rng, config.return_target_rollouts)

        def collect_one(rollout_key):
            rollout_carry, _rollout_rng, transitions = rollout(
                train_state.params,
                carry,
                rollout_key,
                schedule,
                forced_actions=None,
                reset_rewards_seq=reset_rewards_seq,
                training=True,
            )
            return rollout_carry, transitions

        rollout_carries, old_transitions = jax.vmap(collect_one)(rollout_keys)
        collect_carry = jax.tree_util.tree_map(lambda x: x[0], rollout_carries)
        forced_actions = jax.lax.stop_gradient(old_transitions.action)
        old_logp = jax.lax.stop_gradient(old_transitions.log_prob)

        def loss_for_rollout(params, replay_key, rollout_actions, rollout_old_logp):
            _, _, transitions = rollout(
                params,
                carry,
                replay_key,
                schedule,
                forced_actions=rollout_actions,
                reset_rewards_seq=reset_rewards_seq,
                training=True,
            )
            weights = compute_policy_weights(transitions, path_map, task.num_nodes)
            advantages = jax.lax.stop_gradient(transitions.selected_q_target - transitions.policy_value_pred)
            ratio = jnp.exp(jnp.clip(transitions.log_prob - rollout_old_logp, -10.0, 10.0))
            clipped_ratio = jnp.clip(ratio, 1.0 - schedule.ppo_clip, 1.0 + schedule.ppo_clip)
            policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages) * weights
            entropy_loss = transitions.entropy
            expansion_loss = (
                jnp.mean(policy_loss)
                - schedule.expansion_entropy_coef * jnp.mean(entropy_loss)
            )
            critic_err = jnp.square(transitions.q_values - transitions.q_targets) * transitions.legal_mask
            critic_loss = jnp.sum(critic_err) / (jnp.sum(transitions.legal_mask) + 1e-6)
            information_loss = jnp.mean(transitions.paid_kl) / float(task.num_nodes) / 5.0
            reconstruction_loss = jnp.mean(transitions.reconstruction_loss)
            done_mask = (transitions.is_stop > 0) | (
                (transitions.step_index >= (task.num_nodes - 1)) & (transitions.is_observe > 0)
            )
            action_loss = -jnp.sum(transitions.terminal_expected_reward * done_mask.astype(jnp.float32)) / (
                jnp.sum(done_mask.astype(jnp.float32)) + 1e-6
            )
            probe_loss = jnp.sum(transitions.probe_loss) / (jnp.sum(transitions.valid_probe) + 1e-6)
            total_loss = (
                information_loss * schedule.current_beta
                + action_loss * config.lambda_
                + expansion_loss * config.lambda_
                + critic_loss * config.lambda_ * schedule.current_critic_coef
                + reconstruction_loss * config.alpha
                + probe_loss
            )
            expansion_head_loss = (
                expansion_loss * config.lambda_
                + critic_loss * config.lambda_ * schedule.current_critic_coef
                + action_loss * config.lambda_
            )
            metrics_parts = aggregate_best_value_metrics(transitions, path_map, task.num_nodes)
            probe_acc = jnp.sum(transitions.probe_correct) / (jnp.sum(transitions.valid_probe) + 1e-6)
            metrics = UpdateMetrics(
                total_loss=total_loss,
                information_loss=information_loss,
                action_loss=action_loss,
                reconstruction_loss=reconstruction_loss,
                expansion_loss=expansion_loss,
                critic_loss=critic_loss,
                lstm_probe_loss=probe_loss,
                lstm_probe_accuracy=probe_acc,
                stop_rate=jnp.mean(transitions.is_stop),
                continue_rate=jnp.mean(transitions.is_observe),
                entropy_coef=jnp.asarray(schedule.expansion_entropy_coef),
                critic_coef=jnp.asarray(schedule.current_critic_coef),
                current_beta=jnp.asarray(schedule.current_beta),
                learning_rate=learning_rate_at(train_state.step, total_updates * max(config.update_epochs, 1)),
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

        def loss_fn(params):
            losses, _expansion_head_losses, metrics_tree = jax.vmap(
                lambda replay_key, rollout_actions, rollout_old_logp: loss_for_rollout(
                    params,
                    replay_key,
                    rollout_actions,
                    rollout_old_logp,
                )
            )(replay_keys, forced_actions, old_logp)
            first_metrics = jax.tree_util.tree_map(lambda x: x[0], metrics_tree)
            metrics = first_metrics._replace(
                total_loss=jnp.mean(metrics_tree.total_loss),
                information_loss=jnp.mean(metrics_tree.information_loss),
                action_loss=jnp.mean(metrics_tree.action_loss),
                reconstruction_loss=jnp.mean(metrics_tree.reconstruction_loss),
                expansion_loss=jnp.mean(metrics_tree.expansion_loss),
                critic_loss=jnp.mean(metrics_tree.critic_loss),
                lstm_probe_loss=jnp.mean(metrics_tree.lstm_probe_loss),
                lstm_probe_accuracy=jnp.mean(metrics_tree.lstm_probe_accuracy),
                stop_rate=jnp.mean(metrics_tree.stop_rate),
                continue_rate=jnp.mean(metrics_tree.continue_rate),
            )
            return jnp.mean(losses), metrics

        def expansion_head_loss_fn(params):
            _losses, expansion_head_losses, _metrics_tree = jax.vmap(
                lambda replay_key, rollout_actions, rollout_old_logp: loss_for_rollout(
                    params,
                    replay_key,
                    rollout_actions,
                    rollout_old_logp,
                )
            )(replay_keys, forced_actions, old_logp)
            return jnp.mean(expansion_head_losses)

        (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)
        expansion_head_mask = top_level_param_mask(new_train_state.params, {"expansion_head"})

        def ppo_expansion_only_step(state, _):
            grads = jax.grad(expansion_head_loss_fn)(state.params)
            state = apply_masked_gradients(state, grads, expansion_head_mask)
            return state, None

        new_train_state, _ = jax.lax.scan(
            ppo_expansion_only_step,
            new_train_state,
            xs=None,
            length=max(config.update_epochs - 1, 0),
        )
        # Keep the target params field synchronized; the current TensorFlow target
        # critic no longer affects selected observe-action targets after the revert.
        new_train_state = new_train_state.replace(target_params=new_train_state.params)
        return new_train_state, collect_carry, rng, metrics

    if not config.jit_training:
        return update_step
    if backend:
        return jax.jit(update_step, backend=backend)
    return jax.jit(update_step)


def empty_metric_accumulator(time_steps: int) -> dict[str, np.ndarray | float]:
    shape = (time_steps, 9)
    return {
        "total_loss": 0.0,
        "information_loss": 0.0,
        "action_loss": 0.0,
        "reconstruction_loss": 0.0,
        "expansion_loss": 0.0,
        "critic_loss": 0.0,
        "lstm_probe_loss": 0.0,
        "lstm_probe_accuracy": 0.0,
        "stop_rate": 0.0,
        "continue_rate": 0.0,
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
        "critic_loss",
        "lstm_probe_loss",
        "lstm_probe_accuracy",
        "stop_rate",
        "continue_rate",
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
        "steps_per_epoch": config.steps_per_epoch,
        "steps_per_batch": config.num_envs * config.num_steps,
        "updates_per_epoch": updates_per_epoch,
        "rollout_steps": config.num_steps,
        "expansion_return_target_mode": "one_step_observe_then_stop",
        "expansion_lambda_return": float("nan"),
        "target_critic_update_interval": 0,
        "target_critic_tau": float("nan"),
        "forced_continue_epsilon": 0.0,
        "expansion_entropy_coef": float(np.asarray(metrics.entropy_coef)),
        "critic_coef": float(np.asarray(metrics.critic_coef)),
        "current_beta": float(np.asarray(metrics.current_beta)),
    }
    for name in [
        "total_loss",
        "information_loss",
        "action_loss",
        "reconstruction_loss",
        "expansion_loss",
        "critic_loss",
        "lstm_probe_loss",
        "lstm_probe_accuracy",
        "stop_rate",
        "continue_rate",
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
        opportunity_cost=config.opportunity_cost,
        lambda_=config.lambda_,
        alpha=config.alpha,
        beta=config.beta,
    )
    updates_per_epoch = max(1, math.ceil(config.steps_per_epoch / (config.num_envs * config.num_steps)))
    total_updates = config.epochs * updates_per_epoch
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, config, task, init_rng, total_updates)
    carry = initial_carry(config.num_envs, task, config.rnn_units)
    update_fn = build_update_fn(model, task, config, total_updates)
    rows = []
    for epoch in range(config.epochs):
        acc = empty_metric_accumulator(task.num_nodes)
        last_metrics = None
        for update_in_epoch in range(updates_per_epoch):
            update_idx = epoch * updates_per_epoch + update_in_epoch
            schedule = make_schedule(config, update_idx, updates_per_epoch)
            state, carry, rng, metrics = update_fn(state, carry, rng, schedule)
            last_metrics = metrics
            add_metrics(acc, metrics)
        row = finalize_epoch_row(epoch + 1, acc, updates_per_epoch, last_metrics, config, task, updates_per_epoch)
        rows.append(row)
        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"Loss = {row['total_loss']:.4f} | KL = {row['kl_loss']:.4f} | "
            f"Stop = {row['expansion_stop_rate']:.4f} | Continue = {row['expansion_continue_rate']:.4f}",
            flush=True,
        )
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name_for(config, task)
    log_path = model_dir / f"{model_name}_training_logs.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    weights_path = model_dir / f"{model_name}.msgpack"
    weights_path.write_bytes(serialization.to_bytes(state.params))
    print(f"Saved JAX training logs to: {log_path}")
    print(f"Saved JAX parameters to: {weights_path}")
    return model, state


def load_state_for_sim(config: RunConfig, task: TaskSpec) -> tuple[PlanningVAE, object]:
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
        opportunity_cost=config.opportunity_cost,
        lambda_=config.lambda_,
        alpha=config.alpha,
        beta=config.beta,
    )
    rng = jax.random.PRNGKey(config.seed)
    dummy = initial_carry(1, task, config.rnn_units)
    sched = ScheduleValues(1.0, 1.0 / config.beta, 0.0, 0.0, 0.0, 0.0, 0.2)
    params = model.init(rng, dummy, rng, sched, None, False)["params"]
    weights_path = Path(config.model_dir) / f"{model_name_for(config, task)}.msgpack"
    if weights_path.exists():
        params = serialization.from_bytes(params, weights_path.read_bytes())
    else:
        print(f"Warning: {weights_path} not found; simulating initialized JAX model.")
    return model, params


def simulate(config: RunConfig, task: TaskSpec, model: PlanningVAE | None = None, params=None):
    if model is None or params is None:
        model, params = load_state_for_sim(config, task)
    rng = jax.random.PRNGKey(config.seed + 100_000)
    num_trials = int(config.n_sim_trials)
    carry = initial_carry(num_trials, task, config.rnn_units)
    rng, reset_rng = jax.random.split(rng)
    reset_rewards = sample_reward_matrix(reset_rng, num_trials, task.num_nodes, task.reward_values)
    carry = reset_done_envs(carry, reset_rewards)
    sched = ScheduleValues(1.0, 1.0 / config.beta, 0.0, 0.0, 0.0, 0.0, 0.2)
    transitions = []
    for _ in range(task.num_nodes):
        rng, step_rng = jax.random.split(rng)
        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            sched,
            None,
            False,
            True,
            method=PlanningVAE.__call__,
        )
        transitions.append(jax.device_get(trans))
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
            chosen_path = int(np.argmax(terminal_probs_last))
            chosen_value = float(path_rewards[chosen_path])
        trial_row = {
            "graph": trial,
            "chosen_path": chosen_path,
            "V": chosen_value,
            "MI": float(np.nansum([np.asarray(t.paid_kl)[trial] for t in transitions])),
            "opportunity_cost": config.opportunity_cost,
            "expansion_decision_version": config.expansion_decision_version,
        }
        for t, trans in enumerate(transitions, start=1):
            node_idx = int(np.asarray(trans.node_index)[trial])
            trial_row[f"expanded_node_t{t}"] = np.nan if node_idx < 0 else node_idx + 1
            trial_row[f"expanded_reward_t{t}"] = np.asarray(trans.expanded_reward)[trial]
            trial_row[f"stop_t{t}"] = bool(np.asarray(trans.is_stop)[trial] > 0)
            trial_row[f"kl_d_t{t}"] = float(np.asarray(trans.paid_kl)[trial])
            trial_row[f"kl_d_obs_t{t}"] = float(np.asarray(trans.observed_kl)[trial])
        for node in range(task.num_nodes):
            row = dict(trial_row)
            row["node"] = node + 1
            row["actual_reward"] = float(rewards[node])
            for t in range(1, task.num_nodes + 1):
                row[f"estimated_reward_t{t}"] = np.nan
            rows.append(row)
    out_dir = Path(config.sim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name_for(config, task)}_{config.input_type}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved JAX simulation results to: {out_path}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_string")
    parser.add_argument("alpha_string")
    parser.add_argument("beta_string")
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
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--return-target-rollouts",
        type=int,
        default=int(os.environ.get("RETURN_TARGET_ROLLOUTS", "8")),
    )
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT compilation for tiny smoke/debug runs. Full training remains faster with JIT.",
    )
    args = parser.parse_args()

    lambda_values = parse_float_list(args.lambda_string)
    alpha_values = parse_float_list(args.alpha_string)
    beta_values = parse_float_list(args.beta_string)
    opportunity_values = parse_float_list(args.opportunity_cost_string)
    if not (len(lambda_values) == len(alpha_values) == len(beta_values) == len(opportunity_values) == 1):
        raise ValueError("model_jax/planning.py expects one lambda/alpha/beta/opportunity per process.")
    tree_size = int(args.tree_size)
    num_steps = int(args.num_steps or tree_size)
    steps_per_epoch = int(args.steps_per_epoch or (200 * 200 * tree_size))
    sim_dir = args.sim_dir or "outputs/jax_simulations"
    return RunConfig(
        lambda_=lambda_values[0],
        alpha=alpha_values[0],
        beta=beta_values[0],
        model_dir=args.model_dir,
        epochs=int(args.epochs),
        input_type=str(args.input_type),
        seed=int(args.seed),
        tree_size=tree_size,
        train_mode=str(args.train_mode),
        tree_type=str(args.tree_type),
        opportunity_cost=opportunity_values[0],
        expansion_decision_version=normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=normalize_model_variant(args.model_variant),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir=sim_dir,
        n_sim_trials=int(args.n_sim_trials),
        num_envs=int(args.num_envs),
        num_steps=num_steps,
        update_epochs=int(args.update_epochs),
        steps_per_epoch=steps_per_epoch,
        return_target_rollouts=max(int(args.return_target_rollouts), 1),
        backend=args.backend,
        jit_training=not bool(args.no_jit),
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
