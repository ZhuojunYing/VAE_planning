"""
PureJaxRL-style recurrent PPO trainer for the planning tasks.

This file is intentionally separate from the TensorFlow/Keras implementation.
It follows the recurrent PPO structure used in PureJaxRL: collect a fixed
number of environment steps with ``jax.lax.scan``, keep the RNN hidden state in
the runner state, and carry that state across update batches so unfinished
episodes resume rather than being reset at rollout boundaries.

The compiled training function is forced onto the CPU backend. Batches are
defined by ``num_steps * num_envs`` environment steps, not by a number of full
episodes.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from flax import serialization
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


jax.config.update("jax_platform_name", "cpu")


REWARD_VALUES = np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float32)


@dataclass(frozen=True)
class JaxPlanningConfig:
    total_timesteps: int = 1_000_000
    num_envs: int = 128
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    rnn_units: int = 64
    latent_dim: int = 32
    tree_size: int = 2
    tree_type: str = "default"
    input_type: str = "uniform"
    opportunity_cost: float = 0.0
    lambda_: float = 100.0
    alpha: float = 0.0
    beta: float = 1.0
    expansion_decision_version: str = "lstm"
    model_variant: str = "jax_ppo"
    train_mode: str = "train"
    n_sim_trials: int = 2000
    seed: int = 1
    model_dir: str = "outputs/jax_models"
    sim_dir: str = "outputs/simulations"


@dataclass(frozen=True)
class TaskSpec:
    path_map: np.ndarray
    reward_values: np.ndarray
    reward_norm: float
    tree_type: str

    @property
    def num_nodes(self) -> int:
        return int(self.path_map.shape[1])

    @property
    def num_paths(self) -> int:
        return int(self.path_map.shape[0])


class EnvState(NamedTuple):
    rewards: jax.Array
    observed: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    done: jax.Array
    action_mask: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    next_done: jax.Array


class VAEForwardResult(NamedTuple):
    category_outputs: jax.Array
    action_output: jax.Array
    total_loss: jax.Array
    first_decoder_loss: jax.Array
    second_decoder_loss: jax.Array
    action_head_loss: jax.Array
    critic_loss: jax.Array
    information_loss: jax.Array
    action_loss: jax.Array
    reconstruction_loss: jax.Array
    information_cost: jax.Array
    z_means: jax.Array
    node_selections: jax.Array
    stop_decisions: jax.Array
    observed_masks: jax.Array
    action_outputs_sequence: jax.Array
    kl_d_sequence: jax.Array
    expansion_head_loss: jax.Array
    expansion_log_probs: jax.Array
    expansion_loss: jax.Array
    expansion_stop_rate: jax.Array
    expansion_continue_rate: jax.Array
    opportunity_loss: jax.Array
    lstm_probe_loss: jax.Array
    lstm_probe_accuracy: jax.Array
    terminal_path_output: jax.Array
    observation_kl_d_sequence: jax.Array
    lstm_state_sequence: jax.Array
    decoder_state_sequence: jax.Array
    expansion_probs_sequence: jax.Array
    legal_action_masks: jax.Array
    stop_value_preds: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    last_done: jax.Array
    last_action_mask: jax.Array
    hstate: jax.Array
    rng: jax.Array


def normalize_tree_type(tree_type: str, tree_size: int) -> str:
    key = str(tree_type).strip().lower()
    aliases = {
        "": "default",
        "legacy": "default",
        "default": "default",
        "bandit": f"bandit{tree_size}" if tree_size in (3, 4) else "bandit",
        "bandit3": "bandit3",
        "3armed": "bandit3",
        "3_arm": "bandit3",
        "bandit4": "bandit4",
        "4armed": "bandit4",
        "4_arm": "bandit4",
        "disjoint2x2": "disjoint2x2",
        "2x2": "disjoint2x2",
        "disjoint3x2": "disjoint3x2",
        "3x2": "disjoint3x2",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported JAX tree_type={tree_type!r}.")
    normalized = aliases[key]
    if normalized == "bandit":
        raise ValueError("tree_type='bandit' requires tree_size 3 or 4.")
    if normalized == "default" and tree_size not in (2, 6):
        raise ValueError("tree_type='default' is supported for tree_size 2 or 6.")
    return normalized


def build_task(tree_size: int, tree_type: str, input_type: str) -> TaskSpec:
    tree_type = normalize_tree_type(tree_type, tree_size)
    if tree_type == "default" and tree_size == 2:
        path_map = np.eye(2, dtype=np.float32)
    elif tree_type == "default" and tree_size == 6:
        path_map = np.asarray(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
    elif tree_type == "bandit3":
        path_map = np.eye(3, dtype=np.float32)
    elif tree_type == "bandit4":
        path_map = np.eye(4, dtype=np.float32)
    elif tree_type == "disjoint2x2":
        path_map = np.asarray(
            [[1, 1, 0, 0], [0, 0, 1, 1]],
            dtype=np.float32,
        )
    elif tree_type == "disjoint3x2":
        path_map = np.asarray(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported JAX task {tree_type=} {tree_size=}.")

    if str(input_type).strip().lower() == "binary":
        reward_values = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        reward_values = REWARD_VALUES
    reward_norm = expected_max_path_reward(path_map, reward_values)
    return TaskSpec(
        path_map=path_map,
        reward_values=reward_values,
        reward_norm=reward_norm,
        tree_type=tree_type,
    )


def expected_max_path_reward(path_map: np.ndarray, reward_values: np.ndarray) -> float:
    num_nodes = int(path_map.shape[1])
    total = len(reward_values) ** num_nodes
    if total > 2_000_000:
        samples = np.random.default_rng(0).choice(reward_values, size=(2_000_000, num_nodes))
        path_rewards = samples @ path_map.T
        return float(np.mean(np.max(path_rewards, axis=1)))
    grids = np.meshgrid(*([reward_values] * num_nodes), indexing="ij")
    rewards = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    path_rewards = rewards @ path_map.T
    return float(np.mean(np.max(path_rewards, axis=1)))


def sample_rewards(rng: jax.Array, num_envs: int, task: TaskSpec) -> jax.Array:
    values = jnp.asarray(task.reward_values, dtype=jnp.float32)
    idx = jax.random.randint(
        rng,
        shape=(num_envs, task.num_nodes),
        minval=0,
        maxval=len(task.reward_values),
    )
    return values[idx]


def reset_env(rng: jax.Array, num_envs: int, task: TaskSpec) -> EnvState:
    return EnvState(
        rewards=sample_rewards(rng, num_envs, task),
        observed=jnp.zeros((num_envs, task.num_nodes), dtype=jnp.bool_),
    )


def make_obs(state: EnvState, task: TaskSpec) -> jax.Array:
    reward_scale = float(np.max(np.abs(task.reward_values)))
    reward_scale = reward_scale if reward_scale > 0 else 1.0
    observed_rewards = jnp.where(state.observed, state.rewards / reward_scale, 0.0)
    observed = state.observed.astype(jnp.float32)
    progress = jnp.mean(observed, axis=-1, keepdims=True)
    best_observed = jnp.max(
        jnp.where(state.observed, state.rewards, -1e9),
        axis=-1,
        keepdims=True,
    )
    best_observed = jnp.where(best_observed < -1e8, 0.0, best_observed / reward_scale)
    return jnp.concatenate([observed_rewards, observed, progress, best_observed], axis=-1)


def action_mask(state: EnvState, task: TaskSpec) -> jax.Array:
    observe_mask = jnp.logical_not(state.observed)
    stop_mask = jnp.ones((state.rewards.shape[0], task.num_paths), dtype=jnp.bool_)
    return jnp.concatenate([observe_mask, stop_mask], axis=-1)


def tree_where(mask: jax.Array, a, b):
    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
        a,
        b,
    )


def env_step(
    rng: jax.Array,
    state: EnvState,
    action: jax.Array,
    task: TaskSpec,
    opportunity_cost: float,
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    num_nodes = task.num_nodes
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)
    batch = state.rewards.shape[0]

    is_observe = action < num_nodes
    node_action = jnp.clip(action, 0, num_nodes - 1)
    path_action = jnp.clip(action - num_nodes, 0, task.num_paths - 1)
    node_onehot = jax.nn.one_hot(node_action, num_nodes, dtype=jnp.bool_)
    already_observed = jnp.sum(state.observed & node_onehot, axis=-1) > 0
    observe_is_legal = is_observe & jnp.logical_not(already_observed)
    observed_count = jnp.sum(state.observed, axis=-1)

    next_observed = state.observed | (node_onehot & observe_is_legal[:, None])
    path_rewards = state.rewards @ path_map.T
    terminal_reward = jnp.take_along_axis(
        path_rewards,
        path_action[:, None],
        axis=1,
    ).squeeze(-1) / float(task.reward_norm)
    observe_reward = jnp.where(observed_count == 0, 0.0, -float(opportunity_cost))
    invalid_observe_reward = -1.0
    reward = jnp.where(
        is_observe,
        jnp.where(observe_is_legal, observe_reward, invalid_observe_reward),
        terminal_reward,
    )
    done = jnp.logical_not(is_observe)

    next_state = EnvState(rewards=state.rewards, observed=next_observed)
    reset_state = reset_env(rng, batch, task)
    next_state = tree_where(done, reset_state, next_state)
    next_obs = make_obs(next_state, task)
    next_mask = action_mask(next_state, task)
    return next_state, next_obs, next_mask, reward, done


def env_step_no_reset(
    state: EnvState,
    action: jax.Array,
    task: TaskSpec,
    opportunity_cost: float,
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    num_nodes = task.num_nodes
    path_map = jnp.asarray(task.path_map, dtype=jnp.float32)

    is_observe = action < num_nodes
    node_action = jnp.clip(action, 0, num_nodes - 1)
    path_action = jnp.clip(action - num_nodes, 0, task.num_paths - 1)
    node_onehot = jax.nn.one_hot(node_action, num_nodes, dtype=jnp.bool_)
    already_observed = jnp.sum(state.observed & node_onehot, axis=-1) > 0
    observe_is_legal = is_observe & jnp.logical_not(already_observed)
    observed_count = jnp.sum(state.observed, axis=-1)

    next_observed = state.observed | (node_onehot & observe_is_legal[:, None])
    path_rewards = state.rewards @ path_map.T
    terminal_reward = jnp.take_along_axis(
        path_rewards,
        path_action[:, None],
        axis=1,
    ).squeeze(-1) / float(task.reward_norm)
    observe_reward = jnp.where(observed_count == 0, 0.0, -float(opportunity_cost))
    reward = jnp.where(is_observe, jnp.where(observe_is_legal, observe_reward, -1.0), terminal_reward)
    done = jnp.logical_not(is_observe)
    next_state = EnvState(rewards=state.rewards, observed=next_observed)
    next_obs = make_obs(next_state, task)
    next_mask = action_mask(next_state, task)
    return next_state, next_obs, next_mask, reward, done


def masked_categorical_sample(rng: jax.Array, logits: jax.Array) -> jax.Array:
    return jax.random.categorical(rng, logits, axis=-1)


def categorical_log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def scalar_to_categorical(values: jax.Array, num_classes: int = 9) -> jax.Array:
    idx = jnp.floor(4.0 - values + 0.5).astype(jnp.int32)
    idx = jnp.clip(idx, 0, num_classes - 1)
    return jax.nn.one_hot(idx, num_classes, dtype=jnp.float32)


def categorical_cross_entropy_from_probs(target: jax.Array, probs: jax.Array) -> jax.Array:
    return -jnp.sum(target * jnp.log(probs + 1e-7), axis=-1)


def layer_norm_relu_dense(x: jax.Array, features: int, name: str) -> jax.Array:
    x = nn.Dense(features, name=f"{name}_dense")(x)
    x = nn.LayerNorm(name=f"{name}_ln")(x)
    return nn.relu(x)


class CriticMLP(nn.Module):
    rnn_units: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.rnn_units, kernel_init=nn.initializers.glorot_uniform())(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(
            max(self.rnn_units // 2, 16),
            activation=nn.relu,
            kernel_init=nn.initializers.glorot_uniform(),
        )(x)
        return nn.Dense(self.output_dim, kernel_init=nn.initializers.glorot_uniform())(x)


class JaxVariationalRNN(nn.Module):
    task: TaskSpec
    rnn_units: int
    latent_dim: int
    alpha: float
    beta: float
    lambda_: float
    opportunity_cost: float
    input_type: str = "uniform"
    expansion_decision_version: str = "lstm"
    use_autoencoder: bool = True
    num_categories: int = 9
    min_observations_before_stop: int = 0
    belief_rollout_samples: int = 8

    def setup(self):
        joint_dim = self.task.num_nodes + self.task.num_paths
        self.lstm_cell = nn.LSTMCell(features=self.rnn_units, name="lstm_cell")
        self.encoder_dense = nn.Dense(self.rnn_units, name="encoder_dense")
        self.encoder_ln = nn.LayerNorm(name="encoder_ln")
        self.encoder_z_mean = nn.Dense(self.latent_dim, name="encoder_z_mean")
        self.encoder_z_log_var = nn.Dense(self.latent_dim, name="encoder_z_log_var")
        self.decoder_dense = nn.Dense(self.rnn_units, name="decoder_dense")
        self.decoder_ln = nn.LayerNorm(name="decoder_ln")
        self.decoder_output = nn.Dense(2 * self.rnn_units, name="decoder_output")
        self.reconstruction_head = nn.Dense(
            self.task.num_nodes * 2,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.zeros,
            name="reconstruction_head",
        )
        self.expansion_head = nn.Dense(
            joint_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            name="expansion_head",
        )
        self.critic_head = CriticMLP(
            self.rnn_units,
            joint_dim,
            name="critic_head",
        )

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        rng: jax.Array,
        training: bool = True,
        compute_losses: bool = True,
        forced_node_selections: jax.Array | None = None,
        old_expansion_log_probs: jax.Array | None = None,
        use_ppo_loss: bool = False,
        current_beta: float = 1.0,
        current_critic_coef: float = 1.0,
        expansion_entropy_coef: float = 0.01,
        ppo_clip: float = 0.2,
        use_lambda_return: bool = True,
        lambda_return: float = 0.95,
    ) -> VAEForwardResult:
        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        if inputs.ndim == 2:
            inputs = inputs[..., None]
        batch_size = inputs.shape[0]
        time_steps = self.task.num_nodes
        num_paths = self.task.num_paths
        joint_dim = time_steps + num_paths
        path_map = jnp.asarray(self.task.path_map, dtype=jnp.float32)
        reward_norm = float(self.task.reward_norm)

        prior_mu = self.param(
            "prior_mu",
            nn.initializers.zeros,
            (time_steps, self.latent_dim),
        )
        prior_logvar = self.param(
            "prior_logvar",
            nn.initializers.zeros,
            (time_steps, self.latent_dim),
        )

        rewards_flat = jnp.squeeze(inputs, axis=-1)
        path_rewards = rewards_flat @ path_map.T
        best_path_reward = jnp.max(path_rewards, axis=1, keepdims=True)
        best_path_mask = (path_rewards == best_path_reward).astype(jnp.float32)
        categories_onehot = scalar_to_categorical(inputs, self.num_categories)

        h = jnp.zeros((batch_size, self.rnn_units), dtype=jnp.float32)
        c = jnp.zeros((batch_size, self.rnn_units), dtype=jnp.float32)
        active = jnp.ones((batch_size, 1), dtype=jnp.float32)
        visited = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
        observed_mask = jnp.zeros((batch_size, time_steps, 1), dtype=jnp.float32)
        pending_kl = jnp.zeros((batch_size,), dtype=jnp.float32)
        last_reward_onehot = jnp.zeros((batch_size, self.num_categories), dtype=jnp.float32)
        pre_lstm_context = jnp.zeros((batch_size, self.rnn_units + time_steps + 1 + self.num_categories))
        lstm_context = h

        records = []
        info_cost = jnp.array(0.0, dtype=jnp.float32)

        for t in range(time_steps):
            rng, action_rng, z_rng = jax.random.split(rng, 3)
            if self.expansion_decision_version == "decoder":
                expansion_input = h
            elif self.expansion_decision_version == "lstm":
                expansion_input = lstm_context
            else:
                expansion_input = pre_lstm_context

            q_values = self.critic_head(expansion_input)
            expansion_logits = self.expansion_head(expansion_input)
            observed_count = jnp.sum(visited, axis=1, keepdims=True)
            can_stop = (observed_count >= float(self.min_observations_before_stop)).astype(jnp.float32)
            terminal_action_mask = (1.0 - can_stop) * jnp.ones((batch_size, num_paths), dtype=jnp.float32)
            decision_mask = jnp.concatenate([visited, terminal_action_mask], axis=1)
            legal_action_mask = (1.0 - decision_mask) * active
            masked_logits = jnp.where(decision_mask > 0.0, -1e9, expansion_logits)
            terminal_pre = jax.nn.softmax(expansion_logits[:, time_steps:joint_dim], axis=-1)

            if forced_node_selections is not None:
                action = forced_node_selections[:, t].astype(jnp.int32)
            elif training:
                action = jax.random.categorical(action_rng, masked_logits, axis=-1)
            else:
                action = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
            action = jnp.clip(action, 0, joint_dim - 1)
            log_prob = categorical_log_prob(masked_logits, action)
            expansion_probs = jax.nn.softmax(masked_logits, axis=-1)
            expansion_entropy = categorical_entropy(masked_logits)

            is_stop = (action >= time_steps).astype(jnp.float32)
            is_observe = 1.0 - is_stop
            stop_decision = is_stop[:, None] * active
            observe_active = active * is_observe[:, None]
            terminal_path = jnp.where(action >= time_steps, action - time_steps, -1)
            next_active = observe_active

            active_decision = jnp.squeeze(active, axis=-1)
            continue_decision = jnp.squeeze(observe_active, axis=-1)
            previous_reward_mask = last_reward_onehot * active_decision[:, None]

            safe_node = jnp.minimum(action, time_steps - 1)
            chosen_reward = jnp.take_along_axis(rewards_flat, safe_node[:, None], axis=1)
            reward_onehot = jnp.squeeze(scalar_to_categorical(chosen_reward, self.num_categories), axis=1)
            reward_onehot = reward_onehot * observe_active
            last_reward_onehot = reward_onehot + last_reward_onehot * (1.0 - observe_active)

            expansion_token = jnp.where(action < time_steps, safe_node, time_steps)
            node_onehot = jax.nn.one_hot(expansion_token, time_steps + 1, dtype=jnp.float32)
            lstm_input = jnp.concatenate([node_onehot, reward_onehot], axis=1)
            prev_h, prev_c = h, c
            pre_lstm_context = jnp.concatenate([prev_h, lstm_input], axis=1)
            (new_c, new_h), _ = self.lstm_cell((c, h), lstm_input)
            new_h = new_h * observe_active + h * (1.0 - observe_active)
            new_c = new_c * observe_active + c * (1.0 - observe_active)
            lstm_context = new_h

            if self.use_autoencoder:
                enc_in = jnp.concatenate([new_h, new_c], axis=-1)
                enc_x = nn.relu(self.encoder_ln(self.encoder_dense(enc_in)))
                z_mu = self.encoder_z_mean(enc_x)
                z_logvar = self.encoder_z_log_var(enc_x)
                z_logvar = jnp.clip(z_logvar, -10.0, 10.0)
                z = z_mu + jnp.exp(0.5 * z_logvar) * jax.random.normal(z_rng, z_mu.shape)
                dec_x = nn.relu(self.decoder_ln(self.decoder_dense(z)))
                dec_out = self.decoder_output(dec_x)
                dec_h, dec_c = jnp.split(dec_out, 2, axis=-1)
                h = dec_h * observe_active + prev_h * (1.0 - observe_active)
                c = dec_c * observe_active + prev_c * (1.0 - observe_active)
            else:
                z_mu = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
                z_logvar = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
                h, c = new_h, new_c

            if self.expansion_decision_version == "decoder":
                action_input = h
            elif self.expansion_decision_version == "lstm":
                action_input = lstm_context
            else:
                action_input = pre_lstm_context
            post_logits = self.expansion_head(action_input)
            terminal_post = jax.nn.softmax(post_logits[:, time_steps:joint_dim], axis=-1)
            step_action_output = is_stop[:, None] * terminal_pre + is_observe[:, None] * terminal_post

            node_observation = jax.nn.one_hot(safe_node, time_steps, dtype=jnp.float32)[:, :, None]
            node_observation = node_observation * observe_active[:, None, :]
            observed_mask = jnp.minimum(observed_mask + node_observation, 1.0)

            if self.use_autoencoder:
                rec_params = self.reconstruction_head(h)
                rec_params = rec_params.reshape((batch_size, time_steps, 2))
                mu = 5.0 * jnp.tanh(rec_params[:, :, 0:1])
                scale = jax.nn.softplus(rec_params[:, :, 1:2]) + 1e-4
                edges = jnp.asarray([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=jnp.float32)
                cdf = jax.nn.sigmoid((edges.reshape((1, 1, 10)) - mu) / scale)
                cat = cdf[:, :, 1:10] - cdf[:, :, 0:9]
                cat = jnp.flip(cat, axis=-1) + 1e-6
                category_output = cat / (jnp.sum(cat, axis=-1, keepdims=True) + 1e-8)
                prior_mean = jnp.broadcast_to(prior_mu[t], z_mu.shape)
                prior_lv = jnp.clip(jnp.broadcast_to(prior_logvar[t], z_mu.shape), -10.0, 10.0)
                prior_var = jnp.exp(prior_lv) + 1e-6
                z_var = jnp.exp(z_logvar) + 1e-6
                log_ratio = z_logvar - jnp.log(prior_var)
                kl_sample = -0.5 * jnp.mean(
                    1.0 + log_ratio - ((jnp.square(z_mu - prior_mean) + z_var) / prior_var),
                    axis=1,
                )
                observed_kl = kl_sample * jnp.squeeze(observe_active, axis=-1)
                if self.expansion_decision_version in ("lstm", "pre_lstm"):
                    paid_kl = pending_kl * jnp.squeeze(observe_active, axis=-1)
                    pending_kl = observed_kl
                else:
                    paid_kl = observed_kl
            else:
                category_output = jnp.ones((batch_size, time_steps, self.num_categories)) / float(self.num_categories)
                observed_kl = jnp.zeros((batch_size,), dtype=jnp.float32)
                paid_kl = jnp.zeros((batch_size,), dtype=jnp.float32)

            info_cost = info_cost + jnp.mean(paid_kl)
            visited_obs = jax.nn.one_hot(safe_node, time_steps, dtype=jnp.float32) * observe_active
            visited = jnp.minimum(visited + visited_obs, 1.0)
            active = next_active

            records.append({
                "z_mu": z_mu,
                "z_logvar": z_logvar,
                "category_output": category_output,
                "observed_mask": jnp.squeeze(observed_mask, axis=-1),
                "action_output": step_action_output,
                "node_selection": action,
                "terminal_path": terminal_path,
                "stop": stop_decision,
                "q_values": q_values,
                "legal_mask": legal_action_mask,
                "probs": expansion_probs,
                "log_prob": log_prob[:, None],
                "entropy": expansion_entropy[:, None],
                "entropy_mask": active,
                "kl_d": paid_kl[:, None],
                "obs_kl_d": observed_kl[:, None],
                "lstm_state": new_h,
                "decoder_state": h,
                "valid": active_decision[:, None],
                "previous_reward_mask": previous_reward_mask,
            })

        stack = lambda key: jnp.stack([r[key] for r in records], axis=1)
        category_outputs = stack("category_output")
        observed_masks = stack("observed_mask")
        action_outputs_sequence = stack("action_output")
        node_selections = stack("node_selection")
        terminal_paths = stack("terminal_path")
        stop_decisions = stack("stop")
        q_values = stack("q_values")
        legal_masks = stack("legal_mask")
        probs = stack("probs")
        expansion_log_probs = stack("log_prob")
        entropies = stack("entropy")
        entropy_masks = stack("entropy_mask")
        kl_d_sequence = stack("kl_d")
        obs_kl_d_sequence = stack("obs_kl_d")
        valid_masks = stack("valid")

        stop_flags = jnp.squeeze(stop_decisions > 0, axis=-1)
        has_stop = jnp.any(stop_flags, axis=1)
        first_stop = jnp.argmax(stop_flags.astype(jnp.int32), axis=1)
        selected_idx = jnp.where(has_stop, first_stop, time_steps - 1)
        batch_idx = jnp.arange(batch_size)
        action_output = action_outputs_sequence[batch_idx, selected_idx]
        terminal_output = terminal_paths[batch_idx, selected_idx]
        terminal_output = jnp.where(has_stop, terminal_output, -1)

        expected_reward = jnp.sum(action_output * path_rewards, axis=1, keepdims=True) / reward_norm
        action_loss = 1.0 - jnp.mean(expected_reward)
        information_loss = info_cost / float(time_steps)
        target_expanded = jnp.tile(categories_onehot[:, None, :, :, :], (1, time_steps, 1, 1, 1))
        target_expanded = jnp.squeeze(target_expanded, axis=3)
        rec_ce = categorical_cross_entropy_from_probs(target_expanded, category_outputs)
        rec_mask = observed_masks
        reconstruction_loss = (
            jnp.sum(rec_ce * rec_mask) /
            ((jnp.sum(rec_mask) + 1e-6) * jnp.log(float(self.num_categories)))
        )

        target_returns = self.compute_return_targets(
            path_rewards,
            action_outputs_sequence,
            terminal_paths,
            valid_masks,
            stop_decisions,
            kl_d_sequence,
            current_beta,
            q_values,
            probs,
            legal_masks,
            lambda_return,
            use_lambda_return,
            reward_norm,
        )
        joint_targets, joint_masks = self.build_joint_targets(
            rewards_flat,
            observed_masks,
            node_selections,
            legal_masks,
            target_returns,
            valid_masks,
            kl_d_sequence,
            current_beta,
            reward_norm,
            rng,
        )
        critic_loss = jnp.sum(jnp.square(q_values - jax.lax.stop_gradient(joint_targets)) * joint_masks) / (jnp.sum(joint_masks) + 1e-6)
        selected_q = jnp.take_along_axis(q_values, node_selections[..., None], axis=-1)
        selected_target = jnp.take_along_axis(joint_targets, node_selections[..., None], axis=-1)
        policy_value = jnp.sum(probs * q_values, axis=-1, keepdims=True)
        advantages = jax.lax.stop_gradient(selected_target - policy_value)
        if use_ppo_loss and old_expansion_log_probs is not None:
            log_ratio = jnp.clip(expansion_log_probs - jax.lax.stop_gradient(old_expansion_log_probs), -10.0, 10.0)
            ratio = jnp.exp(log_ratio)
            clipped_ratio = jnp.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
            surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
            policy_loss = -jnp.sum(surrogate * valid_masks) / (jnp.sum(valid_masks) + 1e-6)
        else:
            policy_loss = -jnp.sum(expansion_log_probs * advantages * valid_masks) / (jnp.sum(valid_masks) + 1e-6)
        entropy_bonus = jnp.sum(entropies * entropy_masks) / (jnp.sum(entropy_masks) + 1e-6)
        expansion_loss = policy_loss - float(expansion_entropy_coef) * entropy_bonus
        expansion_stop_rate = jnp.sum(stop_decisions) / (jnp.sum(valid_masks) + 1e-6)
        expansion_continue_rate = jnp.sum(valid_masks * (1.0 - stop_decisions)) / (jnp.sum(valid_masks) + 1e-6)

        first_decoder_loss = (
            information_loss * current_beta
            + action_loss * self.lambda_
            + expansion_loss * self.lambda_
            + critic_loss * self.lambda_ * current_critic_coef
            + reconstruction_loss * self.alpha
        )
        second_decoder_loss = reconstruction_loss
        expansion_head_loss = expansion_loss * self.lambda_ + critic_loss * self.lambda_ * current_critic_coef + action_loss * self.lambda_
        total_loss = first_decoder_loss if compute_losses else jnp.array(0.0, dtype=jnp.float32)

        return VAEForwardResult(
            category_outputs=category_outputs,
            action_output=action_output,
            total_loss=total_loss,
            first_decoder_loss=first_decoder_loss,
            second_decoder_loss=second_decoder_loss,
            action_head_loss=jnp.array(0.0, dtype=jnp.float32),
            critic_loss=critic_loss,
            information_loss=information_loss,
            action_loss=action_loss,
            reconstruction_loss=reconstruction_loss,
            information_cost=info_cost,
            z_means=stack("z_mu"),
            node_selections=node_selections,
            stop_decisions=stop_decisions,
            observed_masks=observed_masks,
            action_outputs_sequence=action_outputs_sequence,
            kl_d_sequence=kl_d_sequence,
            expansion_head_loss=expansion_head_loss,
            expansion_log_probs=expansion_log_probs,
            expansion_loss=expansion_loss,
            expansion_stop_rate=expansion_stop_rate,
            expansion_continue_rate=expansion_continue_rate,
            opportunity_loss=jnp.array(0.0, dtype=jnp.float32),
            lstm_probe_loss=jnp.array(0.0, dtype=jnp.float32),
            lstm_probe_accuracy=jnp.array(0.0, dtype=jnp.float32),
            terminal_path_output=terminal_output,
            observation_kl_d_sequence=obs_kl_d_sequence,
            lstm_state_sequence=stack("lstm_state"),
            decoder_state_sequence=stack("decoder_state"),
            expansion_probs_sequence=probs,
            legal_action_masks=legal_masks,
            stop_value_preds=q_values,
        )

    def compute_return_targets(self, path_rewards, action_outputs, terminal_paths, valid_masks, stop_decisions, kl_d, current_beta, q_values, probs, legal_masks, lambda_return, use_lambda_return, reward_norm):
        batch_size, time_steps, _ = valid_masks.shape
        stop_flags = jnp.squeeze(stop_decisions > 0, axis=-1)
        has_stop = jnp.any(stop_flags, axis=1)
        first_stop = jnp.argmax(stop_flags.astype(jnp.int32), axis=1)
        selected_idx = jnp.where(has_stop, first_stop, time_steps - 1)
        batch_idx = jnp.arange(batch_size)
        terminal_action_probs = action_outputs[batch_idx, selected_idx]
        selected_terminal = terminal_paths[batch_idx, selected_idx]
        sampled_probs = jax.nn.one_hot(jnp.maximum(selected_terminal, 0), self.task.num_paths, dtype=jnp.float32)
        terminal_probs = jnp.where((selected_terminal >= 0)[:, None], sampled_probs, terminal_action_probs)
        terminal_reward = jnp.sum(terminal_probs * path_rewards, axis=1, keepdims=True) / reward_norm
        timestep_has_prior = (jnp.arange(time_steps) > 0).astype(jnp.float32).reshape((1, time_steps, 1))
        non_stop = valid_masks * (1.0 - stop_decisions)
        step_cost = non_stop * (self.opportunity_cost * timestep_has_prior + current_beta * kl_d)
        if use_lambda_return:
            legal_policy = probs * legal_masks
            legal_policy = legal_policy / (jnp.sum(legal_policy, axis=-1, keepdims=True) + 1e-6)
            state_value = jnp.sum(q_values * legal_policy, axis=-1, keepdims=True)
            next_target = terminal_reward
            targets = []
            for t in range(time_steps - 1, -1, -1):
                next_bootstrap = terminal_reward if t == time_steps - 1 else ((1.0 - lambda_return) * state_value[:, t + 1, :] + lambda_return * next_target)
                cont_target = -step_cost[:, t, :] + next_bootstrap
                target_t = jnp.where(stop_decisions[:, t, :] > 0.0, terminal_reward, cont_target)
                target_t = jnp.where(valid_masks[:, t, :] > 0.0, target_t, 0.0)
                targets.append(target_t)
                next_target = target_t
            return jnp.stack(targets[::-1], axis=1)
        future_cost = jnp.flip(jnp.cumsum(jnp.flip(step_cost, axis=1), axis=1), axis=1)
        return terminal_reward[:, None, :] - future_cost

    def build_joint_targets(self, rewards_flat, observed_masks, node_selections, legal_masks, selected_returns, valid_masks, kl_d, current_beta, reward_norm, rng):
        observed_before = jnp.concatenate([
            jnp.zeros_like(observed_masks[:, :1, :]),
            observed_masks[:, :-1, :],
        ], axis=1)
        reward_support = jnp.asarray(self.task.reward_values, dtype=jnp.float32)
        prior_mean = jnp.mean(reward_support)
        belief = rewards_flat[:, None, :] * observed_before + (1.0 - observed_before) * prior_mean
        path_map = jnp.asarray(self.task.path_map, dtype=jnp.float32)
        stop_targets = jnp.einsum("btn,pn->btp", belief, path_map) / reward_norm
        timestep_has_prior = (jnp.arange(self.task.num_nodes) > 0).astype(jnp.float32).reshape((1, self.task.num_nodes))
        observe_costs = self.opportunity_cost * timestep_has_prior + current_beta * jnp.squeeze(kl_d, axis=-1)
        if self.task.num_nodes < 4:
            rollout_rewards = reward_support
        else:
            idx = jax.random.randint(rng, (self.belief_rollout_samples,), 0, reward_support.shape[0])
            rollout_rewards = reward_support[idx]
        observe_targets = []
        for node_idx in range(self.task.num_nodes):
            node_onehot = jax.nn.one_hot(node_idx, self.task.num_nodes, dtype=jnp.float32).reshape((1, 1, 1, self.task.num_nodes))
            rr = rollout_rewards.reshape((-1, 1, 1, 1))
            next_belief = belief[None, ...] * (1.0 - node_onehot) + rr * node_onehot
            next_path = jnp.einsum("kbtn,pn->kbtp", next_belief, path_map) / reward_norm
            observe_targets.append(jnp.mean(jnp.max(next_path, axis=-1), axis=0) - observe_costs)
        joint_targets = jnp.concatenate([jnp.stack(observe_targets, axis=-1), stop_targets], axis=-1)
        selected_onehot = jax.nn.one_hot(node_selections, self.task.num_nodes + self.task.num_paths, dtype=jnp.float32)
        joint_targets = joint_targets * (1.0 - selected_onehot) + jnp.squeeze(selected_returns, axis=-1)[..., None] * selected_onehot
        joint_masks = legal_masks * valid_masks
        return joint_targets, joint_masks


class RecurrentActorCritic(nn.Module):
    action_dim: int
    rnn_units: int
    latent_dim: int

    @nn.compact
    def __call__(
        self,
        hstate: jax.Array,
        obs: jax.Array,
        done: jax.Array,
        action_mask_batch: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = nn.Dense(self.rnn_units, kernel_init=orthogonal(math.sqrt(2.0)))(obs)
        x = nn.relu(x)
        cell = nn.GRUCell(features=self.rnn_units)

        def step(carry, inputs):
            x_t, done_t, mask_t = inputs
            carry = jnp.where(done_t[:, None], jnp.zeros_like(carry), carry)
            carry, rnn_out = cell(carry, x_t)
            if self.latent_dim > 0:
                z = nn.Dense(self.latent_dim, kernel_init=orthogonal(math.sqrt(2.0)))(rnn_out)
                z = nn.tanh(z)
                features = nn.Dense(self.rnn_units, kernel_init=orthogonal(math.sqrt(2.0)))(z)
                features = nn.relu(features)
            else:
                features = rnn_out
            actor = nn.Dense(self.rnn_units, kernel_init=orthogonal(math.sqrt(2.0)))(features)
            actor = nn.tanh(actor)
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
            logits = jnp.where(mask_t, logits, -1e9)
            critic = nn.Dense(self.rnn_units, kernel_init=orthogonal(math.sqrt(2.0)))(features)
            critic = nn.tanh(critic)
            value = nn.Dense(1, kernel_init=orthogonal(1.0))(critic).squeeze(-1)
            return carry, (logits, value)

        hstate, (logits, values) = jax.lax.scan(
            step,
            hstate,
            (x, done, action_mask_batch),
        )
        return hstate, logits, values


def orthogonal(scale: float):
    return nn.initializers.orthogonal(scale)


def linear_schedule(init_value: float, count: jax.Array, total_updates: int) -> jax.Array:
    frac = 1.0 - (count // 1) / float(max(total_updates, 1))
    return init_value * frac


def make_train(config: JaxPlanningConfig):
    task = build_task(config.tree_size, config.tree_type, config.input_type)
    action_dim = task.num_nodes + task.num_paths
    obs_dim = 2 * task.num_nodes + 2
    num_updates = max(1, config.total_timesteps // config.num_steps // config.num_envs)
    num_minibatches = min(config.num_minibatches, config.num_envs)
    minibatch_envs = max(1, config.num_envs // num_minibatches)
    clipped_num_envs = minibatch_envs * num_minibatches
    cfg = replace(config, num_envs=clipped_num_envs)
    network = RecurrentActorCritic(
        action_dim=action_dim,
        rnn_units=cfg.rnn_units,
        latent_dim=cfg.latent_dim,
    )

    def train(rng: jax.Array):
        rng, reset_rng, init_rng = jax.random.split(rng, 3)
        env_state = reset_env(reset_rng, cfg.num_envs, task)
        last_obs = make_obs(env_state, task)
        last_done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
        last_action_mask = action_mask(env_state, task)
        hstate = jnp.zeros((cfg.num_envs, cfg.rnn_units), dtype=jnp.float32)

        init_obs = jnp.zeros((1, cfg.num_envs, obs_dim), dtype=jnp.float32)
        init_done = jnp.ones((1, cfg.num_envs), dtype=jnp.bool_)
        init_mask = jnp.ones((1, cfg.num_envs, action_dim), dtype=jnp.bool_)
        params = network.init(init_rng, hstate, init_obs, init_done, init_mask)
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=last_obs,
            last_done=last_done,
            last_action_mask=last_action_mask,
            hstate=hstate,
            rng=rng,
        )

        def _update_step(runner_state: RunnerState, update_idx):
            initial_hstate = runner_state.hstate

            def _env_step(runner_state: RunnerState, _):
                rng, action_rng, env_rng = jax.random.split(runner_state.rng, 3)
                obs_seq = runner_state.last_obs[None, ...]
                done_seq = runner_state.last_done[None, ...]
                mask_seq = runner_state.last_action_mask[None, ...]
                hstate, logits, value = network.apply(
                    runner_state.train_state.params,
                    runner_state.hstate,
                    obs_seq,
                    done_seq,
                    mask_seq,
                )
                logits = logits.squeeze(axis=0)
                value = value.squeeze(axis=0)
                action = masked_categorical_sample(action_rng, logits)
                log_prob = categorical_log_prob(logits, action)
                env_state, obs, mask, reward, done = env_step(
                    env_rng,
                    runner_state.env_state,
                    action,
                    task,
                    cfg.opportunity_cost,
                )
                transition = Transition(
                    obs=runner_state.last_obs,
                    done=runner_state.last_done,
                    action_mask=runner_state.last_action_mask,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    next_done=done,
                )
                next_runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obs,
                    last_done=done,
                    last_action_mask=mask,
                    hstate=hstate,
                    rng=rng,
                )
                return next_runner_state, transition

            runner_state, traj = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                length=cfg.num_steps,
            )

            _, _, last_value_seq = network.apply(
                runner_state.train_state.params,
                runner_state.hstate,
                runner_state.last_obs[None, ...],
                runner_state.last_done[None, ...],
                runner_state.last_action_mask[None, ...],
            )
            last_value = last_value_seq.squeeze(axis=0)

            def _gae_step(carry, transition):
                gae, next_value = carry
                nonterminal = 1.0 - transition.next_done.astype(jnp.float32)
                delta = transition.reward + cfg.gamma * next_value * nonterminal - transition.value
                gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
                return (gae, transition.value), gae

            _, advantages_rev = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_value), last_value),
                traj,
                reverse=True,
            )
            advantages = advantages_rev
            targets = advantages + traj.value

            def _loss_fn(params, hstate_mb, traj_mb, adv_mb, targets_mb):
                _, logits, values = network.apply(
                    params,
                    hstate_mb,
                    traj_mb.obs,
                    traj_mb.done,
                    traj_mb.action_mask,
                )
                log_prob = categorical_log_prob(logits, traj_mb.action)
                entropy = categorical_entropy(logits)
                log_ratio = log_prob - traj_mb.log_prob
                ratio = jnp.exp(log_ratio)
                adv_norm = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                unclipped = ratio * adv_norm
                clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_norm
                actor_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
                value_loss = 0.5 * jnp.mean((values - targets_mb) ** 2)
                entropy_loss = jnp.mean(entropy)
                total_loss = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss
                approx_kl = 0.5 * jnp.mean((log_prob - traj_mb.log_prob) ** 2)
                return total_loss, {
                    "loss": total_loss,
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_loss,
                    "approx_kl": approx_kl,
                }

            def _update_epoch(update_state, _):
                train_state, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                permutation = jax.random.permutation(perm_rng, cfg.num_envs)
                env_batches = permutation.reshape((num_minibatches, minibatch_envs))

                def _update_minibatch(train_state, env_idx):
                    traj_mb = jax.tree_util.tree_map(lambda x: x[:, env_idx], traj)
                    adv_mb = advantages[:, env_idx]
                    target_mb = targets[:, env_idx]
                    hstate_mb = initial_hstate[env_idx]
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (_, metrics), grads = grad_fn(
                        train_state.params,
                        hstate_mb,
                        traj_mb,
                        adv_mb,
                        target_mb,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, metrics

                train_state, metrics = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    env_batches,
                )
                return (train_state, rng), metrics

            (train_state, rng), metrics = jax.lax.scan(
                _update_epoch,
                (runner_state.train_state, runner_state.rng),
                None,
                length=cfg.update_epochs,
            )
            runner_state = RunnerState(
                train_state=train_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                last_done=runner_state.last_done,
                last_action_mask=runner_state.last_action_mask,
                hstate=runner_state.hstate,
                rng=rng,
            )
            flat_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
            flat_metrics["mean_reward"] = jnp.mean(traj.reward)
            flat_metrics["stop_rate"] = jnp.mean(traj.next_done.astype(jnp.float32))
            flat_metrics["update"] = update_idx
            return runner_state, flat_metrics

        runner_state, metrics = jax.lax.scan(
            _update_step,
            runner_state,
            jnp.arange(num_updates),
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, task, num_updates


def run_training(config: JaxPlanningConfig):
    task = build_task(config.tree_size, config.tree_type, config.input_type)
    num_updates = max(1, config.total_timesteps // max(1, config.num_envs * task.num_nodes))
    model = JaxVariationalRNN(
        task=task,
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        alpha=config.alpha,
        beta=config.beta,
        lambda_=config.lambda_,
        opportunity_cost=config.opportunity_cost,
        input_type=config.input_type,
        expansion_decision_version=config.expansion_decision_version,
        use_autoencoder=(config.model_variant != "rnn"),
        min_observations_before_stop=(
            1 if config.expansion_decision_version == "decoder" else 0
        ),
    )

    def sample_batch(rng):
        values = jnp.asarray(task.reward_values, dtype=jnp.float32)
        idx = jax.random.randint(
            rng,
            shape=(config.num_envs, task.num_nodes),
            minval=0,
            maxval=len(task.reward_values),
        )
        return values[idx][..., None]

    def train(rng):
        rng, init_rng, batch_rng, forward_rng = jax.random.split(rng, 4)
        dummy = sample_batch(batch_rng)
        params = model.init(init_rng, dummy, forward_rng, training=True)
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(config.learning_rate, weight_decay=1e-4),
        )
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        def update_step(train_state_and_rng, update_idx):
            train_state, rng = train_state_and_rng
            rng, batch_rng, rollout_rng, ppo_rng = jax.random.split(rng, 4)
            batch = sample_batch(batch_rng)

            rollout = model.apply(
                train_state.params,
                batch,
                rollout_rng,
                training=True,
                compute_losses=False,
            )
            forced_actions = jax.lax.stop_gradient(rollout.node_selections)
            old_log_probs = jax.lax.stop_gradient(rollout.expansion_log_probs)

            def loss_fn(params, rng_for_loss):
                out = model.apply(
                    params,
                    batch,
                    rng_for_loss,
                    training=True,
                    compute_losses=True,
                    forced_node_selections=forced_actions,
                    old_expansion_log_probs=old_log_probs,
                    use_ppo_loss=True,
                    current_beta=1.0 / max(config.beta, 1e-8),
                    current_critic_coef=1.0,
                    expansion_entropy_coef=config.ent_coef,
                    use_lambda_return=True,
                    lambda_return=config.gae_lambda,
                )
                return out.total_loss, out

            (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, ppo_rng)
            train_state = train_state.apply_gradients(grads=grads)
            metrics = {
                "update": update_idx,
                "loss": loss,
                "total_loss": out.total_loss,
                "first_decoder_loss": out.first_decoder_loss,
                "second_decoder_loss": out.second_decoder_loss,
                "critic_loss": out.critic_loss,
                "information_loss": out.information_loss,
                "action_loss": out.action_loss,
                "reconstruction_loss": out.reconstruction_loss,
                "expansion_loss": out.expansion_loss,
                "expansion_stop_rate": out.expansion_stop_rate,
                "expansion_continue_rate": out.expansion_continue_rate,
            }
            return (train_state, rng), metrics

        (train_state, rng), metrics = jax.lax.scan(
            update_step,
            (train_state, rng),
            jnp.arange(num_updates),
        )
        return {"train_state": train_state, "metrics": metrics}

    result = jax.jit(train, backend="cpu")(jax.random.PRNGKey(config.seed))
    return result, task, num_updates


def tree_label(task: TaskSpec) -> str:
    suffix = "" if task.tree_type == "default" else f"_{task.tree_type}"
    return f"{task.num_nodes}n{suffix}"


def model_name_for(config: JaxPlanningConfig, task: TaskSpec) -> str:
    return (
        f"lambda_{config.lambda_}_alpha_{config.alpha}_beta_{config.beta}_"
        f"opportunity_{config.opportunity_cost}_expansion_{config.expansion_decision_version}_"
        f"variant_{config.model_variant}_"
        f"seed_{config.seed}_{tree_label(task)}_"
        f"rnn_{config.rnn_units}_latent_{config.latent_dim}"
    )


def params_path(config: JaxPlanningConfig, task: TaskSpec) -> Path:
    return Path(config.model_dir) / f"{model_name_for(config, task)}.msgpack"


def metrics_path(config: JaxPlanningConfig, task: TaskSpec) -> Path:
    return Path(config.model_dir) / f"{model_name_for(config, task)}_training_logs.csv"


def simulation_path(config: JaxPlanningConfig, task: TaskSpec) -> Path:
    return Path(config.sim_dir) / f"{model_name_for(config, task)}_{config.input_type}.csv"


def write_metrics_csv(path: Path, metrics) -> None:
    rows = []
    metric_np = jax.tree_util.tree_map(np.asarray, metrics)
    keys = sorted(metric_np)
    n_rows = len(metric_np[keys[0]]) if keys else 0
    for idx in range(n_rows):
        row = {key: float(np.ravel(metric_np[key][idx])[0]) for key in keys}
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_params(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def load_params(path: Path, template_params):
    return serialization.from_bytes(template_params, path.read_bytes())


def sample_simulation_rewards(config: JaxPlanningConfig, task: TaskSpec) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 12345)
    return rng.choice(
        np.asarray(task.reward_values, dtype=np.float32),
        size=(config.n_sim_trials, task.num_nodes),
    ).astype(np.float32)


def simulate_policy(
    params,
    config: JaxPlanningConfig,
    task: TaskSpec,
    rewards: np.ndarray,
):
    model = JaxVariationalRNN(
        task=task,
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        alpha=config.alpha,
        beta=config.beta,
        lambda_=config.lambda_,
        opportunity_cost=config.opportunity_cost,
        input_type=config.input_type,
        expansion_decision_version=config.expansion_decision_version,
        use_autoencoder=(config.model_variant != "rnn"),
        min_observations_before_stop=(
            1 if config.expansion_decision_version == "decoder" else 0
        ),
    )
    rewards_j = jnp.asarray(rewards, dtype=jnp.float32)[..., None]

    def _simulate(rng):
        out = model.apply(params, rewards_j, rng, training=False, compute_losses=False)
        rewards_flat = jnp.squeeze(rewards_j, axis=-1)
        path_rewards = rewards_flat @ jnp.asarray(task.path_map, dtype=jnp.float32).T
        fallback_path = jnp.argmax(path_rewards, axis=-1)
        chosen_path = jnp.where(out.terminal_path_output >= 0, out.terminal_path_output, fallback_path)
        chosen_value = jnp.take_along_axis(path_rewards, chosen_path[:, None], axis=1).squeeze(-1)
        node_selection = out.node_selections
        expanded_node = jnp.where(node_selection < task.num_nodes, node_selection + 1, -1)
        expanded_reward = jnp.where(
            node_selection < task.num_nodes,
            jnp.take_along_axis(rewards_flat, jnp.minimum(node_selection, task.num_nodes - 1), axis=1),
            jnp.nan,
        )
        return {
            "expanded_node": jnp.transpose(expanded_node, (1, 0)),
            "expanded_reward": jnp.transpose(expanded_reward, (1, 0)),
            "stop": jnp.transpose(jnp.squeeze(out.stop_decisions > 0, axis=-1), (1, 0)),
            "observed": jnp.transpose(out.observed_masks > 0, (1, 0, 2)),
            "chosen_path": chosen_path,
            "V": chosen_value,
            "kl_d": jnp.transpose(jnp.squeeze(out.kl_d_sequence, axis=-1), (1, 0)),
            "kl_d_obs": jnp.transpose(jnp.squeeze(out.observation_kl_d_sequence, axis=-1), (1, 0)),
            "estimated": jnp.transpose(out.category_outputs, (1, 0, 2, 3)),
        }

    return jax.jit(_simulate, backend="cpu")(jax.random.PRNGKey(config.seed + 99))


def simulation_rows(config: JaxPlanningConfig, task: TaskSpec, rewards: np.ndarray, sim) -> list[dict]:
    sim_np = jax.tree_util.tree_map(np.asarray, sim)
    category_values = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    rows = []
    for graph_index in range(rewards.shape[0]):
        chosen_path = int(sim_np["chosen_path"][graph_index])
        row_template = {
            "graph": graph_index,
            "chosen_path": chosen_path,
            "V": float(sim_np["V"][graph_index]),
            "MI": 0.0,
        }
        stopped_seen = False
        for t in range(task.num_nodes):
            step = t + 1
            stopped_at_step = bool(sim_np["stop"][t, graph_index])
            expanded_node = sim_np["expanded_node"][t, graph_index]
            expanded_reward = sim_np["expanded_reward"][t, graph_index]
            if stopped_seen or int(expanded_node) < 1:
                row_template[f"expanded_node_t{step}"] = np.nan
                row_template[f"expanded_reward_t{step}"] = np.nan
            else:
                row_template[f"expanded_node_t{step}"] = int(expanded_node)
                row_template[f"expanded_reward_t{step}"] = float(expanded_reward)
            row_template[f"stop_t{step}"] = stopped_at_step and not stopped_seen
            row_template[f"kl_d_t{step}"] = (
                np.nan if stopped_seen else float(sim_np["kl_d"][t, graph_index])
            )
            row_template[f"kl_d_obs_t{step}"] = (
                float(sim_np["kl_d_obs"][t, graph_index])
                if not np.isnan(row_template[f"expanded_reward_t{step}"]) else np.nan
            )
            stopped_seen = stopped_seen or stopped_at_step

        observed_by_step = np.asarray(sim_np["observed"][:, graph_index, :], dtype=bool)
        for node_index in range(task.num_nodes):
            row = dict(row_template)
            row["node"] = node_index + 1
            row["actual_reward"] = float(rewards[graph_index, node_index])
            for t in range(task.num_nodes):
                probs = sim_np["estimated"][t, graph_index, node_index]
                row[f"estimated_reward_t{t + 1}"] = float(category_values[int(np.argmax(probs))])
            rows.append(row)
    for row in rows:
        row["opportunity_cost"] = config.opportunity_cost
        row["expansion_decision_version"] = config.expansion_decision_version
    return rows


def run_simulation(config: JaxPlanningConfig, task: TaskSpec, params) -> Path:
    rewards = sample_simulation_rewards(config, task)
    sim = simulate_policy(params, config, task, rewards)
    rows = simulation_rows(config, task, rewards, sim)
    path = simulation_path(config, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_original_positional(argv: list[str]) -> JaxPlanningConfig:
    if len(argv) < 11 or len(argv) > 16:
        raise SystemExit(
            "Usage positional: run_jax_model.sh lambda alpha beta model_dir sim_dir "
            "trial_n input_type seed train opportunity_cost tree_size "
            "[expansion_decision_version] [model_variant] [tree_config] [rnn_units] [latent_dim]"
        )
    lambda_ = float(argv[0])
    alpha = float(argv[1])
    beta = float(argv[2])
    model_dir = argv[3]
    sim_dir = argv[4]
    n_sim_trials = int(argv[5])
    input_type = argv[6]
    seed = int(argv[7])
    train_mode = argv[8]
    opportunity_cost = float(argv[9])
    tree_size = int(argv[10])
    expansion_decision_version = argv[11] if len(argv) > 11 else "lstm"
    model_variant = argv[12] if len(argv) > 12 else "jax_ppo"
    tree_type = argv[13] if len(argv) > 13 and argv[13] else "default"
    rnn_units = int(argv[14]) if len(argv) > 14 else 64
    latent_dim = int(argv[15]) if len(argv) > 15 else 32
    return JaxPlanningConfig(
        lambda_=lambda_,
        alpha=alpha,
        beta=beta,
        model_dir=model_dir,
        sim_dir=sim_dir,
        n_sim_trials=n_sim_trials,
        input_type=input_type,
        seed=seed,
        train_mode=train_mode,
        opportunity_cost=opportunity_cost,
        tree_size=tree_size,
        tree_type=tree_type,
        expansion_decision_version=expansion_decision_version,
        model_variant=model_variant,
        rnn_units=rnn_units,
        latent_dim=latent_dim,
    )


def parse_args() -> argparse.Namespace:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        return parse_original_positional(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", type=str, default="default")
    parser.add_argument("--input-type", type=str, default="uniform")
    parser.add_argument("--opportunity-cost", type=float, default=0.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--expansion-decision-version", type=str, default="lstm")
    parser.add_argument("--model-variant", type=str, default="jax_ppo")
    parser.add_argument("--train-mode", type=str, default="train")
    parser.add_argument("--n-sim-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-dir", type=str, default="outputs/jax_models")
    parser.add_argument("--sim-dir", type=str, default="outputs/simulations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args if isinstance(args, JaxPlanningConfig) else JaxPlanningConfig(**vars(args))
    mode = str(config.train_mode).strip().lower()
    task = build_task(config.tree_size, config.tree_type, config.input_type)

    if mode in ("train", "training"):
        result, task, num_updates = run_training(config)
        params = result["train_state"].params
        save_params(params_path(config, task), params)
        write_metrics_csv(metrics_path(config, task), result["metrics"])
        np.savez(
            Path(config.model_dir) / f"{model_name_for(config, task)}_summary.npz",
            tree_type=task.tree_type,
            path_map=task.path_map,
            reward_norm=task.reward_norm,
            num_updates=num_updates,
            num_envs=config.num_envs,
            num_steps=config.num_steps,
        )
        print(f"Saved JAX training logs to: {metrics_path(config, task)}")
        print(f"Saved JAX parameters to: {params_path(config, task)}")
    else:
        init_rng = jax.random.PRNGKey(config.seed)
        model = JaxVariationalRNN(
            task=task,
            rnn_units=config.rnn_units,
            latent_dim=config.latent_dim,
            alpha=config.alpha,
            beta=config.beta,
            lambda_=config.lambda_,
            opportunity_cost=config.opportunity_cost,
            input_type=config.input_type,
            expansion_decision_version=config.expansion_decision_version,
            use_autoencoder=(config.model_variant != "rnn"),
            min_observations_before_stop=(
                1 if config.expansion_decision_version == "decoder" else 0
            ),
        )
        values = jnp.asarray(task.reward_values, dtype=jnp.float32)
        dummy_idx = jax.random.randint(
            init_rng,
            shape=(config.num_envs, task.num_nodes),
            minval=0,
            maxval=len(task.reward_values),
        )
        dummy = values[dummy_idx][..., None]
        template_params = model.init(init_rng, dummy, init_rng, training=False, compute_losses=False)
        path = params_path(config, task)
        if not path.exists():
            raise FileNotFoundError(f"JAX parameter file not found: {path}")
        params = load_params(path, template_params)

    if mode in ("train", "training", "simulate", "simulation", "inference", "eval", "evaluate"):
        sim_path = run_simulation(config, task, params)
        print(f"Saved JAX simulation results to: {sim_path}")


if __name__ == "__main__":
    main()
