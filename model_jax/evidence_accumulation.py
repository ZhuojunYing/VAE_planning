"""JAX VAE/RNN trainer for a continuous evidence-accumulation task.

This script intentionally lives beside, rather than inside,
``model_jax/planning.py``.  The implementation keeps the same broad training
shape as the planning trainer: recurrent evidence encoding, optional VAE
compression with a timestep prior, PPO-style clipped action updates, a value
critic, rollout replay with forced actions and forced observations, CSV logs,
MsgPack checkpoints, and trial-level simulation output.

Evidence-task timing
--------------------
At reset, each trial receives its first continuous evidence sample.  On each
decision step the model first encodes the current sample and then chooses among
``CONTINUE``, ``CHOOSE_A``, and ``CHOOSE_B``.  ``CONTINUE`` pays opportunity
cost plus the current representation KL cost and advances to the next
pre-sampled evidence sample.  Terminal choices end the trial immediately and do
not create another observation or continuation cost.  At
``max_observations_before_stop`` observations, ``CONTINUE`` is masked out.

This differs from ``planning.py`` in one important way: planning actions choose
which node to observe before the reward is seen, while here an evidence sample
is seen before each continue/choice decision.  The replay machinery still fixes
all sampled trial variables and evidence samples during PPO replay.
"""

from __future__ import annotations

import argparse
import csv
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


CONTINUE = 0
CHOOSE_A = 1
CHOOSE_B = 2
NUM_ACTIONS = 3


@dataclass(frozen=True)
class EvidenceTaskSpec:
    coherence_values: tuple[float, ...]
    observation_noise_std: float
    correct_reward: float
    incorrect_reward: float


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
    enable_reconstruction: bool
    enable_probe: bool
    max_observations_before_stop: int
    coherence_values: tuple[float, ...]
    observation_noise_std: float
    correct_reward: float
    incorrect_reward: float
    kl_start_multiplier: float
    kl_annealing_epochs: int


class ScheduleValues(NamedTuple):
    current_alpha: jax.Array
    current_beta: jax.Array
    current_critic_coef: jax.Array
    expansion_epsilon: jax.Array
    expansion_entropy_coef: jax.Array
    forced_continue_epsilon: jax.Array
    ppo_clip: jax.Array


class EvidenceBatch(NamedTuple):
    correct_choice: jax.Array
    correct_action: jax.Array
    coherence: jax.Array
    signed_coherence: jax.Array
    evidence_samples: jax.Array
    cumulative_evidence: jax.Array
    oracle_cumulative_llr: jax.Array


class EvidenceCarry(NamedTuple):
    correct_choice: jax.Array
    correct_action: jax.Array
    coherence: jax.Array
    signed_coherence: jax.Array
    evidence_samples: jax.Array
    cumulative_evidence_by_time: jax.Array
    oracle_cumulative_llr_by_time: jax.Array
    current_evidence: jax.Array
    cumulative_evidence: jax.Array
    oracle_cumulative_llr: jax.Array
    num_observations: jax.Array
    done: jax.Array
    h: jax.Array
    c: jax.Array
    decoded_h: jax.Array
    decoded_c: jax.Array
    lstm_context: jax.Array
    pre_context: jax.Array
    trial_id: jax.Array


class EvidenceTransition(NamedTuple):
    valid: jax.Array
    correct_choice: jax.Array
    correct_action: jax.Array
    coherence: jax.Array
    signed_coherence: jax.Array
    step_index: jax.Array
    num_observations_before: jax.Array
    num_observations_after: jax.Array
    current_evidence: jax.Array
    cumulative_evidence: jax.Array
    oracle_cumulative_llr: jax.Array
    action: jax.Array
    terminal_action: jax.Array
    is_continue: jax.Array
    is_terminal: jax.Array
    terminal_reward: jax.Array
    step_reward: jax.Array
    opportunity_cost_paid: jax.Array
    memory_cost_paid: jax.Array
    log_prob: jax.Array
    entropy: jax.Array
    probs: jax.Array
    legal_mask: jax.Array
    value_pred: jax.Array
    expansion_input: jax.Array
    z_mu: jax.Array
    z_logvar: jax.Array
    z_sample: jax.Array
    prior_mu: jax.Array
    prior_logvar: jax.Array
    paid_kl: jax.Array
    observed_kl: jax.Array
    reconstruction_loss: jax.Array
    probe_loss: jax.Array
    probe_correct: jax.Array
    valid_probe: jax.Array


class UpdateMetrics(NamedTuple):
    total_loss: jax.Array
    information_loss: jax.Array
    action_loss: jax.Array
    reconstruction_loss: jax.Array
    expansion_loss: jax.Array
    critic_loss: jax.Array
    probe_loss: jax.Array
    probe_accuracy: jax.Array
    stop_rate: jax.Array
    continue_rate: jax.Array
    accuracy: jax.Array
    mean_observations: jax.Array
    entropy_coef: jax.Array
    critic_coef: jax.Array
    current_beta: jax.Array
    learning_rate: jax.Array


class EvidenceTrainState(TrainState):
    target_params: object


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def normalize_return_target_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in {"sampled_lambda", "trajectory_lambda", "lambda_sampled"}:
        return "sampled_lambda"
    if key in {"one_step", "joint_q", "joint", "counterfactual_one_step"}:
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


def label_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def coherence_file_label(values: tuple[float, ...]) -> str:
    if not values:
        return "coh_none"
    return f"coh_n{len(values)}_min_{min(values):g}_max_{max(values):g}"


def model_variant_label(variant: str) -> str:
    return f"variant_{variant}_"


def architecture_file_label(rnn_units: int, latent_dim: int) -> str:
    return f"rnn_{rnn_units}_latent_{latent_dim}"


def model_name_for(config: RunConfig) -> str:
    return (
        f"evidence_lambda_{config.lambda_}_alpha_{config.alpha}_beta_{config.beta}_"
        f"opportunity_{config.opportunity_cost}_expansion_{config.expansion_decision_version}_"
        f"{model_variant_label(config.model_variant)}"
        f"seed_{config.seed}_{coherence_file_label(config.coherence_values)}_"
        f"obsstd_{config.observation_noise_std:g}_maxobs_{config.max_observations_before_stop}_"
        f"{architecture_file_label(config.rnn_units, config.latent_dim)}"
    )


def make_task(config: RunConfig) -> EvidenceTaskSpec:
    return EvidenceTaskSpec(
        coherence_values=tuple(float(v) for v in config.coherence_values),
        observation_noise_std=float(config.observation_noise_std),
        correct_reward=float(config.correct_reward),
        incorrect_reward=float(config.incorrect_reward),
    )


def sample_evidence_batch(
    rng: jax.Array,
    batch_size: int,
    num_steps: int,
    task: EvidenceTaskSpec,
) -> EvidenceBatch:
    """Balanced batch over correct side and coherence magnitude."""
    rng_cond, rng_perm, rng_noise = jax.random.split(rng, 3)
    del rng_cond
    coherence_values = jnp.asarray(task.coherence_values, dtype=jnp.float32)
    n_coh = int(len(task.coherence_values))
    n_conditions = max(2 * n_coh, 1)
    reps = int(math.ceil(batch_size / n_conditions))
    base = jnp.tile(jnp.arange(n_conditions, dtype=jnp.int32), reps)[:batch_size]
    condition_index = jax.random.permutation(rng_perm, base)
    side_index = condition_index // n_coh
    coherence_index = condition_index % n_coh
    correct_choice = jnp.where(side_index == 0, -1.0, 1.0).astype(jnp.float32)
    correct_action = jnp.where(correct_choice < 0.0, CHOOSE_A, CHOOSE_B).astype(jnp.int32)
    coherence = coherence_values[coherence_index]
    signed_coherence = correct_choice * coherence
    noise = float(task.observation_noise_std) * jax.random.normal(
        rng_noise,
        (num_steps, batch_size),
        dtype=jnp.float32,
    )
    evidence_samples = signed_coherence[None, :] + noise
    cumulative_evidence = jnp.cumsum(evidence_samples, axis=0)
    obs_var = float(task.observation_noise_std) ** 2
    oracle_cumulative_llr = (2.0 * coherence[None, :] / obs_var) * cumulative_evidence
    return EvidenceBatch(
        correct_choice=correct_choice,
        correct_action=correct_action,
        coherence=coherence,
        signed_coherence=signed_coherence,
        evidence_samples=evidence_samples,
        cumulative_evidence=cumulative_evidence,
        oracle_cumulative_llr=oracle_cumulative_llr,
    )


def take_evidence_batch(batch: EvidenceBatch, indices: jax.Array) -> EvidenceBatch:
    return EvidenceBatch(
        correct_choice=batch.correct_choice[indices],
        correct_action=batch.correct_action[indices],
        coherence=batch.coherence[indices],
        signed_coherence=batch.signed_coherence[indices],
        evidence_samples=batch.evidence_samples[:, indices],
        cumulative_evidence=batch.cumulative_evidence[:, indices],
        oracle_cumulative_llr=batch.oracle_cumulative_llr[:, indices],
    )


def initial_carry(batch: EvidenceBatch, rnn_units: int) -> EvidenceCarry:
    batch_size = batch.correct_choice.shape[0]
    zeros_h = jnp.zeros((batch_size, rnn_units), dtype=jnp.float32)
    zeros_pre = jnp.zeros((batch_size, rnn_units + 2), dtype=jnp.float32)
    return EvidenceCarry(
        correct_choice=batch.correct_choice,
        correct_action=batch.correct_action,
        coherence=batch.coherence,
        signed_coherence=batch.signed_coherence,
        evidence_samples=batch.evidence_samples,
        cumulative_evidence_by_time=batch.cumulative_evidence,
        oracle_cumulative_llr_by_time=batch.oracle_cumulative_llr,
        current_evidence=batch.evidence_samples[0],
        cumulative_evidence=batch.cumulative_evidence[0],
        oracle_cumulative_llr=batch.oracle_cumulative_llr[0],
        num_observations=jnp.ones((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        h=zeros_h,
        c=zeros_h,
        decoded_h=zeros_h,
        decoded_c=zeros_h,
        lstm_context=zeros_h,
        pre_context=zeros_pre,
        trial_id=jnp.arange(batch_size, dtype=jnp.int32),
    )


def make_schedule(config: RunConfig, update_idx: int, updates_per_epoch: int) -> ScheduleValues:
    epoch = update_idx // max(updates_per_epoch, 1)
    target_beta = 1.0 / config.beta
    if int(config.kl_annealing_epochs) > 0:
        epoch_value = jnp.asarray(epoch, dtype=jnp.float32)
        progress = jnp.minimum(epoch_value / float(max(int(config.kl_annealing_epochs) - 1, 1)), 1.0)
        kl_multiplier = float(config.kl_start_multiplier) + (1.0 - float(config.kl_start_multiplier)) * progress
    else:
        kl_multiplier = 1.0
    current_beta = target_beta * kl_multiplier
    critic_coef = 0.1
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
        current_beta=jnp.asarray(current_beta, dtype=jnp.float32),
        current_critic_coef=jnp.asarray(critic_coef, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(entropy, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )


def learning_rate_at(step: jax.Array, total_steps: int) -> jax.Array:
    step = jnp.asarray(step)
    progress = jnp.minimum(step.astype(jnp.float32) / float(max(total_steps, 1)), 1.0)
    peak = jnp.asarray(5e-4, dtype=jnp.float32)
    floor = peak * 0.1
    return floor + 0.5 * (peak - floor) * (1.0 + jnp.cos(jnp.pi * progress))


def optimizer_steps_per_update(config: RunConfig) -> int:
    return max(int(config.update_epochs), 1) * max(int(config.ppo_minibatches), 1)


class EvidenceVAE(nn.Module):
    rnn_units: int
    latent_dim: int
    max_observations: int
    expansion_decision_version: str
    use_autoencoder: bool
    enable_reconstruction: bool
    enable_probe: bool
    opportunity_cost: float
    correct_reward: float
    incorrect_reward: float

    def setup(self):
        input_dim = 2  # current evidence and normalized observation count
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
        self.expansion_head = nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.glorot_uniform())
        self.value_dense1 = nn.Dense(self.rnn_units, kernel_init=nn.initializers.glorot_uniform())
        self.value_ln = nn.LayerNorm()
        self.value_dense2 = nn.Dense(max(self.rnn_units // 2, 16), kernel_init=nn.initializers.glorot_uniform())
        self.value_out = nn.Dense(1, kernel_init=nn.initializers.glorot_uniform())
        self.reconstruction_head = nn.Dense(2, kernel_init=nn.initializers.glorot_uniform())
        self.probe_head = nn.Dense(2, kernel_init=nn.initializers.glorot_uniform())
        self.prior_mu = self.param("prior_mu", nn.initializers.zeros, (self.max_observations, self.latent_dim))
        self.prior_logvar = self.param("prior_logvar", nn.initializers.zeros, (self.max_observations, self.latent_dim))

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

    def prior(self, num_observations: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        idx = jnp.clip(num_observations - 1, 0, self.max_observations - 1)
        mu = self.prior_mu[idx]
        logvar = self.prior_logvar[idx]
        var = jnp.exp(logvar) + 1e-6
        return mu, logvar, var

    def expansion_input(self, carry: EvidenceCarry) -> jax.Array:
        if self.expansion_decision_version == "decoder":
            return carry.decoded_h
        if self.expansion_decision_version == "lstm":
            return carry.lstm_context
        return carry.pre_context

    def value_critic(self, x: jax.Array) -> jax.Array:
        y = nn.relu(self.value_ln(self.value_dense1(x)))
        y = nn.relu(self.value_dense2(y))
        return self.value_out(y)[:, 0]

    def expansion_logits_from_input(self, x: jax.Array) -> jax.Array:
        return self.expansion_head(x)

    def __call__(
        self,
        carry: EvidenceCarry,
        rng: jax.Array,
        schedule: ScheduleValues,
        forced_action: jax.Array | None = None,
        training: bool = True,
        use_posterior_mean: bool = False,
    ) -> tuple[EvidenceCarry, EvidenceTransition]:
        batch_size = carry.correct_choice.shape[0]
        active = (~carry.done).astype(jnp.float32)
        active_2 = active[:, None]
        obs_count_f = carry.num_observations.astype(jnp.float32)
        normalized_count = obs_count_f / float(max(self.max_observations, 1))
        lstm_input = jnp.stack([carry.current_evidence, normalized_count], axis=-1)
        pre_context = jnp.concatenate([carry.decoded_h, lstm_input], axis=-1)

        raw_h, raw_c = self.lstm_cell(lstm_input, carry.decoded_h, carry.decoded_c)
        raw_h = raw_h * active_2 + carry.h * (1.0 - active_2)
        raw_c = raw_c * active_2 + carry.c * (1.0 - active_2)
        encoder_input = jnp.concatenate([raw_h, raw_c], axis=-1)

        if self.use_autoencoder:
            z_mu, z_logvar, z = self.encode(encoder_input, rng, use_mean=use_posterior_mean)
            prior_mu, prior_logvar, prior_var = self.prior(carry.num_observations)
            dec_h, dec_c = self.decode(z)
            dec_h = dec_h * active_2 + carry.decoded_h * (1.0 - active_2)
            dec_c = dec_c * active_2 + carry.decoded_c * (1.0 - active_2)
            post_var = jnp.exp(jnp.clip(z_logvar, -10.0, 10.0))
            kl_per_dim = 0.5 * (
                jnp.log(prior_var + 1e-6)
                - jnp.log(post_var + 1e-6)
                + (post_var + jnp.square(z_mu - prior_mu)) / (prior_var + 1e-6)
                - 1.0
            )
            observed_kl = jnp.mean(kl_per_dim, axis=-1) * active
        else:
            dec_h, dec_c = raw_h, raw_c
            z_mu = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            z_logvar = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            z = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            prior_mu = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            prior_logvar = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            observed_kl = jnp.zeros((batch_size,), dtype=jnp.float32)

        if self.expansion_decision_version == "decoder":
            action_input = dec_h
        elif self.expansion_decision_version == "lstm":
            action_input = raw_h
        else:
            action_input = pre_context
        logits = self.expansion_head(action_input)
        value_pred = self.value_critic(action_input)

        continue_invalid = carry.num_observations >= int(self.max_observations)
        decision_invalid = jnp.stack(
            [
                continue_invalid.astype(jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            ],
            axis=-1,
        )
        masked_logits = logits + decision_invalid * -1e9
        probs = jax.nn.softmax(masked_logits, axis=-1)
        log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)

        rng_action, rng_eps = jax.random.split(rng)
        sampled_action = jax.random.categorical(rng_action, masked_logits, axis=-1)
        if forced_action is not None:
            action = forced_action.astype(jnp.int32)
        elif training:
            explore = jax.random.uniform(rng_eps, (batch_size,)) < schedule.expansion_epsilon
            uniform_action = jax.random.categorical(rng_eps, decision_invalid * -1e9, axis=-1)
            action = jnp.where(explore, uniform_action, sampled_action)
        else:
            action = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
        action = jnp.clip(action, 0, NUM_ACTIONS - 1)
        log_prob = jnp.take_along_axis(log_probs_all, action[:, None], axis=-1)[:, 0]

        is_continue = (action == CONTINUE) & (active > 0.0)
        is_terminal = (action != CONTINUE) & (active > 0.0)
        terminal_action = jnp.where(is_terminal, action, -jnp.ones_like(action))
        terminal_reward = jnp.where(
            is_terminal & (action == carry.correct_action),
            float(self.correct_reward),
            0.0,
        )
        terminal_reward = jnp.where(
            is_terminal & (action != carry.correct_action),
            float(self.incorrect_reward),
            terminal_reward,
        )
        paid_kl = observed_kl * is_continue.astype(jnp.float32)
        opportunity_cost_paid = float(self.opportunity_cost) * is_continue.astype(jnp.float32)
        memory_cost_paid = schedule.current_beta * paid_kl
        step_reward = terminal_reward - opportunity_cost_paid - memory_cost_paid

        if self.use_autoencoder and self.enable_reconstruction:
            rec_params = self.reconstruction_head(dec_h)
            rec_mean = rec_params[:, 0]
            rec_logvar = jnp.clip(rec_params[:, 1], -8.0, 8.0)
            rec_var = jnp.exp(rec_logvar) + 1e-6
            reconstruction_loss = 0.5 * (
                rec_logvar + jnp.square(carry.current_evidence - rec_mean) / rec_var
            )
            reconstruction_loss = reconstruction_loss * active
        else:
            reconstruction_loss = jnp.zeros((batch_size,), dtype=jnp.float32)

        if self.enable_probe:
            probe_logits = self.probe_head(jax.lax.stop_gradient(raw_h))
            target = (carry.correct_action == CHOOSE_B).astype(jnp.int32)
            probe_loss = optax.softmax_cross_entropy_with_integer_labels(probe_logits, target) * active
            probe_correct = (jnp.argmax(probe_logits, axis=-1) == target).astype(jnp.float32) * active
            valid_probe = active
        else:
            probe_loss = jnp.zeros((batch_size,), dtype=jnp.float32)
            probe_correct = jnp.zeros((batch_size,), dtype=jnp.float32)
            valid_probe = jnp.zeros((batch_size,), dtype=jnp.float32)

        next_obs_index = jnp.clip(carry.num_observations, 0, self.max_observations - 1)
        env_idx = jnp.arange(batch_size)
        next_evidence = carry.evidence_samples[next_obs_index, env_idx]
        next_cumulative = carry.cumulative_evidence_by_time[next_obs_index, env_idx]
        next_oracle_llr = carry.oracle_cumulative_llr_by_time[next_obs_index, env_idx]
        advance = is_continue & (carry.num_observations < int(self.max_observations))
        next_num_observations = carry.num_observations + advance.astype(jnp.int32)
        next_done = carry.done | is_terminal
        next_carry = EvidenceCarry(
            correct_choice=carry.correct_choice,
            correct_action=carry.correct_action,
            coherence=carry.coherence,
            signed_coherence=carry.signed_coherence,
            evidence_samples=carry.evidence_samples,
            cumulative_evidence_by_time=carry.cumulative_evidence_by_time,
            oracle_cumulative_llr_by_time=carry.oracle_cumulative_llr_by_time,
            current_evidence=jnp.where(advance, next_evidence, carry.current_evidence),
            cumulative_evidence=jnp.where(advance, next_cumulative, carry.cumulative_evidence),
            oracle_cumulative_llr=jnp.where(advance, next_oracle_llr, carry.oracle_cumulative_llr),
            num_observations=next_num_observations,
            done=next_done,
            h=raw_h,
            c=raw_c,
            decoded_h=dec_h,
            decoded_c=dec_c,
            lstm_context=raw_h,
            pre_context=pre_context,
            trial_id=carry.trial_id,
        )

        transition = EvidenceTransition(
            valid=active,
            correct_choice=carry.correct_choice,
            correct_action=carry.correct_action,
            coherence=carry.coherence,
            signed_coherence=carry.signed_coherence,
            step_index=carry.num_observations - 1,
            num_observations_before=carry.num_observations,
            num_observations_after=next_num_observations,
            current_evidence=carry.current_evidence,
            cumulative_evidence=carry.cumulative_evidence,
            oracle_cumulative_llr=carry.oracle_cumulative_llr,
            action=action,
            terminal_action=terminal_action,
            is_continue=is_continue.astype(jnp.float32),
            is_terminal=is_terminal.astype(jnp.float32),
            terminal_reward=terminal_reward,
            step_reward=step_reward,
            opportunity_cost_paid=opportunity_cost_paid,
            memory_cost_paid=memory_cost_paid,
            log_prob=log_prob,
            entropy=entropy,
            probs=probs,
            legal_mask=(1.0 - decision_invalid) * active_2,
            value_pred=value_pred,
            expansion_input=action_input,
            z_mu=z_mu,
            z_logvar=z_logvar,
            z_sample=z,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            paid_kl=paid_kl,
            observed_kl=observed_kl,
            reconstruction_loss=reconstruction_loss,
            probe_loss=probe_loss,
            probe_correct=probe_correct,
            valid_probe=valid_probe,
        )
        return next_carry, transition


def build_model(config: RunConfig, task: EvidenceTaskSpec) -> EvidenceVAE:
    return EvidenceVAE(
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        max_observations=config.max_observations_before_stop,
        expansion_decision_version=config.expansion_decision_version,
        use_autoencoder=(config.model_variant == "vae"),
        enable_reconstruction=config.enable_reconstruction,
        enable_probe=config.enable_probe,
        opportunity_cost=config.opportunity_cost,
        correct_reward=task.correct_reward,
        incorrect_reward=task.incorrect_reward,
    )


def build_rollout_fn(model: EvidenceVAE, config: RunConfig):
    def rollout(
        params,
        batch: EvidenceBatch,
        rng: jax.Array,
        schedule: ScheduleValues,
        forced_actions=None,
        training: bool = True,
    ):
        carry = initial_carry(batch, config.rnn_units)

        def scan_step(scan_carry, step_i):
            step_carry, step_rng = scan_carry
            step_rng, model_rng = jax.random.split(step_rng)
            forced_action = None if forced_actions is None else forced_actions[step_i]
            next_carry, transition = model.apply(
                {"params": params},
                step_carry,
                model_rng,
                schedule,
                forced_action,
                training,
                method=EvidenceVAE.__call__,
            )
            return (next_carry, step_rng), transition

        (new_carry, new_rng), transitions = jax.lax.scan(
            scan_step,
            (carry, rng),
            jnp.arange(config.num_steps),
        )
        return new_carry, new_rng, transitions

    return rollout


def compute_lambda_returns(
    rewards: jax.Array,
    values: jax.Array,
    valid: jax.Array,
    lambda_return: float,
) -> tuple[jax.Array, jax.Array]:
    """GAE/lambda returns for [time, batch] transitions with gamma=1."""
    next_values = jnp.concatenate([values[1:], jnp.zeros_like(values[:1])], axis=0)
    next_valid = jnp.concatenate([valid[1:], jnp.zeros_like(valid[:1])], axis=0)

    def backward(carry, inputs):
        reward_t, value_t, next_value_t, valid_t, next_valid_t = inputs
        delta = reward_t + next_value_t * next_valid_t - value_t
        advantage = delta + float(lambda_return) * next_valid_t * carry
        advantage = advantage * valid_t
        return advantage, advantage

    _last, advantages_rev = jax.lax.scan(
        backward,
        jnp.zeros_like(values[0]),
        (rewards[::-1], values[::-1], next_values[::-1], valid[::-1], next_valid[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return returns, advantages


def first_terminal_indices(transitions: EvidenceTransition) -> tuple[jax.Array, jax.Array]:
    final_rollout_step = jax.nn.one_hot(
        transitions.is_terminal.shape[0] - 1,
        transitions.is_terminal.shape[0],
        dtype=jnp.float32,
    )[:, None]
    terminal_flags = (transitions.is_terminal > 0.0) | (
        (final_rollout_step > 0.0) & (transitions.valid > 0.0)
    )
    has_terminal = jnp.any(terminal_flags, axis=0)
    first_terminal = jnp.argmax(terminal_flags.astype(jnp.int32), axis=0)
    last_step = transitions.is_terminal.shape[0] - 1
    selected_step = jnp.where(has_terminal, first_terminal, last_step)
    env_idx = jnp.arange(transitions.is_terminal.shape[1])
    return selected_step, env_idx


def terminal_expected_action_loss(transitions: EvidenceTransition) -> jax.Array:
    selected_step, env_idx = first_terminal_indices(transitions)
    terminal_probs = transitions.probs[selected_step, env_idx, :]
    correct_action = transitions.correct_action[selected_step, env_idx]
    p_correct = jnp.take_along_axis(terminal_probs, correct_action[:, None], axis=-1)[:, 0]
    return 1.0 - jnp.mean(p_correct)


def summarize_rollout_metrics(transitions: EvidenceTransition) -> tuple[jax.Array, jax.Array, jax.Array]:
    selected_step, env_idx = first_terminal_indices(transitions)
    terminal_action = transitions.terminal_action[selected_step, env_idx]
    correct_action = transitions.correct_action[selected_step, env_idx]
    accuracy = jnp.mean((terminal_action == correct_action).astype(jnp.float32))
    mean_observations = jnp.mean(transitions.num_observations_before[selected_step, env_idx].astype(jnp.float32))
    stop_rate = jnp.sum(transitions.is_terminal) / (jnp.sum(transitions.valid) + 1e-6)
    return accuracy, mean_observations, stop_rate


def create_train_state(model: EvidenceVAE, config: RunConfig, task: EvidenceTaskSpec, rng: jax.Array, total_updates: int):
    dummy_batch = sample_evidence_batch(
        rng,
        config.num_envs,
        config.num_steps,
        task,
    )
    dummy_carry = initial_carry(dummy_batch, config.rnn_units)
    dummy_schedule = ScheduleValues(
        current_alpha=1.0,
        current_beta=1.0 / config.beta,
        current_critic_coef=0.1,
        expansion_epsilon=0.0,
        expansion_entropy_coef=1.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    params = model.init(rng, dummy_carry, rng, dummy_schedule, None, True)["params"]
    schedule = lambda step: learning_rate_at(step, total_updates * optimizer_steps_per_update(config))
    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )
    return EvidenceTrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params=params,
        tx=tx,
    )


def soft_update_tree(source, target, tau: float):
    tau_value = jnp.asarray(tau, dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda src, tgt: tau_value * src + (1.0 - tau_value) * tgt,
        source,
        target,
    )


def build_update_fn(model: EvidenceVAE, task: EvidenceTaskSpec, config: RunConfig, total_updates: int):
    rollout = build_rollout_fn(model, config)
    batch_size = config.num_envs * config.return_target_rollouts
    num_minibatches = max(int(config.ppo_minibatches), 1)
    if batch_size % num_minibatches != 0:
        raise ValueError("--ppo-minibatches must divide num_envs * return_target_rollouts.")
    minibatch_size = batch_size // num_minibatches

    def loss_for_batch(
        params,
        batch: EvidenceBatch,
        rng: jax.Array,
        schedule: ScheduleValues,
        forced_actions: jax.Array,
        old_logp: jax.Array,
        returns: jax.Array,
        advantages: jax.Array,
    ):
        _carry, _rng, transitions = rollout(
            params,
            batch,
            rng,
            schedule,
            forced_actions=forced_actions,
            training=True,
        )
        ratio = jnp.exp(jnp.clip(transitions.log_prob - old_logp, -10.0, 10.0))
        clipped_ratio = jnp.clip(ratio, 1.0 - schedule.ppo_clip, 1.0 + schedule.ppo_clip)
        valid = transitions.valid
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages) * valid
        expansion_loss = (
            jnp.sum(policy_loss) / (jnp.sum(valid) + 1e-6)
            - schedule.expansion_entropy_coef
            * (jnp.sum(transitions.entropy * valid) / (jnp.sum(valid) + 1e-6))
        )
        critic_loss = (
            jnp.sum(jnp.square(transitions.value_pred - returns) * valid)
            / (jnp.sum(valid) + 1e-6)
        )
        information_loss = jnp.sum(transitions.paid_kl * valid) / (jnp.sum(valid) + 1e-6)
        reconstruction_loss = jnp.sum(transitions.reconstruction_loss * valid) / (jnp.sum(valid) + 1e-6)
        action_loss = terminal_expected_action_loss(transitions)
        probe_loss = jnp.sum(transitions.probe_loss) / (jnp.sum(transitions.valid_probe) + 1e-6)
        probe_acc = jnp.sum(transitions.probe_correct) / (jnp.sum(transitions.valid_probe) + 1e-6)
        total_loss = (
            expansion_loss * config.lambda_
            + critic_loss * config.lambda_ * schedule.current_critic_coef
            + action_loss * config.lambda_
            + information_loss * schedule.current_beta
            + reconstruction_loss * config.alpha
            + probe_loss
        )
        accuracy, mean_observations, stop_rate = summarize_rollout_metrics(transitions)
        metrics = UpdateMetrics(
            total_loss=total_loss,
            information_loss=information_loss,
            action_loss=action_loss,
            reconstruction_loss=reconstruction_loss,
            expansion_loss=expansion_loss,
            critic_loss=critic_loss,
            probe_loss=probe_loss,
            probe_accuracy=probe_acc,
            stop_rate=stop_rate,
            continue_rate=jnp.sum(transitions.is_continue) / (jnp.sum(valid) + 1e-6),
            accuracy=accuracy,
            mean_observations=mean_observations,
            entropy_coef=jnp.asarray(schedule.expansion_entropy_coef),
            critic_coef=jnp.asarray(schedule.current_critic_coef),
            current_beta=jnp.asarray(schedule.current_beta),
            learning_rate=learning_rate_at(train_state_step_for_lr(), total_updates * optimizer_steps_per_update(config)),
        )
        return total_loss, metrics

    def train_state_step_for_lr():
        return jnp.asarray(0, dtype=jnp.int32)

    def update_step(train_state: EvidenceTrainState, rng: jax.Array, schedule: ScheduleValues):
        rng, batch_rng, rollout_rng, ppo_rng = jax.random.split(rng, 4)
        batch = sample_evidence_batch(batch_rng, batch_size, config.num_steps, task)
        _carry, _rollout_rng, old_transitions = rollout(
            train_state.params,
            batch,
            rollout_rng,
            schedule,
            forced_actions=None,
            training=True,
        )
        old_logp = jax.lax.stop_gradient(old_transitions.log_prob)
        returns, advantages = compute_lambda_returns(
            old_transitions.step_reward,
            old_transitions.value_pred,
            old_transitions.valid,
            config.lambda_return,
        )
        returns = jax.lax.stop_gradient(returns)
        advantages = jax.lax.stop_gradient(
            (advantages - jnp.mean(advantages * old_transitions.valid) / (jnp.mean(old_transitions.valid) + 1e-6))
            / (jnp.sqrt(jnp.mean(jnp.square(advantages) * old_transitions.valid) + 1e-6))
        )

        def ppo_epoch(epoch_state, _):
            state, epoch_rng = epoch_state
            epoch_rng, perm_rng, replay_rng = jax.random.split(epoch_rng, 3)
            permutation = jax.random.permutation(perm_rng, batch_size)
            minibatches = permutation.reshape((num_minibatches, minibatch_size))
            replay_keys = jax.random.split(replay_rng, num_minibatches)

            def minibatch_step(minibatch_state, inputs):
                idx, key = inputs
                sub_batch = take_evidence_batch(batch, idx)
                sub_actions = old_transitions.action[:, idx]
                sub_logp = old_logp[:, idx]
                sub_returns = returns[:, idx]
                sub_adv = advantages[:, idx]

                def loss_fn(params):
                    total_loss, metrics = loss_for_batch(
                        params,
                        sub_batch,
                        key,
                        schedule,
                        sub_actions,
                        sub_logp,
                        sub_returns,
                        sub_adv,
                    )
                    metrics = metrics._replace(
                        learning_rate=learning_rate_at(
                            minibatch_state.step,
                            total_updates * optimizer_steps_per_update(config),
                        )
                    )
                    return total_loss, metrics

                (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    minibatch_state.params
                )
                del loss_value
                new_state = minibatch_state.apply_gradients(grads=grads)
                return new_state, metrics

            state, metrics_tree = jax.lax.scan(
                minibatch_step,
                state,
                (minibatches, replay_keys),
            )
            mean_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_tree)
            return (state, epoch_rng), mean_metrics

        (new_train_state, rng), metrics_tree = jax.lax.scan(
            ppo_epoch,
            (train_state, ppo_rng),
            xs=None,
            length=max(int(config.update_epochs), 1),
        )
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_tree)
        if config.target_critic_update_interval > 0:
            should_update_target = (new_train_state.step % int(config.target_critic_update_interval)) == 0
            updated_target = soft_update_tree(
                new_train_state.params,
                new_train_state.target_params,
                config.target_critic_tau,
            )
            target_params = jax.tree_util.tree_map(
                lambda new, old: jnp.where(should_update_target, new, old),
                updated_target,
                new_train_state.target_params,
            )
            new_train_state = new_train_state.replace(target_params=target_params)
        return new_train_state, rng, metrics

    return jax.jit(update_step) if config.jit_training else update_step


def train(config: RunConfig, task: EvidenceTaskSpec) -> tuple[EvidenceVAE, EvidenceTrainState]:
    setup_start = time.perf_counter()
    rng = jax.random.PRNGKey(config.seed)
    model = build_model(config, task)
    updates_per_epoch = max(1, math.ceil(config.steps_per_epoch / (config.num_envs * config.num_steps)))
    total_updates = config.epochs * updates_per_epoch
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, config, task, init_rng, total_updates)
    update_fn = build_update_fn(model, task, config, total_updates)
    setup_sec = time.perf_counter() - setup_start
    print(
        "Evidence JAX setup timing: "
        f"setup={setup_sec:.3f}s | updates_per_epoch={updates_per_epoch} | "
        f"total_updates={total_updates} | jit_training={config.jit_training} | "
        f"coherence_values={','.join(f'{x:g}' for x in config.coherence_values)} | "
        f"observation_noise_std={config.observation_noise_std} | "
        f"max_observations_before_stop={config.max_observations_before_stop}",
        flush=True,
    )
    rows = []
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        metrics_acc = []
        for update_idx in range(updates_per_epoch):
            global_update = epoch * updates_per_epoch + update_idx
            schedule = make_schedule(config, global_update, updates_per_epoch)
            rng, step_rng = jax.random.split(rng)
            state, rng, metrics = update_fn(state, step_rng, schedule)
            metrics_acc.append(jax.device_get(metrics))
        epoch_sec = time.perf_counter() - epoch_start
        row = finalize_epoch_row(epoch + 1, metrics_acc, config, updates_per_epoch)
        rows.append(row)
        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"Loss = {row['total_loss']:.4f} | KL = {row['kl_loss']:.4f} | "
            f"Accuracy = {row['accuracy']:.4f} | Obs = {row['mean_observations']:.2f} | "
            f"Stop = {row['stop_rate']:.4f} | Continue = {row['continue_rate']:.4f} | "
            f"Timing epoch = {epoch_sec:.2f}s",
            flush=True,
        )
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name_for(config)
    log_path = model_dir / f"{model_name}_training_logs.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    weights_path = model_dir / f"{model_name}.msgpack"
    weights_path.write_bytes(serialization.to_bytes(state.params))
    print(f"Saved evidence training logs to: {log_path}", flush=True)
    print(f"Saved evidence model weights to: {weights_path}", flush=True)
    return model, state


def finalize_epoch_row(epoch: int, metrics_acc: list[UpdateMetrics], config: RunConfig, updates_per_epoch: int):
    stack = jax.tree_util.tree_map(lambda *xs: np.asarray(xs, dtype=float), *metrics_acc)
    return {
        "epoch": epoch,
        "learning_rate": float(np.mean(stack.learning_rate)),
        "steps_per_epoch": config.steps_per_epoch,
        "steps_per_batch": config.num_envs * config.num_steps,
        "updates_per_epoch": updates_per_epoch,
        "rollout_steps": config.num_steps,
        "return_target_rollouts": config.return_target_rollouts,
        "ppo_minibatches": config.ppo_minibatches,
        "update_epochs": config.update_epochs,
        "expansion_return_target_mode": config.return_target_mode,
        "expansion_lambda_return": config.lambda_return,
        "coherence_values": ",".join(f"{x:g}" for x in config.coherence_values),
        "observation_noise_std": config.observation_noise_std,
        "max_observations_before_stop": config.max_observations_before_stop,
        "total_loss": float(np.mean(stack.total_loss)),
        "information_loss": float(np.mean(stack.information_loss)),
        "kl_loss": float(np.mean(stack.information_loss)),
        "action_loss": float(np.mean(stack.action_loss)),
        "reconstruction_loss": float(np.mean(stack.reconstruction_loss)),
        "expansion_loss": float(np.mean(stack.expansion_loss)),
        "critic_loss": float(np.mean(stack.critic_loss)),
        "probe_loss": float(np.mean(stack.probe_loss)),
        "probe_accuracy": float(np.mean(stack.probe_accuracy)),
        "stop_rate": float(np.mean(stack.stop_rate)),
        "continue_rate": float(np.mean(stack.continue_rate)),
        "accuracy": float(np.mean(stack.accuracy)),
        "mean_observations": float(np.mean(stack.mean_observations)),
        "expansion_entropy_coef": float(np.mean(stack.entropy_coef)),
        "critic_coef": float(np.mean(stack.critic_coef)),
        "current_beta": float(np.mean(stack.current_beta)),
    }


def load_state_for_sim(config: RunConfig, task: EvidenceTaskSpec) -> tuple[EvidenceVAE, object]:
    model = build_model(config, task)
    rng = jax.random.PRNGKey(config.seed)
    dummy_batch = sample_evidence_batch(rng, 1, config.num_steps, task)
    dummy = initial_carry(dummy_batch, config.rnn_units)
    sched = ScheduleValues(
        current_alpha=1.0,
        current_beta=1.0 / config.beta,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    params = model.init(rng, dummy, rng, sched, None, False)["params"]
    weights_path = Path(config.model_dir) / f"{model_name_for(config)}.msgpack"
    if weights_path.exists():
        params = serialization.from_bytes(params, weights_path.read_bytes())
    else:
        print(f"Warning: {weights_path} not found; simulating initialized evidence model.", flush=True)
    return model, params


def simulate(config: RunConfig, task: EvidenceTaskSpec, model: EvidenceVAE | None = None, params=None):
    sim_start = time.perf_counter()
    if model is None or params is None:
        model, params = load_state_for_sim(config, task)
    rng = jax.random.PRNGKey(config.seed + 100_000)
    batch = sample_evidence_batch(rng, config.n_sim_trials, config.num_steps, task)
    rollout = build_rollout_fn(model, config)
    sched = ScheduleValues(
        current_alpha=1.0,
        current_beta=1.0 / config.beta,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    rng, rollout_rng = jax.random.split(rng)
    _carry, _rng, transitions = rollout(
        params,
        batch,
        rollout_rng,
        sched,
        forced_actions=None,
        training=True,
    )
    transitions = jax.device_get(transitions)
    batch_np = jax.device_get(batch)
    rows = simulation_rows(config, batch_np, transitions)
    out_dir = Path(config.sim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name_for(config)}_{config.input_type}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    summary_path = out_dir / f"{model_name_for(config)}_{config.input_type}_summary.csv"
    save_condition_summary(rows, summary_path)
    print(f"Saved evidence simulation results to: {out_path}", flush=True)
    print(f"Saved evidence simulation summary to: {summary_path}", flush=True)
    print(f"Evidence simulation timing: total={time.perf_counter() - sim_start:.3f}s", flush=True)


def simulation_rows(config: RunConfig, batch: EvidenceBatch, transitions: EvidenceTransition) -> list[dict]:
    rows = []
    n_trials = int(batch.correct_choice.shape[0])
    for trial in range(n_trials):
        terminal_step = None
        for t in range(config.num_steps):
            if bool(transitions.is_terminal[t, trial] > 0):
                terminal_step = t
                break
        if terminal_step is None:
            terminal_step = config.num_steps - 1
            terminal_action = int(np.argmax(np.asarray(transitions.probs[terminal_step, trial, 1:])) + 1)
            terminal_reward = float(terminal_action == int(batch.correct_action[trial]))
        else:
            terminal_action = int(transitions.terminal_action[terminal_step, trial])
            terminal_reward = float(transitions.terminal_reward[terminal_step, trial])
        num_observations = int(transitions.num_observations_before[terminal_step, trial])
        row = {
            "graph": trial,
            "correct_choice": int(batch.correct_choice[trial]),
            "correct_action": int(batch.correct_action[trial]),
            "coherence": float(batch.coherence[trial]),
            "signed_coherence": float(batch.signed_coherence[trial]),
            "num_observations": num_observations,
            "stopping_time": num_observations,
            "terminal_action": terminal_action,
            "terminal_reward": terminal_reward,
            "choose_right": terminal_action == CHOOSE_B,
            "choose_correct": terminal_action == int(batch.correct_action[trial]),
            "total_reward": float(np.sum(transitions.step_reward[:, trial])),
            "total_opportunity_cost": float(np.sum(transitions.opportunity_cost_paid[:, trial])),
            "total_memory_cost": float(np.sum(transitions.memory_cost_paid[:, trial])),
            "total_kl_paid": float(np.sum(transitions.paid_kl[:, trial])),
            "decision_cumulative_evidence": float(transitions.cumulative_evidence[terminal_step, trial]),
            "decision_oracle_cumulative_llr": float(transitions.oracle_cumulative_llr[terminal_step, trial]),
        }
        for t in range(config.num_steps):
            step = t + 1
            row[f"evidence_sample_t{step}"] = float(batch.evidence_samples[t, trial])
            row[f"cumulative_evidence_t{step}"] = float(batch.cumulative_evidence[t, trial])
            row[f"oracle_cumulative_llr_t{step}"] = float(batch.oracle_cumulative_llr[t, trial])
            row[f"action_t{step}"] = int(transitions.action[t, trial])
            row[f"continue_t{step}"] = bool(transitions.is_continue[t, trial] > 0)
            row[f"stop_t{step}"] = bool(transitions.is_terminal[t, trial] > 0)
            row[f"kl_d_t{step}"] = float(transitions.paid_kl[t, trial])
            row[f"kl_d_obs_t{step}"] = float(transitions.observed_kl[t, trial])
            row[f"opportunity_cost_t{step}"] = float(transitions.opportunity_cost_paid[t, trial])
            row[f"memory_cost_t{step}"] = float(transitions.memory_cost_paid[t, trial])
            row[f"policy_continue_t{step}"] = float(transitions.probs[t, trial, CONTINUE])
            row[f"policy_choose_a_t{step}"] = float(transitions.probs[t, trial, CHOOSE_A])
            row[f"policy_choose_b_t{step}"] = float(transitions.probs[t, trial, CHOOSE_B])
            row[f"value_pred_t{step}"] = float(transitions.value_pred[t, trial])
            row[f"action_policy_entropy_t{step}"] = float(transitions.entropy[t, trial])
            for dim in range(min(config.latent_dim, 16)):
                row[f"z_mu_{dim}_t{step}"] = float(transitions.z_mu[t, trial, dim])
                row[f"z_logvar_{dim}_t{step}"] = float(transitions.z_logvar[t, trial, dim])
                row[f"z_sample_{dim}_t{step}"] = float(transitions.z_sample[t, trial, dim])
                row[f"prior_mu_{dim}_t{step}"] = float(transitions.prior_mu[t, trial, dim])
                row[f"prior_logvar_{dim}_t{step}"] = float(transitions.prior_logvar[t, trial, dim])
        rows.append(row)
    return rows


def save_condition_summary(rows: list[dict], out_path: Path):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    summary = (
        df.groupby(["coherence", "signed_coherence"], dropna=False)
        .agg(
            n=("graph", "count"),
            p_choose_right=("choose_right", "mean"),
            p_choose_correct=("choose_correct", "mean"),
            mean_num_observations=("num_observations", "mean"),
            median_num_observations=("num_observations", "median"),
            mean_stopping_time=("stopping_time", "mean"),
            median_stopping_time=("stopping_time", "median"),
            mean_total_reward=("total_reward", "mean"),
            mean_terminal_reward=("terminal_reward", "mean"),
            mean_decision_cumulative_evidence=("decision_cumulative_evidence", "mean"),
            mean_decision_oracle_cumulative_llr=("decision_oracle_cumulative_llr", "mean"),
            p_choose_a=("terminal_action", lambda x: np.mean(np.asarray(x) == CHOOSE_A)),
            p_choose_b=("terminal_action", lambda x: np.mean(np.asarray(x) == CHOOSE_B)),
        )
        .reset_index()
    )
    summary.to_csv(out_path, index=False)


def validate_task_generator(config: RunConfig):
    task = make_task(config)
    n = 120_000
    rng = jax.random.PRNGKey(config.seed + 777)
    batch = sample_evidence_batch(rng, n, max(4, config.num_steps), task)
    evidence = np.asarray(batch.evidence_samples)
    y = np.asarray(batch.correct_choice)
    c = np.asarray(batch.coherence)
    signed = np.asarray(batch.signed_coherence)
    tol_mean = 0.04
    tol_var = 0.05
    for side in (-1.0, 1.0):
        for coh in task.coherence_values:
            mask = (y == side) & np.isclose(c, coh)
            vals = evidence[:, mask].reshape(-1)
            if vals.size == 0:
                continue
            empirical_mean = float(np.mean(vals))
            empirical_var = float(np.var(vals))
            expected_mean = side * coh
            if abs(empirical_mean - expected_mean) > tol_mean:
                raise AssertionError((side, coh, empirical_mean, expected_mean))
            if abs(empirical_var - task.observation_noise_std ** 2) > tol_var:
                raise AssertionError((side, coh, empirical_var, task.observation_noise_std ** 2))
    zero_mask = np.isclose(c, 0.0)
    if np.any(zero_mask):
        left = evidence[:, zero_mask & (y < 0)].reshape(-1)
        right = evidence[:, zero_mask & (y > 0)].reshape(-1)
        if abs(float(np.mean(left)) - float(np.mean(right))) > tol_mean:
            raise AssertionError("c=0 left/right evidence means differ unexpectedly.")
    for coh in task.coherence_values:
        if coh <= 0:
            continue
        mask = np.isclose(c, coh)
        vals = evidence[:, mask]
        signs = np.sign(vals) == np.sign(signed[mask])[None, :]
        if float(np.mean(signs)) <= 0.5:
            raise AssertionError(f"Evidence sign accuracy did not exceed chance for coherence={coh}.")
    cumulative = np.asarray(batch.cumulative_evidence)
    llr = np.asarray(batch.oracle_cumulative_llr)
    expected_llr = (2.0 * c[None, :] / (task.observation_noise_std ** 2)) * cumulative
    if not np.allclose(llr, expected_llr, atol=1e-5):
        raise AssertionError("Oracle cumulative LLR formula mismatch.")
    correct_action = np.asarray(batch.correct_action)
    reward_a = (correct_action == CHOOSE_A).astype(float)
    reward_b = (correct_action == CHOOSE_B).astype(float)
    if not np.all((reward_a + reward_b) == 1.0):
        raise AssertionError("Terminal reward mapping is inconsistent.")
    print("Evidence task validation passed.", flush=True)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_string", nargs="?", default="100.0")
    parser.add_argument("alpha_string", nargs="?", default="0.0")
    parser.add_argument("beta_string", nargs="?", default="1000.0")
    parser.add_argument("model_dir", nargs="?", default="outputs/jax_models_evi")
    parser.add_argument("epochs", nargs="?", type=int, default=120)
    parser.add_argument("input_type", nargs="?", default="evidence")
    parser.add_argument("seed", nargs="?", type=int, default=1)
    parser.add_argument("tree_size", nargs="?", type=int, default=2)
    parser.add_argument("train_mode", nargs="?", default="train")
    parser.add_argument("tree_type", nargs="?", default="evidence")
    parser.add_argument("opportunity_cost_string", nargs="?", default="0.0")
    parser.add_argument("expansion_decision_version", nargs="?", default="lstm")
    parser.add_argument("model_variant", nargs="?", default="vae")
    parser.add_argument("rnn_units", nargs="?", type=int, default=32)
    parser.add_argument("latent_dim", nargs="?", type=int, default=16)
    parser.add_argument("--sim-dir", default="outputs/jax_simulations_evi")
    parser.add_argument("--n-sim-trials", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatches", type=int, default=int(os.environ.get("PPO_MINIBATCHES", "1")))
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--return-target-rollouts", type=int, default=int(os.environ.get("RETURN_TARGET_ROLLOUTS", "1")))
    parser.add_argument(
        "--return-target-mode",
        default=os.environ.get("EXPANSION_RETURN_TARGET", "sampled_lambda"),
        help="Accepted for launch compatibility; evidence PPO uses sampled trajectory lambda returns.",
    )
    parser.add_argument("--lambda-return", type=float, default=float(os.environ.get("EXPANSION_LAMBDA_RETURN", "0.95")))
    parser.add_argument(
        "--sampled-lambda-critic",
        choices=["value", "q"],
        default=os.environ.get("SAMPLED_LAMBDA_CRITIC", "value").strip().lower(),
        help="Accepted for filename/log compatibility. Evidence PPO uses scalar V(h_t).",
    )
    parser.add_argument("--target-critic-update-interval", type=int, default=int(os.environ.get("TARGET_CRITIC_UPDATE_INTERVAL", "100")))
    parser.add_argument("--target-critic-tau", type=float, default=float(os.environ.get("TARGET_CRITIC_TAU", "1.0")))
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--enable-reconstruction", action="store_true")
    parser.add_argument("--enable-probe", action="store_true")
    parser.add_argument("--max-observations-before-stop", type=int, default=int(os.environ.get("MAX_OBSERVATIONS_BEFORE_STOP", "10")))
    parser.add_argument("--coherence-values", default=os.environ.get("COHERENCE_VALUES", "0,0.05,0.1,0.2,0.4,0.8"))
    parser.add_argument("--observation-noise-std", type=float, default=float(os.environ.get("OBSERVATION_NOISE_STD", "1.0")))
    parser.add_argument("--correct-reward", type=float, default=1.0)
    parser.add_argument("--incorrect-reward", type=float, default=0.0)
    parser.add_argument("--kl-start-multiplier", type=float, default=float(os.environ.get("KL_START_MULTIPLIER", "1.0")))
    parser.add_argument("--kl-annealing-epochs", type=int, default=int(os.environ.get("KL_ANNEALING_EPOCHS", "0")))
    parser.add_argument("--validate-task", action="store_true")
    args = parser.parse_args()
    lambda_values = parse_float_list(args.lambda_string)
    alpha_values = parse_float_list(args.alpha_string)
    beta_values = parse_float_list(args.beta_string)
    opportunity_values = parse_float_list(args.opportunity_cost_string)
    if not (len(lambda_values) == len(alpha_values) == len(beta_values) == len(opportunity_values) == 1):
        raise ValueError("evidence_accumulation.py expects one lambda/alpha/beta/opportunity per process.")
    max_observations = max(int(args.max_observations_before_stop), 1)
    num_steps = int(args.num_steps or max_observations)
    if num_steps != max_observations:
        raise ValueError(
            "Evidence timing uses exactly one decision per possible observation. "
            f"Set --num-steps equal to --max-observations-before-stop ({max_observations}) or omit it."
        )
    coherence_values = tuple(parse_float_list(args.coherence_values))
    if not coherence_values:
        raise ValueError("--coherence-values must contain at least one nonnegative value.")
    if any(v < 0 for v in coherence_values):
        raise ValueError("--coherence-values must be nonnegative magnitudes.")
    num_envs = int(args.num_envs)
    return_target_rollouts = max(int(args.return_target_rollouts), 1)
    ppo_minibatches = max(int(args.ppo_minibatches), 1)
    if num_envs * return_target_rollouts % ppo_minibatches != 0:
        raise ValueError("--ppo-minibatches must evenly divide num_envs * return_target_rollouts.")
    steps_per_epoch = int(args.steps_per_epoch or (200 * 200 * num_steps))
    return RunConfig(
        lambda_=lambda_values[0],
        alpha=alpha_values[0],
        beta=beta_values[0],
        model_dir=str(args.model_dir),
        epochs=int(args.epochs),
        input_type=str(args.input_type),
        seed=int(args.seed),
        tree_size=int(args.tree_size),
        train_mode=str(args.train_mode),
        tree_type=str(args.tree_type),
        opportunity_cost=opportunity_values[0],
        expansion_decision_version=normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=normalize_model_variant(args.model_variant),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir=str(args.sim_dir),
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
        enable_reconstruction=bool(args.enable_reconstruction),
        enable_probe=bool(args.enable_probe),
        max_observations_before_stop=max_observations,
        coherence_values=coherence_values,
        observation_noise_std=max(float(args.observation_noise_std), 1e-6),
        correct_reward=float(args.correct_reward),
        incorrect_reward=float(args.incorrect_reward),
        kl_start_multiplier=max(float(args.kl_start_multiplier), 0.0),
        kl_annealing_epochs=max(int(args.kl_annealing_epochs), 0),
    )


def main():
    config = parse_args()
    if config.backend:
        jax.config.update("jax_platform_name", config.backend)
    task = make_task(config)
    if "--validate-task" in sys.argv:
        validate_task_generator(config)
        return
    if config.train_mode in {"train", "training"}:
        model, state = train(config, task)
        simulate(config, task, model, state.params)
    else:
        simulate(config, task)


if __name__ == "__main__":
    main()
