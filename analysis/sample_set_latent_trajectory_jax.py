#!/usr/bin/env python3
"""Sample-set latent trajectory plots for two-node revisit JAX models.

For each actual node-1 x node-2 reward combination, this script generates
multiple noisy observation streams per node, runs the trained revisit policy,
and records the latent posterior trajectory at each observed reward.

The default checkpoint grid matches:

  beta sweep: beta = 10,20,80; opportunity = 0
  opportunity sweep: beta = 100000; opportunity = 0.06,0.2,0.4
  sigmas = 0,0.5,1,2; seeds = 1,2,3

Outputs are one figure per model parameter and sigma. By default, the script
writes the shaded z_mu trajectory and action-logit plots; the 3D and contour
plots are optional via --latent-trajectory-plot-types.

  latent_mu_sigma_3d_by_node_rewards.png
      Rows are actual node-1 reward, columns are actual node-2 reward. Within
      each panel, x = current observation index, y = mean z_mu, z = mean z_sigma.

  action_logit_by_node_rewards.png
      Same row/column panels, with x = current observation index and
      y = terminal action logit.

  latent_mu_sem_shaded_error_by_node_rewards.png
      Same as the mean-z shaded plot, but the shaded band is +/- SEM of the
      plotted z_mu value across sample sets/seeds in that cell.

  action_logit_sem_shaded_error_by_node_rewards.png
      Same as the action-logit plot, but the shaded band is +/- SEM.

With --aggregate-only, only three pooled delta plots are written under
aggregate_delta_plots/.  Rows pair high/medium/low beta-memory cost with
high/medium/low opportunity cost, columns are observation sigma, and each panel
overlays the paired beta-vary and opportunity-vary models.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR), str(REPO_ROOT / "model_jax")):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis import sample_set_pairwise_last_paid_kl_jax as sample_base  # noqa: E402
from model_jax import evidence_accumulation as ev  # noqa: E402
from model_jax import planning as jp  # noqa: E402


PLOT_FONT_SIZE = 7
EPS = 1e-8


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "figure.dpi": 300,
        }
    )


def value_token(value) -> str:
    return sample_base.value_token(value)


def values_token(values) -> str:
    return sample_base.values_token(values)


def terminal_action_logit(action_output: np.ndarray, mode: str) -> np.ndarray:
    probs = np.asarray(action_output, dtype=float)
    probs = np.where(np.isfinite(probs), probs, np.nan)
    if probs.ndim != 2 or probs.shape[1] < 2:
        return np.full((probs.shape[0],), np.nan, dtype=float)
    if probs.shape[1] == 2:
        logit = np.log(np.clip(probs[:, 0], EPS, 1.0)) - np.log(np.clip(probs[:, 1], EPS, 1.0))
        if mode == "abs_path1_minus_path2":
            return np.abs(logit)
        return logit
    log_probs = np.log(np.clip(probs, EPS, 1.0))
    sorted_log_probs = np.sort(log_probs, axis=1)
    return sorted_log_probs[:, -1] - sorted_log_probs[:, -2]


def rollout_latent_trajectory_rows(
    *,
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    streams: np.ndarray,
    metadata: pd.DataFrame,
    seed_offset: int,
    force_first_observe_node: int,
    action_logit_mode: str,
    progress_label: str,
) -> pd.DataFrame:
    n_trials = int(rewards.shape[0])
    reward_feature_dim = int(model.reward_feature_dim_override) or jp.reward_feature_dim_for_sigma(
        config.observation_sigma
    )
    carry = jp.initial_carry(
        n_trials,
        task,
        int(config.rnn_units),
        reward_feature_dim,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    carry = jp.reset_done_envs(carry, jnp.asarray(rewards, dtype=jnp.float32))
    sched = sample_base.schedule_for(config.beta)
    rng = jax.random.PRNGKey(int(config.seed) + 970_000 + int(seed_offset))
    stream_counts = np.zeros((n_trials, task.num_nodes), dtype=np.int32)
    observation_count = np.zeros(n_trials, dtype=np.int32)
    stopped = np.zeros(n_trials, dtype=bool)
    rows: list[dict] = []
    reward_node_1 = rewards[:, 0].astype(float)
    reward_node_2 = rewards[:, 1].astype(float) if rewards.shape[1] > 1 else np.full(n_trials, np.nan)
    condition_index = metadata["condition_index"].to_numpy(dtype=int) if "condition_index" in metadata else np.arange(n_trials)
    original_condition_index = (
        metadata["original_condition_index"].to_numpy(dtype=int)
        if "original_condition_index" in metadata
        else condition_index
    )
    sample_set = metadata["sample_set"].to_numpy(dtype=int) if "sample_set" in metadata else np.arange(n_trials)

    print(
        f"{progress_label}: trajectory rollout starts with {n_trials} trial(s), "
        f"num_steps={int(config.num_steps)}",
        flush=True,
    )
    for timestep in range(1, int(config.num_steps) + 1):
        rng, step_rng = jax.random.split(rng)
        if timestep == 1 and 1 <= int(force_first_observe_node) <= int(task.num_nodes):
            action = np.full(n_trials, int(force_first_observe_node) - 1, dtype=np.int32)
        else:
            _, probe_trans = model.apply(
                {"params": params},
                carry,
                step_rng,
                sched,
                forced_action=None,
                training=True,
                use_posterior_mean=False,
                compute_targets=False,
                method=jp.PlanningVAE.__call__,
            )
            action = np.asarray(jax.device_get(probe_trans.action), dtype=np.int32)

        forced_observation = np.full(n_trials, np.nan, dtype=np.float32)
        sample_position = np.full(n_trials, -1, dtype=np.int32)
        carry_done = np.asarray(jax.device_get(carry.done), dtype=bool)
        for trial in range(n_trials):
            if carry_done[trial]:
                continue
            if action[trial] < int(task.num_nodes):
                node = int(action[trial])
                sample_idx = min(stream_counts[trial, node], streams.shape[2] - 1)
                forced_observation[trial] = streams[trial, node, sample_idx]
                sample_position[trial] = sample_idx
                stream_counts[trial, node] += 1

        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            sched,
            forced_action=jnp.asarray(action, dtype=jnp.int32),
            forced_observation=jnp.asarray(forced_observation, dtype=jnp.float32),
            training=True,
            use_posterior_mean=False,
            compute_targets=False,
            method=jp.PlanningVAE.__call__,
        )
        trans_np = jax.device_get(trans)
        is_observe = np.asarray(trans_np.is_observe, dtype=float) > 0.5
        is_stop = np.asarray(trans_np.is_stop, dtype=float) > 0.5
        z_mu = np.asarray(trans_np.z_mu, dtype=float)
        z_sigma = np.exp(0.5 * np.clip(np.asarray(trans_np.z_logvar, dtype=float), -30.0, 30.0))
        prior_mu = np.asarray(trans_np.prior_mu, dtype=float)
        prior_sigma = np.exp(0.5 * np.clip(np.asarray(trans_np.prior_logvar, dtype=float), -30.0, 30.0))
        action_logit = terminal_action_logit(np.asarray(trans_np.action_output, dtype=float), action_logit_mode)
        node_index = np.asarray(trans_np.node_index, dtype=np.int32)

        include = (~stopped) & is_observe
        for trial in np.where(include)[0]:
            observation_count[trial] += 1
            node = int(node_index[trial])
            rows.append(
                {
                    "transition_type": "observe",
                    "is_stop_decision": False,
                    "trial_index": int(trial),
                    "condition_index": int(condition_index[trial]),
                    "original_condition_index": int(original_condition_index[trial]),
                    "sample_set": int(sample_set[trial]),
                    "timestep": int(timestep),
                    "observation_index": int(observation_count[trial]),
                    "action_plot_timestep": int(timestep),
                    "observed_node": int(node + 1),
                    "sample_position": int(sample_position[trial]) if sample_position[trial] >= 0 else np.nan,
                    "sampled_observed_reward": float(forced_observation[trial]),
                    "actual_observed_reward": float(rewards[trial, node]),
                    "node1_reward": float(reward_node_1[trial]),
                    "node2_reward": float(reward_node_2[trial]),
                    "z_mu_mean": float(np.nanmean(z_mu[trial])),
                    "z_sigma_mean": float(np.nanmean(z_sigma[trial])),
                    "prior_mu_mean": float(np.nanmean(prior_mu[trial])),
                    "prior_sigma_mean": float(np.nanmean(prior_sigma[trial])),
                    "action_logit": float(action_logit[trial]),
                }
            )
        stop_include = (~stopped) & is_stop
        for trial in np.where(stop_include)[0]:
            rows.append(
                {
                    "transition_type": "stop",
                    "is_stop_decision": True,
                    "trial_index": int(trial),
                    "condition_index": int(condition_index[trial]),
                    "original_condition_index": int(original_condition_index[trial]),
                    "sample_set": int(sample_set[trial]),
                    "timestep": int(timestep),
                    "observation_index": int(observation_count[trial]),
                    "action_plot_timestep": int(timestep),
                    "observed_node": np.nan,
                    "sample_position": np.nan,
                    "sampled_observed_reward": np.nan,
                    "actual_observed_reward": np.nan,
                    "node1_reward": float(reward_node_1[trial]),
                    "node2_reward": float(reward_node_2[trial]),
                    "z_mu_mean": float(np.nanmean(z_mu[trial])),
                    "z_sigma_mean": float(np.nanmean(z_sigma[trial])),
                    "prior_mu_mean": float(np.nanmean(prior_mu[trial])),
                    "prior_sigma_mean": float(np.nanmean(prior_sigma[trial])),
                    "action_logit": float(action_logit[trial]),
                }
            )
        stopped |= is_stop
        print(
            f"{progress_label}: timestep {timestep}/{int(config.num_steps)}; "
            f"observations={int(np.sum(observation_count))}; stopped={int(np.sum(stopped))}/{n_trials}",
            flush=True,
        )
    out = pd.DataFrame(rows)
    print(f"{progress_label}: trajectory rows={len(out)}", flush=True)
    return out


def evidence_terminal_action_logit(probs: np.ndarray, correct_action: np.ndarray, mode: str) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < ev.NUM_ACTIONS:
        return np.full((probs.shape[0],), np.nan, dtype=float)
    p_a = probs[:, ev.CHOOSE_A]
    p_b = probs[:, ev.CHOOSE_B]
    if mode == "abs_path1_minus_path2":
        return np.abs(np.log(np.clip(p_a, EPS, 1.0)) - np.log(np.clip(p_b, EPS, 1.0)))
    if mode == "max_minus_second":
        terminal_probs = np.stack([p_a, p_b], axis=1)
        log_probs = np.log(np.clip(terminal_probs, EPS, 1.0))
        sorted_log_probs = np.sort(log_probs, axis=1)
        return sorted_log_probs[:, -1] - sorted_log_probs[:, -2]
    if mode in {"path1_minus_path2", "choose_a_minus_choose_b"}:
        return np.log(np.clip(p_a, EPS, 1.0)) - np.log(np.clip(p_b, EPS, 1.0))
    if mode == "choose_b_minus_choose_a":
        return np.log(np.clip(p_b, EPS, 1.0)) - np.log(np.clip(p_a, EPS, 1.0))
    correct_action = np.asarray(correct_action, dtype=int)
    p_correct = np.where(correct_action == ev.CHOOSE_B, p_b, p_a)
    p_incorrect = np.where(correct_action == ev.CHOOSE_B, p_a, p_b)
    return np.log(np.clip(p_correct, EPS, 1.0)) - np.log(np.clip(p_incorrect, EPS, 1.0))


def make_evidence_config(
    args: argparse.Namespace,
    *,
    seed: int,
    beta: float,
    opportunity: float,
    observation_noise_std: float,
) -> ev.RunConfig:
    max_observations = max(int(args.max_observations_before_stop), 1)
    return ev.RunConfig(
        loss_scale=float(args.loss_scale_value),
        alpha=float(args.alpha),
        beta=float(beta),
        memory_lambda=float(args.memory_lambda) if args.memory_lambda is not None else float(beta),
        model_dir=str(args.checkpoint_root),
        epochs=int(args.num_updates),
        input_type=str(args.input_type),
        seed=int(seed),
        tree_size=int(args.tree_size),
        train_mode="sim",
        tree_type=str(args.tree_type),
        opportunity_cost=float(opportunity),
        expansion_decision_version=ev.normalize_expansion_decision_version(str(args.expansion_decision_version)),
        model_variant=ev.normalize_model_variant(str(args.model_variant)),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir=str(args.sim_dir),
        n_sim_trials=int(args.n_sample_sets),
        num_envs=int(args.num_envs),
        num_steps=max_observations,
        update_epochs=int(args.update_epochs),
        ppo_minibatches=int(args.ppo_minibatches),
        steps_per_epoch=int(args.steps_per_epoch),
        return_target_rollouts=int(args.return_target_rollouts),
        return_target_mode=ev.normalize_return_target_mode(str(args.return_target_mode)),
        sampled_lambda_critic=str(args.sampled_lambda_critic),
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=int(args.target_critic_update_interval),
        target_critic_tau=float(args.target_critic_tau),
        backend=args.backend,
        jit_training=not bool(args.no_jit),
        enable_reconstruction=bool(args.enable_reconstruction),
        enable_probe=bool(args.enable_probe),
        max_observations_before_stop=max_observations,
        coherence_values=tuple(float(v) for v in args.coherence_values),
        observation_noise_std=max(float(observation_noise_std), 1e-6),
        correct_reward=float(args.correct_reward),
        incorrect_reward=float(args.incorrect_reward),
        kl_start_multiplier=max(float(args.kl_start_multiplier), 0.0),
        kl_annealing_epochs=max(int(args.kl_annealing_epochs), 0),
        critic_huber_delta=max(float(args.critic_huber_delta), 0.0),
        advantage_clip=max(float(args.advantage_clip), 0.0),
        pay_kl_on_stop=bool(args.pay_kl_on_stop),
        choice_at_end_only=False,
    )


def build_evidence_sample_set_batch(
    *,
    coherence_values: list[float],
    observation_noise_std: float,
    n_sample_sets: int,
    max_observations: int,
    seed: int,
) -> tuple[ev.EvidenceBatch, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict] = []
    evidence_streams = []
    correct_choice_values = []
    correct_action_values = []
    coherence_out = []
    signed_coherence_out = []
    condition_index = 0
    for correct_choice in (-1.0, 1.0):
        for coherence in coherence_values:
            signed_coherence = float(correct_choice) * float(coherence)
            correct_action = ev.CHOOSE_A if correct_choice < 0 else ev.CHOOSE_B
            for sample_set in range(int(n_sample_sets)):
                if float(observation_noise_std) <= 0:
                    samples = np.full((int(max_observations),), signed_coherence, dtype=np.float32)
                else:
                    samples = rng.normal(
                        loc=signed_coherence,
                        scale=float(observation_noise_std),
                        size=int(max_observations),
                    ).astype(np.float32)
                evidence_streams.append(samples)
                correct_choice_values.append(float(correct_choice))
                correct_action_values.append(int(correct_action))
                coherence_out.append(float(coherence))
                signed_coherence_out.append(float(signed_coherence))
                rows.append(
                    {
                        "condition_index": int(condition_index),
                        "original_condition_index": int(condition_index),
                        "sample_set": int(sample_set),
                        "correct_choice": float(correct_choice),
                        "correct_action": int(correct_action),
                        "coherence": float(coherence),
                        "signed_coherence": float(signed_coherence),
                    }
                )
            condition_index += 1
    evidence_arr = np.asarray(evidence_streams, dtype=np.float32).T
    cumulative = np.cumsum(evidence_arr, axis=0).astype(np.float32)
    coherence_arr = np.asarray(coherence_out, dtype=np.float32)
    obs_var = max(float(observation_noise_std) ** 2, 1e-12)
    oracle_llr = ((2.0 * coherence_arr[None, :]) / obs_var * cumulative).astype(np.float32)
    batch = ev.EvidenceBatch(
        correct_choice=jnp.asarray(correct_choice_values, dtype=jnp.float32),
        correct_action=jnp.asarray(correct_action_values, dtype=jnp.int32),
        coherence=jnp.asarray(coherence_out, dtype=jnp.float32),
        signed_coherence=jnp.asarray(signed_coherence_out, dtype=jnp.float32),
        evidence_samples=jnp.asarray(evidence_arr, dtype=jnp.float32),
        cumulative_evidence=jnp.asarray(cumulative, dtype=jnp.float32),
        oracle_cumulative_llr=jnp.asarray(oracle_llr, dtype=jnp.float32),
    )
    return batch, pd.DataFrame(rows)


def rollout_evidence_trajectory_rows(
    *,
    model: ev.EvidenceVAE,
    params,
    config: ev.RunConfig,
    batch: ev.EvidenceBatch,
    metadata: pd.DataFrame,
    seed_offset: int,
    action_logit_mode: str,
    progress_label: str,
) -> pd.DataFrame:
    batch_np = jax.device_get(batch)
    n_trials = int(batch_np.correct_choice.shape[0])
    carry = ev.initial_carry(batch, int(config.rnn_units))
    sched = ev.ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(1.0 / max(float(config.beta), EPS), dtype=jnp.float32),
        current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )
    rng = jax.random.PRNGKey(int(config.seed) + 1_730_000 + int(seed_offset))
    rows: list[dict] = []
    stopped = np.zeros(n_trials, dtype=bool)
    condition_index = metadata["condition_index"].to_numpy(dtype=int) if "condition_index" in metadata else np.arange(n_trials)
    original_condition_index = (
        metadata["original_condition_index"].to_numpy(dtype=int)
        if "original_condition_index" in metadata
        else condition_index
    )
    sample_set = metadata["sample_set"].to_numpy(dtype=int) if "sample_set" in metadata else np.arange(n_trials)
    print(
        f"{progress_label}: evidence rollout starts with {n_trials} trial(s), "
        f"num_steps={int(config.num_steps)}",
        flush=True,
    )
    for timestep in range(1, int(config.num_steps) + 1):
        rng, step_rng = jax.random.split(rng)
        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            sched,
            forced_action=None,
            training=True,
            use_posterior_mean=False,
            method=ev.EvidenceVAE.__call__,
        )
        trans_np = jax.device_get(trans)
        valid = np.asarray(trans_np.valid, dtype=float) > 0.5
        is_terminal = np.asarray(trans_np.is_terminal, dtype=float) > 0.5
        z_mu = np.asarray(trans_np.z_mu, dtype=float)
        z_sigma = np.exp(0.5 * np.clip(np.asarray(trans_np.z_logvar, dtype=float), -30.0, 30.0))
        prior_mu = np.asarray(trans_np.prior_mu, dtype=float)
        prior_sigma = np.exp(0.5 * np.clip(np.asarray(trans_np.prior_logvar, dtype=float), -30.0, 30.0))
        probs = np.asarray(trans_np.probs, dtype=float)
        correct_action = np.asarray(trans_np.correct_action, dtype=int)
        action_logit = evidence_terminal_action_logit(probs, correct_action, action_logit_mode)
        include = (~stopped) & valid
        for trial in np.where(include)[0]:
            rows.append(
                {
                    "transition_type": "observe",
                    "is_stop_decision": bool(is_terminal[trial]),
                    "trial_index": int(trial),
                    "condition_index": int(condition_index[trial]),
                    "original_condition_index": int(original_condition_index[trial]),
                    "sample_set": int(sample_set[trial]),
                    "timestep": int(timestep),
                    "observation_index": int(trans_np.num_observations_before[trial]),
                    "action_plot_timestep": int(timestep),
                    "observed_node": np.nan,
                    "sample_position": int(timestep - 1),
                    "sampled_observed_reward": float(trans_np.current_evidence[trial]),
                    "actual_observed_reward": float(trans_np.signed_coherence[trial]),
                    "node1_reward": float(trans_np.correct_choice[trial]),
                    "node2_reward": float(trans_np.coherence[trial]),
                    "correct_choice": float(trans_np.correct_choice[trial]),
                    "correct_action": int(trans_np.correct_action[trial]),
                    "coherence": float(trans_np.coherence[trial]),
                    "signed_coherence": float(trans_np.signed_coherence[trial]),
                    "cumulative_evidence": float(trans_np.cumulative_evidence[trial]),
                    "oracle_cumulative_llr": float(trans_np.oracle_cumulative_llr[trial]),
                    "action": int(trans_np.action[trial]),
                    "policy_continue": float(probs[trial, ev.CONTINUE]),
                    "policy_choose_a": float(probs[trial, ev.CHOOSE_A]),
                    "policy_choose_b": float(probs[trial, ev.CHOOSE_B]),
                    "kl_paid": float(trans_np.paid_kl[trial]),
                    "observed_kl": float(trans_np.observed_kl[trial]),
                    "z_mu_mean": float(np.nanmean(z_mu[trial])),
                    "z_sigma_mean": float(np.nanmean(z_sigma[trial])),
                    "prior_mu_mean": float(np.nanmean(prior_mu[trial])),
                    "prior_sigma_mean": float(np.nanmean(prior_sigma[trial])),
                    "action_logit": float(action_logit[trial]),
                }
            )
        stopped |= is_terminal
        print(
            f"{progress_label}: timestep {timestep}/{int(config.num_steps)}; "
            f"active_rows={int(np.sum(include))}; stopped={int(np.sum(stopped))}/{n_trials}",
            flush=True,
        )
    out = pd.DataFrame(rows)
    print(f"{progress_label}: evidence trajectory rows={len(out)}", flush=True)
    return out


def run_one_evidence(
    args: argparse.Namespace,
    *,
    family: str,
    parameter_name: str,
    parameter_value: float,
    beta: float,
    opportunity: float,
    sigma: float,
    seed: int,
) -> pd.DataFrame:
    config = make_evidence_config(
        args,
        seed=seed,
        beta=beta,
        opportunity=opportunity,
        observation_noise_std=sigma,
    )
    task = ev.make_task(config)
    model_name = ev.model_name_for(config)
    weights_path = Path(config.model_dir) / f"{model_name}.msgpack"
    label = (
        f"evidence {family} {parameter_name}={parameter_value:g} beta={beta:g} "
        f"opp={opportunity:g} obsstd={sigma:g} seed={seed}"
    )
    print(f"{label}: loading {weights_path}", flush=True)
    model, params = ev.load_state_for_sim(config, task)
    batch, metadata = build_evidence_sample_set_batch(
        coherence_values=list(args.coherence_values),
        observation_noise_std=float(sigma),
        n_sample_sets=int(args.n_sample_sets),
        max_observations=int(args.max_observations_before_stop),
        seed=int(seed + round(10_000 * sigma)),
    )
    print(
        f"{label}: generated {len(metadata)} evidence trial(s) "
        f"({metadata['condition_index'].nunique()} condition(s) x {int(args.n_sample_sets)} sample set(s))",
        flush=True,
    )
    rows = rollout_evidence_trajectory_rows(
        model=model,
        params=params,
        config=config,
        batch=batch,
        metadata=metadata,
        seed_offset=int(round(10_000 * sigma) + round(beta) + round(10_000 * opportunity)),
        action_logit_mode=str(args.action_logit_mode),
        progress_label=label,
    )
    for col, value in [
        ("family", family),
        ("parameter_name", parameter_name),
        ("parameter_value", parameter_value),
        ("beta", beta),
        ("opportunity", opportunity),
        ("sigma", sigma),
        ("seed", seed),
        ("checkpoint", weights_path.name),
        ("task_mode", "evidence"),
    ]:
        rows[col] = value
    return rows


def run_one(
    args: argparse.Namespace,
    *,
    family: str,
    parameter_name: str,
    parameter_value: float,
    beta: float,
    opportunity: float,
    sigma: float,
    seed: int,
) -> pd.DataFrame:
    task = jp.build_task(int(args.tree_size), str(args.tree_type), str(args.input_type))
    if int(task.num_nodes) != 2:
        raise ValueError("This trajectory reward-grid diagnostic currently expects the two-node default task.")
    config = sample_base.make_config(args, seed=seed, beta=beta, opportunity=opportunity, sigma=sigma)
    model_name = jp.model_name_for(config, task)
    weights_path = Path(config.model_dir) / f"{model_name}.msgpack"
    label = (
        f"{family} {parameter_name}={parameter_value:g} beta={beta:g} "
        f"opp={opportunity:g} sigma={sigma:g} seed={seed}"
    )
    print(f"{label}: loading {weights_path}", flush=True)
    model, params = jp.load_state_for_sim(config, task)
    rewards, streams, metadata = sample_base.build_reward_combination_trials(
        np.asarray(task.reward_values, dtype=float),
        num_nodes=int(task.num_nodes),
        sigma=float(sigma),
        n_sample_sets=int(args.n_sample_sets),
        max_observations=int(args.max_observations_before_stop),
        seed=int(seed + round(1000 * sigma)),
        n_reward_combinations=int(args.n_reward_combinations),
        reward_combination_seed=int(seed),
    )
    print(
        f"{label}: generated {len(rewards)} trial(s) "
        f"({metadata['condition_index'].nunique()} reward condition(s) x {int(args.n_sample_sets)} sample set(s))",
        flush=True,
    )
    rows = rollout_latent_trajectory_rows(
        model=model,
        params=params,
        config=config,
        task=task,
        rewards=rewards,
        streams=streams,
        metadata=metadata,
        seed_offset=int(round(10_000 * sigma) + round(beta) + round(10_000 * opportunity)),
        force_first_observe_node=int(args.force_first_observe_node),
        action_logit_mode=str(args.action_logit_mode),
        progress_label=label,
    )
    for col, value in [
        ("family", family),
        ("parameter_name", parameter_name),
        ("parameter_value", parameter_value),
        ("beta", beta),
        ("opportunity", opportunity),
        ("sigma", sigma),
        ("seed", seed),
        ("checkpoint", weights_path.name),
    ]:
        rows[col] = value
    return rows


def summarize_trajectories(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    if "task_mode" not in rows.columns:
        rows["task_mode"] = "revisit"
    if "transition_type" in rows.columns:
        rows = rows[rows["transition_type"].astype(str) == "observe"].copy()
        if rows.empty:
            return pd.DataFrame()
    if "prior_mu_mean" not in rows.columns:
        print(
            "Warning: latent_trajectory_rows.csv has no prior_mu_mean column; "
            "using prior mean 0 for backward-compatible plotting. Rerun simulations "
            "to get true learned-prior normalization.",
            flush=True,
        )
        rows["prior_mu_mean"] = 0.0
    if "prior_sigma_mean" not in rows.columns:
        print(
            "Warning: latent_trajectory_rows.csv has no prior_sigma_mean column; "
            "using prior sigma 1 for backward-compatible plotting. Rerun simulations "
            "to get true learned-prior normalization.",
            flush=True,
        )
        rows["prior_sigma_mean"] = 1.0
    rows["z_sigma_sq"] = pd.to_numeric(rows["z_sigma_mean"], errors="coerce") ** 2
    rows["prior_sigma_sq"] = pd.to_numeric(rows["prior_sigma_mean"], errors="coerce") ** 2
    row_prior_sigma = pd.to_numeric(rows["prior_sigma_mean"], errors="coerce").clip(lower=1e-8)
    rows["prior_normalized_z_mu_value"] = (
        pd.to_numeric(rows["z_mu_mean"], errors="coerce")
        - pd.to_numeric(rows["prior_mu_mean"], errors="coerce")
    ) / row_prior_sigma
    summary = (
        rows.groupby(
            [
                "family",
                "task_mode",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "node1_reward",
                "node2_reward",
                "observation_index",
            ],
            dropna=False,
        )
        .agg(
            z_mu_mean=("z_mu_mean", "mean"),
            z_mu_sd=("z_mu_mean", "std"),
            prior_mu_mean=("prior_mu_mean", "mean"),
            z_sigma_mean=("z_sigma_mean", "mean"),
            z_sigma_sum_sq=("z_sigma_sq", "sum"),
            prior_sigma_mean=("prior_sigma_mean", "mean"),
            prior_sigma_sum_sq=("prior_sigma_sq", "sum"),
            prior_normalized_z_mu_row_mean=("prior_normalized_z_mu_value", "mean"),
            prior_normalized_z_mu_sd=("prior_normalized_z_mu_value", "std"),
            action_logit=("action_logit", "mean"),
            action_logit_sd=("action_logit", "std"),
            n=("z_mu_mean", "size"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    n = pd.to_numeric(summary["n"], errors="coerce").clip(lower=1)
    summary["z_mu_sem"] = (pd.to_numeric(summary["z_mu_sd"], errors="coerce") / np.sqrt(n)).fillna(0.0)
    summary["prior_normalized_z_mu_sem"] = (
        pd.to_numeric(summary["prior_normalized_z_mu_sd"], errors="coerce") / np.sqrt(n)
    ).fillna(0.0)
    summary["action_logit_sem"] = (
        pd.to_numeric(summary["action_logit_sd"], errors="coerce") / np.sqrt(n)
    ).fillna(0.0)
    summary["z_sigma_of_mean"] = np.sqrt(
        pd.to_numeric(summary["z_sigma_sum_sq"], errors="coerce")
    ) / pd.to_numeric(summary["n"], errors="coerce").clip(lower=1)
    summary["prior_sigma_of_mean"] = np.sqrt(
        pd.to_numeric(summary["prior_sigma_sum_sq"], errors="coerce")
    ) / pd.to_numeric(summary["n"], errors="coerce").clip(lower=1)
    summary["z_variance_of_mean"] = pd.to_numeric(summary["z_sigma_of_mean"], errors="coerce") ** 2
    summary["prior_variance_of_mean"] = pd.to_numeric(summary["prior_sigma_of_mean"], errors="coerce") ** 2
    prior_sigma = pd.to_numeric(summary["prior_sigma_of_mean"], errors="coerce").clip(lower=1e-8)
    prior_variance = pd.to_numeric(summary["prior_variance_of_mean"], errors="coerce").clip(lower=1e-12)
    summary["prior_normalized_z_mu_of_mean"] = (
        pd.to_numeric(summary["z_mu_mean"], errors="coerce")
        - pd.to_numeric(summary["prior_mu_mean"], errors="coerce")
    ) / prior_sigma
    # Use the mean of per-row prior-normalized posterior means as the plotted
    # center. This keeps all shaded variants on the same dot/line positions.
    summary["prior_normalized_z_mu_mean"] = summary["prior_normalized_z_mu_row_mean"]
    summary["prior_normalized_z_variance_of_mean"] = (
        pd.to_numeric(summary["z_variance_of_mean"], errors="coerce") / prior_variance
    )
    summary["prior_normalized_z_sigma_of_mean"] = (
        np.sqrt(pd.to_numeric(summary["prior_normalized_z_variance_of_mean"], errors="coerce").clip(lower=0.0))
    )
    return summary


def summarize_action_logits(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    if "task_mode" not in rows.columns:
        rows["task_mode"] = "revisit"
    if "transition_type" in rows.columns:
        rows = rows[rows["transition_type"].astype(str).isin(["observe", "stop"])].copy()
    if rows.empty or "action_logit" not in rows.columns:
        return pd.DataFrame()
    if "is_stop_decision" not in rows.columns:
        rows["is_stop_decision"] = False
    if "action_plot_timestep" not in rows.columns:
        rows["action_plot_timestep"] = rows.get("timestep", rows.get("observation_index", np.nan))
    summary = (
        rows.groupby(
            [
                "family",
                "task_mode",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "node1_reward",
                "node2_reward",
                "action_plot_timestep",
            ],
            dropna=False,
        )
        .agg(
            action_logit=("action_logit", "mean"),
            action_logit_sd=("action_logit", "std"),
            n=("action_logit", "size"),
            n_seeds=("seed", "nunique"),
            n_stop_decisions=(
                "is_stop_decision",
                lambda x: int(np.nansum(pd.to_numeric(x, errors="coerce").fillna(0).to_numpy(dtype=float))),
            ),
        )
        .reset_index()
        .rename(columns={"action_plot_timestep": "observation_index"})
    )
    n = pd.to_numeric(summary["n"], errors="coerce").clip(lower=1)
    summary["action_logit_sem"] = (
        pd.to_numeric(summary["action_logit_sd"], errors="coerce") / np.sqrt(n)
    ).fillna(0.0)
    return summary


def latent_mu_column(summary: pd.DataFrame) -> str:
    return "prior_normalized_z_mu_mean" if "prior_normalized_z_mu_mean" in summary.columns else "z_mu_mean"


def latent_sigma_column(summary: pd.DataFrame) -> str:
    if "prior_normalized_z_sigma_of_mean" in summary.columns:
        return "prior_normalized_z_sigma_of_mean"
    return "z_sigma_of_mean" if "z_sigma_of_mean" in summary.columns else "z_sigma_mean"


def latent_mu_axis_label(summary: pd.DataFrame) -> str:
    return "prior-norm\nmean z_mu" if latent_mu_column(summary).startswith("prior_normalized") else "mean z_mu"


def latent_mu_sem_column(summary: pd.DataFrame) -> str:
    if latent_mu_column(summary).startswith("prior_normalized") and "prior_normalized_z_mu_sem" in summary.columns:
        return "prior_normalized_z_mu_sem"
    return "z_mu_sem" if "z_mu_sem" in summary.columns else ""


def latent_sigma_axis_label(summary: pd.DataFrame) -> str:
    return (
        "prior-norm\nsigma(mean z)"
        if latent_sigma_column(summary).startswith("prior_normalized")
        else "sigma(mean z)"
    )


def reward_values_for_plot(summary: pd.DataFrame, col: str) -> list[float]:
    return sorted(pd.to_numeric(summary[col], errors="coerce").dropna().unique())


def grid_axis_names(data: pd.DataFrame) -> tuple[str, str]:
    if "task_mode" in data.columns and data["task_mode"].astype(str).eq("evidence").any():
        return "choice", "coh"
    return "R1", "R2"


def filter_summary_for_min_samples(summary: pd.DataFrame, min_samples_per_dot: int) -> pd.DataFrame:
    if int(min_samples_per_dot) <= 1 or "n" not in summary.columns:
        return summary
    keep = pd.to_numeric(summary["n"], errors="coerce").fillna(0) >= int(min_samples_per_dot)
    dropped = int((~keep).sum())
    if dropped:
        print(
            f"Filtering {dropped} trajectory summary point(s) with n < {int(min_samples_per_dot)}.",
            flush=True,
        )
    return summary[keep].copy()


def plot_latent_3d_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameter_value: float,
    sigma: float,
    outpath: Path,
) -> None:
    data = summary[
        (summary["family"].astype(str) == family)
        & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), float(parameter_value))
        & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
    ].copy()
    if data.empty:
        return
    configure_plotting()
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
    row_label, col_label = grid_axis_names(data)
    fig = plt.figure(figsize=(len(node2_values) * 1.28 + 1.0, len(node1_values) * 1.15 + 0.9))
    mu_col = latent_mu_column(data)
    z_mu = pd.to_numeric(data[mu_col], errors="coerce")
    sigma_col = latent_sigma_column(data)
    z_sigma = pd.to_numeric(data[sigma_col], errors="coerce")
    mu_lim = (
        float(np.nanmin(z_mu)) if np.isfinite(z_mu).any() else -1.0,
        float(np.nanmax(z_mu)) if np.isfinite(z_mu).any() else 1.0,
    )
    sigma_lim = (
        max(0.0, float(np.nanmin(z_sigma)) if np.isfinite(z_sigma).any() else 0.0),
        float(np.nanmax(z_sigma)) if np.isfinite(z_sigma).any() else 1.0,
    )
    if abs(mu_lim[1] - mu_lim[0]) < 1e-9:
        mu_lim = (mu_lim[0] - 0.5, mu_lim[1] + 0.5)
    if abs(sigma_lim[1] - sigma_lim[0]) < 1e-9:
        sigma_lim = (max(0.0, sigma_lim[0] - 0.05), sigma_lim[1] + 0.05)
    for row_i, node1 in enumerate(node1_values):
        for col_i, node2 in enumerate(node2_values):
            ax = fig.add_subplot(len(node1_values), len(node2_values), row_i * len(node2_values) + col_i + 1, projection="3d")
            panel = data[
                np.isclose(pd.to_numeric(data["node1_reward"], errors="coerce"), float(node1))
                & np.isclose(pd.to_numeric(data["node2_reward"], errors="coerce"), float(node2))
            ].sort_values("observation_index")
            if panel.empty:
                ax.set_axis_off()
                continue
            ax.plot(
                panel["observation_index"],
                panel[mu_col],
                panel[sigma_col],
                color="#2b2b2b",
                marker="o",
                markersize=2.0,
                linewidth=0.8,
            )
            ax.set_xlim(0.8, max(1.0, float(data["observation_index"].max())) + 0.2)
            ax.set_ylim(*mu_lim)
            ax.set_zlim(*sigma_lim)
            x_max = int(pd.to_numeric(data["observation_index"], errors="coerce").max())
            if x_max <= 6:
                ax.set_xticks(np.arange(1, x_max + 1, 1))
            else:
                ax.set_xticks(np.unique(np.linspace(1, x_max, 4).round().astype(int)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.view_init(elev=22, azim=-55)
            if row_i == 0:
                ax.set_title(f"{col_label}={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={node1:g}\n{latent_mu_axis_label(data)}", labelpad=0)
            else:
                ax.set_ylabel(latent_mu_axis_label(data), labelpad=0)
            ax.set_xlabel("t", labelpad=0)
            ax.set_zlabel(latent_sigma_axis_label(data), labelpad=0)
            ax.tick_params(length=1.5, pad=0, labelsize=PLOT_FONT_SIZE)
    fig.suptitle(f"{family_param_title(family, parameter_value)}, sigma={sigma:g}: latent trajectory", y=0.995)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_latent_contour_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameter_value: float,
    sigma: float,
    outpath: Path,
) -> None:
    data = summary[
        (summary["family"].astype(str) == family)
        & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), float(parameter_value))
        & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
    ].copy()
    if data.empty:
        return
    configure_plotting()
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
    row_label, col_label = grid_axis_names(data)
    fig, axes = plt.subplots(
        len(node1_values),
        len(node2_values),
        figsize=(len(node2_values) * 1.05 + 1.25, len(node1_values) * 0.9 + 0.75),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    mu_col = latent_mu_column(data)
    z_mu = pd.to_numeric(data[mu_col], errors="coerce")
    sigma_col = latent_sigma_column(data)
    z_sigma = pd.to_numeric(data[sigma_col], errors="coerce")
    finite_mu = z_mu[np.isfinite(z_mu)]
    finite_sigma = z_sigma[np.isfinite(z_sigma)]
    if len(finite_mu):
        y_min, y_max = float(finite_mu.min()), float(finite_mu.max())
    else:
        y_min, y_max = -1.0, 1.0
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    if len(finite_sigma):
        norm = Normalize(vmin=float(finite_sigma.min()), vmax=float(finite_sigma.max()))
        if abs(norm.vmax - norm.vmin) < 1e-12:
            norm = Normalize(vmin=max(0.0, norm.vmin - 0.05), vmax=norm.vmax + 0.05)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")
    x_max = int(pd.to_numeric(data["observation_index"], errors="coerce").max())
    for row_i, node1 in enumerate(node1_values):
        for col_i, node2 in enumerate(node2_values):
            ax = axes[row_i, col_i]
            panel = data[
                np.isclose(pd.to_numeric(data["node1_reward"], errors="coerce"), float(node1))
                & np.isclose(pd.to_numeric(data["node2_reward"], errors="coerce"), float(node2))
            ].sort_values("observation_index")
            if panel.empty:
                ax.axis("off")
                continue
            x = pd.to_numeric(panel["observation_index"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(panel[mu_col], errors="coerce").to_numpy(dtype=float)
            color_value = pd.to_numeric(panel[sigma_col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(color_value)
            x, y, color_value = x[ok], y[ok], color_value[ok]
            if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 1e-8:
                try:
                    ax.tricontourf(x, y, color_value, levels=8, cmap=cmap, norm=norm, alpha=0.55)
                    ax.tricontour(x, y, color_value, levels=8, colors="#555555", linewidths=0.25, alpha=0.45)
                except Exception:
                    pass
            if len(x):
                order = np.argsort(x)
                ax.plot(x[order], y[order], color="#2b2b2b", linewidth=0.55, alpha=0.8, zorder=2)
                ax.scatter(
                    x,
                    y,
                    c=color_value,
                    cmap=cmap,
                    norm=norm,
                    s=8,
                    edgecolors="#2b2b2b",
                    linewidths=0.2,
                    zorder=3,
                )
            if row_i == 0:
                ax.set_title(f"{col_label}={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={node1:g}\n{latent_mu_axis_label(data)}")
            ax.set_xlim(0.8, x_max + 0.2)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(1, x_max + 1, 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(
                length=2,
                pad=1,
                labelbottom=(row_i == len(node1_values) - 1),
                labelleft=(col_i == 0),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015, label="sigma(mean z)")
    fig.suptitle(f"{family_param_title(family, parameter_value)}, sigma={sigma:g}: latent contour trajectory", y=0.995)
    fig.supxlabel("current observation timestep", y=0.02)
    fig.supylabel(latent_mu_axis_label(data), x=0.005)
    fig.tight_layout(rect=(0.025, 0.045, 0.93, 0.965), h_pad=0.25, w_pad=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latent_shaded_error_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameter_value: float,
    sigma: float,
    outpath: Path,
    shade_error_scale: float = 25.0,
) -> None:
    data = summary[
        (summary["family"].astype(str) == family)
        & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), float(parameter_value))
        & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
    ].copy()
    if data.empty:
        return
    configure_plotting()
    mu_col = latent_mu_column(data)
    sigma_col = latent_sigma_column(data)
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
    row_label, col_label = grid_axis_names(data)
    fig, axes = plt.subplots(
        len(node1_values),
        len(node2_values),
        figsize=(len(node2_values) * 1.05 + 0.8, len(node1_values) * 0.9 + 0.75),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    center = pd.to_numeric(data[mu_col], errors="coerce")
    spread = pd.to_numeric(data[sigma_col], errors="coerce")
    display_spread = spread * float(shade_error_scale)
    finite = np.isfinite(center) & np.isfinite(spread)
    if np.any(finite):
        y_min = float(np.nanmin(center[finite] - display_spread[finite]))
        y_max = float(np.nanmax(center[finite] + display_spread[finite]))
    else:
        y_min, y_max = -1.0, 1.0
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    x_max = int(pd.to_numeric(data["observation_index"], errors="coerce").max())
    for row_i, node1 in enumerate(node1_values):
        for col_i, node2 in enumerate(node2_values):
            ax = axes[row_i, col_i]
            panel = data[
                np.isclose(pd.to_numeric(data["node1_reward"], errors="coerce"), float(node1))
                & np.isclose(pd.to_numeric(data["node2_reward"], errors="coerce"), float(node2))
            ].sort_values("observation_index")
            if panel.empty:
                ax.axis("off")
                continue
            x = pd.to_numeric(panel["observation_index"], errors="coerce").to_numpy(dtype=float)
            mu = pd.to_numeric(panel[mu_col], errors="coerce").to_numpy(dtype=float)
            sig = pd.to_numeric(panel[sigma_col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(mu) & np.isfinite(sig)
            x, mu, sig = x[ok], mu[ok], sig[ok]
            if len(x):
                order = np.argsort(x)
                x, mu, sig = x[order], mu[order], sig[order]
                display_sig = sig * float(shade_error_scale)
                ax.fill_between(
                    x,
                    mu - display_sig,
                    mu + display_sig,
                    color="#6baed6",
                    alpha=0.38,
                    linewidth=0,
                )
                ax.plot(x, mu, color="#08306b", marker="o", markersize=2.0, linewidth=0.85)
            if row_i == 0:
                ax.set_title(f"{col_label}={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={node1:g}")
            ax.set_xlim(0.8, x_max + 0.2)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(1, x_max + 1, 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(
                length=2,
                pad=1,
                labelbottom=(row_i == len(node1_values) - 1),
                labelleft=(col_i == 0),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{family_param_title(family, parameter_value)}, sigma={sigma:g}: mean z trajectory", y=0.995)
    fig.supxlabel("current observation timestep", y=0.02)
    shade_label = (
        "shade = +/- sigma(mean z)"
        if abs(float(shade_error_scale) - 1.0) < 1e-12
        else f"shade = +/- {float(shade_error_scale):g} x sigma(mean z)"
    )
    fig.supylabel(f"{latent_mu_axis_label(data)}\n{shade_label}", x=0.005)
    fig.tight_layout(rect=(0.025, 0.045, 1.0, 0.965), h_pad=0.25, w_pad=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latent_sem_shaded_error_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameter_value: float,
    sigma: float,
    outpath: Path,
) -> None:
    data = summary[
        (summary["family"].astype(str) == family)
        & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), float(parameter_value))
        & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
    ].copy()
    if data.empty:
        return
    configure_plotting()
    mu_col = latent_mu_column(data)
    sem_col = latent_mu_sem_column(data)
    if not sem_col or sem_col not in data.columns:
        return
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
    row_label, col_label = grid_axis_names(data)
    fig, axes = plt.subplots(
        len(node1_values),
        len(node2_values),
        figsize=(len(node2_values) * 1.05 + 0.8, len(node1_values) * 0.9 + 0.75),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    center = pd.to_numeric(data[mu_col], errors="coerce")
    sem = pd.to_numeric(data[sem_col], errors="coerce").fillna(0.0)
    finite = np.isfinite(center) & np.isfinite(sem)
    if np.any(finite):
        y_min = float(np.nanmin(center[finite] - sem[finite]))
        y_max = float(np.nanmax(center[finite] + sem[finite]))
    else:
        y_min, y_max = -1.0, 1.0
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    x_max = int(pd.to_numeric(data["observation_index"], errors="coerce").max())
    for row_i, node1 in enumerate(node1_values):
        for col_i, node2 in enumerate(node2_values):
            ax = axes[row_i, col_i]
            panel = data[
                np.isclose(pd.to_numeric(data["node1_reward"], errors="coerce"), float(node1))
                & np.isclose(pd.to_numeric(data["node2_reward"], errors="coerce"), float(node2))
            ].sort_values("observation_index")
            if panel.empty:
                ax.axis("off")
                continue
            x = pd.to_numeric(panel["observation_index"], errors="coerce").to_numpy(dtype=float)
            mu = pd.to_numeric(panel[mu_col], errors="coerce").to_numpy(dtype=float)
            sem_values = pd.to_numeric(panel[sem_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(mu) & np.isfinite(sem_values)
            x, mu, sem_values = x[ok], mu[ok], sem_values[ok]
            if len(x):
                order = np.argsort(x)
                x, mu, sem_values = x[order], mu[order], sem_values[order]
                ax.fill_between(
                    x,
                    mu - sem_values,
                    mu + sem_values,
                    color="#9ecae1",
                    alpha=0.48,
                    linewidth=0,
                )
                ax.plot(x, mu, color="#08519c", marker="o", markersize=2.0, linewidth=0.85)
            if row_i == 0:
                ax.set_title(f"{col_label}={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={node1:g}")
            ax.set_xlim(0.8, x_max + 0.2)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(1, x_max + 1, 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(
                length=2,
                pad=1,
                labelbottom=(row_i == len(node1_values) - 1),
                labelleft=(col_i == 0),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{family_param_title(family, parameter_value)}, sigma={sigma:g}: mean z trajectory SEM", y=0.995)
    fig.supxlabel("current observation timestep", y=0.02)
    fig.supylabel(f"{latent_mu_axis_label(data)}\nshade = +/- SEM", x=0.005)
    fig.tight_layout(rect=(0.025, 0.045, 1.0, 0.965), h_pad=0.25, w_pad=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def family_param_title(family: str, parameter_value: float) -> str:
    return f"beta={parameter_value:g}" if family == "vary_beta" else f"opp={parameter_value:g}"


def plot_action_logit_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameter_value: float,
    sigma: float,
    outpath: Path,
) -> None:
    data = summary[
        (summary["family"].astype(str) == family)
        & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), float(parameter_value))
        & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
    ].copy()
    if data.empty:
        return
    configure_plotting()
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
    row_label, col_label = grid_axis_names(data)
    fig, axes = plt.subplots(
        len(node1_values),
        len(node2_values),
        figsize=(len(node2_values) * 1.05 + 0.8, len(node1_values) * 0.9 + 0.75),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    y = pd.to_numeric(data["action_logit"], errors="coerce")
    finite_y = y[np.isfinite(y)]
    if len(finite_y):
        y_min, y_max = float(finite_y.min()), float(finite_y.max())
    else:
        y_min, y_max = -1.0, 1.0
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)
    pad = 0.08 * max(y_max - y_min, 1.0)
    x_max = int(pd.to_numeric(data["observation_index"], errors="coerce").max())
    for row_i, node1 in enumerate(node1_values):
        for col_i, node2 in enumerate(node2_values):
            ax = axes[row_i, col_i]
            panel = data[
                np.isclose(pd.to_numeric(data["node1_reward"], errors="coerce"), float(node1))
                & np.isclose(pd.to_numeric(data["node2_reward"], errors="coerce"), float(node2))
            ].sort_values("observation_index")
            if panel.empty:
                ax.axis("off")
                continue
            ax.plot(
                panel["observation_index"],
                panel["action_logit"],
                color="#2b2b2b",
                marker="o",
                markersize=1.8,
                linewidth=0.75,
            )
            ax.axhline(0.0, color="#9e9e9e", linewidth=0.4, zorder=0)
            if row_i == 0:
                ax.set_title(f"{col_label}={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={node1:g}")
            ax.set_xlim(0.8, x_max + 0.2)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xticks(np.arange(1, x_max + 1, 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(
                length=2,
                pad=1,
                labelbottom=(row_i == len(node1_values) - 1),
                labelleft=(col_i == 0),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{family_param_title(family, parameter_value)}, sigma={sigma:g}: terminal action logit", y=0.995)
    fig.supxlabel("decision timestep", y=0.02)
    fig.supylabel("action logit", x=0.005)
    fig.tight_layout(rect=(0.025, 0.045, 1.0, 0.965), h_pad=0.25, w_pad=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_action_logit_sem_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameter_value: float,
    sigma: float,
    outpath: Path,
) -> None:
    data = summary[
        (summary["family"].astype(str) == family)
        & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), float(parameter_value))
        & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
    ].copy()
    if data.empty or "action_logit_sem" not in data.columns:
        return
    configure_plotting()
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
    row_label, col_label = grid_axis_names(data)
    fig, axes = plt.subplots(
        len(node1_values),
        len(node2_values),
        figsize=(len(node2_values) * 1.05 + 0.8, len(node1_values) * 0.9 + 0.75),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    y = pd.to_numeric(data["action_logit"], errors="coerce")
    sem = pd.to_numeric(data["action_logit_sem"], errors="coerce").fillna(0.0)
    finite = np.isfinite(y) & np.isfinite(sem)
    if np.any(finite):
        y_min = float(np.nanmin(y[finite] - sem[finite]))
        y_max = float(np.nanmax(y[finite] + sem[finite]))
    else:
        y_min, y_max = -1.0, 1.0
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)
    pad = 0.08 * max(y_max - y_min, 1.0)
    x_max = int(pd.to_numeric(data["observation_index"], errors="coerce").max())
    for row_i, node1 in enumerate(node1_values):
        for col_i, node2 in enumerate(node2_values):
            ax = axes[row_i, col_i]
            panel = data[
                np.isclose(pd.to_numeric(data["node1_reward"], errors="coerce"), float(node1))
                & np.isclose(pd.to_numeric(data["node2_reward"], errors="coerce"), float(node2))
            ].sort_values("observation_index")
            if panel.empty:
                ax.axis("off")
                continue
            x = pd.to_numeric(panel["observation_index"], errors="coerce").to_numpy(dtype=float)
            logit = pd.to_numeric(panel["action_logit"], errors="coerce").to_numpy(dtype=float)
            logit_sem = pd.to_numeric(panel["action_logit_sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(logit) & np.isfinite(logit_sem)
            x, logit, logit_sem = x[ok], logit[ok], logit_sem[ok]
            if len(x):
                order = np.argsort(x)
                x, logit, logit_sem = x[order], logit[order], logit_sem[order]
                ax.fill_between(
                    x,
                    logit - logit_sem,
                    logit + logit_sem,
                    color="#bdbdbd",
                    alpha=0.45,
                    linewidth=0,
                )
                ax.plot(x, logit, color="#252525", marker="o", markersize=1.8, linewidth=0.75)
            ax.axhline(0.0, color="#9e9e9e", linewidth=0.4, zorder=0)
            if row_i == 0:
                ax.set_title(f"{col_label}={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={node1:g}")
            ax.set_xlim(0.8, x_max + 0.2)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xticks(np.arange(1, x_max + 1, 1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(
                length=2,
                pad=1,
                labelbottom=(row_i == len(node1_values) - 1),
                labelleft=(col_i == 0),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{family_param_title(family, parameter_value)}, sigma={sigma:g}: terminal action logit SEM", y=0.995)
    fig.supxlabel("decision timestep", y=0.02)
    fig.supylabel("action logit\nshade = +/- SEM", x=0.005)
    fig.tight_layout(rect=(0.025, 0.045, 1.0, 0.965), h_pad=0.25, w_pad=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_plots(
    summary: pd.DataFrame,
    outdir: Path,
    latent_plot_types: set[str],
    *,
    action_summary: pd.DataFrame | None = None,
    shade_error_scale: float = 25.0,
    min_samples_per_dot: int = 1,
) -> None:
    summary = filter_summary_for_min_samples(summary, int(min_samples_per_dot))
    action_plot_summary = summary if action_summary is None else filter_summary_for_min_samples(
        action_summary,
        int(min_samples_per_dot),
    )
    if summary.empty and action_plot_summary.empty:
        return
    families = sorted(
        set(summary["family"].astype(str).unique() if not summary.empty else [])
        | set(action_plot_summary["family"].astype(str).unique() if not action_plot_summary.empty else [])
    )
    for family in families:
        family_df = summary[summary["family"].astype(str) == str(family)] if not summary.empty else pd.DataFrame()
        action_family_df = (
            action_plot_summary[action_plot_summary["family"].astype(str) == str(family)]
            if not action_plot_summary.empty
            else pd.DataFrame()
        )
        family_dir = outdir / str(family)
        parameter_values = sorted(
            set(pd.to_numeric(family_df.get("parameter_value", pd.Series(dtype=float)), errors="coerce").dropna().unique())
            | set(pd.to_numeric(action_family_df.get("parameter_value", pd.Series(dtype=float)), errors="coerce").dropna().unique())
        )
        for parameter_value in parameter_values:
            param_dir = family_dir / (
                f"beta_{value_token(parameter_value)}"
                if str(family) == "vary_beta"
                else f"opp_{value_token(parameter_value)}"
            )
            sigma_values = sorted(
                set(pd.to_numeric(family_df.get("sigma", pd.Series(dtype=float)), errors="coerce").dropna().unique())
                | set(pd.to_numeric(action_family_df.get("sigma", pd.Series(dtype=float)), errors="coerce").dropna().unique())
            )
            for sigma in sigma_values:
                sigma_dir = param_dir / f"sigma_{value_token(sigma)}"
                if not family_df.empty and "line3d" in latent_plot_types:
                    plot_latent_3d_grid(
                        summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "latent_mu_sigma_3d_by_node_rewards.png",
                    )
                if not family_df.empty and "shade" in latent_plot_types:
                    plot_latent_shaded_error_grid(
                        summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "latent_mu_sigma_shaded_error_by_node_rewards.png",
                        shade_error_scale=float(shade_error_scale),
                    )
                    plot_latent_sem_shaded_error_grid(
                        summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "latent_mu_sem_shaded_error_by_node_rewards.png",
                    )
                if not family_df.empty and "contour" in latent_plot_types:
                    plot_latent_contour_grid(
                        summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "latent_mu_sigma_contour_by_node_rewards.png",
                    )
                if not action_plot_summary.empty:
                    plot_action_logit_grid(
                        action_plot_summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "action_logit_by_node_rewards.png",
                    )
                    plot_action_logit_sem_grid(
                        action_plot_summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "action_logit_sem_shaded_error_by_node_rewards.png",
                    )


def write_seed_plots(
    raw: pd.DataFrame,
    outdir: Path,
    latent_plot_types: set[str],
    *,
    shade_error_scale: float = 25.0,
    min_samples_per_dot: int = 1,
) -> None:
    if raw.empty or "seed" not in raw.columns:
        return
    for seed in sorted(pd.to_numeric(raw["seed"], errors="coerce").dropna().astype(int).unique()):
        seed_rows = raw[np.isclose(pd.to_numeric(raw["seed"], errors="coerce"), seed)].copy()
        if seed_rows.empty:
            continue
        seed_dir = outdir / f"seed_{value_token(seed)}"
        seed_summary = summarize_trajectories(seed_rows)
        seed_summary_path = seed_dir / "latent_trajectory_summary.csv"
        seed_summary_path.parent.mkdir(parents=True, exist_ok=True)
        seed_summary.to_csv(seed_summary_path, index=False)
        seed_action_summary = summarize_action_logits(seed_rows)
        seed_action_summary_path = seed_dir / "latent_trajectory_action_logit_summary.csv"
        seed_action_summary.to_csv(seed_action_summary_path, index=False)
        print(
            f"Writing seed-specific latent trajectory plots for seed={seed} to {seed_dir}",
            flush=True,
        )
        write_plots(
            seed_summary,
            seed_dir,
            latent_plot_types,
            action_summary=seed_action_summary,
            shade_error_scale=shade_error_scale,
            min_samples_per_dot=int(min_samples_per_dot),
        )


def paired_cost_levels(beta_values: list[float], opportunity_values: list[float]) -> list[dict]:
    betas = sorted(float(x) for x in beta_values)
    opps = sorted((float(x) for x in opportunity_values), reverse=True)
    n_levels = min(len(betas), len(opps))
    if n_levels == 0:
        return []
    if n_levels == 3:
        labels = ["High cost", "Medium cost", "Low cost"]
    else:
        labels = [f"Cost level {i + 1}" for i in range(n_levels)]
    return [
        {
            "label": labels[i],
            "beta": betas[i],
            "opportunity": opps[i],
        }
        for i in range(n_levels)
    ]


def trial_key_columns(raw: pd.DataFrame) -> list[str]:
    preferred = [
        "task_mode",
        "family",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "trial_index",
    ]
    return [col for col in preferred if col in raw.columns]


def add_stop_alignment(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    out = raw.copy()
    key_cols = trial_key_columns(out)
    if not key_cols:
        out["steps_before_stop"] = np.nan
        return out
    if "is_stop_decision" not in out.columns:
        out["is_stop_decision"] = False
    if "action_plot_timestep" not in out.columns:
        out["action_plot_timestep"] = out.get("timestep", np.nan)
    stop_mask = out["is_stop_decision"].astype(str).str.lower().isin(["true", "1", "yes"])
    stop_rows = out[stop_mask].copy()
    if stop_rows.empty:
        out["steps_before_stop"] = np.nan
        return out
    stop_rows["_stop_timestep"] = pd.to_numeric(stop_rows["action_plot_timestep"], errors="coerce")
    stop_times = (
        stop_rows.groupby(key_cols, dropna=False)["_stop_timestep"]
        .min()
        .reset_index()
        .rename(columns={"_stop_timestep": "stop_timestep"})
    )
    out = out.merge(stop_times, on=key_cols, how="left")
    out["event_timestep"] = pd.to_numeric(out["action_plot_timestep"], errors="coerce")
    out["steps_before_stop"] = pd.to_numeric(out["stop_timestep"], errors="coerce") - out["event_timestep"]
    stop_timestep = pd.to_numeric(out["stop_timestep"], errors="coerce")
    out["normalized_steps_before_stop"] = out["steps_before_stop"] / stop_timestep.where(stop_timestep > 0)
    return out


def filter_trials_continued_after_second_observation(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep aggregate-delta trials that continued past the second observed reward."""
    if raw.empty:
        return raw.copy()
    key_cols = trial_key_columns(raw)
    if not key_cols or "is_stop_decision" not in raw.columns or "observation_index" not in raw.columns:
        return raw.copy()
    stop_mask = raw["is_stop_decision"].astype(str).str.lower().isin(["true", "1", "yes"])
    stop_rows = raw[stop_mask].copy()
    if stop_rows.empty:
        return raw.copy()
    stop_rows["_terminal_observation_count"] = pd.to_numeric(
        stop_rows["observation_index"],
        errors="coerce",
    )
    terminal_counts = (
        stop_rows.groupby(key_cols, dropna=False)["_terminal_observation_count"]
        .min()
        .reset_index()
    )
    terminal_counts["_continued_after_second_observation"] = (
        pd.to_numeric(terminal_counts["_terminal_observation_count"], errors="coerce") > 2
    )
    out = raw.merge(
        terminal_counts[key_cols + ["_continued_after_second_observation"]],
        on=key_cols,
        how="left",
    )
    keep = out["_continued_after_second_observation"].fillna(True).astype(bool)
    before_trials = raw[key_cols].drop_duplicates().shape[0]
    after_trials = out.loc[keep, key_cols].drop_duplicates().shape[0]
    dropped = before_trials - after_trials
    if dropped > 0:
        print(
            "Aggregate delta plots: excluding "
            f"{dropped} trial(s) that stopped at or before the second observed reward.",
            flush=True,
        )
    return out.loc[keep].drop(columns=["_continued_after_second_observation"])


def add_delta_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    out = add_stop_alignment(raw)
    if "prior_mu_mean" not in out.columns:
        out["prior_mu_mean"] = 0.0
    if "prior_sigma_mean" not in out.columns:
        out["prior_sigma_mean"] = 1.0
    prior_sigma = pd.to_numeric(out["prior_sigma_mean"], errors="coerce").clip(lower=1e-8)
    out["prior_normalized_z_mu"] = (
        pd.to_numeric(out["z_mu_mean"], errors="coerce") - pd.to_numeric(out["prior_mu_mean"], errors="coerce")
    ) / prior_sigma
    out["prior_normalized_z_sigma"] = pd.to_numeric(out["z_sigma_mean"], errors="coerce") / prior_sigma
    key_cols = trial_key_columns(out)
    if not key_cols:
        return out
    out = out.sort_values(key_cols + ["event_timestep"]).copy()
    action_rows = out["transition_type"].astype(str).isin(["observe", "stop"]) if "transition_type" in out.columns else pd.Series(True, index=out.index)
    observe_rows = out["transition_type"].astype(str).eq("observe") if "transition_type" in out.columns else pd.Series(True, index=out.index)
    out["delta_action_logit"] = np.nan
    out.loc[action_rows, "delta_action_logit"] = (
        out.loc[action_rows]
        .groupby(key_cols, dropna=False)["action_logit"]
        .diff()
        .to_numpy()
    )
    out["delta_prior_normalized_z_mu"] = np.nan
    out.loc[observe_rows, "delta_prior_normalized_z_mu"] = (
        out.loc[observe_rows]
        .groupby(key_cols, dropna=False)["prior_normalized_z_mu"]
        .diff()
        .to_numpy()
    )
    out["delta_prior_normalized_z_sigma"] = np.nan
    out.loc[observe_rows, "delta_prior_normalized_z_sigma"] = (
        out.loc[observe_rows]
        .groupby(key_cols, dropna=False)["prior_normalized_z_sigma"]
        .diff()
        .to_numpy()
    )
    return out


def aggregate_delta_metric(
    raw: pd.DataFrame,
    *,
    metric_col: str,
    include_stop: bool,
    min_samples: int,
    normalized_time: bool = False,
) -> pd.DataFrame:
    if raw.empty or metric_col not in raw.columns:
        return pd.DataFrame()
    data = raw.copy()
    if "steps_before_stop" not in data.columns:
        data = add_stop_alignment(data)
    if not include_stop:
        data = data[
            data["transition_type"].astype(str).eq("observe")
            & (pd.to_numeric(data["steps_before_stop"], errors="coerce") > 0)
        ].copy()
    else:
        data = data[data["transition_type"].astype(str).isin(["observe", "stop"])].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data[metric_col] = np.abs(data[metric_col])
    data["steps_before_stop"] = pd.to_numeric(data["steps_before_stop"], errors="coerce")
    if "normalized_steps_before_stop" not in data.columns:
        if "stop_timestep" in data.columns:
            stop_timestep = pd.to_numeric(data["stop_timestep"], errors="coerce")
        else:
            stop_timestep = pd.Series(np.nan, index=data.index)
        data["normalized_steps_before_stop"] = data["steps_before_stop"] / stop_timestep.where(stop_timestep > 0)
    x_col = "normalized_steps_before_stop" if normalized_time else "steps_before_stop"
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data = data[np.isfinite(data[metric_col]) & np.isfinite(data[x_col])].copy()
    if data.empty:
        return pd.DataFrame()
    group_cols = ["family", "parameter_value", "beta", "opportunity", "sigma", x_col]
    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            mean=(metric_col, "mean"),
            sd=(metric_col, "std"),
            n=(metric_col, "size"),
            n_seeds=("seed", "nunique") if "seed" in data.columns else (metric_col, "size"),
        )
        .reset_index()
    )
    summary["sem"] = (pd.to_numeric(summary["sd"], errors="coerce") / np.sqrt(summary["n"].clip(lower=1))).fillna(0.0)
    if int(min_samples) > 1:
        before = len(summary)
        summary = summary[pd.to_numeric(summary["n"], errors="coerce").fillna(0) >= int(min_samples)].copy()
        dropped = before - len(summary)
        if dropped:
            print(
                f"Filtering {dropped} aggregate delta point(s) for {metric_col} with n < {int(min_samples)}.",
                flush=True,
            )
    return summary


def aggregate_ylim(summary: pd.DataFrame) -> tuple[float, float]:
    if summary.empty:
        return (-1.0, 1.0)
    values = pd.to_numeric(summary["mean"], errors="coerce")
    sem = pd.to_numeric(summary["sem"], errors="coerce").fillna(0.0)
    candidates = pd.concat([values - sem, values + sem, values], ignore_index=True)
    candidates = candidates[np.isfinite(candidates)]
    if candidates.empty:
        return (-1.0, 1.0)
    y_min, y_max = float(candidates.min()), float(candidates.max())
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)
    if abs(y_max - y_min) < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    pad = 0.08 * max(y_max - y_min, 1e-9)
    return y_min - pad, y_max + pad


def aggregate_xlim(summary: pd.DataFrame, include_stop: bool, normalized_time: bool = False) -> tuple[float, float]:
    if summary.empty:
        if normalized_time:
            return ((-1.0, 0.0) if include_stop else (-1.0, -0.1))
        return ((-1.0, 0.0) if include_stop else (-2.0, -1.0))
    x_col = "normalized_steps_before_stop" if normalized_time else "steps_before_stop"
    x = pd.to_numeric(summary[x_col], errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return ((-1.0, 0.0) if include_stop else (-2.0, -1.0))
    x_plot = -x
    left = float(x_plot.min())
    right = float(x_plot.max())
    if not include_stop and not normalized_time:
        right = min(right, -1.0)
    if abs(right - left) < 1e-9:
        left -= 0.5
        right += 0.5
    return left - 0.08, right + 0.08


def plot_aggregate_delta_metric(
    summary: pd.DataFrame,
    *,
    levels: list[dict],
    sigma_values: list[float],
    metric_col: str,
    y_label: str,
    include_stop: bool,
    outpath: Path,
    normalized_time: bool = False,
) -> None:
    if summary.empty or not levels or not sigma_values:
        print(f"Skipping {outpath.name}: no aggregate data.", flush=True)
        return
    configure_plotting()
    n_rows = len(levels)
    n_cols = len(sigma_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.55 + 0.95, n_rows * 1.35 + 0.85),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    y_lim = aggregate_ylim(summary)
    x_col = "normalized_steps_before_stop" if normalized_time else "steps_before_stop"
    x_lim = aggregate_xlim(summary, include_stop=include_stop, normalized_time=normalized_time)
    beta_color = "#238b45"
    opp_color = "#2171b5"
    for row_i, level in enumerate(levels):
        for col_i, sigma in enumerate(sigma_values):
            ax = axes[row_i, col_i]
            ax.axhline(0.0, color="#bdbdbd", linewidth=0.45, zorder=0)
            for family, param_value, color, marker, linestyle, label in [
                ("vary_beta", float(level["beta"]), beta_color, "o", "-", f"beta {level['beta']:g}"),
                (
                    "vary_opportunity",
                    float(level["opportunity"]),
                    opp_color,
                    "^",
                    "--",
                    f"opp {level['opportunity']:g}",
                ),
            ]:
                panel = summary[
                    summary["family"].astype(str).eq(family)
                    & np.isclose(pd.to_numeric(summary["parameter_value"], errors="coerce"), param_value)
                    & np.isclose(pd.to_numeric(summary["sigma"], errors="coerce"), float(sigma))
                ].sort_values(x_col)
                if panel.empty:
                    continue
                x = -pd.to_numeric(panel[x_col], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(panel["mean"], errors="coerce").to_numpy(dtype=float)
                sem = pd.to_numeric(panel["sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sem)
                x, y, sem = x[ok], y[ok], sem[ok]
                if len(x) == 0:
                    continue
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    markersize=2.2,
                    linewidth=0.85,
                    label=label,
                )
            if row_i == 0:
                ax.set_title(f"sigma={sigma:g}", pad=2)
            if col_i == 0:
                ax.set_ylabel(
                    f"{level['label']}\n"
                    f"beta {level['beta']:g} / opp {level['opportunity']:g}\n"
                    f"{y_label}"
                )
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.xaxis.set_major_locator(MaxNLocator(integer=not normalized_time, nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(color="#ededed", linewidth=0.45)
            ax.tick_params(length=2, pad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.supxlabel("normalized time until stop" if normalized_time else "time until stop")
    fig.tight_layout(rect=(0.03, 0.035, 1.0, 0.94), h_pad=0.35, w_pad=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}", flush=True)


def write_aggregate_delta_plots(
    raw: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    *,
    min_samples_per_dot: int,
) -> None:
    if raw.empty:
        print("Skipping aggregate delta plots: no raw rows.", flush=True)
        return
    raw = filter_trials_continued_after_second_observation(raw)
    if raw.empty:
        print(
            "Skipping aggregate delta plots: no trials continued beyond the second observed reward.",
            flush=True,
        )
        return
    enriched = add_delta_columns(raw)
    levels = paired_cost_levels(list(args.vary_beta_values), list(args.vary_opportunity_values))
    sigma_values = sorted(float(x) for x in args.sigmas)
    aggregate_dir = outdir / "aggregate_delta_plots"
    specs = [
        {
            "metric_col": "delta_action_logit",
            "filename": "aggregate_delta_action_logit_by_timestep_before_stop.png",
            "ylabel": "|delta action logit|",
            "include_stop": False,
        },
        {
            "metric_col": "delta_prior_normalized_z_mu",
            "filename": "aggregate_delta_prior_normalized_z_mu_by_timestep_before_stop.png",
            "ylabel": "|delta prior-norm z_mu|",
            "include_stop": False,
        },
        {
            "metric_col": "delta_prior_normalized_z_sigma",
            "filename": "aggregate_delta_prior_normalized_z_sigma_by_timestep_before_stop.png",
            "ylabel": "|delta prior-norm z_sigma|",
            "include_stop": False,
        },
    ]
    for spec in specs:
        for normalized_time in (False, True):
            summary = aggregate_delta_metric(
                enriched,
                metric_col=spec["metric_col"],
                include_stop=bool(spec["include_stop"]),
                min_samples=int(min_samples_per_dot),
                normalized_time=normalized_time,
            )
            filename = spec["filename"]
            if normalized_time:
                filename = filename.replace(".png", "_normalized_time_until_stop.png")
            summary_path = aggregate_dir / filename.replace(".png", ".csv")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
            plot_aggregate_delta_metric(
                summary,
                levels=levels,
                sigma_values=sigma_values,
                metric_col=spec["metric_col"],
                y_label=spec["ylabel"],
                include_stop=bool(spec["include_stop"]),
                outpath=aggregate_dir / filename,
                normalized_time=normalized_time,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", nargs="?", default="default")
    parser.add_argument(
        "--task-mode",
        choices=["auto", "revisit", "evidence"],
        default="auto",
        help="Use evidence for model_jax/evidence_accumulation.py; auto selects evidence when tree/input/tree-type is evidence.",
    )
    parser.add_argument("--vary-beta-values", default="10,20,80")
    parser.add_argument("--vary-opportunity-values", default="0.06,0.2,0.4")
    parser.add_argument("--beta-sweep-opportunity", type=float, default=0.0)
    parser.add_argument("--opportunity-sweep-beta", type=float, default=100000.0)
    parser.add_argument("--sigmas", default="0,0.5,1,2")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--plot-seeds",
        "--include-seeds",
        dest="plot_seeds",
        default="",
        help=(
            "Optional comma/space separated seed list to include in summaries and plots. "
            "This is most useful with --plot-only: the full cached rows file is read, "
            "then filtered, and outputs are written under plot_seeds_<ids>/."
        ),
    )
    parser.add_argument(
        "--loss-scale",
        "--loss-scale-value",
        "--lambda-value",
        "--lambdas",
        dest="loss_scale_value",
        default="100.0",
        help="Scale applied to task/action/critic losses for evidence runs. --lambda-value is kept as a legacy alias.",
    )
    parser.add_argument(
        "--memory-lambda",
        "--kl-lambda",
        type=float,
        default=None if "MEMORY_LAMBDA" not in os.environ else float(os.environ["MEMORY_LAMBDA"]),
        help="Direct coefficient on paid KL for evidence runs. Defaults to the beta value.",
    )
    parser.add_argument("--alpha", "--alphas", dest="alpha", default="0.0")
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform")
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae")
    parser.add_argument("--rnn-units", "--rnn-dim", "--rnn-dims", dest="rnn_units", default="16")
    parser.add_argument("--latent-dim", "--latent-dims", dest="latent_dim", default="1")
    parser.add_argument("--max-observations-before-stop", type=int, default=10)
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--sim-dir", default="outputs/jax_simulations")
    parser.add_argument("--outdir", default="analysis_outputs/sample_set_latent_trajectory_jax")
    parser.add_argument("--num-updates", type=int, default=42000)
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--ppo-minibatches", type=int, default=1)
    parser.add_argument("--return-target-rollouts", type=int, default=1)
    parser.add_argument("--return-target-mode", default=os.environ.get("EXPANSION_RETURN_TARGET", "sampled_lambda"))
    parser.add_argument("--target-critic-update-interval", type=int, default=100)
    parser.add_argument("--target-critic-tau", type=float, default=1.0)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--enable-reconstruction", action="store_true")
    parser.add_argument("--enable-probe", action="store_true")
    parser.add_argument("--coherence-values", default="0,0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--correct-reward", type=float, default=5.0)
    parser.add_argument("--incorrect-reward", type=float, default=0.0)
    parser.add_argument("--pay-kl-on-stop", "--stop-paid", dest="pay_kl_on_stop", action="store_const", const=True, default=None)
    parser.add_argument("--no-pay-kl-on-stop", "--no-stop-paid", dest="pay_kl_on_stop", action="store_const", const=False)
    parser.add_argument("--n-sample-sets", type=int, default=50)
    parser.add_argument("--n-reward-combinations", type=int, default=0)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--sampled-lambda-critic", choices=["q", "value"], default="q")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
    parser.add_argument("--critic-huber-delta", type=float, default=float(os.environ.get("CRITIC_HUBER_DELTA", "10.0")))
    parser.add_argument("--advantage-clip", type=float, default=float(os.environ.get("ADVANTAGE_CLIP", "10.0")))
    parser.add_argument("--node-coverage-aux-coef", type=float, default=0.0)
    parser.add_argument("--node-coverage-aux-epochs", type=int, default=0)
    parser.add_argument(
        "--force-first-observe-node",
        "--force-first-probe-node",
        dest="force_first_observe_node",
        type=int,
        default=1,
        help=(
            "Force only the first action to observe/probe this 1-indexed node, then "
            "let the policy choose later actions. Default: 1. Use 0 to sample all actions."
        ),
    )
    parser.add_argument(
        "--no-force-first-observe-node",
        "--no-force-first-probe-node",
        dest="force_first_observe_node",
        action="store_const",
        const=0,
        help="Disable the default first-observe/probe-node-1 intervention.",
    )
    parser.add_argument(
        "--action-logit-mode",
        choices=[
            "path1_minus_path2",
            "abs_path1_minus_path2",
            "max_minus_second",
            "correct_minus_incorrect",
            "choose_a_minus_choose_b",
            "choose_b_minus_choose_a",
        ],
        default="path1_minus_path2",
        help=(
            "For evidence mode, the default path1_minus_path2 is remapped to "
            "choose_b_minus_choose_a, so positive means evidence/action toward choice +1."
        ),
    )
    parser.add_argument(
        "--latent-trajectory-plot-types",
        default="shade",
        help=(
            "Comma/space separated latent trajectory plot types: line3d, shade, "
            "contour, or any combination. Default: shade. 'ribbon' is accepted "
            "as an alias for shade."
        ),
    )
    parser.add_argument(
        "--shade-error-scale",
        type=float,
        default=25.0,
        help=(
            "Visual multiplier for the shaded error band. The underlying summary "
            "still stores true sigma(mean z); the plot label reports this scale. "
            "Use 1 for the exact unscaled band."
        ),
    )
    parser.add_argument(
        "--min-samples-per-dot",
        "--min-samples",
        dest="min_samples_per_dot",
        type=int,
        default=1,
        help=(
            "Minimum raw trajectory rows required for a plotted mean point. "
            "Default 1 preserves the historical behavior of plotting every cell."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read latent_trajectory_rows.csv and regenerate plots without rerunning simulations.",
    )
    parser.add_argument(
        "--aggregate-delta-plots",
        action="store_true",
        help=(
            "Write aggregate delta plots with rows for high/medium/low paired "
            "beta/opportunity costs and columns for sigma."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        "--aggregated-plots-only",
        dest="aggregate_only",
        action="store_true",
        help=(
            "Only write the aggregate delta plots. Combine with --plot-only to "
            "reuse latent_trajectory_rows.csv without regenerating model rollouts."
        ),
    )
    parser.add_argument(
        "--no-seed-plots",
        action="store_true",
        help="Only write pooled plots; skip seed_<id> plot folders.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.tree = sample_base.normalize_tree_name(args.tree)
    task_mode = str(args.task_mode).strip().lower()
    if task_mode == "auto":
        task_mode = "evidence" if (
            str(args.tree).strip().lower() == "evidence"
            or str(args.tree_type).strip().lower() == "evidence"
            or str(args.input_type).strip().lower() == "evidence"
        ) else "revisit"
    args.task_mode = task_mode
    if args.task_mode == "evidence":
        generic_beta_default = args.vary_beta_values == sample_base.parse_values("10,20,80", float)
        generic_opp_default = args.vary_opportunity_values == sample_base.parse_values("0.06,0.2,0.4", float)
        generic_sigma_default = args.sigmas == sample_base.parse_values("0,0.5,1,2", float)
        generic_seed_default = args.seeds == sample_base.parse_values("1,2,3", int)
        if generic_beta_default:
            args.vary_beta_values = [1000.0, 500.0, 100.0]
        if generic_opp_default:
            args.vary_opportunity_values = [0.001, 0.005, 0.01]
        if generic_sigma_default:
            args.sigmas = [0.1, 0.5, 1.0]
        if generic_seed_default:
            args.seeds = [4, 5, 6]
        if int(args.num_updates) == 42000:
            args.num_updates = 36000
        if args.pay_kl_on_stop is None:
            args.pay_kl_on_stop = True
        if str(args.input_type).strip().lower() == "uniform":
            args.input_type = "evidence"
        if str(args.tree_type).strip().lower() == "default":
            args.tree_type = "evidence"
        args.tree_size = 2
        if str(args.checkpoint_root) == "outputs/jax_models":
            args.checkpoint_root = "outputs/jax_models_evi"
        if str(args.sim_dir) == "outputs/jax_simulations":
            args.sim_dir = "outputs/jax_simulations_evi"
        if str(args.outdir) == "analysis_outputs/sample_set_latent_trajectory_jax":
            args.outdir = "analysis_outputs/sample_set_latent_trajectory_evidence_jax"
        if str(args.action_logit_mode) == "path1_minus_path2":
            args.action_logit_mode = "choose_b_minus_choose_a"
    args.vary_beta_values = sample_base.parse_values(args.vary_beta_values, float)
    args.vary_opportunity_values = sample_base.parse_values(args.vary_opportunity_values, float)
    args.sigmas = sample_base.parse_values(args.sigmas, float)
    args.seeds = sample_base.parse_values(args.seeds, int)
    args.plot_seeds = sample_base.parse_values(args.plot_seeds, int)
    args.loss_scale_value = float((sample_base.parse_values(args.loss_scale_value, float) or [100.0])[0])
    args.alpha = float((sample_base.parse_values(args.alpha, float) or [0.0])[0])
    args.rnn_units = int((sample_base.parse_values(args.rnn_units, int) or [16])[0])
    args.latent_dim = int((sample_base.parse_values(args.latent_dim, int) or [1])[0])
    args.coherence_values = sample_base.parse_values(args.coherence_values, float)
    if args.task_mode == "evidence" and not args.coherence_values:
        raise ValueError("--coherence-values must contain at least one value for evidence mode.")
    if args.steps_per_epoch is None:
        args.steps_per_epoch = 200 * 200 * max(int(args.max_observations_before_stop), 1)
    plot_types = set(str(x).lower() for x in sample_base.parse_values(args.latent_trajectory_plot_types, str))
    if "ribbon" in plot_types:
        plot_types.remove("ribbon")
        plot_types.add("shade")
    valid_types = {"line3d", "shade", "contour"}
    bad_types = sorted(plot_types - valid_types)
    if bad_types:
        raise ValueError(f"Unknown --latent-trajectory-plot-types value(s): {bad_types}")
    args.latent_trajectory_plot_types = plot_types or {"shade"}
    args.parameter_combos = []
    if args.pay_kl_on_stop is None:
        args.pay_kl_on_stop = False
    for beta in args.vary_beta_values:
        args.parameter_combos.append(("vary_beta", "beta", float(beta), float(beta), float(args.beta_sweep_opportunity)))
    for opp in args.vary_opportunity_values:
        args.parameter_combos.append(
            ("vary_opportunity", "opportunity", float(opp), float(args.opportunity_sweep_beta), float(opp))
        )
    return args


def output_dir(args: argparse.Namespace) -> Path:
    tree_label = "evidence" if str(args.task_mode) == "evidence" else sample_base.normalize_tree_name(args.tree)
    label = (
        f"{tree_label}"
        f"_vary_beta_{values_token(args.vary_beta_values)}"
        f"_vary_opp_{values_token(args.vary_opportunity_values)}"
    )
    if str(args.task_mode) == "evidence":
        label += f"_coh_{values_token(args.coherence_values)}"
        if not (math.isclose(float(args.correct_reward), 1.0) and math.isclose(float(args.incorrect_reward), 0.0)):
            label += f"_correctreward_{value_token(args.correct_reward)}"
            if not math.isclose(float(args.incorrect_reward), 0.0):
                label += f"_incorrectreward_{value_token(args.incorrect_reward)}"
    if str(args.task_mode) != "evidence" and int(args.force_first_observe_node) > 0:
        label += f"_force_first_node_{int(args.force_first_observe_node)}"
    if str(args.task_mode) == "evidence" and bool(args.pay_kl_on_stop):
        label += "_stop_paid"
    return Path(args.outdir) / label


def plot_output_dir(base_outdir: Path, args: argparse.Namespace) -> Path:
    if not args.plot_seeds:
        return base_outdir
    return base_outdir / f"plot_seeds_{values_token(args.plot_seeds)}"


def filter_raw_for_plot_seeds(raw: pd.DataFrame, plot_seeds: list[int]) -> pd.DataFrame:
    if not plot_seeds:
        return raw
    if "seed" not in raw.columns:
        print(
            "Warning: --plot-seeds was provided, but latent trajectory rows have no seed column; "
            "using all rows.",
            flush=True,
        )
        return raw
    seed_values = set(int(seed) for seed in plot_seeds)
    seed_col = pd.to_numeric(raw["seed"], errors="coerce")
    filtered = raw[seed_col.isin(seed_values)].copy()
    print(
        f"Filtering trajectory rows to plot seed(s) {','.join(str(x) for x in plot_seeds)}: "
        f"{len(filtered)}/{len(raw)} rows retained.",
        flush=True,
    )
    if filtered.empty:
        print(
            "Warning: seed filtering removed all rows. Check --plot-seeds against the cached CSV.",
            flush=True,
        )
    return filtered


def main() -> None:
    args = normalize_args(parse_args())
    base_outdir = output_dir(args)
    base_outdir.mkdir(parents=True, exist_ok=True)
    raw_path = base_outdir / "latent_trajectory_rows.csv"
    outdir = plot_output_dir(base_outdir, args)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "latent_trajectory_summary.csv"
    action_summary_path = outdir / "latent_trajectory_action_logit_summary.csv"
    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot --plot-only because {raw_path} does not exist.")
        print(f"Plot-only: reading {raw_path}", flush=True)
        raw = pd.read_csv(raw_path)
    else:
        parts = []
        for family, parameter_name, parameter_value, beta, opportunity in args.parameter_combos:
            for sigma in args.sigmas:
                for seed in args.seeds:
                    runner = run_one_evidence if str(args.task_mode) == "evidence" else run_one
                    parts.append(
                        runner(
                            args,
                            family=family,
                            parameter_name=parameter_name,
                            parameter_value=parameter_value,
                            beta=beta,
                            opportunity=opportunity,
                            sigma=sigma,
                            seed=seed,
                        )
                    )
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        raw.to_csv(raw_path, index=False)
        print(f"Saved raw trajectory rows to {raw_path}", flush=True)
    raw = filter_raw_for_plot_seeds(raw, args.plot_seeds)
    summary = summarize_trajectories(raw)
    summary.to_csv(summary_path, index=False)
    print(f"Saved trajectory summary to {summary_path}", flush=True)
    action_summary = summarize_action_logits(raw)
    action_summary.to_csv(action_summary_path, index=False)
    print(f"Saved trajectory action-logit summary to {action_summary_path}", flush=True)
    if bool(args.aggregate_delta_plots) or bool(args.aggregate_only):
        write_aggregate_delta_plots(
            raw,
            outdir,
            args,
            min_samples_per_dot=int(args.min_samples_per_dot),
        )
    if bool(args.aggregate_only):
        print(f"Saved aggregate-only latent trajectory plots to {outdir / 'aggregate_delta_plots'}", flush=True)
        return
    write_plots(
        summary,
        outdir,
        set(args.latent_trajectory_plot_types),
        action_summary=action_summary,
        shade_error_scale=float(args.shade_error_scale),
        min_samples_per_dot=int(args.min_samples_per_dot),
    )
    if not bool(args.no_seed_plots):
        write_seed_plots(
            raw,
            outdir,
            set(args.latent_trajectory_plot_types),
            shade_error_scale=float(args.shade_error_scale),
            min_samples_per_dot=int(args.min_samples_per_dot),
        )
    print(f"Saved latent trajectory plots to {outdir}", flush=True)


if __name__ == "__main__":
    main()
