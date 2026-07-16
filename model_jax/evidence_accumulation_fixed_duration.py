"""Duration-controlled evaluation for the JAX continuous evidence model.

This script reuses the architecture and checkpoint format from
``model_jax/evidence_accumulation.py`` but changes the evaluation task to a
psychophysical-kernel setting.  For checkpoints trained with observer/end-choice
mode, each trial receives exactly ``max_observations_before_stop`` evidence
samples, with default 10.  Steps 1..T-1 are deterministic forced ``CONTINUE``
transitions; no terminal choice can occur before the final sample has been
processed.  At step T, ``CONTINUE`` is masked out by the original model and the
terminal A/B choice is sampled from the post-mask A/B policy.

For checkpoints not trained in observer/end-choice mode, the evaluator leaves
the stop/continue decision to the learned policy.  The wide output still has
columns up to ``max_observations_before_stop`` but columns after the first
terminal decision are marked invalid/NA, so downstream analyses can condition
on the realized total number of observations before stopping.

Intermediate forced continuation steps are not used to create a policy-gradient
objective here: this script is evaluation-only.  Opportunity cost is preserved
as an input and saved using the learned timing convention.  Because every trial
has exactly T observations, it is a constant offset of (T - 1) *
opportunity_cost within a condition and does not affect stopping behavior.

The important analysis output is the raw pre-mask action head at every
timestep.  ``raw_logit_choose_b_t - raw_logit_choose_a_t`` is saved at all
timesteps, including the forced-continuation timesteps, so downstream analyses
can measure the model's evolving terminal choice preference without
contamination from action masks or forced-action overrides.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from model_jax import evidence_accumulation as ev


DEFAULT_NUM_OBSERVATIONS = 10


def parse_float_values(raw_values: Iterable[str] | str | None) -> list[float]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    out: list[float] = []
    for raw in raw_values:
        for piece in str(raw).replace(",", " ").split():
            if piece.strip():
                out.append(float(piece))
    return out


def parse_int_values(raw_values: Iterable[str] | str | None) -> list[int]:
    return [int(round(v)) for v in parse_float_values(raw_values)]


def value_token(value) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "p", text)
    text = re.sub(r"^p|p$", "", text)
    return text or "value"


def checkpoint_training_step(label: str) -> float:
    nums = re.findall(r"([0-9]+)", str(label))
    if not nums:
        return float("nan")
    return float(nums[-1])


def duration_suffix(checkpoint_label: str, n_obs: int, observer_only: bool) -> str:
    duration_mode = "fixed_duration" if observer_only else "policy_duration"
    return f"_{duration_mode}_{n_obs}_checkpoint_{value_token(checkpoint_label)}"


def fixed_duration_output_stem(config: ev.RunConfig, checkpoint_label: str) -> str:
    """Compact output stem; full metadata is still written inside each CSV."""
    pieces = [
        "evidence",
        f"loss_scale_{value_token(config.loss_scale)}",
        f"beta_{value_token(config.beta)}",
        f"memorylambda_{value_token(config.memory_lambda)}",
        f"opp_{value_token(config.opportunity_cost)}",
        f"seed_{int(config.seed)}",
        f"obsstd_{value_token(config.observation_noise_std)}",
        f"maxobs_{int(config.num_steps)}",
        f"rnn_{int(config.rnn_units)}",
        f"latent_{int(config.latent_dim)}",
        f"correctreward_{value_token(config.correct_reward)}",
    ]
    if bool(config.pay_kl_on_stop):
        pieces.append("stop_paid")
    if bool(config.choice_at_end_only):
        pieces.append("observer_endchoice")
    pieces.append(duration_suffix(checkpoint_label, config.num_steps, bool(config.choice_at_end_only)).lstrip("_"))
    pieces.append(config.input_type)
    return "_".join(pieces)


def make_task(config: ev.RunConfig) -> ev.EvidenceTaskSpec:
    return ev.make_task(config)


def make_config(args, beta: float, opportunity: float, obsstd: float, seed: int) -> ev.RunConfig:
    max_obs = int(args.max_observations_before_stop)
    coherence_values = tuple(parse_float_values(args.coherence_values))
    if not coherence_values:
        raise ValueError("--coherence-values must contain at least one nonnegative value.")
    if any(c < 0 for c in coherence_values):
        raise ValueError("--coherence-values must be nonnegative magnitudes.")
    return ev.RunConfig(
        loss_scale=float(args.loss_scale_string),
        alpha=float(args.alpha_string),
        beta=float(beta),
        memory_lambda=float(args.memory_lambda) if args.memory_lambda is not None else float(beta),
        model_dir=str(args.model_dir),
        epochs=int(args.epochs),
        input_type=str(args.input_type),
        seed=int(seed),
        tree_size=int(args.tree_size),
        train_mode=str(args.eval_mode),
        tree_type=str(args.tree_type),
        opportunity_cost=float(opportunity),
        expansion_decision_version=ev.normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=ev.normalize_model_variant(args.model_variant),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir=str(args.sim_dir),
        n_sim_trials=int(args.n_sim_trials),
        num_envs=int(args.num_envs),
        num_steps=max_obs,
        update_epochs=int(args.update_epochs),
        ppo_minibatches=max(int(args.ppo_minibatches), 1),
        steps_per_epoch=int(args.steps_per_epoch or (200 * 200 * max_obs)),
        return_target_rollouts=max(int(args.return_target_rollouts), 1),
        return_target_mode=ev.normalize_return_target_mode(args.return_target_mode),
        sampled_lambda_critic=str(args.sampled_lambda_critic),
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=max(int(args.target_critic_update_interval), 0),
        target_critic_tau=float(args.target_critic_tau),
        backend=args.backend,
        jit_training=not bool(args.no_jit),
        enable_reconstruction=bool(args.enable_reconstruction),
        enable_probe=bool(args.enable_probe),
        max_observations_before_stop=max_obs,
        coherence_values=coherence_values,
        observation_noise_std=max(float(obsstd), 1e-6),
        correct_reward=float(args.correct_reward),
        incorrect_reward=float(args.incorrect_reward),
        kl_start_multiplier=max(float(args.kl_start_multiplier), 0.0),
        kl_annealing_epochs=max(int(args.kl_annealing_epochs), 0),
        critic_huber_delta=max(float(args.critic_huber_delta), 0.0),
        advantage_clip=max(float(args.advantage_clip), 0.0),
        pay_kl_on_stop=bool(args.pay_kl_on_stop),
        choice_at_end_only=bool(args.choice_at_end_only),
    )


def initialize_model_and_params(config: ev.RunConfig, task: ev.EvidenceTaskSpec):
    model = ev.build_model(config, task)
    rng = jax.random.PRNGKey(config.seed)
    dummy_batch = ev.sample_evidence_batch(rng, 1, config.num_steps, task)
    dummy = ev.initial_carry(dummy_batch, config.rnn_units)
    sched = ev.ScheduleValues(
        current_alpha=1.0,
        current_beta=config.memory_lambda,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    params = model.init(rng, dummy, rng, sched, None, False)["params"]
    return model, params


def resolve_checkpoint_path(config: ev.RunConfig, checkpoint: str) -> tuple[str, Path, list[Path]]:
    checkpoint = str(checkpoint).strip()
    tried: list[Path] = []
    if not checkpoint or checkpoint.lower() in {"final", "default"}:
        candidate = Path(config.model_dir) / f"{ev.model_name_for(config)}.msgpack"
        tried.append(candidate)
        if candidate.exists():
            return "final", candidate, tried
        candidate = Path(config.model_dir) / f"{ev.legacy_model_name_for(config)}.msgpack"
        tried.append(candidate)
        return "final", candidate, tried
    raw = Path(checkpoint)
    tried.append(raw)
    if raw.exists():
        return raw.stem, raw, tried
    candidate = Path(config.model_dir) / checkpoint
    tried.append(candidate)
    if candidate.exists():
        return candidate.stem, candidate, tried
    if not checkpoint.endswith(".msgpack"):
        candidate = Path(config.model_dir) / f"{checkpoint}.msgpack"
        tried.append(candidate)
        if candidate.exists():
            return candidate.stem, candidate, tried
        candidate = Path(config.model_dir) / f"{ev.model_name_for(config)}_{checkpoint}.msgpack"
        tried.append(candidate)
        if candidate.exists():
            return checkpoint, candidate, tried
        candidate = Path(config.model_dir) / f"{ev.legacy_model_name_for(config)}_{checkpoint}.msgpack"
        tried.append(candidate)
        if candidate.exists():
            return checkpoint, candidate, tried
    return checkpoint, Path(config.model_dir) / checkpoint, tried


def load_params_for_checkpoint(config: ev.RunConfig, task: ev.EvidenceTaskSpec, checkpoint: str):
    model, params = initialize_model_and_params(config, task)
    checkpoint_label, checkpoint_path, tried_paths = resolve_checkpoint_path(config, checkpoint)
    if checkpoint_path.exists():
        params = serialization.from_bytes(params, checkpoint_path.read_bytes())
    else:
        attempted = "; ".join(str(path) for path in tried_paths)
        print(
            "Warning: no checkpoint found for fixed-duration evidence evaluation; "
            f"evaluating initialized evidence model. Tried: {attempted}",
            flush=True,
        )
    return model, params, checkpoint_label, checkpoint_path


def fixed_duration_rollout(
    model: ev.EvidenceVAE,
    params,
    batch: ev.EvidenceBatch,
    config: ev.RunConfig,
    rng: jax.Array,
    use_posterior_mean: bool = False,
):
    force_fixed_duration = bool(config.choice_at_end_only)
    schedule = ev.ScheduleValues(
        current_alpha=1.0,
        current_beta=config.memory_lambda,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    carry = ev.initial_carry(batch, config.rnn_units)
    transitions = []
    raw_logits = []
    for step in range(config.num_steps):
        rng, step_rng = jax.random.split(rng)
        if force_fixed_duration and step < config.num_steps - 1:
            forced_action = jnp.full((batch.correct_choice.shape[0],), ev.CONTINUE, dtype=jnp.int32)
        else:
            forced_action = None
        carry, transition = model.apply(
            {"params": params},
            carry,
            step_rng,
            schedule,
            forced_action,
            True,
            use_posterior_mean,
            method=ev.EvidenceVAE.__call__,
        )
        logits = model.apply(
            {"params": params},
            transition.expansion_input,
            method=ev.EvidenceVAE.expansion_logits_from_input,
        )
        transitions.append(transition)
        raw_logits.append(logits)
    stacked_transitions = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *transitions)
    return jax.device_get(stacked_transitions), np.asarray(jax.device_get(jnp.stack(raw_logits, axis=0)))


def first_terminal_step(transitions: ev.EvidenceTransition, trial: int, T: int) -> tuple[int, bool]:
    terminal_flags = np.asarray(transitions.is_terminal[:, trial]) > 0
    if np.any(terminal_flags):
        return int(np.argmax(terminal_flags)), True
    valid_flags = np.asarray(transitions.valid[:, trial]) > 0
    valid_steps = np.flatnonzero(valid_flags)
    if len(valid_steps) > 0:
        return int(valid_steps[-1]), False
    return T - 1, False


def terminal_action_and_reward(
    config: ev.RunConfig,
    batch_np: ev.EvidenceBatch,
    transitions: ev.EvidenceTransition,
    trial: int,
    terminal_step: int,
    has_terminal: bool,
) -> tuple[int, float]:
    if has_terminal:
        terminal_action = int(transitions.terminal_action[terminal_step, trial])
        terminal_reward = float(transitions.terminal_reward[terminal_step, trial])
        return terminal_action, terminal_reward
    terminal_probs = np.asarray(transitions.probs[terminal_step, trial, 1:], dtype=float)
    terminal_action = int(np.argmax(terminal_probs) + 1)
    terminal_reward = (
        float(config.correct_reward)
        if terminal_action == int(batch_np.correct_action[trial])
        else float(config.incorrect_reward)
    )
    return terminal_action, terminal_reward


def finite_or_none(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def rows_from_fixed_duration(
    config: ev.RunConfig,
    batch: ev.EvidenceBatch,
    transitions: ev.EvidenceTransition,
    raw_logits: np.ndarray,
    checkpoint_label: str,
    checkpoint_path: Path,
) -> tuple[list[dict], list[dict]]:
    batch_np = jax.device_get(batch)
    n_trials = int(batch_np.correct_choice.shape[0])
    T = int(config.num_steps)
    training_step = checkpoint_training_step(checkpoint_label)
    run_id = (
        f"loss_scale={config.loss_scale:g}|memory_lambda={config.memory_lambda:g}|"
        f"alpha={config.alpha:g}|beta={config.beta:g}|opp={config.opportunity_cost:g}|seed={config.seed}|"
        f"obsstd={config.observation_noise_std:g}|observer_only={int(bool(config.choice_at_end_only))}|"
        f"checkpoint={checkpoint_label}"
    )
    wide_rows: list[dict] = []
    long_rows: list[dict] = []
    for trial in range(n_trials):
        terminal_step, has_terminal = first_terminal_step(transitions, trial, T)
        terminal_action, terminal_reward = terminal_action_and_reward(
            config,
            batch_np,
            transitions,
            trial,
            terminal_step,
            has_terminal,
        )
        num_observations = int(transitions.num_observations_before[terminal_step, trial])
        num_observations = max(1, min(num_observations, T))
        num_continue_actions = int(np.sum(np.asarray(transitions.is_continue[:, trial]) > 0))
        total_opp = float(np.sum(transitions.opportunity_cost_paid[:, trial]))
        total_memory = float(np.sum(transitions.memory_cost_paid[:, trial]))
        total_kl = float(np.sum(transitions.paid_kl[:, trial]))
        decision_cum = float(transitions.cumulative_evidence[terminal_step, trial])
        row = {
            "graph": trial,
            "trial_id": trial,
            "run_id": run_id,
            "seed": int(config.seed),
            "checkpoint": checkpoint_label,
            "checkpoint_path": str(checkpoint_path),
            "training_step": training_step,
            "loss_scale": float(config.loss_scale),
            "memory_lambda": float(config.memory_lambda),
            "alpha": float(config.alpha),
            "beta": float(config.beta),
            "opportunity_cost": float(config.opportunity_cost),
            "coherence": float(batch_np.coherence[trial]),
            "signed_coherence": float(batch_np.signed_coherence[trial]),
            "observation_noise_std": float(config.observation_noise_std),
            "correct_reward": float(config.correct_reward),
            "incorrect_reward": float(config.incorrect_reward),
            "max_observations_before_stop": int(T),
            "choice_at_end_only": bool(config.choice_at_end_only),
            "duration_mode": "fixed" if bool(config.choice_at_end_only) else "policy",
            "num_observations": int(num_observations),
            "num_continue_actions": int(num_continue_actions),
            "stopping_time": int(num_observations),
            "correct_choice": int(batch_np.correct_choice[trial]),
            "correct_action": int(batch_np.correct_action[trial]),
            "terminal_action": terminal_action,
            "choose_right": terminal_action == ev.CHOOSE_B,
            "correct": terminal_action == int(batch_np.correct_action[trial]),
            "choose_correct": terminal_action == int(batch_np.correct_action[trial]),
            "terminal_reward": terminal_reward,
            "total_opportunity_cost": total_opp,
            "total_memory_cost": total_memory,
            "total_kl_paid": total_kl,
            "total_return": terminal_reward - total_opp - total_memory,
            "total_reward": terminal_reward - total_opp - total_memory,
            "decision_cumulative_evidence": decision_cum,
            "cumulative_evidence_at_decision": decision_cum,
            "decision_oracle_cumulative_llr": float(transitions.oracle_cumulative_llr[terminal_step, trial]),
            "terminal_choice_entropy": float(transitions.entropy[terminal_step, trial]),
        }
        for t in range(T):
            step = t + 1
            is_valid_step = bool(t <= terminal_step and transitions.valid[t, trial] > 0)
            if is_valid_step:
                logits = raw_logits[t, trial, :]
                ab_shift = logits[1:3] - np.max(logits[1:3])
                ab_probs = np.exp(ab_shift) / np.sum(np.exp(ab_shift))
                z_logvar = np.asarray(transitions.z_logvar[t, trial, :], dtype=float)
                prior_logvar = np.asarray(transitions.prior_logvar[t, trial, :], dtype=float)
                z_sigma = np.exp(0.5 * np.clip(z_logvar, -10.0, 10.0))
                prior_sigma = np.sqrt(np.exp(prior_logvar) + 1e-6)
                final_probs = np.asarray(transitions.probs[t, trial, :], dtype=float)
                observation_value = float(batch_np.evidence_samples[t, trial])
                cumulative_value = float(batch_np.cumulative_evidence[t, trial])
                oracle_value = float(batch_np.oracle_cumulative_llr[t, trial])
                action_value = int(transitions.action[t, trial])
                continue_value = bool(transitions.is_continue[t, trial] > 0)
                stop_value = bool(transitions.is_terminal[t, trial] > 0)
            else:
                logits = np.full((ev.NUM_ACTIONS,), np.nan, dtype=float)
                ab_probs = np.full((2,), np.nan, dtype=float)
                z_logvar = np.full((config.latent_dim,), np.nan, dtype=float)
                prior_logvar = np.full((config.latent_dim,), np.nan, dtype=float)
                z_sigma = np.full((config.latent_dim,), np.nan, dtype=float)
                prior_sigma = np.full((config.latent_dim,), np.nan, dtype=float)
                final_probs = np.full((ev.NUM_ACTIONS,), np.nan, dtype=float)
                observation_value = float("nan")
                cumulative_value = float("nan")
                oracle_value = float("nan")
                action_value = -1
                continue_value = False
                stop_value = False
            row[f"valid_t{step}"] = is_valid_step
            row[f"observation_{step}"] = observation_value
            row[f"evidence_sample_t{step}"] = observation_value
            row[f"cumulative_evidence_t{step}"] = cumulative_value
            row[f"oracle_cumulative_llr_t{step}"] = oracle_value
            row[f"action_t{step}"] = action_value
            row[f"continue_t{step}"] = continue_value
            row[f"stop_t{step}"] = stop_value
            row[f"kl_d_t{step}"] = float(transitions.paid_kl[t, trial]) if is_valid_step else float("nan")
            row[f"kl_d_obs_t{step}"] = float(transitions.observed_kl[t, trial]) if is_valid_step else float("nan")
            row[f"opportunity_cost_t{step}"] = float(transitions.opportunity_cost_paid[t, trial]) if is_valid_step else float("nan")
            row[f"memory_cost_t{step}"] = float(transitions.memory_cost_paid[t, trial]) if is_valid_step else float("nan")
            row[f"raw_logit_continue_t{step}"] = finite_or_none(logits[ev.CONTINUE])
            row[f"raw_logit_choose_a_t{step}"] = finite_or_none(logits[ev.CHOOSE_A])
            row[f"raw_logit_choose_b_t{step}"] = finite_or_none(logits[ev.CHOOSE_B])
            row[f"choice_logit_t{step}"] = finite_or_none(logits[ev.CHOOSE_B] - logits[ev.CHOOSE_A])
            row[f"p_choose_b_given_terminal_t{step}"] = finite_or_none(ab_probs[1])
            row[f"policy_continue_t{step}"] = finite_or_none(final_probs[ev.CONTINUE])
            row[f"policy_choose_a_t{step}"] = finite_or_none(final_probs[ev.CHOOSE_A])
            row[f"policy_choose_b_t{step}"] = finite_or_none(final_probs[ev.CHOOSE_B])
            row[f"value_pred_t{step}"] = float(transitions.value_pred[t, trial]) if is_valid_step else float("nan")
            row[f"action_policy_entropy_t{step}"] = float(transitions.entropy[t, trial]) if is_valid_step else float("nan")
            for dim in range(config.latent_dim):
                row[f"z_mu_{dim}_t{step}"] = float(transitions.z_mu[t, trial, dim]) if is_valid_step else float("nan")
                row[f"z_logvar_{dim}_t{step}"] = float(z_logvar[dim])
                row[f"z_sigma_{dim}_t{step}"] = float(z_sigma[dim])
                row[f"z_sample_{dim}_t{step}"] = float(transitions.z_sample[t, trial, dim]) if is_valid_step else float("nan")
                row[f"prior_mu_{dim}_t{step}"] = float(transitions.prior_mu[t, trial, dim]) if is_valid_step else float("nan")
                row[f"prior_logvar_{dim}_t{step}"] = float(prior_logvar[dim])
                row[f"prior_sigma_{dim}_t{step}"] = float(prior_sigma[dim])
            if not is_valid_step:
                continue
            long = {
                "trial_id": trial,
                "run_id": run_id,
                "seed": int(config.seed),
                "checkpoint": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                "training_step": training_step,
                "loss_scale": float(config.loss_scale),
                "memory_lambda": float(config.memory_lambda),
                "alpha": float(config.alpha),
                "beta": float(config.beta),
                "opportunity_cost": float(config.opportunity_cost),
                "coherence": float(batch_np.coherence[trial]),
                "signed_coherence": float(batch_np.signed_coherence[trial]),
                "observation_noise_std": float(config.observation_noise_std),
                "choice_at_end_only": bool(config.choice_at_end_only),
                "duration_mode": "fixed" if bool(config.choice_at_end_only) else "policy",
                "num_observations": int(num_observations),
                "correct_choice": int(batch_np.correct_choice[trial]),
                "correct_action": int(batch_np.correct_action[trial]),
                "terminal_action": terminal_action,
                "choose_right": terminal_action == ev.CHOOSE_B,
                "correct": terminal_action == int(batch_np.correct_action[trial]),
                "timestep": step,
                "observation": float(batch_np.evidence_samples[t, trial]),
                "cumulative_evidence": float(batch_np.cumulative_evidence[t, trial]),
                "oracle_cumulative_llr": float(batch_np.oracle_cumulative_llr[t, trial]),
                "raw_logit_continue": float(logits[ev.CONTINUE]),
                "raw_logit_choose_a": float(logits[ev.CHOOSE_A]),
                "raw_logit_choose_b": float(logits[ev.CHOOSE_B]),
                "choice_logit": float(logits[ev.CHOOSE_B] - logits[ev.CHOOSE_A]),
                "p_choose_b_given_terminal": float(ab_probs[1]),
                "final_policy_continue": float(final_probs[ev.CONTINUE]),
                "final_policy_choose_a": float(final_probs[ev.CHOOSE_A]),
                "final_policy_choose_b": float(final_probs[ev.CHOOSE_B]),
                "action": int(transitions.action[t, trial]),
                "is_forced_continue": bool(config.choice_at_end_only and t < T - 1),
                "is_terminal_step": bool(t == terminal_step),
                "kl_paid": float(transitions.paid_kl[t, trial]),
                "observed_kl": float(transitions.observed_kl[t, trial]),
                "opportunity_cost_paid": float(transitions.opportunity_cost_paid[t, trial]),
                "memory_cost_paid": float(transitions.memory_cost_paid[t, trial]),
                "value_pred": float(transitions.value_pred[t, trial]),
            }
            for dim in range(config.latent_dim):
                long[f"z_mu_{dim}"] = float(transitions.z_mu[t, trial, dim])
                long[f"z_logvar_{dim}"] = float(z_logvar[dim])
                long[f"z_sigma_{dim}"] = float(z_sigma[dim])
                long[f"z_sample_{dim}"] = float(transitions.z_sample[t, trial, dim])
                long[f"prior_mu_{dim}"] = float(transitions.prior_mu[t, trial, dim])
                long[f"prior_logvar_{dim}"] = float(prior_logvar[dim])
                long[f"prior_sigma_{dim}"] = float(prior_sigma[dim])
            long_rows.append(long)
        wide_rows.append(row)
    return wide_rows, long_rows


def validate_rows(wide: pd.DataFrame, long: pd.DataFrame, config: ev.RunConfig) -> None:
    T = int(config.num_steps)
    obs_cols = [f"observation_{i}" for i in range(1, T + 1)]
    missing = [c for c in obs_cols if c not in wide.columns]
    if missing:
        raise AssertionError(f"Missing observation columns: {missing}")
    num_obs = pd.to_numeric(wide["num_observations"], errors="coerce").to_numpy(dtype=float)
    if bool(config.choice_at_end_only):
        if not np.all(num_obs == T):
            raise AssertionError("Every fixed-duration trial must have exactly T observations.")
        for t in range(1, T):
            if not np.all(wide[f"action_t{t}"].to_numpy(dtype=int) == ev.CONTINUE):
                raise AssertionError(f"Found non-CONTINUE action before final timestep t={t}.")
            if np.any(wide[f"stop_t{t}"].astype(bool)):
                raise AssertionError(f"Found terminal action before final timestep t={t}.")
        if np.any(wide[f"action_t{T}"].to_numpy(dtype=int) == ev.CONTINUE):
            raise AssertionError("Final fixed-duration action cannot be CONTINUE.")
    else:
        if np.any((num_obs < 1) | (num_obs > T) | ~np.isfinite(num_obs)):
            raise AssertionError("Self-timed trials must have num_observations in [1, T].")
    for t in range(1, T + 1):
        raw = wide[f"raw_logit_choose_b_t{t}"] - wide[f"raw_logit_choose_a_t{t}"]
        valid = wide.get(f"valid_t{t}", pd.Series(True, index=wide.index)).astype(bool).to_numpy()
        if not np.allclose(raw[valid], wide[f"choice_logit_t{t}"][valid], equal_nan=False):
            raise AssertionError(f"choice_logit_t{t} mismatch.")
        p = pd.to_numeric(wide[f"p_choose_b_given_terminal_t{t}"], errors="coerce")
        if np.any(((p < -1e-8) | (p > 1 + 1e-8) | ~np.isfinite(p))[valid]):
            raise AssertionError(f"Invalid conditional terminal choice probability at t={t}.")
        logit_cols = [f"raw_logit_continue_t{t}", f"raw_logit_choose_a_t{t}", f"raw_logit_choose_b_t{t}"]
        if not np.all(np.isfinite(wide.loc[valid, logit_cols].to_numpy(dtype=float))):
            raise AssertionError(f"Non-finite raw logits at t={t}.")
    if bool(config.choice_at_end_only) and not long.empty and long["timestep"].nunique() != T:
        raise AssertionError("Long trajectory output does not contain all timesteps.")
    for dim in range(config.latent_dim):
        for t in range(1, T + 1):
            valid = wide.get(f"valid_t{t}", pd.Series(True, index=wide.index)).astype(bool).to_numpy()
            sigma_expected = np.exp(0.5 * np.clip(wide[f"z_logvar_{dim}_t{t}"], -10.0, 10.0))
            if not np.allclose(wide[f"z_sigma_{dim}_t{t}"][valid], sigma_expected[valid], rtol=1e-5, atol=1e-7):
                raise AssertionError(f"z_sigma dim={dim} t={t} inconsistent with z_logvar.")


def save_outputs(config: ev.RunConfig, checkpoint_label: str, wide_rows: list[dict], long_rows: list[dict]) -> tuple[Path, Path, Path]:
    def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    out_dir = Path(config.sim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = fixed_duration_output_stem(config, checkpoint_label)
    wide_path = out_dir / f"{base}_wide.csv"
    long_path = out_dir / f"{base}_trajectory_long.csv"
    summary_path = out_dir / f"{base}_summary.csv"
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    validate_rows(wide, long, config)
    write_csv_atomic(wide, wide_path)
    write_csv_atomic(long, long_path)
    summary = (
        wide.groupby(
            [
                "run_id",
                "checkpoint",
                "training_step",
                "loss_scale",
                "memory_lambda",
                "choice_at_end_only",
                "duration_mode",
                "beta",
                "opportunity_cost",
                "coherence",
                "signed_coherence",
                "observation_noise_std",
            ],
            dropna=False,
        )
        .agg(
            n=("trial_id", "count"),
            p_choose_right=("choose_right", "mean"),
            p_choose_correct=("choose_correct", "mean"),
            mean_terminal_reward=("terminal_reward", "mean"),
            mean_total_opportunity_cost=("total_opportunity_cost", "mean"),
            mean_total_memory_cost=("total_memory_cost", "mean"),
            mean_total_return=("total_return", "mean"),
            mean_terminal_choice_entropy=("terminal_choice_entropy", "mean"),
        )
        .reset_index()
    )
    write_csv_atomic(summary, summary_path)
    return wide_path, long_path, summary_path


def evaluate_one(config: ev.RunConfig, checkpoint: str, use_posterior_mean: bool = False) -> None:
    task = make_task(config)
    model, params, checkpoint_label, checkpoint_path = load_params_for_checkpoint(config, task, checkpoint)
    rng = jax.random.PRNGKey(config.seed + 250_000 + int(checkpoint_training_step(checkpoint_label) if np.isfinite(checkpoint_training_step(checkpoint_label)) else 0))
    batch = ev.sample_evidence_batch(rng, config.n_sim_trials, config.num_steps, task)
    rng, rollout_rng = jax.random.split(rng)
    transitions, raw_logits = fixed_duration_rollout(
        model,
        params,
        batch,
        config,
        rollout_rng,
        use_posterior_mean=use_posterior_mean,
    )
    batch_np = jax.device_get(batch)
    wide_rows, long_rows = rows_from_fixed_duration(
        config,
        batch_np,
        transitions,
        raw_logits,
        checkpoint_label,
        checkpoint_path,
    )
    wide_path, long_path, summary_path = save_outputs(config, checkpoint_label, wide_rows, long_rows)
    wide = pd.DataFrame(wide_rows)
    print(
        f"{'Fixed-duration' if bool(config.choice_at_end_only) else 'Self-timed'} evidence evaluation: "
        f"trials={len(wide)}, checkpoint={checkpoint_label}, beta={config.beta:g}, "
        f"loss_scale={config.loss_scale:g}, memory_lambda={config.memory_lambda:g}, "
        f"opp={config.opportunity_cost:g}, obsstd={config.observation_noise_std:g}, "
        f"observer_only={bool(config.choice_at_end_only)}, "
        f"coherence_values={','.join(f'{x:g}' for x in config.coherence_values)}",
        flush=True,
    )
    print(
        f"Array shapes: raw_logits={raw_logits.shape}, z_mu=[{len(wide)}, {config.num_steps}, {config.latent_dim}]",
        flush=True,
    )
    print(
        f"P(choose right)={wide['choose_right'].mean():.4f}; "
        f"accuracy={wide['choose_correct'].mean():.4f}; "
        f"terminal entropy={wide['terminal_choice_entropy'].mean():.4f}; "
        f"mean observations={wide['num_observations'].mean():.3f}",
        flush=True,
    )
    print(f"Saved duration-controlled wide trials to: {wide_path}", flush=True)
    print(f"Saved duration-controlled long trajectories to: {long_path}", flush=True)
    print(f"Saved duration-controlled summary to: {summary_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Duration-controlled evaluator for model_jax/evidence_accumulation.py checkpoints.")
    parser.add_argument("loss_scale_string", nargs="?", default="100.0")
    parser.add_argument("alpha_string", nargs="?", default="0.0")
    parser.add_argument("beta_string", nargs="?", default="1000.0")
    parser.add_argument("model_dir", nargs="?", default="outputs/jax_models_evi")
    parser.add_argument("epochs", nargs="?", type=int, default=120)
    parser.add_argument("input_type", nargs="?", default="evidence")
    parser.add_argument("seed", nargs="?", type=int, default=1)
    parser.add_argument("tree_size", nargs="?", type=int, default=2)
    parser.add_argument("eval_mode", nargs="?", default="sim")
    parser.add_argument("tree_type", nargs="?", default="evidence")
    parser.add_argument("opportunity_cost_string", nargs="?", default="0.0")
    parser.add_argument("expansion_decision_version", nargs="?", default="lstm")
    parser.add_argument("model_variant", nargs="?", default="vae")
    parser.add_argument("rnn_units", nargs="?", type=int, default=16)
    parser.add_argument("latent_dim", nargs="?", type=int, default=1)
    parser.add_argument("--sim-dir", default="outputs/jax_simulations_evi_fixed_duration")
    parser.add_argument("--n-sim-trials", "--num-trials", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=None, help="Must equal --max-observations-before-stop when supplied.")
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatches", type=int, default=int(os.environ.get("PPO_MINIBATCHES", "1")))
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--return-target-rollouts", type=int, default=int(os.environ.get("RETURN_TARGET_ROLLOUTS", "1")))
    parser.add_argument("--return-target-mode", default=os.environ.get("EXPANSION_RETURN_TARGET", "sampled_lambda"))
    parser.add_argument("--sampled-lambda-critic", choices=["value", "q"], default=os.environ.get("SAMPLED_LAMBDA_CRITIC", "value").strip().lower())
    parser.add_argument("--lambda-return", type=float, default=float(os.environ.get("EXPANSION_LAMBDA_RETURN", "0.95")))
    parser.add_argument("--target-critic-update-interval", type=int, default=int(os.environ.get("TARGET_CRITIC_UPDATE_INTERVAL", "100")))
    parser.add_argument("--target-critic-tau", type=float, default=float(os.environ.get("TARGET_CRITIC_TAU", "1.0")))
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--enable-reconstruction", action="store_true")
    parser.add_argument("--enable-probe", action="store_true")
    parser.add_argument("--max-observations-before-stop", type=int, default=DEFAULT_NUM_OBSERVATIONS)
    parser.add_argument("--coherence-values", default=os.environ.get("COHERENCE_VALUES", "0,0.05,0.1,0.2,0.4,0.8"))
    parser.add_argument("--observation-noise-std", "--observation-noise-std-list", "--obsstd", nargs="*", default=[os.environ.get("OBSERVATION_NOISE_STD", "1.0")])
    parser.add_argument("--correct-reward", type=float, default=float(os.environ.get("CORRECT_REWARD", "1.0")))
    parser.add_argument("--incorrect-reward", type=float, default=0.0)
    parser.add_argument("--kl-start-multiplier", type=float, default=float(os.environ.get("KL_START_MULTIPLIER", "1.0")))
    parser.add_argument("--kl-annealing-epochs", type=int, default=int(os.environ.get("KL_ANNEALING_EPOCHS", "0")))
    parser.add_argument(
        "--critic-huber-delta",
        type=float,
        default=float(os.environ.get("CRITIC_HUBER_DELTA", "10.0")),
        help="Training compatibility field for evidence RunConfig. Default: 10.",
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=float(os.environ.get("ADVANTAGE_CLIP", "10.0")),
        help="Training compatibility field for evidence RunConfig. Default: 10.",
    )
    parser.add_argument(
        "--memory-lambda",
        "--kl-lambda",
        type=float,
        default=None if "MEMORY_LAMBDA" not in os.environ else float(os.environ["MEMORY_LAMBDA"]),
        help="Direct coefficient on paid KL memory cost. Defaults to the beta positional value.",
    )
    parser.add_argument("--pay-kl-on-stop", "--pay-memory-cost-on-stop", action="store_true", default=os.environ.get("PAY_KL_ON_STOP", "").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument(
        "--choice-at-end-only",
        "--observer-only",
        "--observer-end-choice",
        "--trained-observer-only",
        action="store_true",
        default=os.environ.get("CHOICE_AT_END_ONLY", "").strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "Evaluate checkpoints trained with observer/end-choice mode. In this "
            "mode the evaluator forces fixed-duration observation and uses the "
            "_observer_endchoice checkpoint/output filename suffix. Without this "
            "flag, the model chooses continue/stop itself."
        ),
    )
    parser.add_argument("--checkpoints", nargs="*", default=["final"], help="Checkpoint labels or paths. Default evaluates the final learned-model checkpoint.")
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--use-posterior-mean", action="store_true", help="Use posterior mean z instead of sampling z during evaluation.")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if args.num_steps is not None and int(args.num_steps) != int(args.max_observations_before_stop):
        raise ValueError("--num-steps must equal --max-observations-before-stop in this evaluator.")
    if int(args.max_observations_before_stop) != DEFAULT_NUM_OBSERVATIONS:
        print(
            f"Note: evaluator cap is {int(args.max_observations_before_stop)} observations "
            f"(default is {DEFAULT_NUM_OBSERVATIONS}); observer-only checkpoints are forced to this duration.",
            flush=True,
        )
    return args


def run_self_tests() -> None:
    obs = np.random.default_rng(0).normal(size=(3000, DEFAULT_NUM_OBSERVATIONS))
    weights = np.linspace(-1.0, 1.0, DEFAULT_NUM_OBSERVATIONS)
    logits = obs @ weights
    prob = 1.0 / (1.0 + np.exp(-logits))
    choice = np.random.default_rng(1).binomial(1, prob)

    # Tiny dependency-free logistic IRLS check used only by --run-tests.  The
    # R analysis does the canonical glm() fit for the real integration weights.
    x = np.column_stack([np.ones(obs.shape[0]), obs])
    beta = np.zeros(x.shape[1])
    ridge = np.eye(x.shape[1]) * 1e-6
    ridge[0, 0] = 0.0
    for _ in range(50):
        eta = x @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
        w = np.maximum(mu * (1.0 - mu), 1e-6)
        z = eta + (choice - mu) / w
        xtw = x.T * w
        beta_next = np.linalg.solve(xtw @ x + ridge, xtw @ z)
        if np.max(np.abs(beta_next - beta)) < 1e-8:
            beta = beta_next
            break
        beta = beta_next
    corr = np.corrcoef(weights, beta[1:])[0, 1]
    if corr < 0.98:
        raise AssertionError("Synthetic logistic-regression integration weights were not recovered.")
    print("Fixed-duration synthetic integration-weight self-test passed.", flush=True)


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_self_tests()
        return
    if args.backend:
        jax.config.update("jax_platform_name", args.backend)
    beta_values = parse_float_values(args.beta_string)
    opportunity_values = parse_float_values(args.opportunity_cost_string)
    obsstd_values = parse_float_values(args.observation_noise_std)
    seeds = parse_int_values(args.seeds) if args.seeds is not None else [int(args.seed)]
    checkpoint_values = [x for raw in args.checkpoints for x in str(raw).replace(",", " ").split() if x]
    if not beta_values:
        raise ValueError("beta_string must contain at least one value.")
    if not opportunity_values:
        raise ValueError("opportunity_cost_string must contain at least one value.")
    if not obsstd_values:
        raise ValueError("--observation-noise-std must contain at least one value.")
    total = len(beta_values) * len(opportunity_values) * len(obsstd_values) * len(seeds) * len(checkpoint_values)
    print(
        f"Evidence evaluation grid: {total} combo(s); beta={beta_values}; "
        f"opportunity={opportunity_values}; obsstd={obsstd_values}; seeds={seeds}; "
        f"checkpoints={checkpoint_values}; max observations={int(args.max_observations_before_stop)}; "
        f"trained observer-only={bool(args.choice_at_end_only)}",
        flush=True,
    )
    for beta in beta_values:
        for opportunity in opportunity_values:
            for obsstd in obsstd_values:
                for seed in seeds:
                    config = make_config(args, beta, opportunity, obsstd, seed)
                    for checkpoint in checkpoint_values:
                        evaluate_one(config, checkpoint, use_posterior_mean=bool(args.use_posterior_mean))


if __name__ == "__main__":
    main()
