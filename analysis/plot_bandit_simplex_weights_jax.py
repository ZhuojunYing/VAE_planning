#!/usr/bin/env python3
"""Simplex temporal weights for JAX bandit/revisit planning models.

This is the bandit analogue of ``analysis/plot_evidence_accumulation_simplex_weights.py``.
It evaluates trained ``model_jax/planning.py`` checkpoints on freshly sampled
reward trials, records observations, terminal-choice log odds, and latent
statistics at each observation timestep, then fits

    y_hat = bias + X @ c,  c_i >= 0

with one model using only observations from the target path and another using
only observations from non-target paths.  Relative weights are ``c / sum(c)``;
effective weights are the raw nonnegative coefficients ``c``.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.optimize import lsq_linear
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script needs scipy. Please run it inside vae_env.") from exc

from model_jax import planning as jp


def parse_csv_values(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x for x in re.split(r"[, \t\n]+", str(value).strip()) if x]


def numeric_values(value: str | None) -> list[float]:
    out = []
    for raw in parse_csv_values(value):
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def int_values(value: str | None) -> list[int]:
    return [int(round(v)) for v in numeric_values(value)]


def num_label(value) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return str(value)
    return f"{x:.7g}"


def value_token(value) -> str:
    return re.sub(r"(^p|p$)", "", re.sub(r"[^A-Za-z0-9]+", "p", num_label(value))) or "value"


def parameter_equal(values, target: float, tol: float = 1e-6) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(arr) & (np.abs(arr - float(target)) <= tol)


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


@dataclass(frozen=True)
class Combo:
    family: str
    parameter_value: float
    memory_lambda: float
    opportunity_cost: float
    sigma: float
    seed: int


def make_config(args: argparse.Namespace, combo: Combo) -> jp.RunConfig:
    tree_size = int(args.tree_size)
    tree_type = jp.normalize_tree_type(args.tree_type, tree_size)
    max_obs = int(args.max_observations_before_stop)
    if args.num_steps is not None:
        num_steps = int(args.num_steps)
    elif args.rollout_mode == "policy":
        num_steps = max_obs + 1 if args.allow_node_revisit else max(tree_size, max_obs)
    else:
        num_steps = max_obs
    return jp.RunConfig(
        loss_scale=float(args.loss_scale),
        alpha=float(args.alpha),
        memory_lambda=float(combo.memory_lambda),
        model_dir=str(args.model_dir),
        epochs=int(args.epochs),
        input_type=str(args.input_type),
        seed=int(combo.seed),
        tree_size=tree_size,
        train_mode="sim",
        tree_type=tree_type,
        opportunity_cost=float(combo.opportunity_cost),
        expansion_decision_version=jp.normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=jp.normalize_model_variant(args.model_variant),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir=str(args.outdir),
        n_sim_trials=int(args.n_trials),
        num_envs=int(args.num_envs),
        num_steps=int(num_steps),
        update_epochs=int(args.update_epochs),
        ppo_minibatches=max(int(args.ppo_minibatches), 1),
        steps_per_epoch=int(args.steps_per_epoch),
        return_target_rollouts=max(int(args.return_target_rollouts), 1),
        return_target_mode=jp.normalize_return_target_mode(args.return_target_mode),
        sampled_lambda_critic=str(args.sampled_lambda_critic),
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=max(int(args.target_critic_update_interval), 0),
        target_critic_tau=float(args.target_critic_tau),
        backend=args.backend,
        jit_training=not bool(args.no_jit),
        profile_update_components=False,
        profile_update_components_every=1,
        enable_reconstruction=bool(args.enable_reconstruction),
        enable_probe=bool(args.enable_probe),
        allow_node_revisit=bool(args.allow_node_revisit),
        max_observations_before_stop=max_obs,
        observation_sigma=max(float(combo.sigma), 0.0),
        kl_start_multiplier=max(float(args.kl_start_multiplier), 0.0),
        kl_annealing_epochs=max(int(args.kl_annealing_epochs), 0),
        node_coverage_aux_coef=max(float(args.node_coverage_aux_coef), 0.0),
        node_coverage_aux_epochs=max(int(args.node_coverage_aux_epochs), 0),
        critic_huber_delta=max(float(args.critic_huber_delta), 0.0),
        advantage_clip=max(float(args.advantage_clip), 0.0),
        learning_rate=max(float(args.learning_rate), 0.0),
        min_learning_rate=None if args.min_learning_rate is None else max(float(args.min_learning_rate), 0.0),
        pay_kl_on_stop=bool(args.pay_kl_on_stop),
        choice_at_end_only=bool(args.choice_at_end_only),
    )


def make_output_dir(args: argparse.Namespace) -> Path:
    memory_tag = "_".join(value_token(v) for v in numeric_values(args.vary_memory_lambda_values))
    opp_tag = "_".join(value_token(v) for v in numeric_values(args.vary_opportunity_values))
    sigma_tag = "_".join(value_token(v) for v in numeric_values(args.sigmas))
    folder = (
        f"{jp.normalize_tree_type(args.tree_type, int(args.tree_size))}"
        f"_simplex_memory_{memory_tag}_opp_{opp_tag}_sigma_{sigma_tag}_{args.rollout_mode}"
    )
    return Path(args.outdir) / folder


def forced_action_for_step(step: int, config: jp.RunConfig, task: jp.TaskSpec, mode: str, n_trials: int) -> jax.Array | None:
    if mode == "policy":
        return None
    if mode == "round_robin":
        node = step % task.num_nodes
    elif mode == "random_nodes":
        rng = np.random.default_rng(config.seed + 9917 + step)
        return jnp.asarray(rng.integers(0, task.num_nodes, size=n_trials), dtype=jnp.int32)
    else:
        raise ValueError(f"Unsupported rollout mode: {mode}")
    return jnp.full((n_trials,), int(node), dtype=jnp.int32)


def rollout_combo(config: jp.RunConfig, task: jp.TaskSpec, args: argparse.Namespace) -> tuple[np.ndarray, list[jp.StepTransition]]:
    model, params = jp.load_state_for_sim(config, task)
    rng = jax.random.PRNGKey(config.seed + 310_000)
    n_trials = int(config.n_sim_trials)
    reward_feature_dim = jp.reward_feature_dim_for_sigma(config.observation_sigma)
    carry = jp.initial_carry(
        n_trials,
        task,
        config.rnn_units,
        reward_feature_dim,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    rng, reset_rng = jax.random.split(rng)
    reset_rewards = jp.sample_reward_matrix(reset_rng, n_trials, task.num_nodes, task.reward_values)
    carry = jp.reset_done_envs(carry, reset_rewards)
    sched = jp.ScheduleValues(
        current_alpha=1.0,
        current_beta=config.memory_lambda,
        current_critic_coef=0.0,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.0,
        node_coverage_aux_coef=0.0,
        forced_continue_epsilon=0.0,
        ppo_clip=0.3,
    )
    transitions: list[jp.StepTransition] = []
    for step in range(int(config.num_steps)):
        rng, step_rng = jax.random.split(rng)
        forced_action = forced_action_for_step(step, config, task, args.rollout_mode, n_trials)
        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            sched,
            forced_action=forced_action,
            training=True,
            use_posterior_mean=bool(args.use_posterior_mean),
            compute_targets=False,
            method=jp.PlanningVAE.__call__,
        )
        transitions.append(jax.device_get(trans))
    return np.asarray(reset_rewards), transitions


def path_for_observed_node(path_map: np.ndarray, node_index: int) -> int:
    if node_index < 0:
        return -1
    memberships = np.flatnonzero(path_map[:, node_index] > 0)
    if memberships.size == 0:
        return -1
    return int(memberships[0])


def logit_target_vs_other(path_probs: np.ndarray, target_path: int) -> float:
    probs = np.asarray(path_probs, dtype=float)
    if target_path < 0 or target_path >= probs.size:
        return float("nan")
    other = np.delete(probs, target_path)
    if other.size == 0:
        return float("nan")
    return float(np.log(probs[target_path] + 1e-12) - np.log(np.mean(other) + 1e-12))


def rows_for_combo(
    combo: Combo,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    transitions: list[jp.StepTransition],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    path_map = np.asarray(task.path_map, dtype=float)
    path_values = rewards @ path_map.T
    best_paths = np.argmax(path_values, axis=1)
    n_trials = rewards.shape[0]
    max_obs = int(args.max_observations_before_stop)
    trial_rows: list[dict] = []
    example_rows: list[dict] = []
    for trial in range(n_trials):
        terminal_probs_last = None
        chosen_path = None
        stop_timestep = None
        for step_idx, trans in enumerate(transitions):
            terminal_probs_last = np.asarray(trans.action_output[trial], dtype=float)
            if bool(np.asarray(trans.is_stop)[trial] > 0) and chosen_path is None:
                chosen_path = int(np.asarray(trans.terminal_path_index)[trial])
                stop_timestep = step_idx + 1
        if chosen_path is None:
            chosen_path = int(np.argmax(terminal_probs_last)) if terminal_probs_last is not None else int(best_paths[trial])
        if stop_timestep is None:
            stop_timestep = len(transitions)
        target_path = int(best_paths[trial]) if args.target_path_mode == "best" else int(chosen_path)
        base = {
            "family": combo.family,
            "parameter_value": combo.parameter_value,
            "memory_lambda": combo.memory_lambda,
            "opportunity_cost": combo.opportunity_cost,
            "observation_sigma": combo.sigma,
            "seed": combo.seed,
            "trial": trial,
            "target_path_mode": args.target_path_mode,
            "target_path": target_path + 1,
            "chosen_path": chosen_path + 1,
            "best_path": int(best_paths[trial]) + 1,
            "stop_timestep": int(stop_timestep),
            "rollout_mode": args.rollout_mode,
        }
        for node in range(task.num_nodes):
            base[f"reward_node_{node + 1}"] = float(rewards[trial, node])
        for path in range(task.num_paths):
            base[f"path_value_{path + 1}"] = float(path_values[trial, path])
        target_seq: list[float] = []
        nontarget_seq: list[float] = []
        trial_row = dict(base)
        for step_idx, trans in enumerate(transitions):
            step = step_idx + 1
            node_idx = int(np.asarray(trans.node_index)[trial])
            is_observe = bool(np.asarray(trans.is_observe)[trial] > 0)
            obs_value = float(np.asarray(trans.expanded_reward)[trial]) if is_observe else float("nan")
            obs_path = path_for_observed_node(path_map, node_idx)
            is_target_obs = is_observe and bool(path_map[target_path, node_idx] > 0)
            if is_observe:
                if is_target_obs:
                    target_seq.append(obs_value)
                else:
                    nontarget_seq.append(obs_value)
            terminal_probs = np.asarray(trans.action_output[trial], dtype=float)
            action_logit = logit_target_vs_other(terminal_probs, target_path)
            trial_row[f"observed_node_t{step}"] = node_idx + 1 if is_observe else np.nan
            trial_row[f"observed_path_t{step}"] = obs_path + 1 if obs_path >= 0 else np.nan
            trial_row[f"observation_t{step}"] = obs_value
            trial_row[f"is_target_observation_t{step}"] = bool(is_target_obs)
            trial_row[f"target_logit_t{step}"] = action_logit
            for dim in range(min(config.latent_dim, int(args.max_latent_dims))):
                z_logvar = float(np.asarray(trans.z_logvar)[trial, dim])
                trial_row[f"z_mu_{dim}_t{step}"] = float(np.asarray(trans.z_mu)[trial, dim])
                trial_row[f"z_sigma_{dim}_t{step}"] = float(math.exp(0.5 * np.clip(z_logvar, -10.0, 10.0)))
            if not is_observe:
                continue
            example = dict(base)
            example.update(
                outcome_timestep=step,
                observed_node=node_idx + 1,
                observed_path=obs_path + 1 if obs_path >= 0 else np.nan,
                observed_value=obs_value,
                action_logit=action_logit,
            )
            for pos in range(1, max_obs + 1):
                example[f"target_obs_{pos}"] = target_seq[pos - 1] if pos <= len(target_seq) else 0.0
                example[f"target_obs_present_{pos}"] = pos <= len(target_seq)
                example[f"nontarget_obs_{pos}"] = nontarget_seq[pos - 1] if pos <= len(nontarget_seq) else 0.0
                example[f"nontarget_obs_present_{pos}"] = pos <= len(nontarget_seq)
            for dim in range(min(config.latent_dim, int(args.max_latent_dims))):
                z_logvar = float(np.asarray(trans.z_logvar)[trial, dim])
                example[f"z_mu_{dim}"] = float(np.asarray(trans.z_mu)[trial, dim])
                example[f"z_sigma_{dim}"] = float(math.exp(0.5 * np.clip(z_logvar, -10.0, 10.0)))
            example_rows.append(example)
        trial_rows.append(trial_row)
    return trial_rows, example_rows


def fit_nonnegative(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, dict]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    if X.shape[0] == 0:
        raise ValueError("No finite rows.")
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    result = lsq_linear(
        X_aug,
        y,
        bounds=(np.r_[-np.inf, np.zeros(X.shape[1])], np.inf),
        max_iter=500,
        tol=1e-8,
        lsmr_tol="auto",
    )
    coef = np.asarray(result.x[1:], dtype=float)
    bias = float(result.x[0])
    y_hat = X_aug @ result.x
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    metrics = {
        "n_trials": int(X.shape[0]),
        "bias": bias,
        "gain": float(np.sum(coef)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "converged": bool(result.success),
        "optimizer_status": int(result.status),
    }
    return coef, bias, metrics


def outcome_specs(examples: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, str, int | None]]:
    specs: list[tuple[str, str, int | None]] = []
    requested = {x.strip() for x in parse_csv_values(args.outcomes)}
    if "action_logit" in requested and "action_logit" in examples.columns:
        specs.append(("action_logit", "action_logit", None))
    dims = sorted(
        {
            int(m.group(1))
            for col in examples.columns
            for m in [re.match(r"^z_mu_(\d+)$", str(col))]
            if m
        }
    )
    if args.z_dims:
        keep = set(int_values(args.z_dims))
        dims = [d for d in dims if d in keep]
    for dim in dims:
        if "z_mu" in requested and f"z_mu_{dim}" in examples.columns:
            specs.append((f"z_mu_dim{dim}", f"z_mu_{dim}", dim))
        if "z_sigma" in requested and f"z_sigma_{dim}" in examples.columns:
            specs.append((f"z_sigma_dim{dim}", f"z_sigma_{dim}", dim))
    return specs


def fit_all(examples: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_obs = int(args.max_observations_before_stop)
    roles = ["target", "nontarget"]
    run_rows: list[dict] = []
    metric_rows: list[dict] = []
    specs = outcome_specs(examples, args)
    group_cols = [
        "family",
        "parameter_value",
        "memory_lambda",
        "opportunity_cost",
        "observation_sigma",
        "seed",
        "target_path_mode",
        "rollout_mode",
        "outcome_timestep",
    ]
    for group_values, group in examples.groupby(group_cols, dropna=False):
        group_meta = dict(zip(group_cols, group_values))
        for outcome_name, outcome_col, latent_dim in specs:
            y = pd.to_numeric(group[outcome_col], errors="coerce").to_numpy(dtype=float)
            for role in roles:
                cols = [f"{role}_obs_{pos}" for pos in range(1, max_obs + 1)]
                present_cols = [f"{role}_obs_present_{pos}" for pos in range(1, max_obs + 1)]
                X = group[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                present = group[present_cols].astype(bool).to_numpy(dtype=bool)
                if role == "nontarget" and outcome_name == "action_logit" and args.flip_nontarget_logit_predictors:
                    X = -X
                keep = np.isfinite(y) & (present.any(axis=1))
                if int(np.sum(keep)) < int(args.min_trials_per_fit):
                    metric_rows.append(
                        {
                            **group_meta,
                            "outcome": outcome_name,
                            "outcome_col": outcome_col,
                            "latent_dim": latent_dim if latent_dim is not None else np.nan,
                            "predictor_role": role,
                            "n_trials": int(np.sum(keep)),
                            "skipped_reason": f"too_few_trials:{int(np.sum(keep))}",
                        }
                    )
                    continue
                try:
                    coef, _bias, metrics = fit_nonnegative(X[keep], y[keep])
                except Exception as exc:
                    metric_rows.append(
                        {
                            **group_meta,
                            "outcome": outcome_name,
                            "outcome_col": outcome_col,
                            "latent_dim": latent_dim if latent_dim is not None else np.nan,
                            "predictor_role": role,
                            "n_trials": int(np.sum(keep)),
                            "skipped_reason": str(exc),
                        }
                    )
                    continue
                gain = float(np.sum(coef))
                weights = coef / gain if gain > 0 else np.full_like(coef, np.nan)
                for pos, (w, c) in enumerate(zip(weights, coef), start=1):
                    run_rows.append(
                        {
                            **group_meta,
                            "outcome": outcome_name,
                            "outcome_col": outcome_col,
                            "latent_dim": latent_dim if latent_dim is not None else np.nan,
                            "predictor_role": role,
                            "predictor_position": pos,
                            "simplex_weight": float(w),
                            "effective_coefficient": float(c),
                        }
                    )
                metric_rows.append(
                    {
                        **group_meta,
                        "outcome": outcome_name,
                        "outcome_col": outcome_col,
                        "latent_dim": latent_dim if latent_dim is not None else np.nan,
                        "predictor_role": role,
                        **metrics,
                        "skipped_reason": "",
                    }
                )
    return pd.DataFrame(run_rows), pd.DataFrame(metric_rows)


def summarize_weights(run_level: pd.DataFrame) -> pd.DataFrame:
    if run_level.empty:
        return pd.DataFrame()
    group_cols = [
        "family",
        "parameter_value",
        "memory_lambda",
        "opportunity_cost",
        "observation_sigma",
        "target_path_mode",
        "rollout_mode",
        "outcome_timestep",
        "outcome",
        "outcome_col",
        "latent_dim",
        "predictor_role",
        "predictor_position",
    ]
    rows = []
    for values, group in run_level.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, values))
        row.update(
            n_runs=int(group["seed"].nunique()),
            mean_simplex_weight=float(group["simplex_weight"].mean()),
            se_simplex_weight=sem(group["simplex_weight"]),
            mean_effective_coefficient=float(group["effective_coefficient"].mean()),
            se_effective_coefficient=sem(group["effective_coefficient"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def palette(values: list[float], family: str) -> dict[float, tuple[float, float, float, float]]:
    values = sorted(values)
    cmap = plt.get_cmap("Greens" if family == "memory" else "Blues")
    if len(values) == 1:
        levels = [0.75]
    else:
        levels = np.linspace(0.45, 0.9, len(values))
    # Larger memory lambda and larger opportunity cost are darker.
    return {v: cmap(levels[i]) for i, v in enumerate(values)}


def plot_profile(summary: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    if summary.empty:
        print("No successful fits to plot.", flush=True)
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": float(args.font_size)})
    plot_timesteps = sorted(pd.to_numeric(summary["outcome_timestep"], errors="coerce").dropna().astype(int).unique())
    if args.plot_max_timestep is not None:
        plot_timesteps = [t for t in plot_timesteps if t <= int(args.plot_max_timestep)]
    sigmas = sorted(pd.to_numeric(summary["observation_sigma"], errors="coerce").dropna().unique())
    families = ["memory", "opportunity"]
    roles = ["target", "nontarget"]
    y_specs = [
        ("relative_temporal_weights", "mean_simplex_weight", "se_simplex_weight", "Relative\ntemporal weight"),
        ("effective_coefficients", "mean_effective_coefficient", "se_effective_coefficient", "Effective\ncoefficient"),
    ]
    for outcome in sorted(summary["outcome"].dropna().unique()):
        outcome_data = summary[summary["outcome"] == outcome]
        for sigma in sigmas:
            sigma_data = outcome_data[parameter_equal(outcome_data["observation_sigma"], sigma)]
            if sigma_data.empty:
                continue
            for timestep in plot_timesteps:
                step_data = sigma_data[pd.to_numeric(sigma_data["outcome_timestep"], errors="coerce") == timestep]
                if step_data.empty:
                    continue
                for slug, y_col, se_col, y_label in y_specs:
                    fig, axes = plt.subplots(
                        len(roles),
                        len(families),
                        figsize=(0.6 + len(families) * 1.55, 0.55 + len(roles) * 1.55),
                        squeeze=False,
                    )
                    for row_i, role in enumerate(roles):
                        for col_i, family in enumerate(families):
                            ax = axes[row_i, col_i]
                            panel = step_data[
                                (step_data["predictor_role"] == role) &
                                (step_data["family"] == family)
                            ]
                            params = sorted(pd.to_numeric(panel["parameter_value"], errors="coerce").dropna().unique())
                            colors = palette(list(params), family)
                            for param in params:
                                line = panel[parameter_equal(panel["parameter_value"], param)].sort_values("predictor_position")
                                if line.empty:
                                    continue
                                x = pd.to_numeric(line["predictor_position"], errors="coerce").to_numpy(dtype=float)
                                y = pd.to_numeric(line[y_col], errors="coerce").to_numpy(dtype=float)
                                se = pd.to_numeric(line[se_col], errors="coerce").to_numpy(dtype=float)
                                ax.plot(
                                    x,
                                    y,
                                    marker="o" if family == "memory" else "^",
                                    ms=2.4,
                                    lw=1.0,
                                    color=colors[float(param)],
                                    label=num_label(param),
                                )
                                if np.any(np.isfinite(se)):
                                    ax.errorbar(x, y, yerr=se, fmt="none", lw=0.6, capsize=1.5, color=colors[float(param)])
                            if slug == "relative_temporal_weights":
                                ax.axhline(1.0 / max(int(args.max_observations_before_stop), 1), color="0.75", ls="--", lw=0.7)
                                ax.set_ylim(-0.02, 1.02)
                            ax.set_xlim(0.75, int(args.max_observations_before_stop) + 0.25)
                            ax.set_xticks(range(1, int(args.max_observations_before_stop) + 1))
                            ax.grid(color="0.9", lw=0.5)
                            if row_i == 0:
                                ax.set_title("Memory lambda" if family == "memory" else "Opportunity")
                            if col_i == 0:
                                ax.set_ylabel(f"{role}\n{y_label}")
                            if row_i == len(roles) - 1:
                                ax.set_xlabel("Observation\nposition")
                            if row_i == 0 and col_i == len(families) - 1 and params:
                                ax.legend(frameon=False, fontsize=max(float(args.font_size) - 1, 5))
                    fig.suptitle(f"{outcome}, t={timestep}, sigma={num_label(sigma)}", fontsize=float(args.font_size))
                    fig.tight_layout()
                    out = fig_dir / f"bandit_simplex_{outcome}_t{timestep}_{slug}_sigma_{value_token(sigma)}.png"
                    fig.savefig(out, dpi=300)
                    plt.close(fig)
                    print(f"Saved {out}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bandit/revisit simplex temporal weights for model_jax/planning.py checkpoints.")
    parser.add_argument("tree", nargs="?", default="default", help="Tree type alias: default, bandit3, disjoint2x2, disjoint3x2.")
    parser.add_argument("--tree-size", type=int, default=None)
    parser.add_argument("--tree-type", default=None)
    parser.add_argument("--input-type", default="uniform")
    parser.add_argument("--loss-scale", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--vary-memory-lambda-values", "--vary-beta-values", default="10,20,80")
    parser.add_argument("--fixed-opportunity", type=float, default=0.0)
    parser.add_argument("--vary-opportunity-values", default="0.06,0.2,0.4")
    parser.add_argument("--fixed-memory-lambda", "--fixed-beta", type=float, default=0.0)
    parser.add_argument("--sigmas", "--sigma-list", default="0,0.5,1.0,2.0")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model-dir", default="outputs/jax_models")
    parser.add_argument("--outdir", default="analysis_outputs/bandit_simplex_weights")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--rnn-units", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--max-latent-dims", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--max-observations-before-stop", type=int, default=10)
    parser.add_argument("--rollout-mode", choices=["round_robin", "random_nodes", "policy"], default="round_robin")
    parser.add_argument("--target-path-mode", choices=["best", "chosen"], default="best")
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae")
    parser.add_argument("--allow-node-revisit", action="store_true", default=True)
    parser.add_argument("--no-allow-node-revisit", dest="allow_node_revisit", action="store_false")
    parser.add_argument("--pay-kl-on-stop", action="store_true")
    parser.add_argument("--choice-at-end-only", "--observer-only", action="store_true")
    parser.add_argument("--sampled-lambda-critic", choices=["value", "q"], default="q")
    parser.add_argument("--return-target-mode", default="sampled_lambda")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--return-target-rollouts", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--ppo-minibatches", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--target-critic-update-interval", type=int, default=100)
    parser.add_argument("--target-critic-tau", type=float, default=1.0)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--enable-reconstruction", action="store_true")
    parser.add_argument("--enable-probe", action="store_true")
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
    parser.add_argument("--node-coverage-aux-coef", type=float, default=0.0)
    parser.add_argument("--node-coverage-aux-epochs", type=int, default=0)
    parser.add_argument("--critic-huber-delta", type=float, default=10.0)
    parser.add_argument("--advantage-clip", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--use-posterior-mean", action="store_true")
    parser.add_argument("--outcomes", default="action_logit,z_mu,z_sigma")
    parser.add_argument("--z-dims", default=None)
    parser.add_argument("--min-trials-per-fit", type=int, default=25)
    parser.add_argument("--plot-max-timestep", type=int, default=5)
    parser.add_argument("--font-size", type=float, default=7.0)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--flip-nontarget-logit-predictors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit action-logit non-target predictors as -observation so positive coefficients mean evidence for the target.",
    )
    return parser


def normalize_tree_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.tree_type is None:
        args.tree_type = args.tree
    if args.tree_size is None:
        tree = jp.normalize_tree_type(args.tree_type, 2 if args.tree_type == "default" else 3)
        if tree == "legacy":
            args.tree_size = 2
        elif tree == "bandit3":
            args.tree_size = 3
        elif tree == "bandit4":
            args.tree_size = 4
        elif tree == "disjoint2x2":
            args.tree_size = 4
        elif tree == "disjoint3x2":
            args.tree_size = 6
        else:
            args.tree_size = 2
    args.tree_type = jp.normalize_tree_type(args.tree_type, int(args.tree_size))
    return args


def build_combos(args: argparse.Namespace) -> list[Combo]:
    seeds = int_values(args.seeds)
    sigmas = numeric_values(args.sigmas)
    memory_values = numeric_values(args.vary_memory_lambda_values)
    opp_values = numeric_values(args.vary_opportunity_values)
    combos: list[Combo] = []
    for memory in memory_values:
        for sigma in sigmas:
            for seed in seeds:
                combos.append(
                    Combo("memory", float(memory), float(memory), float(args.fixed_opportunity), float(sigma), int(seed))
                )
    for opp in opp_values:
        for sigma in sigmas:
            for seed in seeds:
                combos.append(
                    Combo("opportunity", float(opp), float(args.fixed_memory_lambda), float(opp), float(sigma), int(seed))
                )
    return combos


def main() -> None:
    parser = build_arg_parser()
    args = normalize_tree_args(parser.parse_args())
    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "bandit_simplex_examples.csv"
    trials_path = output_dir / "bandit_simplex_trials.csv"
    if args.plot_only:
        if not examples_path.exists():
            raise SystemExit(f"--plot-only requested, but {examples_path} does not exist.")
        examples = pd.read_csv(examples_path)
        print(f"Loaded existing examples from {examples_path}", flush=True)
    else:
        combos = build_combos(args)
        print(f"Running {len(combos)} bandit simplex rollout combo(s).", flush=True)
        all_trials: list[dict] = []
        all_examples: list[dict] = []
        for i, combo in enumerate(combos, start=1):
            config = make_config(args, combo)
            task = jp.build_task(config.tree_size, config.tree_type, config.input_type)
            print(
                f"[{i}/{len(combos)}] {combo.family}: param={combo.parameter_value:g}, "
                f"lambda={combo.memory_lambda:g}, opp={combo.opportunity_cost:g}, "
                f"sigma={combo.sigma:g}, seed={combo.seed}",
                flush=True,
            )
            rewards, transitions = rollout_combo(config, task, args)
            trial_rows, example_rows = rows_for_combo(combo, config, task, rewards, transitions, args)
            all_trials.extend(trial_rows)
            all_examples.extend(example_rows)
        trials = pd.DataFrame(all_trials)
        examples = pd.DataFrame(all_examples)
        trials.to_csv(trials_path, index=False)
        examples.to_csv(examples_path, index=False)
        print(f"Saved trial table to {trials_path}", flush=True)
        print(f"Saved example table to {examples_path}", flush=True)
    run_level, fit_metrics = fit_all(examples, args)
    summary = summarize_weights(run_level)
    run_path = output_dir / "bandit_simplex_weights_run_level.csv"
    summary_path = output_dir / "bandit_simplex_weights_summary.csv"
    metrics_path = output_dir / "bandit_simplex_fit_metrics.csv"
    run_level.to_csv(run_path, index=False)
    summary.to_csv(summary_path, index=False)
    fit_metrics.to_csv(metrics_path, index=False)
    print(f"Saved run-level weights to {run_path}", flush=True)
    print(f"Saved summary weights to {summary_path}", flush=True)
    print(f"Saved fit metrics to {metrics_path}", flush=True)
    plot_profile(summary, output_dir, args)
    print("Bandit simplex model: y_hat = bias + X @ c, c >= 0; relative=c/sum(c).", flush=True)


if __name__ == "__main__":
    main()
