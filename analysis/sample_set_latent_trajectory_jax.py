#!/usr/bin/env python3
"""Sample-set latent trajectory plots for two-node revisit JAX models.

For each actual node-1 x node-2 reward combination, this script generates
multiple noisy observation streams per node, runs the trained revisit policy,
and records the latent posterior trajectory at each observed reward.

The default checkpoint grid matches:

  beta sweep: beta = 10,20,80; opportunity = 0
  opportunity sweep: beta = 100000; opportunity = 0.06,0.2,0.4
  sigmas = 0,0.5,1,2; seeds = 1,2,3

Outputs are one figure per model parameter and sigma:

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
"""

from __future__ import annotations

import argparse
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
    summary["prior_normalized_z_mu_mean"] = (
        pd.to_numeric(summary["z_mu_mean"], errors="coerce")
        - pd.to_numeric(summary["prior_mu_mean"], errors="coerce")
    ) / prior_sigma
    # The original prior-normalized center above uses the prior of the mean
    # distribution.  Keep it for backward-compatible plots, but also store the
    # mean of per-row prior-normalized values for SEM diagnostics.
    summary["prior_normalized_z_mu_sample_mean"] = summary["prior_normalized_z_mu_row_mean"]
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
                ax.set_title(f"R2={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"R1={node1:g}\n{latent_mu_axis_label(data)}", labelpad=0)
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
                ax.set_title(f"R2={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"R1={node1:g}\n{latent_mu_axis_label(data)}")
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
                ax.set_title(f"R2={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"R1={node1:g}")
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
    if mu_col.startswith("prior_normalized") and "prior_normalized_z_mu_sample_mean" in data.columns:
        mu_col = "prior_normalized_z_mu_sample_mean"
    sem_col = latent_mu_sem_column(data)
    if not sem_col or sem_col not in data.columns:
        return
    node1_values = reward_values_for_plot(data, "node1_reward")
    node2_values = reward_values_for_plot(data, "node2_reward")
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
                ax.set_title(f"R2={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"R1={node1:g}")
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
                ax.set_title(f"R2={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"R1={node1:g}")
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
    fig.supxlabel("current observation timestep", y=0.02)
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
                ax.set_title(f"R2={node2:g}", pad=1)
            if col_i == 0:
                ax.set_ylabel(f"R1={node1:g}")
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
    fig.supxlabel("current observation timestep", y=0.02)
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
    shade_error_scale: float = 25.0,
    min_samples_per_dot: int = 1,
) -> None:
    summary = filter_summary_for_min_samples(summary, int(min_samples_per_dot))
    if summary.empty:
        return
    for family, family_df in summary.groupby("family", sort=False):
        family_dir = outdir / str(family)
        for parameter_value in sorted(pd.to_numeric(family_df["parameter_value"], errors="coerce").dropna().unique()):
            param_dir = family_dir / (
                f"beta_{value_token(parameter_value)}"
                if str(family) == "vary_beta"
                else f"opp_{value_token(parameter_value)}"
            )
            for sigma in sorted(pd.to_numeric(family_df["sigma"], errors="coerce").dropna().unique()):
                sigma_dir = param_dir / f"sigma_{value_token(sigma)}"
                if "line3d" in latent_plot_types:
                    plot_latent_3d_grid(
                        summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "latent_mu_sigma_3d_by_node_rewards.png",
                    )
                if "shade" in latent_plot_types:
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
                if "contour" in latent_plot_types:
                    plot_latent_contour_grid(
                        summary,
                        family=str(family),
                        parameter_value=float(parameter_value),
                        sigma=float(sigma),
                        outpath=sigma_dir / "latent_mu_sigma_contour_by_node_rewards.png",
                    )
                plot_action_logit_grid(
                    summary,
                    family=str(family),
                    parameter_value=float(parameter_value),
                    sigma=float(sigma),
                    outpath=sigma_dir / "action_logit_by_node_rewards.png",
                )
                plot_action_logit_sem_grid(
                    summary,
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
        print(
            f"Writing seed-specific latent trajectory plots for seed={seed} to {seed_dir}",
            flush=True,
        )
        write_plots(
            seed_summary,
            seed_dir,
            latent_plot_types,
            shade_error_scale=shade_error_scale,
            min_samples_per_dot=int(min_samples_per_dot),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", nargs="?", default="default")
    parser.add_argument("--vary-beta-values", default="10,20,80")
    parser.add_argument("--vary-opportunity-values", default="0.06,0.2,0.4")
    parser.add_argument("--beta-sweep-opportunity", type=float, default=0.0)
    parser.add_argument("--opportunity-sweep-beta", type=float, default=100000.0)
    parser.add_argument("--sigmas", default="0,0.5,1,2")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--lambda-value", "--lambdas", dest="lambda_value", default="100.0")
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
    parser.add_argument("--outdir", default="analysis_outputs/sample_set_latent_trajectory_jax")
    parser.add_argument("--n-sample-sets", type=int, default=50)
    parser.add_argument("--n-reward-combinations", type=int, default=0)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--sampled-lambda-critic", choices=["q", "value"], default="q")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
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
        choices=["path1_minus_path2", "abs_path1_minus_path2", "max_minus_second"],
        default="path1_minus_path2",
    )
    parser.add_argument(
        "--latent-trajectory-plot-types",
        default="line3d,shade",
        help=(
            "Comma/space separated latent trajectory plot types: line3d, shade, "
            "contour, or any combination. 'ribbon' is accepted as an alias for shade."
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
        "--no-seed-plots",
        action="store_true",
        help="Only write pooled plots; skip seed_<id> plot folders.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.tree = sample_base.normalize_tree_name(args.tree)
    args.vary_beta_values = sample_base.parse_values(args.vary_beta_values, float)
    args.vary_opportunity_values = sample_base.parse_values(args.vary_opportunity_values, float)
    args.sigmas = sample_base.parse_values(args.sigmas, float)
    args.seeds = sample_base.parse_values(args.seeds, int)
    args.lambda_value = float((sample_base.parse_values(args.lambda_value, float) or [100.0])[0])
    args.alpha = float((sample_base.parse_values(args.alpha, float) or [0.0])[0])
    args.rnn_units = int((sample_base.parse_values(args.rnn_units, int) or [16])[0])
    args.latent_dim = int((sample_base.parse_values(args.latent_dim, int) or [1])[0])
    plot_types = set(str(x).lower() for x in sample_base.parse_values(args.latent_trajectory_plot_types, str))
    if "ribbon" in plot_types:
        plot_types.remove("ribbon")
        plot_types.add("shade")
    valid_types = {"line3d", "shade", "contour"}
    bad_types = sorted(plot_types - valid_types)
    if bad_types:
        raise ValueError(f"Unknown --latent-trajectory-plot-types value(s): {bad_types}")
    args.latent_trajectory_plot_types = plot_types or {"line3d", "shade"}
    args.parameter_combos = []
    for beta in args.vary_beta_values:
        args.parameter_combos.append(("vary_beta", "beta", float(beta), float(beta), float(args.beta_sweep_opportunity)))
    for opp in args.vary_opportunity_values:
        args.parameter_combos.append(
            ("vary_opportunity", "opportunity", float(opp), float(args.opportunity_sweep_beta), float(opp))
        )
    return args


def output_dir(args: argparse.Namespace) -> Path:
    label = (
        f"{sample_base.normalize_tree_name(args.tree)}"
        f"_vary_beta_{values_token(args.vary_beta_values)}"
        f"_vary_opp_{values_token(args.vary_opportunity_values)}"
    )
    if int(args.force_first_observe_node) > 0:
        label += f"_force_first_node_{int(args.force_first_observe_node)}"
    return Path(args.outdir) / label


def main() -> None:
    args = normalize_args(parse_args())
    outdir = output_dir(args)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "latent_trajectory_rows.csv"
    summary_path = outdir / "latent_trajectory_summary.csv"
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
                    parts.append(
                        run_one(
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
    summary = summarize_trajectories(raw)
    summary.to_csv(summary_path, index=False)
    print(f"Saved trajectory summary to {summary_path}", flush=True)
    write_plots(
        summary,
        outdir,
        set(args.latent_trajectory_plot_types),
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
