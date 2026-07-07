#!/usr/bin/env python3
"""Plot revisit JAX latent variance and terminal action logits by observation timestep.

This is a narrow companion to ``plot_revisit_latent_density_gaussian_pga_jax.py``.
It loads revisit-enabled JAX checkpoints, simulates trials, and writes four
summary figures:

1. beta rows x sigma columns: posterior variance of z_t vs observation timestep
2. opportunity rows x sigma columns: posterior variance of z_t vs observation timestep
3. beta rows x sigma columns: terminal action logit vs observation timestep
4. opportunity rows x sigma columns: terminal action logit vs observation timestep
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR), str(REPO_ROOT / "model_jax")):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis import plot_revisit_latent_density_jax as base  # noqa: E402
from model_jax import planning as jp  # noqa: E402


PLOT_FONT_SIZE = 7
PANEL_SIZE_IN = 33.0 / 25.4
EPS = 1e-8


def parse_list(raw: str | Iterable, typ=float) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        pieces = re.split(r"[,\s]+", raw.strip())
    else:
        pieces = []
        for item in raw:
            pieces.extend(re.split(r"[,\s]+", str(item).strip()))
    return [typ(piece) for piece in pieces if piece != ""]


def value_token(value) -> str:
    text = f"{float(value):g}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value)
    return text.replace("-", "m").replace(".", "p")


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


def make_schedule(beta: float) -> jp.ScheduleValues:
    return jp.ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(1.0 / float(beta), dtype=jnp.float32),
        current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
        node_coverage_aux_coef=jnp.asarray(0.0, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )


def terminal_action_logit(action_output: np.ndarray, mode: str) -> np.ndarray:
    probs = np.asarray(action_output, dtype=float)
    probs = np.where(np.isfinite(probs), probs, np.nan)
    if probs.ndim != 2 or probs.shape[1] < 2:
        return np.full((probs.shape[0],), np.nan, dtype=float)
    if probs.shape[1] == 2:
        logit = np.log(np.clip(probs[:, 0], EPS, 1.0)) - np.log(np.clip(probs[:, 1], EPS, 1.0))
        if mode == "abs_path1_minus_path2":
            logit = np.abs(logit)
        return logit
    log_probs = np.log(np.clip(probs, EPS, 1.0))
    sorted_log_probs = np.sort(log_probs, axis=1)
    margin = sorted_log_probs[:, -1] - sorted_log_probs[:, -2]
    return margin


def rollout_combo_rows(
    *,
    model: jp.PlanningVAE,
    params,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    batch_size: int,
    seed: int,
    beta: float,
    max_observations_before_stop: int,
    force_first_observe_node: int,
    action_logit_mode: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    schedule = make_schedule(beta)
    reward_feature_dim = int(model.reward_feature_dim())
    visited_feature_dim = int(model.visited_feature_dim())
    num_steps = int(max_observations_before_stop) + 1
    for batch_i, start in enumerate(range(0, rewards.shape[0], int(batch_size))):
        batch_rewards = rewards[start : start + int(batch_size)]
        carry = jp.initial_carry(
            batch_rewards.shape[0],
            task,
            int(model.rnn_units),
            reward_feature_dim,
            visited_feature_dim,
        )
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        rng = jax.random.PRNGKey(int(seed) + 10_000 * int(batch_i))
        stopped = np.zeros(batch_rewards.shape[0], dtype=bool)
        for step_i in range(num_steps):
            rng, step_rng = jax.random.split(rng)
            forced_action = None
            if step_i == 0 and 1 <= int(force_first_observe_node) <= int(task.num_nodes):
                forced_action = jnp.full(
                    (batch_rewards.shape[0],),
                    int(force_first_observe_node) - 1,
                    dtype=jnp.int32,
                )
            carry, trans = model.apply(
                {"params": params},
                carry,
                step_rng,
                schedule,
                forced_action,
                True,
                False,
                False,
                method=jp.PlanningVAE.__call__,
            )
            trans = jax.device_get(trans)
            is_observe = np.asarray(trans.is_observe, dtype=float) > 0.5
            include = (~stopped) & is_observe
            if np.any(include):
                z_var = np.exp(np.clip(np.asarray(trans.z_logvar, dtype=float), -30.0, 30.0))
                z_var_mean = np.mean(z_var, axis=1)
                logits = terminal_action_logit(np.asarray(trans.action_output, dtype=float), action_logit_mode)
                node_index = np.asarray(trans.node_index, dtype=int)
                observed_reward = np.asarray(trans.expanded_reward, dtype=float)
                for local_i in np.where(include)[0]:
                    rows.append(
                        {
                            "trial_id": int(start + local_i),
                            "timestep": int(step_i + 1),
                            "observed_node": int(node_index[local_i] + 1),
                            "sampled_observed_reward": float(observed_reward[local_i]),
                            "z_variance": float(z_var_mean[local_i]),
                            "action_logit": float(logits[local_i]),
                        }
                    )
            stopped |= np.asarray(trans.is_stop, dtype=float) > 0.5
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    seed_summary = (
        raw.groupby(["family", "parameter_value", "sigma", "seed", "timestep"], dropna=False)
        .agg(
            z_variance=("z_variance", "mean"),
            action_logit=("action_logit", "mean"),
            n_trials=("z_variance", "size"),
        )
        .reset_index()
    )
    rows = []
    for keys, group in seed_summary.groupby(["family", "parameter_value", "sigma", "timestep"], dropna=False):
        family, parameter_value, sigma, timestep = keys
        out = {
            "family": family,
            "parameter_value": float(parameter_value),
            "sigma": float(sigma),
            "timestep": int(timestep),
            "n_seeds": int(group["seed"].nunique()),
            "n_trials": int(group["n_trials"].sum()),
        }
        for metric in ["z_variance", "action_logit"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            out[metric] = float(np.mean(values)) if len(values) else np.nan
            out[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        rows.append(out)
    return seed_summary, pd.DataFrame(rows)


def metric_limits(summary: pd.DataFrame, metric: str, sem_col: str) -> tuple[float, float]:
    y = pd.to_numeric(summary[metric], errors="coerce").to_numpy(dtype=float)
    sem = pd.to_numeric(summary[sem_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return (0.0, 1.0)
    lo = float(np.nanmin(y[finite] - sem[finite]))
    hi = float(np.nanmax(y[finite] + sem[finite]))
    if metric == "z_variance":
        lo = min(0.0, lo)
    if metric == "action_logit":
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if abs(hi - lo) < 1e-9:
        pad = 0.5 if metric == "action_logit" else 0.05
    else:
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def plot_grid(
    summary: pd.DataFrame,
    *,
    family: str,
    parameters: list[float],
    sigmas: list[float],
    metric: str,
    ylabel: str,
    row_label: str,
    outpath: Path,
) -> None:
    data = summary[summary["family"] == family].copy()
    if data.empty:
        return
    configure_plotting()
    n_rows = max(1, len(parameters))
    n_cols = max(1, len(sigmas))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * PANEL_SIZE_IN + 0.95, n_rows * PANEL_SIZE_IN + 0.85),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    sem_col = f"{metric}_sem"
    y_min, y_max = metric_limits(data, metric, sem_col)
    x_values = pd.to_numeric(data["timestep"], errors="coerce").dropna().astype(int)
    x_max = int(x_values.max()) if len(x_values) else 1
    for row_i, parameter in enumerate(parameters):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = data[
                np.isclose(pd.to_numeric(data["parameter_value"], errors="coerce"), float(parameter))
                & np.isclose(pd.to_numeric(data["sigma"], errors="coerce"), float(sigma))
            ].sort_values("timestep")
            if panel.empty:
                ax.axis("off")
                continue
            ax.errorbar(
                panel["timestep"],
                panel[metric],
                yerr=panel[sem_col],
                color="#2b2b2b",
                marker="o",
                markersize=2.4,
                linewidth=0.9,
                capsize=1.4,
            )
            if metric == "action_logit":
                ax.axhline(0.0, color="#9e9e9e", linewidth=0.45, zorder=0)
            if row_i == 0:
                ax.set_title(f"sigma={sigma:g}", pad=2)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}={parameter:g}")
            else:
                ax.set_yticklabels([])
            if row_i != n_rows - 1:
                ax.set_xticklabels([])
            ax.set_xlim(0.8, x_max + 0.2)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(1, x_max + 1, 1))
            ax.tick_params(length=2, pad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.supxlabel("current observation\ntimestep", y=0.025)
    fig.supylabel(ylabel, x=0.01)
    fig.tight_layout(rect=(0.04, 0.05, 1.0, 0.98), h_pad=0.45, w_pad=0.35)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_family(
    *,
    family: str,
    parameter_values: list[float],
    fixed_beta: float | None,
    fixed_opportunity: float | None,
    args: argparse.Namespace,
    task: jp.TaskSpec,
) -> list[pd.DataFrame]:
    rows = []
    checkpoint_root = Path(args.checkpoint_root)
    for parameter in parameter_values:
        beta = float(parameter) if family == "vary_beta" else float(fixed_beta)
        opportunity = float(parameter) if family == "vary_opportunity" else float(fixed_opportunity)
        for sigma in parse_list(args.sigmas, float):
            for seed in parse_list(args.seeds, int):
                checkpoint, message = base.find_revisit_checkpoint(
                    checkpoint_root,
                    lambda_value=float(args.lambda_value),
                    alpha=float(args.alpha),
                    beta=beta,
                    opportunity_cost=opportunity,
                    seed=int(seed),
                    task=task,
                    tree_size=int(args.tree_size),
                    rnn_dim=int(args.rnn_dim),
                    latent_dim=int(args.latent_dim),
                    expansion_decision_version=str(args.expansion_decision_version),
                    model_variant=str(args.model_variant),
                    max_observations_before_stop=int(args.max_observations_before_stop),
                    observation_sigma=float(sigma),
                    kl_start_multiplier=args.kl_start_multiplier,
                    kl_annealing_epochs=args.kl_annealing_epochs,
                )
                if checkpoint is None:
                    print(
                        f"Skipping {family}, beta={beta:g}, opp={opportunity:g}, "
                        f"sigma={sigma:g}, seed={seed}: {message}"
                    )
                    continue
                model, params = base.build_model_and_params(
                    checkpoint,
                    task=task,
                    lambda_value=float(args.lambda_value),
                    alpha=float(args.alpha),
                    beta=beta,
                    opportunity_cost=opportunity,
                    rnn_dim=int(args.rnn_dim),
                    latent_dim=int(args.latent_dim),
                    expansion_decision_version=str(args.expansion_decision_version),
                    model_variant=str(args.model_variant),
                    max_observations_before_stop=int(args.max_observations_before_stop),
                    observation_sigma=float(sigma),
                )
                rewards = base.sample_rewards(
                    int(args.n_trials),
                    task,
                    int(args.analysis_seed_offset)
                    + 1_000_003 * int(seed)
                    + int(round(1000.0 * beta))
                    + int(round(100_000.0 * opportunity))
                    + int(round(10_000.0 * float(sigma))),
                )
                combo = rollout_combo_rows(
                    model=model,
                    params=params,
                    task=task,
                    rewards=rewards,
                    batch_size=int(args.batch_size),
                    seed=int(args.analysis_seed_offset) + 97 * int(seed),
                    beta=beta,
                    max_observations_before_stop=int(args.max_observations_before_stop),
                    force_first_observe_node=int(args.force_first_observe_node),
                    action_logit_mode=str(args.action_logit_mode),
                )
                if combo.empty:
                    continue
                combo["family"] = family
                combo["parameter_value"] = float(parameter)
                combo["beta"] = beta
                combo["opportunity"] = opportunity
                combo["sigma"] = float(sigma)
                combo["seed"] = int(seed)
                combo["checkpoint"] = checkpoint.name
                rows.append(combo)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vary-beta-values", default="10,20,80")
    parser.add_argument("--vary-opportunity-values", default="0.06,0.2,0.4")
    parser.add_argument("--beta-sweep-opportunity", type=float, default=0.0)
    parser.add_argument("--opportunity-sweep-beta", type=float, default=100000.0)
    parser.add_argument("--sigmas", default="0,0.5,1,2")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--lambda-value", "--lambdas", dest="lambda_value", type=float, default=100.0)
    parser.add_argument("--alpha", "--alphas", dest="alpha", type=float, default=0.0)
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform", choices=["uniform", "binary"])
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae", choices=["vae", "rnn"])
    parser.add_argument("--rnn-dim", "--rnn-dims", dest="rnn_dim", type=int, default=16)
    parser.add_argument("--latent-dim", "--latent-dims", dest="latent_dim", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-observations-before-stop", type=int, default=10)
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--outdir", default="analysis_outputs/revisit_latent_variance_action_logits_jax")
    parser.add_argument("--analysis-seed-offset", type=int, default=920_000)
    parser.add_argument("--force-first-observe-node", type=int, default=1)
    parser.add_argument("--kl-start-multiplier", type=float, default=None)
    parser.add_argument("--kl-annealing-epochs", type=int, default=None)
    parser.add_argument(
        "--action-logit-mode",
        choices=["path1_minus_path2", "abs_path1_minus_path2", "max_minus_second"],
        default="path1_minus_path2",
        help=(
            "For two-path tasks, plot log P(path1)-log P(path2), optionally "
            "absolute value. For larger tasks, max_minus_second is used."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate PNGs from existing latent_variance_action_logit_raw.csv in the output folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beta_values = parse_list(args.vary_beta_values, float)
    opportunity_values = parse_list(args.vary_opportunity_values, float)
    sigmas = parse_list(args.sigmas, float)
    task = jp.build_task(int(args.tree_size), str(args.tree_type), str(args.input_type))
    outdir = (
        Path(args.outdir)
        / f"{int(args.tree_size)}n{task.tree_name_suffix or '_default'}"
        / (
            f"beta_{'_'.join(value_token(x) for x in beta_values)}"
            f"_opp_{'_'.join(value_token(x) for x in opportunity_values)}"
            f"_sigma_{'_'.join(value_token(x) for x in sigmas)}"
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "latent_variance_action_logit_raw.csv"
    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot --plot-only because {raw_path} does not exist.")
        raw = pd.read_csv(raw_path)
    else:
        all_rows = []
        all_rows.extend(
            run_family(
                family="vary_beta",
                parameter_values=beta_values,
                fixed_beta=None,
                fixed_opportunity=float(args.beta_sweep_opportunity),
                args=args,
                task=task,
            )
        )
        all_rows.extend(
            run_family(
                family="vary_opportunity",
                parameter_values=opportunity_values,
                fixed_beta=float(args.opportunity_sweep_beta),
                fixed_opportunity=None,
                args=args,
                task=task,
            )
        )
        raw = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        raw.to_csv(raw_path, index=False)
    seed_summary, summary = summarize(raw)
    seed_summary.to_csv(outdir / "latent_variance_action_logit_seed_summary.csv", index=False)
    summary.to_csv(outdir / "latent_variance_action_logit_summary.csv", index=False)
    plot_grid(
        summary,
        family="vary_beta",
        parameters=beta_values,
        sigmas=sigmas,
        metric="z_variance",
        ylabel="mean posterior\nVar(z_t)",
        row_label="beta",
        outpath=outdir / "vary_beta_z_variance_by_observation_timestep.png",
    )
    plot_grid(
        summary,
        family="vary_opportunity",
        parameters=opportunity_values,
        sigmas=sigmas,
        metric="z_variance",
        ylabel="mean posterior\nVar(z_t)",
        row_label="opp",
        outpath=outdir / "vary_opportunity_z_variance_by_observation_timestep.png",
    )
    plot_grid(
        summary,
        family="vary_beta",
        parameters=beta_values,
        sigmas=sigmas,
        metric="action_logit",
        ylabel="terminal action\nlogit",
        row_label="beta",
        outpath=outdir / "vary_beta_action_logit_by_observation_timestep.png",
    )
    plot_grid(
        summary,
        family="vary_opportunity",
        parameters=opportunity_values,
        sigmas=sigmas,
        metric="action_logit",
        ylabel="terminal action\nlogit",
        row_label="opp",
        outpath=outdir / "vary_opportunity_action_logit_by_observation_timestep.png",
    )
    print(f"Saved latent variance/action-logit summaries to {outdir}")


if __name__ == "__main__":
    main()
