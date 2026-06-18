#!/usr/bin/env python
"""Plot revisit-task JAX latent z0/z1 densities by node value role.

This script is intentionally narrower than analyze_latent_angle_planning_jax.py:
it loads revisit-enabled JAX checkpoints, simulates trials with node revisits
allowed, and writes only z_mu_0-vs-z_mu_1 density plots. Each output PNG is for
one seed/beta/opportunity/lambda/sigma combination, one timestep, and one node
role: whether the observed node was the better or worse of the two nodes in that
trial. It also writes one role-colored density plot per timestep. Density
contours are empirical KDEs over posterior means and plotted dots are centroids,
grouped by either true reward or better/worse node role. Rows where the model
already stopped before the observation are excluded.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_JAX_DIR = REPO_ROOT / "model_jax"
for path in (str(REPO_ROOT), str(MODEL_JAX_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
from flax import serialization

try:
    from model_jax import planning as jp
except ModuleNotFoundError:
    import planning as jp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument(
        "--scalers",
        "--betas",
        dest="beta_values",
        nargs="+",
        type=float,
        required=True,
        help="Reward/action/critic scalers; --betas is accepted as a legacy alias.",
    )
    parser.add_argument("--lambdas", dest="lambda_values", nargs="+", type=float, required=True)
    parser.add_argument("--opportunity-costs", "--opportunity-cost", dest="opportunity_costs", nargs="+", type=float, default=[0.0])
    parser.add_argument("--sigmas", "--observation-sigmas", dest="observation_sigmas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--rnn-dims", nargs="+", type=int, default=[16])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[2])
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", default="analysis_outputs/revisit_latent_density_jax")
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform", choices=["uniform", "binary"])
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae", choices=["vae", "rnn"])
    parser.add_argument("--max-observations-before-stop", type=int, default=10)
    parser.add_argument("--latent-density-grid-n", type=int, default=120)
    parser.add_argument("--max-density-points", type=int, default=1500)
    parser.add_argument("--analysis-seed-offset", type=int, default=300_000)
    parser.add_argument("--kl-start-multiplier", type=float, default=None)
    parser.add_argument("--kl-annealing-epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Accepted for CLI symmetry; plotting uses JAX default device.")
    return parser.parse_args()


def file_token(value) -> str:
    text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _extract_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1).rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def _extract_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _close(found: Optional[float], expected: float, tol: float = 1e-8) -> bool:
    return found is not None and abs(float(found) - float(expected)) <= tol


def normalize_variant_for_file(variant: str) -> str:
    return "vae" if str(variant).strip().lower() in {"jax", "jax_vae"} else str(variant).strip().lower()


def find_revisit_checkpoint(
    root: Path,
    *,
    lambda_value: float,
    alpha: float,
    beta: float,
    opportunity_cost: float,
    seed: int,
    task: jp.TaskSpec,
    tree_size: int,
    rnn_dim: int,
    latent_dim: int,
    expansion_decision_version: str,
    model_variant: str,
    max_observations_before_stop: int,
    observation_sigma: float,
    kl_start_multiplier: Optional[float],
    kl_annealing_epochs: Optional[int],
) -> Tuple[Optional[Path], str]:
    if not root.exists():
        return None, f"checkpoint root does not exist: {root}"
    expected_tree = f"{tree_size}n{task.tree_name_suffix}"
    expected_revisit = f"_revisit_maxobs_{int(max_observations_before_stop)}"
    expected_sigma = f"_obs_sigma_{observation_sigma:g}"
    variant = normalize_variant_for_file(model_variant)
    matches = []
    for path in root.rglob("*.msgpack"):
        name = path.name
        found_lambda = _extract_float(r"lambda_([0-9eE.+-]+)", name)
        found_scaler = _extract_float(r"scaler_([0-9eE.+-]+)", name)
        found_beta = _extract_float(r"beta_([0-9eE.+-]+)", name)
        beta_match = _close(found_scaler, beta) or _close(found_beta, beta)
        reversed_legacy_match = _close(found_lambda, beta) and _close(found_beta, lambda_value)
        if not ((_close(found_lambda, lambda_value) and beta_match) or reversed_legacy_match):
            continue
        if not _close(_extract_float(r"alpha_([0-9eE.+-]+)", name), alpha):
            continue
        if not _close(_extract_float(r"opportunity_([0-9eE.+-]+)", name), opportunity_cost):
            continue
        if _extract_int(r"seed_([0-9]+)", name) != int(seed):
            continue
        if expected_tree not in name or expected_revisit not in name:
            continue
        found_sigma = _extract_float(r"obs_sigma_([0-9eE.+-]+)", name)
        if abs(float(observation_sigma)) > 1e-12:
            if not _close(found_sigma, float(observation_sigma)):
                continue
        elif found_sigma is not None and not _close(found_sigma, 0.0):
            continue
        if _extract_int(r"rnn_([0-9]+)", name) != int(rnn_dim):
            continue
        if _extract_int(r"latent_([0-9]+)", name) != int(latent_dim):
            continue
        if f"expansion_{expansion_decision_version}" not in name:
            continue
        if f"variant_{variant}" not in name:
            continue
        if kl_start_multiplier is not None:
            found_klstart = _extract_float(r"klstart_([0-9eE.+-]+)", name)
            if not _close(found_klstart, float(kl_start_multiplier)):
                continue
        if kl_annealing_epochs is not None:
            found_klanneal = _extract_int(r"klanneal_([0-9]+)", name)
            if found_klanneal != int(kl_annealing_epochs):
                continue
        matches.append(path)
    if not matches:
        return None, f"no revisit checkpoint matched tree={expected_tree}, revisit={expected_revisit}, sigma={observation_sigma:g}"
    matches.sort(key=lambda p: -p.stat().st_mtime)
    return matches[0], f"matched {matches[0].name}"


def build_model_and_params(
    checkpoint_path: Path,
    *,
    task: jp.TaskSpec,
    lambda_value: float,
    alpha: float,
    beta: float,
    opportunity_cost: float,
    rnn_dim: int,
    latent_dim: int,
    expansion_decision_version: str,
    model_variant: str,
    max_observations_before_stop: int,
    observation_sigma: float,
) -> Tuple[jp.PlanningVAE, object]:
    path_tuple = tuple(tuple(float(v) for v in row) for row in task.path_map)
    reward_tuple = tuple(float(v) for v in task.reward_values)
    model = jp.PlanningVAE(
        rnn_units=int(rnn_dim),
        latent_dim=int(latent_dim),
        time_steps=task.num_nodes,
        num_paths=task.num_paths,
        path_map=path_tuple,
        reward_values=reward_tuple,
        reward_norm_value=float(task.reward_norm),
        expansion_decision_version=jp.normalize_expansion_decision_version(expansion_decision_version),
        use_autoencoder=(normalize_variant_for_file(model_variant) != "rnn"),
        enable_reconstruction=False,
        enable_probe=False,
        allow_node_revisit=True,
        max_observations_before_stop=int(max_observations_before_stop),
        opportunity_cost=float(opportunity_cost),
        observation_sigma=float(observation_sigma),
        lambda_=float(lambda_value),
        alpha=float(alpha),
        beta=float(beta),
    )
    reward_feature_dim = jp.reward_feature_dim_for_sigma(observation_sigma)
    dummy = jp.initial_carry(1, task, int(rnn_dim), reward_feature_dim)
    schedule = jp.ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(1.0 / float(beta), dtype=jnp.float32),
        current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )
    params_template = model.init(jax.random.PRNGKey(0), dummy, jax.random.PRNGKey(1), schedule, None, False)["params"]
    params = serialization.from_bytes(params_template, checkpoint_path.read_bytes())
    return model, params


def sample_rewards(n_trials: int, task: jp.TaskSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reward_values = np.asarray(task.reward_values, dtype=np.float32)
    return rng.choice(reward_values, size=(int(n_trials), int(task.num_nodes))).astype(np.float32)


def rollout_revisit_rows(
    *,
    model: jp.PlanningVAE,
    params,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    batch_size: int,
    seed: int,
    beta: float,
    max_observations_before_stop: int,
) -> pd.DataFrame:
    rows = []
    schedule = jp.ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(1.0 / float(beta), dtype=jnp.float32),
        current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )
    reward_feature_dim = jp.reward_feature_dim_for_sigma(model.observation_sigma)
    num_steps = int(max_observations_before_stop) + 1
    for batch_i, start in enumerate(range(0, rewards.shape[0], batch_size)):
        batch_rewards = rewards[start:start + batch_size]
        carry = jp.initial_carry(batch_rewards.shape[0], task, model.rnn_units, reward_feature_dim)
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        rng = jax.random.PRNGKey(seed + batch_i)
        stopped = np.zeros(batch_rewards.shape[0], dtype=bool)
        for step_i in range(num_steps):
            rng, step_rng = jax.random.split(rng)
            carry, trans = model.apply(
                {"params": params},
                carry,
                step_rng,
                schedule,
                None,
                True,
                False,
                False,
                method=jp.PlanningVAE.__call__,
            )
            trans = jax.device_get(trans)
            node_index = np.asarray(trans.node_index, dtype=int)
            is_observe = np.asarray(trans.is_observe, dtype=float) > 0.5
            is_stop = np.asarray(trans.is_stop, dtype=float) > 0.5
            sampled_observed_reward = np.asarray(trans.expanded_reward, dtype=float)
            z_mu = np.asarray(trans.z_mu, dtype=float)
            z_sigma = np.exp(0.5 * np.clip(np.asarray(trans.z_logvar, dtype=float), -10.0, 10.0))
            include = (~stopped) & is_observe & (node_index >= 0)
            for local_i in np.where(include)[0]:
                node_i = int(node_index[local_i])
                if task.num_nodes != 2:
                    continue
                other_i = 1 - node_i
                node_reward = float(batch_rewards[local_i, node_i])
                other_reward = float(batch_rewards[local_i, other_i])
                if math.isclose(node_reward, other_reward):
                    continue
                rows.append(
                    {
                        "trial_id": int(start + local_i),
                        "timestep": int(step_i + 1),
                        "observed_node": int(node_i + 1),
                        "node_role": "better" if node_reward > other_reward else "worse",
                        "actual_node_reward": node_reward,
                        "node_reward": node_reward,
                        "sampled_observed_reward": float(sampled_observed_reward[local_i]),
                        "other_node_reward": other_reward,
                        "z_mu_0": float(z_mu[local_i, 0]),
                        "z_mu_1": float(z_mu[local_i, 1]),
                        "z_sigma_0": float(z_sigma[local_i, 0]),
                        "z_sigma_1": float(z_sigma[local_i, 1]),
                    }
                )
            stopped |= is_stop
    return pd.DataFrame(rows)


def axis_limits(df: pd.DataFrame) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if df.empty:
        return None
    x = pd.to_numeric(df["z_mu_0"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["z_mu_1"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return None
    xmin, xmax = float(np.min(x[ok])), float(np.max(x[ok]))
    ymin, ymax = float(np.min(y[ok])), float(np.max(y[ok]))
    xpad = max((xmax - xmin) * 0.08, 0.25)
    ypad = max((ymax - ymin) * 0.08, 0.25)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def empirical_mu_kde_density(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    max_points: int,
    seed: int,
) -> Optional[np.ndarray]:
    cols = ["z_mu_0", "z_mu_1"]
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=seed)
    mu_x = data["z_mu_0"].to_numpy(dtype=float)
    mu_y = data["z_mu_1"].to_numpy(dtype=float)
    panel_span = min(float(x_grid[-1] - x_grid[0]), float(y_grid[-1] - y_grid[0]))
    min_bandwidth = max(0.06 * panel_span, 0.08)
    if len(data) > 1:
        pooled_sd = math.sqrt(
            max(float(np.nanvar(mu_x, ddof=1)) + float(np.nanvar(mu_y, ddof=1)), 0.0) / 2.0
        )
        bandwidth = pooled_sd * (float(len(data)) ** (-1.0 / 6.0))
    else:
        bandwidth = min_bandwidth
    bandwidth = max(float(bandwidth), min_bandwidth, 1e-3)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(xx, dtype=float)
    for start in range(0, len(data), 200):
        sl = slice(start, start + 200)
        dx = (xx[None, :, :] - mu_x[sl, None, None]) / bandwidth
        dy = (yy[None, :, :] - mu_y[sl, None, None]) / bandwidth
        norm = 1.0 / (2.0 * np.pi * bandwidth * bandwidth)
        density += np.sum(norm * np.exp(-0.5 * (dx * dx + dy * dy)), axis=0)
    density /= float(len(data))
    return density


def positive_contour_levels(density: np.ndarray) -> Optional[np.ndarray]:
    values = density[np.isfinite(density) & (density > 0)]
    if len(values) == 0:
        return None
    max_density = float(np.max(values))
    min_density = float(np.min(values))
    levels = np.unique(max_density * np.linspace(0.12, 0.72, 5))
    levels = levels[(levels > min_density) & (levels < max_density)]
    return levels if len(levels) > 0 else None


def reward_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """One plotted marker per true observed-node reward value."""
    reward_col = "actual_node_reward" if "actual_node_reward" in df.columns else "node_reward"
    work = df[[reward_col, "z_mu_0", "z_mu_1"]].copy()
    work[reward_col] = pd.to_numeric(work[reward_col], errors="coerce")
    work["z_mu_0"] = pd.to_numeric(work["z_mu_0"], errors="coerce")
    work["z_mu_1"] = pd.to_numeric(work["z_mu_1"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(columns=["actual_node_reward", "z_mu_0", "z_mu_1", "n"])
    out = (
        work.groupby(reward_col, as_index=False)
        .agg(z_mu_0=("z_mu_0", "mean"), z_mu_1=("z_mu_1", "mean"), n=("z_mu_0", "size"))
        .rename(columns={reward_col: "actual_node_reward"})
        .sort_values("actual_node_reward")
    )
    return out


def reward_contour_seed(timestep: float, reward_value: float, role: str) -> int:
    reward_token = int(round((float(reward_value) + 10.0) * 100.0))
    return 17 + int(round(float(timestep))) * 101 + reward_token * 13 + (0 if role == "better" else 1)


def role_contour_seed(timestep: float, role: str) -> int:
    return 7919 + int(round(float(timestep))) * 101 + (0 if role == "better" else 1)


def role_centroids(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["node_role", "z_mu_0", "z_mu_1"]].copy()
    work["z_mu_0"] = pd.to_numeric(work["z_mu_0"], errors="coerce")
    work["z_mu_1"] = pd.to_numeric(work["z_mu_1"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(columns=["node_role", "z_mu_0", "z_mu_1", "n"])
    return (
        work.groupby("node_role", as_index=False)
        .agg(z_mu_0=("z_mu_0", "mean"), z_mu_1=("z_mu_1", "mean"), n=("z_mu_0", "size"))
        .sort_values("node_role")
    )


def combo_output_dir(outdir: Path, *, seed: int, beta: float, opportunity: float, lambda_value: float, sigma: float, rnn_dim: int, latent_dim: int, tree_type: str) -> Path:
    return outdir / (
        f"seed_{file_token(seed)}_beta_{file_token(beta)}_opp_{file_token(opportunity)}_"
        f"lambda_{file_token(lambda_value)}_sigma_{file_token(sigma)}_"
        f"rnn_{file_token(rnn_dim)}_latent_{file_token(latent_dim)}_tree_{file_token(tree_type)}"
    )


def plot_combo_density(
    df: pd.DataFrame,
    figdir: Path,
    *,
    combo_label: str,
    grid_n: int,
    max_density_points: int,
):
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    limits = axis_limits(df)
    if limits is None:
        return
    xlim, ylim = limits
    x_grid = np.linspace(xlim[0], xlim[1], max(40, min(int(grid_n), 250)))
    y_grid = np.linspace(ylim[0], ylim[1], max(40, min(int(grid_n), 250)))
    reward_col = "actual_node_reward" if "actual_node_reward" in df.columns else "node_reward"
    reward = pd.to_numeric(df[reward_col], errors="coerce")
    reward_min = float(np.nanmin(reward))
    reward_max = float(np.nanmax(reward))
    if math.isclose(reward_min, reward_max):
        reward_min -= 0.5
        reward_max += 0.5
    norm = Normalize(vmin=reward_min, vmax=reward_max)
    cmap = plt.get_cmap("viridis")
    role_colors = {"better": "#1f78b4", "worse": "#e66101"}
    figdir.mkdir(parents=True, exist_ok=True)
    for role in ["better", "worse"]:
        role_df = df[df["node_role"] == role].copy()
        if role_df.empty:
            continue
        for timestep in sorted(pd.to_numeric(role_df["timestep"], errors="coerce").dropna().unique()):
            piece = role_df[np.isclose(pd.to_numeric(role_df["timestep"], errors="coerce"), timestep)].copy()
            if piece.empty:
                continue
            fig, ax = plt.subplots(figsize=(2.9, 2.6))
            piece_rewards = pd.to_numeric(piece[reward_col], errors="coerce")
            for reward_value in sorted(piece_rewards.dropna().unique()):
                reward_piece = piece[np.isclose(piece_rewards, reward_value)].copy()
                if reward_piece.empty:
                    continue
                density = empirical_mu_kde_density(
                    reward_piece,
                    x_grid,
                    y_grid,
                    max_points=max_density_points,
                    seed=reward_contour_seed(timestep, reward_value, role),
                )
                if density is None:
                    continue
                levels = positive_contour_levels(density)
                if levels is not None:
                    ax.contour(
                        x_grid,
                        y_grid,
                        density,
                        levels=levels,
                        colors=[cmap(norm(float(reward_value)))],
                        linewidths=0.8,
                        alpha=0.9,
                    )
            centroids = reward_centroids(piece)
            scatter = ax.scatter(
                pd.to_numeric(centroids["z_mu_0"], errors="coerce"),
                pd.to_numeric(centroids["z_mu_1"], errors="coerce"),
                c=pd.to_numeric(centroids["actual_node_reward"], errors="coerce"),
                cmap=cmap,
                norm=norm,
                s=34,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.35,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel("z_mu_0", fontsize=7)
            ax.set_ylabel("z_mu_1", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_title(f"{combo_label}\n{role} node, timestep {int(timestep)}", fontsize=7)
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
            cbar.set_label("actual observed-node reward", fontsize=7)
            cbar.ax.tick_params(labelsize=7)
            fig.tight_layout()
            out_name = f"revisit_latent_z0_z1_density_{role}_node_t{int(timestep)}.png"
            fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
            plt.close(fig)

    for timestep in sorted(pd.to_numeric(df["timestep"], errors="coerce").dropna().unique()):
        piece = df[np.isclose(pd.to_numeric(df["timestep"], errors="coerce"), timestep)].copy()
        if piece.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.25, 2.6))
        for role in ["better", "worse"]:
            role_piece = piece[piece["node_role"] == role].copy()
            if role_piece.empty:
                continue
            color = role_colors[role]
            density = empirical_mu_kde_density(
                role_piece,
                x_grid,
                y_grid,
                max_points=max_density_points,
                seed=role_contour_seed(timestep, role),
            )
            if density is not None:
                levels = positive_contour_levels(density)
                if levels is not None:
                    ax.contour(
                        x_grid,
                        y_grid,
                        density,
                        levels=levels,
                        colors=[color],
                        linewidths=0.9,
                        alpha=0.9,
                    )
            centroids = role_centroids(role_piece)
            if not centroids.empty:
                ax.scatter(
                    pd.to_numeric(centroids["z_mu_0"], errors="coerce"),
                    pd.to_numeric(centroids["z_mu_1"], errors="coerce"),
                    c=color,
                    s=38,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.35,
                    label=role,
                )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z_mu_0", fontsize=7)
        ax.set_ylabel("z_mu_1", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{combo_label}\nnode role, timestep {int(timestep)}", fontsize=7)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                title="node role",
                fontsize=7,
                title_fontsize=7,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                borderaxespad=0.0,
            )
        fig.tight_layout()
        out_name = f"revisit_latent_z0_z1_density_node_role_t{int(timestep)}.png"
        fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    checkpoint_root = Path(args.checkpoint_root)
    failures = []
    for lambda_value in args.lambda_values:
        for alpha in args.alphas:
            for beta in args.beta_values:
                for opportunity in args.opportunity_costs:
                    for sigma in args.observation_sigmas:
                        for seed in args.seeds:
                            for rnn_dim in args.rnn_dims:
                                for latent_dim in args.latent_dims:
                                    task = jp.build_task(args.tree_size, args.tree_type, args.input_type)
                                    if task.num_nodes != 2 or task.num_paths != 2:
                                        raise ValueError(
                                            "better/worse node revisit density plots are currently defined for two one-node paths only."
                                        )
                                    checkpoint, note = find_revisit_checkpoint(
                                        checkpoint_root,
                                        lambda_value=lambda_value,
                                        alpha=alpha,
                                        beta=beta,
                                        opportunity_cost=opportunity,
                                        seed=seed,
                                        task=task,
                                        tree_size=args.tree_size,
                                        rnn_dim=rnn_dim,
                                        latent_dim=latent_dim,
                                        expansion_decision_version=args.expansion_decision_version,
                                        model_variant=args.model_variant,
                                        max_observations_before_stop=args.max_observations_before_stop,
                                        observation_sigma=sigma,
                                        kl_start_multiplier=args.kl_start_multiplier,
                                        kl_annealing_epochs=args.kl_annealing_epochs,
                                    )
                                    combo_label = (
                                        f"seed={seed}, beta={beta:g}, lambda={lambda_value:g}, "
                                        f"opp={opportunity:g}, sigma={sigma:g}"
                                    )
                                    if checkpoint is None:
                                        failures.append({"combo": combo_label, "reason": note})
                                        print(f"Skipping {combo_label}: {note}")
                                        continue
                                    model, params = build_model_and_params(
                                        checkpoint,
                                        task=task,
                                        lambda_value=lambda_value,
                                        alpha=alpha,
                                        beta=beta,
                                        opportunity_cost=opportunity,
                                        rnn_dim=rnn_dim,
                                        latent_dim=latent_dim,
                                        expansion_decision_version=args.expansion_decision_version,
                                        model_variant=args.model_variant,
                                        max_observations_before_stop=args.max_observations_before_stop,
                                        observation_sigma=sigma,
                                    )
                                    rewards = sample_rewards(args.n_trials, task, args.analysis_seed_offset + seed)
                                    rows = rollout_revisit_rows(
                                        model=model,
                                        params=params,
                                        task=task,
                                        rewards=rewards,
                                        batch_size=args.batch_size,
                                        seed=args.analysis_seed_offset + seed + 10_000,
                                        beta=beta,
                                        max_observations_before_stop=args.max_observations_before_stop,
                                    )
                                    combo_dir = combo_output_dir(
                                        outdir,
                                        seed=seed,
                                        beta=beta,
                                        opportunity=opportunity,
                                        lambda_value=lambda_value,
                                        sigma=sigma,
                                        rnn_dim=rnn_dim,
                                        latent_dim=latent_dim,
                                        tree_type=args.tree_type,
                                    )
                                    combo_dir.mkdir(parents=True, exist_ok=True)
                                    rows.to_csv(combo_dir / "revisit_latent_density_rows.csv", index=False)
                                    plot_combo_density(
                                        rows,
                                        combo_dir / "figures",
                                        combo_label=combo_label,
                                        grid_n=args.latent_density_grid_n,
                                        max_density_points=args.max_density_points,
                                    )
                                    print(f"Saved revisit latent density plots for {combo_label} to {combo_dir / 'figures'}")
    if failures:
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(outdir / "revisit_latent_density_failures.csv", index=False)


if __name__ == "__main__":
    main()
