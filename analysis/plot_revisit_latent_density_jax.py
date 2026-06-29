#!/usr/bin/env python
"""Plot revisit-task JAX latent z0/z1 densities by observed node index.

This script is intentionally narrower than analyze_latent_angle_planning_jax.py:
it loads revisit-enabled JAX checkpoints, simulates trials with node revisits
allowed, and writes z_mu_0-vs-z_mu_1 density plots for the two-node task plus
compact path-context heatmaps for KL, terminal-choice entropy, and stopping
time. Each output PNG is for one seed/beta/opportunity/lambda/sigma
combination. Rows where the model already stopped before the observation are
excluded.
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

PANEL_SIZE_IN = 33.0 / 25.4
PLOT_FONT_SIZE = 7
PANEL_MARGIN_X_IN = 1.15
PANEL_MARGIN_Y_IN = 1.05
COLORBAR_WIDTH_IN = 0.45
PANEL_GAP_IN = 0.26


def configure_plot_text(plt) -> None:
    plt.rcParams.update({
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
    })


def single_panel_figsize(colorbar: bool = False, legend: bool = False) -> Tuple[float, float]:
    extra = COLORBAR_WIDTH_IN if colorbar else 0.0
    extra += 0.7 if legend else 0.0
    return (PANEL_SIZE_IN + PANEL_MARGIN_X_IN + extra, PANEL_SIZE_IN + PANEL_MARGIN_Y_IN)


def stacked_panel_figsize(n_panels: int, colorbar: bool = False) -> Tuple[float, float]:
    extra = COLORBAR_WIDTH_IN if colorbar else 0.0
    return (
        PANEL_SIZE_IN + PANEL_MARGIN_X_IN + extra,
        n_panels * PANEL_SIZE_IN + PANEL_MARGIN_Y_IN + max(0, n_panels - 1) * PANEL_GAP_IN,
    )


def reward_density_grid_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    # Match the old full 8x8 grid's per-panel scale without letting smaller
    # subsets expand to fill the previous minimum whole-figure size.
    panel_w = 1.0
    panel_h = 0.95
    return (max(1, n_cols) * panel_w + 1.1, max(1, n_rows) * panel_h + 1.0)


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
    parser.add_argument(
        "--min-density-samples",
        type=int,
        default=1,
        help="Minimum number of states required to draw each latent KDE contour.",
    )
    parser.add_argument("--analysis-seed-offset", type=int, default=300_000)
    parser.add_argument("--kl-start-multiplier", type=float, default=None)
    parser.add_argument("--kl-annealing-epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Accepted for CLI symmetry; plotting uses JAX default device.")
    return parser.parse_args()


def file_token(value) -> str:
    text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def observed_path_order_label(order: int, unit: str = "path") -> str:
    names = {
        1: "first_observed",
        2: "second_observed",
        3: "third_observed",
        4: "fourth_observed",
        5: "fifth_observed",
        6: "sixth_observed",
    }
    prefix = names.get(int(order), f"observed_order_{int(order)}")
    return f"{prefix}_{unit}"


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
    checkpoint_reward_dim = jp.infer_reward_feature_dim_from_checkpoint(
        checkpoint_path,
        task.num_nodes,
    )
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
        reward_feature_dim_override=int(checkpoint_reward_dim),
    )
    reward_feature_dim = (
        int(checkpoint_reward_dim)
        if int(checkpoint_reward_dim) > 0
        else jp.reward_feature_dim_for_sigma(observation_sigma)
    )
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


def node_to_path_indices(path_map: np.ndarray) -> np.ndarray:
    node_to_path = np.full(path_map.shape[1], -1, dtype=int)
    for node_i in range(path_map.shape[1]):
        containing_paths = np.where(path_map[:, node_i] > 0)[0]
        if containing_paths.size > 0:
            node_to_path[node_i] = int(containing_paths[0])
    return node_to_path


def bin_nearest_integer_value(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.rint(value))


def bin_width_two_away_from_zero_value(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if abs(float(value)) < 1e-12:
        return 0.0
    return float(np.sign(value) * np.ceil(abs(value) / 2.0) * 2.0)


def default_mean_other_path_value(value: float, task: jp.TaskSpec) -> float:
    if task.tree_type in {"bandit3", "disjoint3x2"}:
        return bin_nearest_integer_value(value)
    return float(value) if np.isfinite(value) else np.nan


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
    path_map = np.asarray(task.path_map, dtype=float)
    node_to_path = node_to_path_indices(path_map)
    has_two_one_node_paths = task.num_nodes == 2 and task.num_paths == 2
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
        path_rewards = batch_rewards @ path_map.T
        path_reward_sums = np.sum(path_rewards, axis=1)
        carry = jp.initial_carry(batch_rewards.shape[0], task, model.rnn_units, reward_feature_dim)
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        rng = jax.random.PRNGKey(seed + batch_i)
        stopped = np.zeros(batch_rewards.shape[0], dtype=bool)
        stop_decision_timestep = np.full(batch_rewards.shape[0], num_steps, dtype=int)
        first_observed_node = np.full(batch_rewards.shape[0], -1, dtype=int)
        first_observed_path = np.full(batch_rewards.shape[0], -1, dtype=int)
        path_visit_order = np.full((batch_rewards.shape[0], task.num_paths), -1, dtype=int)
        next_path_visit_order = np.ones(batch_rewards.shape[0], dtype=int)
        batch_rows = []
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
            prior_mu = np.asarray(trans.prior_mu, dtype=float)
            prior_sigma = np.exp(0.5 * np.clip(np.asarray(trans.prior_logvar, dtype=float), -10.0, 10.0))
            paid_kl = np.asarray(trans.paid_kl, dtype=float)
            action_output = np.asarray(trans.action_output, dtype=float)
            prob_sums = np.nansum(np.where(np.isfinite(action_output), action_output, 0.0), axis=1)
            terminal_entropy = np.full(batch_rewards.shape[0], np.nan, dtype=float)
            valid_probs = np.isfinite(prob_sums) & (prob_sums > 0)
            if np.any(valid_probs):
                probs = np.zeros_like(action_output, dtype=float)
                probs[valid_probs] = np.where(
                    np.isfinite(action_output[valid_probs]),
                    action_output[valid_probs],
                    0.0,
                ) / prob_sums[valid_probs, None]
                terminal_entropy[valid_probs] = -np.sum(
                    np.where(probs[valid_probs] > 0, probs[valid_probs] * np.log(probs[valid_probs] + 1e-12), 0.0),
                    axis=1,
                )
            include = (~stopped) & is_observe & (node_index >= 0)
            for local_i in np.where(include)[0]:
                node_i = int(node_index[local_i])
                if node_i < 0 or node_i >= len(node_to_path):
                    continue
                path_i = int(node_to_path[node_i])
                if path_i < 0:
                    continue
                if first_observed_node[local_i] < 0:
                    first_observed_node[local_i] = node_i
                if first_observed_path[local_i] < 0:
                    first_observed_path[local_i] = path_i
                if path_visit_order[local_i, path_i] < 0:
                    path_visit_order[local_i, path_i] = next_path_visit_order[local_i]
                    next_path_visit_order[local_i] += 1
                observed_path_order = int(path_visit_order[local_i, path_i])
                first_path_i = int(first_observed_path[local_i])
                node_reward = float(batch_rewards[local_i, node_i])
                observed_path_reward = float(path_rewards[local_i, path_i])
                first_path_reward = float(path_rewards[local_i, first_path_i])
                mean_other_path_reward = (
                    float(path_reward_sums[local_i] - first_path_reward) / float(task.num_paths - 1)
                    if task.num_paths > 1
                    else np.nan
                )
                mean_other_observed_path_reward = (
                    float(path_reward_sums[local_i] - observed_path_reward) / float(task.num_paths - 1)
                    if task.num_paths > 1
                    else np.nan
                )
                row = {
                    "trial_id": int(start + local_i),
                    "batch_row": int(local_i),
                    "timestep": int(step_i + 1),
                    "observed_node": int(node_i + 1),
                    "observed_path": int(path_i + 1),
                    "observed_path_order": observed_path_order,
                    "observed_path_order_label": observed_path_order_label(observed_path_order, unit="path"),
                    "first_observed_node": int(first_observed_node[local_i] + 1),
                    "first_observed_path": int(first_path_i + 1),
                    "actual_node_reward": node_reward,
                    "node_reward": node_reward,
                    "observed_path_actual_reward": observed_path_reward,
                    "first_observed_path_actual_reward_raw": first_path_reward,
                    "mean_other_path_actual_reward_raw": mean_other_path_reward,
                    "first_observed_path_actual_reward": first_path_reward,
                    "mean_other_path_actual_reward": default_mean_other_path_value(mean_other_path_reward, task),
                    "first_observed_path_actual_reward_integer": bin_nearest_integer_value(first_path_reward),
                    "mean_other_path_actual_reward_integer": bin_nearest_integer_value(mean_other_path_reward),
                    "first_observed_path_actual_reward_bin2": bin_width_two_away_from_zero_value(first_path_reward),
                    "mean_other_path_actual_reward_bin2": bin_width_two_away_from_zero_value(mean_other_path_reward),
                    "observed_path_actual_reward_raw": observed_path_reward,
                    "mean_other_observed_path_actual_reward_raw": mean_other_observed_path_reward,
                    "observed_path_actual_reward_integer": bin_nearest_integer_value(observed_path_reward),
                    "mean_other_observed_path_actual_reward_integer": bin_nearest_integer_value(mean_other_observed_path_reward),
                    "observed_path_actual_reward_bin2": bin_width_two_away_from_zero_value(observed_path_reward),
                    "mean_other_observed_path_actual_reward_bin2": bin_width_two_away_from_zero_value(mean_other_observed_path_reward),
                    "mean_other_observed_path_actual_reward": default_mean_other_path_value(mean_other_observed_path_reward, task),
                    "sampled_observed_reward": float(sampled_observed_reward[local_i]),
                    "kl_paid_at_timestep": float(paid_kl[local_i]),
                    "terminal_choice_entropy_at_timestep": float(terminal_entropy[local_i]),
                    "z_mu_0": float(z_mu[local_i, 0]),
                    "z_mu_1": float(z_mu[local_i, 1]),
                    "z_sigma_0": float(z_sigma[local_i, 0]),
                    "z_sigma_1": float(z_sigma[local_i, 1]),
                    "prior_mu_0": float(prior_mu[local_i, 0]),
                    "prior_mu_1": float(prior_mu[local_i, 1]),
                    "prior_sigma_0": float(prior_sigma[local_i, 0]),
                    "prior_sigma_1": float(prior_sigma[local_i, 1]),
                    "prior_normalized_z_mu_0": float(
                        (z_mu[local_i, 0] - prior_mu[local_i, 0])
                        / max(float(prior_sigma[local_i, 0]), 1e-6)
                    ),
                    "prior_normalized_z_mu_1": float(
                        (z_mu[local_i, 1] - prior_mu[local_i, 1])
                        / max(float(prior_sigma[local_i, 1]), 1e-6)
                    ),
                }
                if has_two_one_node_paths:
                    visit_order = "first_observed" if observed_path_order == 1 else "second_observed"
                    other_i = 1 - node_i
                    other_reward = float(batch_rewards[local_i, other_i])
                    node1_reward = float(batch_rewards[local_i, 0])
                    node2_reward = float(batch_rewards[local_i, 1])
                    first_minus_second_reward = (
                        node_reward - other_reward
                        if visit_order == "first_observed"
                        else other_reward - node_reward
                    )
                    first_observed_reward = node_reward if visit_order == "first_observed" else other_reward
                    second_observed_reward = other_reward if visit_order == "first_observed" else node_reward
                    row.update(
                        {
                            "node_visit_order": visit_order,
                            "node_role": "better" if node_reward > other_reward else "worse",
                            "node1_actual_reward": node1_reward,
                            "node2_actual_reward": node2_reward,
                            "first_observed_actual_reward": first_observed_reward,
                            "second_observed_actual_reward": second_observed_reward,
                            "first_observed_minus_second_actual_reward": first_minus_second_reward,
                            "other_node_reward": other_reward,
                        }
                    )
                batch_rows.append(row)
            new_stop = (~stopped) & is_stop
            stop_decision_timestep[new_stop] = int(step_i + 1)
            stopped |= is_stop
        if batch_rows:
            for row in batch_rows:
                local_i = int(row["batch_row"])
                timestep_before_stop = int(max(stop_decision_timestep[local_i] - 1, 0))
                row["timestep_before_stop"] = timestep_before_stop
                row["continued_after_observation"] = int(int(row["timestep"]) < timestep_before_stop)
                row.pop("batch_row", None)
            rows.extend(row for row in batch_rows if row["continued_after_observation"] == 1)
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


GLOBAL_Z0_Z1_LIMITS: Tuple[Tuple[float, float], Tuple[float, float]] = ((-5.0, 5.0), (-5.0, 5.0))
GLOBAL_Z0_Z1_TICKS = np.asarray([-5.0, 0.0, 5.0], dtype=float)


def apply_global_z0_z1_axes(ax) -> None:
    ax.set_xlim(*GLOBAL_Z0_Z1_LIMITS[0])
    ax.set_ylim(*GLOBAL_Z0_Z1_LIMITS[1])
    ax.set_xticks(GLOBAL_Z0_Z1_TICKS)
    ax.set_yticks(GLOBAL_Z0_Z1_TICKS)


def square_axis_limits(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_mid = 0.5 * (float(xlim[0]) + float(xlim[1]))
    y_mid = 0.5 * (float(ylim[0]) + float(ylim[1]))
    span = max(float(xlim[1]) - float(xlim[0]), float(ylim[1]) - float(ylim[0]))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    half = 0.5 * span
    return (x_mid - half, x_mid + half), (y_mid - half, y_mid + half)


def empirical_mu_kde_density(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    max_points: int,
    seed: int,
    min_samples: int = 1,
) -> Optional[np.ndarray]:
    cols = ["z_mu_0", "z_mu_1"]
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < max(1, int(min_samples)):
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


def positive_contour_levels(
    density: np.ndarray,
    masses: Tuple[float, ...] = (0.50, 0.90),
) -> Optional[np.ndarray]:
    """Return KDE density thresholds enclosing the requested probability masses."""
    values = density[np.isfinite(density) & (density > 0)]
    if len(values) == 0:
        return None
    sorted_values = np.sort(values.astype(float))[::-1]
    total = float(np.sum(sorted_values))
    if not np.isfinite(total) or total <= 0.0:
        return None
    cumulative_mass = np.cumsum(sorted_values) / total
    levels = []
    for mass in masses:
        if not (0.0 < float(mass) < 1.0):
            continue
        idx = int(np.searchsorted(cumulative_mass, float(mass), side="left"))
        idx = min(max(idx, 0), len(sorted_values) - 1)
        levels.append(float(sorted_values[idx]))
    min_density = float(np.min(values))
    max_density = float(np.max(values))
    levels = np.unique(np.asarray(levels, dtype=float))
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


def value_centroids(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """One plotted marker per value of value_col."""
    work = df[[value_col, "z_mu_0", "z_mu_1"]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["z_mu_0"] = pd.to_numeric(work["z_mu_0"], errors="coerce")
    work["z_mu_1"] = pd.to_numeric(work["z_mu_1"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(columns=[value_col, "z_mu_0", "z_mu_1", "n"])
    return (
        work.groupby(value_col, as_index=False)
        .agg(z_mu_0=("z_mu_0", "mean"), z_mu_1=("z_mu_1", "mean"), n=("z_mu_0", "size"))
        .sort_values(value_col)
    )


def reward_contour_seed(timestep: float, reward_value: float, group: str) -> int:
    reward_token = int(round((float(reward_value) + 10.0) * 100.0))
    group_token = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(group)))
    return 17 + int(round(float(timestep))) * 101 + reward_token * 13 + group_token


def value_contour_seed(timestep: float, value: float, group: str) -> int:
    value_token = int(round((float(value) + 50.0) * 100.0))
    group_token = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(group)))
    return 104729 + int(round(float(timestep))) * 101 + value_token * 17 + group_token


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


def reward_axis_values_for_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    reward_pair_cols = {"node1_actual_reward", "node2_actual_reward"}
    if not reward_pair_cols.issubset(df.columns):
        return None
    pair_rewards = pd.concat(
        [
            pd.to_numeric(df["node1_actual_reward"], errors="coerce"),
            pd.to_numeric(df["node2_actual_reward"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if pair_rewards.empty:
        return None
    if float(pair_rewards.min()) < 0.0 or float(pair_rewards.max()) > 1.0:
        return np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)
    return np.asarray([0, 1], dtype=float)


def reward_pair_mean_grid(
    panel: pd.DataFrame,
    value_col: str,
    y_col: str,
    x_col: str,
    y_values: np.ndarray,
    x_values: np.ndarray,
) -> np.ndarray:
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    if panel.empty or not {y_col, x_col, value_col}.issubset(panel.columns):
        return grid
    work = panel[[y_col, x_col, value_col]].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return grid
    summary = (
        work.groupby([y_col, x_col], as_index=False)[value_col]
        .mean()
    )
    for row_i, y_value in enumerate(y_values):
        for col_i, x_value in enumerate(x_values):
            match = summary[
                np.isclose(summary[y_col], y_value)
                & np.isclose(summary[x_col], x_value)
            ]
            if not match.empty:
                grid[row_i, col_i] = float(match[value_col].iloc[0])
    return grid


def fixed_metric_color_limits(value_col: str) -> Optional[Tuple[float, float]]:
    if value_col == "kl_paid_at_timestep":
        return (0.0, 6.0)
    if value_col == "terminal_choice_entropy_at_timestep":
        return (0.0, 0.6)
    return None


def plot_reward_grid_metric_panels(
    df: pd.DataFrame,
    figdir: Path,
    *,
    combo_label: str,
    split_label: str,
    split_stub: str,
    value_col: str,
    value_label: str,
    file_stub: str,
    y_col: str,
    x_col: str,
    y_label: str,
    x_label: str,
    y_values: np.ndarray,
    x_values: np.ndarray,
    split_by_timestep: bool,
    cmap_name: str,
) -> None:
    if df.empty or value_col not in df.columns:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_plot_text(plt)
    configure_plot_text(plt)

    metric_values = pd.to_numeric(df[value_col], errors="coerce")
    if not np.isfinite(metric_values).any():
        return

    if split_by_timestep:
        timestep_values = sorted(pd.to_numeric(df["timestep"], errors="coerce").dropna().unique())
    else:
        timestep_values = [None]
    if not timestep_values:
        return

    grids = []
    titles = []
    for timestep in timestep_values:
        panel = df
        if timestep is not None:
            panel = df[np.isclose(pd.to_numeric(df["timestep"], errors="coerce"), timestep)].copy()
            titles.append(f"timestep {int(timestep)}")
        else:
            titles.append("all observed timesteps")
        grids.append(reward_pair_mean_grid(panel, value_col, y_col, x_col, y_values, x_values))

    finite_values = np.concatenate([grid[np.isfinite(grid)] for grid in grids if np.isfinite(grid).any()])
    if finite_values.size == 0:
        return
    fixed_limits = fixed_metric_color_limits(value_col)
    if fixed_limits is not None:
        vmin, vmax = fixed_limits
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if math.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    n_panels = len(grids)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=stacked_panel_figsize(n_panels, colorbar=True),
        squeeze=False,
    )
    cmap = plt.get_cmap(cmap_name)
    image = None
    for ax, grid, title in zip(axes[:, 0], grids, titles):
        image = ax.imshow(
            grid,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([f"{v:g}" for v in x_values], fontsize=PLOT_FONT_SIZE)
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_yticklabels([f"{v:g}" for v in y_values], fontsize=PLOT_FONT_SIZE)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel(y_label, fontsize=7)
        ax.set_title(title, fontsize=7, pad=8)
        ax.tick_params(length=1.5, pad=1)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.045, pad=0.035)
        cbar.set_label(value_label, fontsize=7)
        cbar.ax.tick_params(labelsize=PLOT_FONT_SIZE)
    fig.suptitle(
        f"{combo_label}\n{split_label}",
        fontsize=7,
        y=0.995,
    )
    fig.subplots_adjust(top=0.78 if n_panels == 1 else 0.88, hspace=0.52)
    out_name = (
        f"revisit_latent_reward_grid_{file_stub}_"
        f"{split_stub}.png"
    )
    fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def coordinate_axis_values(df: pd.DataFrame, column: str) -> Optional[np.ndarray]:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return np.asarray(sorted(values), dtype=float)


def observed_index_split(df: pd.DataFrame) -> Tuple[str, str]:
    """Choose whether index-split plots should use observed node or path IDs."""
    observed_path_max = pd.to_numeric(
        df.get("observed_path", pd.Series(dtype=float)),
        errors="coerce",
    ).max()
    observed_node_max = pd.to_numeric(
        df.get("observed_node", pd.Series(dtype=float)),
        errors="coerce",
    ).max()
    if (
        np.isfinite(observed_path_max) and float(observed_path_max) > 2.0
    ) or (
        np.isfinite(observed_node_max)
        and np.isfinite(observed_path_max)
        and float(observed_node_max) > float(observed_path_max)
    ):
        return "observed_path", "path"
    if "observed_node" in df.columns:
        return "observed_node", "node"
    return "observed_path", "path"


def coordinate_mean_grid(
    panel: pd.DataFrame,
    *,
    y_col: str,
    x_col: str,
    value_col: str,
    y_values: np.ndarray,
    x_values: np.ndarray,
) -> np.ndarray:
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    if panel.empty or not {y_col, x_col, value_col}.issubset(panel.columns):
        return grid
    work = panel[[y_col, x_col, value_col]].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return grid
    summary = work.groupby([y_col, x_col], as_index=False)[value_col].mean()
    for row_i, y_value in enumerate(y_values):
        for col_i, x_value in enumerate(x_values):
            match = summary[
                np.isclose(summary[y_col], y_value)
                & np.isclose(summary[x_col], x_value)
            ]
            if not match.empty:
                grid[row_i, col_i] = float(match[value_col].iloc[0])
    return grid


def plot_path_context_metric_panels(
    df: pd.DataFrame,
    figdir: Path,
    *,
    combo_label: str,
    value_col: str,
    value_label: str,
    file_stub: str,
    y_col: str,
    x_col: str,
    y_label: str,
    x_label: str,
    split_by_timestep: bool,
    cmap_name: str,
) -> None:
    if df.empty or not {y_col, x_col, value_col}.issubset(df.columns):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_plot_text(plt)

    y_values = coordinate_axis_values(df, y_col)
    x_values = coordinate_axis_values(df, x_col)
    if y_values is None or x_values is None:
        return
    metric_values = pd.to_numeric(df[value_col], errors="coerce")
    if not np.isfinite(metric_values).any():
        return

    if split_by_timestep and "timestep" in df.columns:
        timestep_values = sorted(pd.to_numeric(df["timestep"], errors="coerce").dropna().unique())
    else:
        timestep_values = [None]
    if not timestep_values:
        return

    grids = []
    titles = []
    for timestep in timestep_values:
        panel = df
        if timestep is not None:
            panel = df[np.isclose(pd.to_numeric(df["timestep"], errors="coerce"), timestep)].copy()
            titles.append(f"timestep {int(timestep)}")
        else:
            titles.append("all observed timesteps")
        grids.append(
            coordinate_mean_grid(
                panel,
                y_col=y_col,
                x_col=x_col,
                value_col=value_col,
                y_values=y_values,
                x_values=x_values,
            )
        )

    finite_values = np.concatenate([grid[np.isfinite(grid)] for grid in grids if np.isfinite(grid).any()])
    if finite_values.size == 0:
        return
    fixed_limits = fixed_metric_color_limits(value_col)
    if fixed_limits is not None:
        vmin, vmax = fixed_limits
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if math.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    n_panels = len(grids)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=stacked_panel_figsize(n_panels, colorbar=True),
        squeeze=False,
    )
    cmap = plt.get_cmap(cmap_name)
    image = None
    for ax, grid, title in zip(axes[:, 0], grids, titles):
        image = ax.imshow(
            grid,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([f"{v:g}" for v in x_values], fontsize=PLOT_FONT_SIZE, rotation=90)
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_yticklabels([f"{v:g}" for v in y_values], fontsize=PLOT_FONT_SIZE)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel(y_label, fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(length=1.5, pad=1)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.045, pad=0.035)
        cbar.set_label(value_label, fontsize=7)
        cbar.ax.tick_params(labelsize=PLOT_FONT_SIZE)
    fig.suptitle(combo_label, fontsize=7, y=0.995)
    fig.savefig(figdir / f"revisit_path_context_grid_{file_stub}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    task_tree_type: str = "",
    latent_file_prefix: str = "revisit_latent_z0_z1_density",
    plot_context_heatmaps: bool = True,
    x_axis_label: str = "z_mu_0",
    y_axis_label: str = "z_mu_1",
    x_panel_label: str = "z0",
    min_density_samples: int = 1,
):
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    if axis_limits(df) is None:
        return
    xlim, ylim = GLOBAL_Z0_Z1_LIMITS
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
    for old_plot in figdir.glob(f"{latent_file_prefix}_*.png"):
        old_plot.unlink()
    if plot_context_heatmaps:
        for old_plot in figdir.glob("revisit_latent_reward_grid_*.png"):
            old_plot.unlink()
        for old_plot in figdir.glob("revisit_path_context_grid_*.png"):
            old_plot.unlink()
    tree_type_key = str(task_tree_type).strip().lower()

    def plot_path_context_set(
        suffix: str,
        y_col: str,
        x_col: str,
        y_label: str,
        x_label: str,
        file_context: str = "first_observed_path_and_mean_other_path",
        data: Optional[pd.DataFrame] = None,
        title_combo_label: Optional[str] = None,
    ) -> None:
        plot_df = df if data is None else data
        plot_title = combo_label if title_combo_label is None else title_combo_label
        plot_path_context_metric_panels(
            plot_df,
            figdir,
            combo_label=plot_title,
            value_col="kl_paid_at_timestep",
            value_label="KL paid at timestep",
            file_stub=f"kl_paid_at_timestep_by_{file_context}{suffix}",
            y_col=y_col,
            x_col=x_col,
            y_label=y_label,
            x_label=x_label,
            split_by_timestep=True,
            cmap_name="plasma",
        )
        plot_path_context_metric_panels(
            plot_df,
            figdir,
            combo_label=plot_title,
            value_col="terminal_choice_entropy_at_timestep",
            value_label="Terminal choice entropy",
            file_stub=f"terminal_choice_entropy_by_{file_context}{suffix}",
            y_col=y_col,
            x_col=x_col,
            y_label=y_label,
            x_label=x_label,
            split_by_timestep=True,
            cmap_name="viridis",
        )
        plot_path_context_metric_panels(
            plot_df,
            figdir,
            combo_label=plot_title,
            value_col="timestep_before_stop",
            value_label="Timestep before stopping",
            file_stub=f"timestep_before_stop_by_{file_context}{suffix}",
            y_col=y_col,
            x_col=x_col,
            y_label=y_label,
            x_label=x_label,
            split_by_timestep=False,
            cmap_name="cividis",
        )

    def plot_observed_path_context_sets(
        data: pd.DataFrame,
        suffix_extra: str = "",
        title_combo_label: Optional[str] = None,
    ) -> None:
        plot_path_context_set(
            suffix_extra,
            "observed_path_actual_reward",
            "mean_other_observed_path_actual_reward",
            "R(observed path)",
            "Mean R(other paths)",
            file_context="observed_path_and_mean_other_path",
            data=data,
            title_combo_label=title_combo_label,
        )
        if tree_type_key in {"disjoint2x2", "disjoint3x2"}:
            plot_path_context_set(
                f"_path_value_mean_other_integer{suffix_extra}",
                "observed_path_actual_reward_raw",
                "mean_other_observed_path_actual_reward_integer",
                "R(observed path)",
                "Mean R(other paths), rounded",
                file_context="observed_path_and_mean_other_path",
                data=data,
                title_combo_label=title_combo_label,
            )
            plot_path_context_set(
                f"_path_value_bin2{suffix_extra}",
                "observed_path_actual_reward_bin2",
                "mean_other_observed_path_actual_reward_bin2",
                "R(observed path), bin size 2",
                "Mean R(other paths), bin size 2",
                file_context="observed_path_and_mean_other_path",
                data=data,
                title_combo_label=title_combo_label,
            )

    index_split_col, index_split_unit = observed_index_split(df)
    if plot_context_heatmaps and tree_type_key in {"bandit3", "disjoint2x2", "disjoint3x2"} and index_split_col in df.columns:
        split_values = sorted(
            int(v)
            for v in pd.to_numeric(df[index_split_col], errors="coerce").dropna().unique()
            if int(v) >= 1
        )
        for split_value in split_values:
            path_df = df[
                pd.to_numeric(df[index_split_col], errors="coerce") == split_value
            ].copy()
            if path_df.empty:
                continue
            split_stub = f"observed_{index_split_unit}_{split_value}"
            split_suffix = f"_{split_stub}"
            split_title = f"{combo_label}\nobserved {index_split_unit} {split_value}"
            plot_observed_path_context_sets(
                path_df,
                suffix_extra=split_suffix,
                title_combo_label=split_title,
            )

    reward_y_col = "observed_path_actual_reward"
    reward_x_col = "mean_other_observed_path_actual_reward"
    reward_y_label = "R(observed path)"
    reward_x_label = "Mean R(other paths)"
    if not {reward_y_col, reward_x_col}.issubset(df.columns):
        return
    reward_y_values = coordinate_axis_values(df, reward_y_col)
    reward_x_values = coordinate_axis_values(df, reward_x_col)
    if reward_y_values is None or reward_x_values is None:
        return

    def plot_timestep_split_value_density(
        split_df: pd.DataFrame,
        *,
        split_label: str,
        split_stub: str,
        timestep: float,
        value_col: str,
        value_label: str,
        value_stub: str,
        cmap_name: str = "viridis",
    ) -> None:
        piece = split_df[
            np.isclose(pd.to_numeric(split_df["timestep"], errors="coerce"), timestep)
        ].copy()
        if piece.empty or value_col not in piece.columns:
            return
        values = pd.to_numeric(piece[value_col], errors="coerce")
        keep = np.isfinite(values)
        piece = piece[keep].copy()
        values = values[keep]
        if piece.empty:
            return
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        if math.isclose(value_min, value_max):
            value_min -= 0.5
            value_max += 0.5
        value_norm = Normalize(vmin=value_min, vmax=value_max)
        value_cmap = plt.get_cmap(cmap_name)
        fig, ax = plt.subplots(figsize=single_panel_figsize(colorbar=True))
        for value in sorted(values.dropna().unique()):
            value_piece = piece[np.isclose(pd.to_numeric(piece[value_col], errors="coerce"), value)].copy()
            if value_piece.empty:
                continue
            color = value_cmap(value_norm(float(value)))
            density = empirical_mu_kde_density(
                value_piece,
                x_grid,
                y_grid,
                max_points=max_density_points,
                seed=value_contour_seed(timestep, value, f"{split_stub}_{value_stub}"),
                min_samples=min_density_samples,
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
                        linewidths=0.8,
                        alpha=0.9,
                    )
            z0 = pd.to_numeric(value_piece["z_mu_0"], errors="coerce")
            z1 = pd.to_numeric(value_piece["z_mu_1"], errors="coerce")
            finite = np.isfinite(z0) & np.isfinite(z1)
            if int(np.sum(finite)) >= max(1, int(min_density_samples)):
                ax.scatter(
                    float(z0[finite].mean()),
                    float(z1[finite].mean()),
                    c=[color],
                    s=28,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.35,
                )
        apply_global_z0_z1_axes(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(x_axis_label, fontsize=7)
        ax.set_ylabel(y_axis_label, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{combo_label}\n{split_label}, timestep {int(timestep)}",
            fontsize=7,
        )
        sm = plt.cm.ScalarMappable(norm=value_norm, cmap=value_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label(value_label, fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        out_name = (
            f"{latent_file_prefix}_by_{value_stub}_"
            f"t{int(timestep)}_{split_stub}.png"
        )
        fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    timestep_values = sorted(pd.to_numeric(df["timestep"], errors="coerce").dropna().unique())
    if timestep_values:
        if index_split_col not in df.columns:
            return
        split_values = sorted(
            int(v)
            for v in pd.to_numeric(df[index_split_col], errors="coerce").dropna().unique()
            if int(v) >= 1
        )
        for split_value in split_values:
            split_df = df[
                pd.to_numeric(df[index_split_col], errors="coerce") == split_value
            ].copy()
            if split_df.empty:
                continue
            split_stub = f"observed_{index_split_unit}_{split_value}"
            split_label = f"observed {index_split_unit} {split_value}"
            if plot_context_heatmaps:
                plot_reward_grid_metric_panels(
                    split_df,
                    figdir,
                    combo_label=combo_label,
                    split_label=split_label,
                    split_stub=split_stub,
                    value_col="kl_paid_at_timestep",
                    value_label="KL paid at timestep",
                    file_stub="kl_paid_at_timestep_by_observed_and_mean_other_path",
                    y_col=reward_y_col,
                    x_col=reward_x_col,
                    y_label=reward_y_label,
                    x_label=reward_x_label,
                    y_values=reward_y_values,
                    x_values=reward_x_values,
                    split_by_timestep=True,
                    cmap_name="plasma",
                )
                plot_reward_grid_metric_panels(
                    split_df,
                    figdir,
                    combo_label=combo_label,
                    split_label=split_label,
                    split_stub=split_stub,
                    value_col="terminal_choice_entropy_at_timestep",
                    value_label="Terminal choice entropy",
                    file_stub="terminal_choice_entropy_by_observed_and_mean_other_path",
                    y_col=reward_y_col,
                    x_col=reward_x_col,
                    y_label=reward_y_label,
                    x_label=reward_x_label,
                    y_values=reward_y_values,
                    x_values=reward_x_values,
                    split_by_timestep=True,
                    cmap_name="viridis",
                )
                plot_reward_grid_metric_panels(
                    split_df,
                    figdir,
                    combo_label=combo_label,
                    split_label=split_label,
                    split_stub=split_stub,
                    value_col="timestep_before_stop",
                    value_label="Timestep before stopping",
                    file_stub="timestep_before_stop_by_observed_and_mean_other_path",
                    y_col=reward_y_col,
                    x_col=reward_x_col,
                    y_label=reward_y_label,
                    x_label=reward_x_label,
                    y_values=reward_y_values,
                    x_values=reward_x_values,
                    split_by_timestep=False,
                    cmap_name="cividis",
                )
            for timestep in timestep_values:
                plot_timestep_split_value_density(
                    split_df,
                    split_label=split_label,
                    split_stub=split_stub,
                    timestep=timestep,
                    value_col=reward_y_col,
                    value_label=reward_y_label,
                    value_stub="current_path_value",
                    cmap_name="viridis",
                )
                plot_timestep_split_value_density(
                    split_df,
                    split_label=split_label,
                    split_stub=split_stub,
                    timestep=timestep,
                    value_col=reward_x_col,
                    value_label=reward_x_label,
                    value_stub="mean_other_path_value",
                    cmap_name="magma",
                )
        return

    visit_order_levels = ["first_observed", "second_observed"]
    if "node_visit_order" not in df.columns:
        visit_order_levels = []

    for visit_order in visit_order_levels:
        visit_df = df[df["node_visit_order"] == visit_order].copy()
        if visit_df.empty:
            continue
        for timestep in sorted(pd.to_numeric(visit_df["timestep"], errors="coerce").dropna().unique()):
            piece = visit_df[np.isclose(pd.to_numeric(visit_df["timestep"], errors="coerce"), timestep)].copy()
            if piece.empty:
                continue
            fig, ax = plt.subplots(figsize=single_panel_figsize(colorbar=True))
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
                    seed=reward_contour_seed(timestep, reward_value, visit_order),
                    min_samples=min_density_samples,
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
            if "n" in centroids.columns:
                centroids = centroids[pd.to_numeric(centroids["n"], errors="coerce") >= max(1, int(min_density_samples))]
            if centroids.empty:
                plt.close(fig)
                continue
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
            apply_global_z0_z1_axes(ax)
            ax.set_xlabel(x_axis_label, fontsize=7)
            ax.set_ylabel(y_axis_label, fontsize=7)
            ax.tick_params(labelsize=7)
            title_label = visit_order.replace("_", " ")
            ax.set_title(f"{combo_label}\n{title_label} node, timestep {int(timestep)}", fontsize=7)
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
            cbar.set_label("actual observed-node reward", fontsize=7)
            cbar.ax.tick_params(labelsize=7)
            fig.tight_layout()
            out_name = f"{latent_file_prefix}_{visit_order}_node_t{int(timestep)}.png"
            fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
            plt.close(fig)

    diff_col = "first_observed_minus_second_actual_reward"
    if diff_col in df.columns:
        diff_values_all = pd.to_numeric(df[diff_col], errors="coerce")
        diff_values_all = diff_values_all[np.isfinite(diff_values_all)]
        if len(diff_values_all) > 0:
            diff_min = float(np.nanmin(diff_values_all))
            diff_max = float(np.nanmax(diff_values_all))
            if math.isclose(diff_min, diff_max):
                diff_min -= 0.5
                diff_max += 0.5
            if diff_min < 0.0 < diff_max:
                diff_norm = TwoSlopeNorm(vmin=diff_min, vcenter=0.0, vmax=diff_max)
            else:
                diff_norm = Normalize(vmin=diff_min, vmax=diff_max)
            diff_cmap = plt.get_cmap("coolwarm")
            for visit_order in visit_order_levels:
                visit_df = df[df["node_visit_order"] == visit_order].copy()
                if visit_df.empty:
                    continue
                for timestep in sorted(pd.to_numeric(visit_df["timestep"], errors="coerce").dropna().unique()):
                    piece = visit_df[np.isclose(pd.to_numeric(visit_df["timestep"], errors="coerce"), timestep)].copy()
                    if piece.empty:
                        continue
                    fig, ax = plt.subplots(figsize=single_panel_figsize(colorbar=True))
                    piece_diff = pd.to_numeric(piece[diff_col], errors="coerce")
                    for diff_value in sorted(piece_diff.dropna().unique()):
                        diff_piece = piece[np.isclose(piece_diff, diff_value)].copy()
                        if diff_piece.empty:
                            continue
                        density = empirical_mu_kde_density(
                            diff_piece,
                            x_grid,
                            y_grid,
                            max_points=max_density_points,
                            seed=value_contour_seed(timestep, diff_value, visit_order),
                            min_samples=min_density_samples,
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
                                colors=[diff_cmap(diff_norm(float(diff_value)))],
                                linewidths=0.8,
                                alpha=0.9,
                            )
                    centroids = value_centroids(piece, diff_col)
                    if "n" in centroids.columns:
                        centroids = centroids[pd.to_numeric(centroids["n"], errors="coerce") >= max(1, int(min_density_samples))]
                    if centroids.empty:
                        plt.close(fig)
                        continue
                    scatter = ax.scatter(
                        pd.to_numeric(centroids["z_mu_0"], errors="coerce"),
                        pd.to_numeric(centroids["z_mu_1"], errors="coerce"),
                        c=pd.to_numeric(centroids[diff_col], errors="coerce"),
                        cmap=diff_cmap,
                        norm=diff_norm,
                        s=34,
                        alpha=0.95,
                        edgecolors="black",
                        linewidths=0.35,
                    )
                    apply_global_z0_z1_axes(ax)
                    ax.set_xlabel(x_axis_label, fontsize=7)
                    ax.set_ylabel(y_axis_label, fontsize=7)
                    ax.tick_params(labelsize=7)
                    title_label = visit_order.replace("_", " ")
                    ax.set_title(
                        f"{combo_label}\n{title_label} node, timestep {int(timestep)}",
                        fontsize=7,
                    )
                    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
                    cbar.set_label("R(first observed) - R(second observed)", fontsize=7)
                    cbar.ax.tick_params(labelsize=7)
                    fig.tight_layout()
                    out_name = (
                        f"{latent_file_prefix}_first_minus_second_reward_"
                        f"{visit_order}_node_t{int(timestep)}.png"
                    )
                    fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
                    plt.close(fig)

    for timestep in sorted(pd.to_numeric(df["timestep"], errors="coerce").dropna().unique()):
        piece = df[np.isclose(pd.to_numeric(df["timestep"], errors="coerce"), timestep)].copy()
        if piece.empty:
            continue
        fig, ax = plt.subplots(figsize=single_panel_figsize(legend=True))
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
                min_samples=min_density_samples,
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
            if "n" in centroids.columns:
                centroids = centroids[pd.to_numeric(centroids["n"], errors="coerce") >= max(1, int(min_density_samples))]
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
        apply_global_z0_z1_axes(ax)
        ax.set_xlabel(x_axis_label, fontsize=7)
        ax.set_ylabel(y_axis_label, fontsize=7)
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
        out_name = f"{latent_file_prefix}_node_role_t{int(timestep)}.png"
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
                                        task_tree_type=task.tree_type,
                                        min_density_samples=args.min_density_samples,
                                    )
                                    if {"prior_normalized_z_mu_0", "prior_normalized_z_mu_1"}.issubset(rows.columns):
                                        prior_norm_rows = rows.copy()
                                        prior_norm_rows["z_mu_0"] = pd.to_numeric(
                                            prior_norm_rows["prior_normalized_z_mu_0"],
                                            errors="coerce",
                                        )
                                        prior_norm_rows["z_mu_1"] = pd.to_numeric(
                                            prior_norm_rows["prior_normalized_z_mu_1"],
                                            errors="coerce",
                                        )
                                        plot_combo_density(
                                            prior_norm_rows,
                                            combo_dir / "figures",
                                            combo_label=f"{combo_label}, prior-normalized posterior",
                                            grid_n=args.latent_density_grid_n,
                                            max_density_points=args.max_density_points,
                                            task_tree_type=task.tree_type,
                                            latent_file_prefix="revisit_prior_normalized_z0_z1_density",
                                            plot_context_heatmaps=False,
                                            x_axis_label="(z_mu_0 - prior_mu_0) / prior_sigma_0",
                                            y_axis_label="(z_mu_1 - prior_mu_1) / prior_sigma_1",
                                            x_panel_label="prior-norm z0",
                                            min_density_samples=args.min_density_samples,
                                        )
                                    print(f"Saved revisit latent density plots for {combo_label} to {combo_dir / 'figures'}")
    if failures:
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(outdir / "revisit_latent_density_failures.csv", index=False)


if __name__ == "__main__":
    main()
