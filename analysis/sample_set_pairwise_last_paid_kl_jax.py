#!/usr/bin/env python3
"""Sample-set latent variability diagnostic for revisit JAX models.

For each actual node-reward combination, this script generates multiple
pre-sampled noisy observation streams per node.  The model chooses observe/stop
actions normally; whenever it observes a node, the next value from that node's
pre-generated stream is supplied as the observation.

With ``--force-round-robin-observations``, the diagnostic instead forces a
matched observation history across sample sets: node 1, node 2, ..., all nodes,
then the second sample from node 1, node 2, and so on until the observation
limit.  This isolates latent variability due to sampled observation values from
variability due to the policy choosing different node orders.

The diagnostic extracts the diagonal-Gaussian latent parameters for the last
paid KL state: the latent after the second-to-last observed reward in a trial.
It then computes pairwise symmetric KL among sample sets with the same actual
reward pair.  Large within-pair KL means the representation varies with sample
history/order even after conditioning on the underlying node rewards.

It also computes within-sample-set latent movement: symmetric KL between each
paid latent and the previous paid latent in the same trial/sample stream.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_jax import planning as jp  # noqa: E402

try:
    from analysis import plot_revisit_latent_density_gaussian_pga_jax as gpga  # noqa: E402
except ModuleNotFoundError:
    import plot_revisit_latent_density_gaussian_pga_jax as gpga  # noqa: E402


PLOT_FONT_SIZE_PT = 7
PANEL_WIDTH_IN = 33 / 25.4
PANEL_HEIGHT_IN = 33 / 25.4
LOG_KL_EPS = 1e-12
BETA_COLOR_RAMP = ["#00441b", "#238b45", "#74c476"]
OPPORTUNITY_COLOR_RAMP = ["#6baed6", "#2171b5", "#08306b"]
FAMILY_COLOR_RAMPS = {
    "vary_beta": BETA_COLOR_RAMP,
    "vary_opportunity": OPPORTUNITY_COLOR_RAMP,
}
POSTERIOR_TRIAL_CONTOUR_MASS = 0.90


def parse_values(raw: str | None, typ=float) -> list:
    if raw is None:
        return []
    return [typ(x.strip()) for x in str(raw).replace(",", " ").split() if x.strip()]


def normalize_tree_name(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "2": "default",
        "2n": "default",
        "default2": "default",
        "default": "default",
        "3": "bandit3",
        "3n": "bandit3",
        "bandit3": "bandit3",
        "4": "disjoint2x2",
        "4n": "disjoint2x2",
        "disjoint2x2": "disjoint2x2",
        "6": "disjoint3x2",
        "6n": "disjoint3x2",
        "disjoint3x2": "disjoint3x2",
    }
    return aliases.get(key, key)


def preset_file_default() -> Path:
    return ROOT / "analyses" / "exp_binary" / "revisit_plot_presets.csv"


def preset_value(row: pd.Series, name: str, default: str | None = None) -> str:
    if name not in row or pd.isna(row[name]):
        if default is None:
            raise KeyError(f"Preset column {name!r} is missing.")
        return default
    value = str(row[name]).strip()
    return value if value else (default or "")


def value_token(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value)
    else:
        text = f"{numeric:g}" if math.isfinite(numeric) else str(value)
    return (
        text.strip()
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
        .replace(",", "_")
        .replace(" ", "_")
    )


def values_token(values) -> str:
    values = list(values or [])
    if not values:
        return "none"
    return "_".join(value_token(value) for value in values)


def dynamic_log_kl_limits(values, sem_values=None) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    candidates = [arr[np.isfinite(arr) & (arr > 0.0)]]
    if sem_values is not None:
        sem = np.asarray(sem_values, dtype=float)
        valid = np.isfinite(arr) & np.isfinite(sem)
        upper = arr[valid] + np.maximum(sem[valid], 0.0)
        lower = arr[valid] - np.maximum(sem[valid], 0.0)
        candidates.extend([upper[np.isfinite(upper) & (upper > 0.0)], lower[np.isfinite(lower) & (lower > 0.0)]])
    positive = np.concatenate([x for x in candidates if len(x)]) if any(len(x) for x in candidates) else np.asarray([])
    if len(positive) == 0:
        return LOG_KL_EPS, 1.0
    lo = max(10.0 ** math.floor(math.log10(float(np.nanmin(positive)))), LOG_KL_EPS)
    hi = 10.0 ** math.ceil(math.log10(float(np.nanmax(positive))))
    if not math.isfinite(hi) or hi <= lo:
        hi = lo * 10.0
    return lo, hi


def positive_kl_values(values, floor: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr) & (arr > floor), arr, floor)


def log_kl_yerr(y_values, sem_values, floor: float) -> np.ndarray:
    y = positive_kl_values(y_values, floor)
    sem = np.asarray(sem_values, dtype=float)
    lower = np.where(
        np.isfinite(sem),
        np.minimum(np.maximum(sem, 0.0), np.maximum(y - floor, 0.0)),
        np.nan,
    )
    upper = np.where(np.isfinite(sem), np.maximum(sem, 0.0), np.nan)
    return np.vstack([lower, upper])


def set_log_kl_axis(ax, lower: float, upper: float) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lower, upper)


def dynamic_linear_limits(values, sem_values=None) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    candidates = [arr[np.isfinite(arr)]]
    if sem_values is not None:
        sem = np.asarray(sem_values, dtype=float)
        valid = np.isfinite(arr) & np.isfinite(sem)
        candidates.extend([arr[valid] - np.maximum(sem[valid], 0.0), arr[valid] + np.maximum(sem[valid], 0.0)])
    finite = np.concatenate([x for x in candidates if len(x)]) if any(len(x) for x in candidates) else np.asarray([])
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    lo = min(0.0, float(np.nanmin(finite)))
    hi = float(np.nanmax(finite))
    if not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)
    return lo, hi + pad


def metric_y_values(values, lower: float, log_y: bool) -> np.ndarray:
    if log_y:
        return positive_kl_values(values, lower)
    return np.asarray(values, dtype=float)


def metric_yerr(y_values, sem_values, lower: float, log_y: bool) -> np.ndarray:
    if log_y:
        return log_kl_yerr(y_values, sem_values, lower)
    sem = np.asarray(sem_values, dtype=float)
    return np.where(np.isfinite(sem), np.maximum(sem, 0.0), np.nan)


def set_metric_axis(ax, lower: float, upper: float, log_y: bool) -> None:
    if log_y:
        set_log_kl_axis(ax, lower, upper)
    else:
        ax.set_ylim(lower, upper)


def metric_limits(frame: pd.DataFrame, mean_col: str, sem_col: str, log_y: bool) -> tuple[float, float]:
    if log_y:
        return dynamic_log_kl_limits(frame[mean_col].to_numpy(), frame[sem_col].to_numpy())
    return dynamic_linear_limits(frame[mean_col].to_numpy(), frame[sem_col].to_numpy())


def output_combo_label(args: argparse.Namespace) -> str:
    tree = normalize_tree_name(getattr(args, "output_tree_label", getattr(args, "tree", "default")))
    beta_values = getattr(args, "output_vary_beta_values", [])
    opportunity_values = getattr(args, "output_vary_opportunity_values", [])
    label = (
        f"{tree}_vary_beta_{values_token(beta_values)}"
        f"_vary_opp_{values_token(opportunity_values)}"
    )
    n_reward_combinations = int(getattr(args, "n_reward_combinations", 0) or 0)
    if n_reward_combinations > 0:
        label += f"_rewardcombos_{n_reward_combinations}"
    if getattr(args, "force_round_robin_observations", False):
        label += "_forced_round_robin"
    force_first_observe_node = int(getattr(args, "force_first_observe_node", 0) or 0)
    if force_first_observe_node > 0:
        label += f"_force_first_node_{force_first_observe_node}"
    return label


def resolve_output_dir(args: argparse.Namespace) -> Path:
    base = Path(args.outdir)
    if getattr(args, "no_combo_subdir", False):
        return base
    label = output_combo_label(args)
    return base if base.name == label else base / label


def should_write_normalized_z_aggregate_plots(args: argparse.Namespace) -> bool:
    tree_label = normalize_tree_name(getattr(args, "output_tree_label", getattr(args, "tree", "default")))
    return tree_label == "default"


def load_default_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset_file = Path(args.preset_file)
    presets = pd.read_csv(preset_file)
    presets["tree_key"] = presets["tree"].map(normalize_tree_name)
    tree = normalize_tree_name(args.tree)
    beta_rows = presets[(presets["tree_key"] == tree) & (presets["vary"] == "beta")]
    opp_rows = presets[(presets["tree_key"] == tree) & (presets["vary"] == "opportunity")]
    if beta_rows.empty or opp_rows.empty:
        raise ValueError(f"Could not find beta/opportunity preset rows for tree={tree}.")
    beta_row = beta_rows.iloc[0]
    opp_row = opp_rows.iloc[0]
    shared = beta_row

    args.tree_size = int(preset_value(shared, "tree_size", "2"))
    args.tree_type = preset_value(shared, "tree_config", "default")
    args.input_type = preset_value(shared, "input_type", "uniform")
    args.expansion_decision_version = preset_value(shared, "expansion_decision_version", "lstm")
    args.model_variant = preset_value(shared, "model_variant", "vae")
    args.lambda_value = float((parse_values(args.lambda_value, float) or parse_values(preset_value(shared, "lambda_arg"), float))[0])
    args.alpha = float((parse_values(args.alpha, float) or parse_values(preset_value(shared, "alpha_arg"), float))[0])
    args.seeds = parse_values(args.seeds, int) or parse_values(preset_value(shared, "seed_arg"), int)
    args.sigmas = parse_values(args.sigmas, float) or parse_values(preset_value(shared, "sigma_arg", "0"), float)
    args.rnn_units = int((parse_values(args.rnn_units, int) or parse_values(preset_value(shared, "rnn_units_arg"), int))[0])
    args.latent_dim = int((parse_values(args.latent_dim, int) or parse_values(preset_value(shared, "latent_dim_arg"), int))[0])
    args.max_observations_before_stop = int(
        args.max_observations_before_stop
        if args.max_observations_before_stop is not None
        else preset_value(shared, "max_observations_arg", "10")
    )

    beta_values = parse_values(args.vary_beta_values, float) or parse_values(preset_value(beta_row, "beta_arg"), float)
    beta_family_opps = parse_values(preset_value(beta_row, "opportunity_arg", "0.0"), float)
    opp_values = parse_values(args.vary_opportunity_values, float) or parse_values(
        preset_value(opp_row, "opportunity_arg"), float
    )
    opp_family_betas = parse_values(preset_value(opp_row, "beta_arg", "1000.0"), float)
    combos = []
    for beta in beta_values:
        for opp in beta_family_opps:
            combos.append(("vary_beta", "beta", float(beta), float(beta), float(opp)))
    for beta in opp_family_betas:
        for opp in opp_values:
            combos.append(("vary_opportunity", "opportunity", float(opp), float(beta), float(opp)))
    args.parameter_combos = combos
    args.output_tree_label = tree
    args.output_vary_beta_values = beta_values
    args.output_vary_opportunity_values = opp_values
    if args.outdir is None:
        args.outdir = str(Path("results") / "revisit_hypothesis2" / "sample_set_pairwise_last_paid_kl")
    print(f"Using preset rows from {preset_file} for tree={tree}", flush=True)
    return args


def make_config(
    args: argparse.Namespace,
    *,
    seed: int,
    beta: float,
    opportunity: float,
    sigma: float,
) -> jp.RunConfig:
    return jp.RunConfig(
        lambda_=float(args.lambda_value),
        alpha=float(args.alpha),
        beta=float(beta),
        model_dir=str(args.checkpoint_root),
        epochs=120,
        input_type=args.input_type,
        seed=int(seed),
        tree_size=int(args.tree_size),
        train_mode="sim",
        tree_type=args.tree_type,
        opportunity_cost=float(opportunity),
        expansion_decision_version=jp.normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=jp.normalize_model_variant(args.model_variant),
        rnn_units=int(args.rnn_units),
        latent_dim=int(args.latent_dim),
        sim_dir="outputs/jax_simulations",
        n_sim_trials=0,
        num_envs=200,
        num_steps=int(args.max_observations_before_stop) + 1,
        update_epochs=int(args.update_epochs),
        ppo_minibatches=1,
        steps_per_epoch=0,
        return_target_rollouts=8,
        return_target_mode="sampled_lambda",
        sampled_lambda_critic=args.sampled_lambda_critic,
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=100,
        target_critic_tau=1.0,
        backend=args.backend,
        jit_training=False,
        profile_update_components=False,
        profile_update_components_every=0,
        enable_reconstruction=False,
        enable_probe=False,
        allow_node_revisit=True,
        max_observations_before_stop=int(args.max_observations_before_stop),
        observation_sigma=float(sigma),
        kl_start_multiplier=float(args.kl_start_multiplier),
        kl_annealing_epochs=int(args.kl_annealing_epochs),
        node_coverage_aux_coef=0.0,
        node_coverage_aux_epochs=0,
    )


def schedule_for(beta: float) -> jp.ScheduleValues:
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


def build_reward_combination_trials(
    reward_values: np.ndarray,
    num_nodes: int,
    sigma: float,
    n_sample_sets: int,
    max_observations: int,
    seed: int,
    n_reward_combinations: int = 0,
    reward_combination_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    combo_rng = np.random.default_rng(seed if reward_combination_seed is None else reward_combination_seed)
    reward_combos = [
        tuple(float(value) for value in combo)
        for combo in itertools.product(reward_values, repeat=int(num_nodes))
    ]
    total_reward_combos = len(reward_combos)
    n_reward_combinations = int(n_reward_combinations or 0)
    if 0 < n_reward_combinations < total_reward_combos:
        selected_indices = np.sort(
            combo_rng.choice(total_reward_combos, size=n_reward_combinations, replace=False)
        )
        reward_combos = [reward_combos[int(idx)] for idx in selected_indices]
        original_condition_indices = [int(idx) for idx in selected_indices]
    else:
        original_condition_indices = list(range(total_reward_combos))
    rows = []
    rewards = []
    streams = []
    for sampled_condition_idx, (original_condition_idx, combo) in enumerate(
        zip(original_condition_indices, reward_combos)
    ):
        for set_idx in range(n_sample_sets):
            rewards.append(list(combo))
            if abs(float(sigma)) <= 1e-12:
                node_streams = np.asarray(
                    [np.full(max_observations, reward, dtype=float) for reward in combo]
                )
            else:
                node_streams = np.vstack(
                    [
                        rng.normal(loc=reward, scale=float(sigma), size=max_observations)
                        for reward in combo
                    ]
                )
            streams.append(node_streams)
            row = {
                "condition_index": sampled_condition_idx,
                "original_condition_index": original_condition_idx,
                "sample_set": set_idx,
                "n_sampled_reward_combinations": len(reward_combos),
                "n_total_reward_combinations": total_reward_combos,
            }
            for node_idx, reward in enumerate(combo, start=1):
                row[f"reward_node_{node_idx}"] = reward
            if len(combo) >= 2:
                row["node1_reward"] = combo[0]
                row["node2_reward"] = combo[1]
            rows.append(row)
    return np.asarray(rewards, dtype=np.float32), np.asarray(streams, dtype=np.float32), pd.DataFrame(rows)


def rollout_with_streams(
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    streams: np.ndarray,
    seed_offset: int,
    force_round_robin_observations: bool = False,
    force_first_observe_node: int = 0,
    progress_label: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_trials = rewards.shape[0]
    if progress_label:
        print(
            f"{progress_label}: starting rollout with {n_trials} total trial(s), "
            f"num_steps={int(config.num_steps)}, max_observations={int(config.max_observations_before_stop)}",
            flush=True,
        )
    reward_feature_dim = int(model.reward_feature_dim_override) or jp.reward_feature_dim_for_sigma(
        config.observation_sigma
    )
    carry = jp.initial_carry(
        n_trials,
        task,
        config.rnn_units,
        reward_feature_dim,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    carry = jp.reset_done_envs(carry, jnp.asarray(rewards, dtype=jnp.float32))
    sched = schedule_for(config.beta)
    rng = jax.random.PRNGKey(config.seed + 910_000 + int(seed_offset))
    stream_counts = np.zeros((n_trials, task.num_nodes), dtype=np.int32)
    observed_latents = [[] for _ in range(n_trials)]
    stop_t = np.full(n_trials, np.nan)
    obs_count = np.zeros(n_trials, dtype=np.int32)
    first_observed_node = np.full(n_trials, -1, dtype=np.int32)
    first_observed_timestep = np.full(n_trials, np.nan, dtype=float)

    for timestep in range(1, config.num_steps + 1):
        rng, step_rng = jax.random.split(rng)
        if force_round_robin_observations:
            if timestep <= int(config.max_observations_before_stop):
                forced_node = (timestep - 1) % int(task.num_nodes)
                action = np.full(n_trials, forced_node, dtype=np.int32)
            else:
                action = np.full(n_trials, int(task.num_nodes), dtype=np.int32)
        elif timestep == 1 and 1 <= int(force_first_observe_node) <= int(task.num_nodes):
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
        for trial in range(n_trials):
            if bool(np.asarray(carry.done)[trial]):
                continue
            if action[trial] < task.num_nodes:
                node = int(action[trial])
                sample_idx = min(stream_counts[trial, node], streams.shape[2] - 1)
                forced_observation[trial] = streams[trial, node, sample_idx]
                sample_position[trial] = int(sample_idx)
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
        is_observe = np.asarray(trans_np.is_observe) > 0.5
        is_stop = np.asarray(trans_np.is_stop) > 0.5
        z_mu = np.asarray(trans_np.z_mu, dtype=float)
        z_logvar = np.asarray(trans_np.z_logvar, dtype=float)
        prior_mu = np.asarray(trans_np.prior_mu, dtype=float)
        prior_logvar = np.asarray(trans_np.prior_logvar, dtype=float)
        node_index = np.asarray(trans_np.node_index, dtype=np.int32)
        for trial in range(n_trials):
            if is_observe[trial]:
                obs_count[trial] += 1
                node = int(node_index[trial])
                if first_observed_node[trial] < 0:
                    first_observed_node[trial] = node
                    first_observed_timestep[trial] = float(timestep)
                observed_latents[trial].append((
                    timestep,
                    node + 1,
                    int(sample_position[trial]) if sample_position[trial] >= 0 else np.nan,
                    float(forced_observation[trial]),
                    float(rewards[trial, node]),
                    z_mu[trial].copy(),
                    z_logvar[trial].copy(),
                    prior_mu[trial].copy(),
                    prior_logvar[trial].copy(),
                ))
            if is_stop[trial] and not np.isfinite(stop_t[trial]):
                stop_t[trial] = float(timestep)
        if progress_label:
            done_now = np.asarray(carry.done, dtype=bool)
            print(
                f"{progress_label}: simulated timestep {timestep}/{int(config.num_steps)}; "
                f"active={int(np.sum(~done_now))}/{n_trials}; "
                f"stopped={int(np.sum(np.isfinite(stop_t)))}/{n_trials}; "
                f"observed_events={int(np.sum(obs_count))}",
                flush=True,
            )

    rows = []
    paid_rows = []
    for trial, latents in enumerate(observed_latents):
        if len(latents) < 2:
            rows.append(
                {
                    "trial_index": trial,
                    "valid_last_paid": False,
                    "force_round_robin_observations": bool(force_round_robin_observations),
                    "force_first_observe_node": int(force_first_observe_node),
                    "first_observed_node": (
                        int(first_observed_node[trial]) + 1
                        if first_observed_node[trial] >= 0
                        else np.nan
                    ),
                    "first_observed_timestep": first_observed_timestep[trial],
                    "stop_timestep": stop_t[trial],
                    "observations": int(obs_count[trial]),
                    "last_paid_timestep": np.nan,
                    "z_mu": None,
                    "z_logvar": None,
                    "prior_mu": None,
                    "prior_logvar": None,
                }
            )
            continue
        for paid_idx, (
            paid_timestep,
            observed_node,
            sample_idx,
            observed_reward,
            actual_observed_reward,
            mu,
            logvar,
            prior_mu_i,
            prior_logvar_i,
        ) in enumerate(latents[:-1], start=1):
            paid_rows.append(
                {
                    "trial_index": trial,
                    "force_round_robin_observations": bool(force_round_robin_observations),
                    "force_first_observe_node": int(force_first_observe_node),
                    "first_observed_node": (
                        int(first_observed_node[trial]) + 1
                        if first_observed_node[trial] >= 0
                        else np.nan
                    ),
                    "first_observed_timestep": first_observed_timestep[trial],
                    "paid_observation_index": int(paid_idx),
                    "n_paid_latents": int(len(latents) - 1),
                    "stop_timestep": stop_t[trial],
                    "observations": int(obs_count[trial]),
                    "paid_timestep": int(paid_timestep),
                    "observed_node": int(observed_node),
                    "sample_position": sample_idx,
                    "observed_reward": observed_reward,
                    "actual_observed_reward": actual_observed_reward,
                    "z_mu": mu,
                    "z_logvar": logvar,
                    "prior_mu": prior_mu_i,
                    "prior_logvar": prior_logvar_i,
                }
            )
        (
            last_paid_timestep,
            observed_node,
            sample_idx,
            observed_reward,
            actual_observed_reward,
            mu,
            logvar,
            prior_mu_i,
            prior_logvar_i,
        ) = latents[-2]
        rows.append(
            {
                "trial_index": trial,
                "valid_last_paid": True,
                "force_round_robin_observations": bool(force_round_robin_observations),
                "force_first_observe_node": int(force_first_observe_node),
                "first_observed_node": (
                    int(first_observed_node[trial]) + 1
                    if first_observed_node[trial] >= 0
                    else np.nan
                ),
                "first_observed_timestep": first_observed_timestep[trial],
                "stop_timestep": stop_t[trial],
                "observations": int(obs_count[trial]),
                "last_paid_timestep": int(last_paid_timestep),
                "last_paid_observed_node": int(observed_node),
                "last_paid_sample_position": sample_idx,
                "last_paid_observed_reward": observed_reward,
                "last_paid_actual_observed_reward": actual_observed_reward,
                "z_mu": mu,
                "z_logvar": logvar,
                "prior_mu": prior_mu_i,
                "prior_logvar": prior_logvar_i,
            }
        )
    latent_df = pd.DataFrame(rows)
    paid_latent_df = pd.DataFrame(paid_rows)
    if progress_label:
        print(
            f"{progress_label}: rollout finished; valid_last_paid="
            f"{int(latent_df['valid_last_paid'].sum()) if 'valid_last_paid' in latent_df.columns else 0}/"
            f"{len(latent_df)}; paid_latent_rows={len(paid_latent_df)}",
            flush=True,
        )
    return latent_df, paid_latent_df


def symmetric_diag_gaussian_kl(mu: np.ndarray, logvar: np.ndarray) -> float:
    var = np.exp(np.clip(logvar, -20.0, 20.0))
    diff = mu[:, None, :] - mu[None, :, :]
    var_i = var[:, None, :]
    var_j = var[None, :, :]
    logvar_i = logvar[:, None, :]
    logvar_j = logvar[None, :, :]
    kl_ij = 0.5 * (
        logvar_j
        - logvar_i
        + (var_i + diff**2) / (var_j + 1e-12)
        - 1.0
    )
    kl_ji = 0.5 * (
        logvar_i
        - logvar_j
        + (var_j + diff**2) / (var_i + 1e-12)
        - 1.0
    )
    sym = 0.5 * (np.mean(kl_ij, axis=-1) + np.mean(kl_ji, axis=-1))
    upper = np.triu_indices(sym.shape[0], k=1)
    if len(upper[0]) == 0:
        return np.nan
    return float(np.nanmean(sym[upper]))


def symmetric_diag_gaussian_kl_pair(
    mu_a: np.ndarray,
    logvar_a: np.ndarray,
    mu_b: np.ndarray,
    logvar_b: np.ndarray,
) -> float:
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    logvar_a = np.asarray(logvar_a, dtype=float)
    logvar_b = np.asarray(logvar_b, dtype=float)
    var_a = np.exp(np.clip(logvar_a, -20.0, 20.0))
    var_b = np.exp(np.clip(logvar_b, -20.0, 20.0))
    diff2 = (mu_a - mu_b) ** 2
    kl_ab = 0.5 * (
        logvar_b
        - logvar_a
        + (var_a + diff2) / (var_b + 1e-12)
        - 1.0
    )
    kl_ba = 0.5 * (
        logvar_a
        - logvar_b
        + (var_b + diff2) / (var_a + 1e-12)
        - 1.0
    )
    return float(np.nanmean(0.5 * (kl_ab + kl_ba)))


def pairwise_mean_abs_difference(values: np.ndarray) -> float:
    diff = np.abs(values[:, None, :] - values[None, :, :])
    per_pair = np.mean(diff, axis=-1)
    upper = np.triu_indices(per_pair.shape[0], k=1)
    if len(upper[0]) == 0:
        return np.nan
    return float(np.nanmean(per_pair[upper]))


def sem_finite(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def family_param_label(row: pd.Series) -> str:
    if row["family"] == "vary_beta":
        return f"beta={float(row['parameter_value']):g}"
    if row["family"] == "vary_opportunity":
        return f"opp={float(row['parameter_value']):g}"
    return f"{row['parameter_name']}={float(row['parameter_value']):g}"


def family_colors_for_values(family: str, values: list[float]) -> dict[float, object]:
    ramp = FAMILY_COLOR_RAMPS.get(family, ["#333333"])
    colors = color_values(len(values), ramp)
    return {float(value): color for value, color in zip(values, colors)}


def symmetric_observed_difference_xlim(frame: pd.DataFrame, x_col: str) -> tuple[float, float]:
    values = pd.to_numeric(frame.get(x_col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        max_abs = 1.0
    else:
        max_abs = float(np.nanmax(np.abs(finite)))
        if not math.isfinite(max_abs) or max_abs <= 0.0:
            max_abs = 1.0
    pad = 0.05 * max_abs
    return -(max_abs + pad), max_abs + pad


def assign_observed_difference_bins(
    detail: pd.DataFrame,
    bin_width: float,
    num_bins: int,
    *,
    include_seed: bool = False,
) -> tuple[pd.DataFrame, str, str]:
    detail = detail.copy()
    signed = "current_minus_previous_same_node_value" in detail.columns
    raw_col = (
        "current_minus_previous_same_node_value"
        if signed
        else "abs_current_minus_previous_same_node_value"
    )
    if raw_col not in detail.columns:
        fallback = "delta_bin" if "delta_bin" in detail.columns else "abs_delta_bin"
        fallback_id = f"{fallback}_id"
        detail[fallback_id] = np.arange(len(detail), dtype=int)
        return detail, fallback, fallback_id

    bin_col = "delta_bin" if signed else "abs_delta_bin"
    bin_id_col = "delta_bin_id" if signed else "abs_delta_bin_id"
    values = pd.to_numeric(detail[raw_col], errors="coerce")
    num_bins = int(num_bins)
    if num_bins > 0:
        time_col = "timestep" if "timestep" in detail.columns else "paid_timestep"
        panel_cols = [col for col in ["sigma", time_col] if col in detail.columns]
        if include_seed and "seed" in detail.columns:
            panel_cols.insert(0, "seed")
        if not panel_cols:
            panel_cols = [pd.Series(np.zeros(len(detail), dtype=int), index=detail.index)]
        detail[bin_col] = np.nan
        detail[bin_id_col] = np.nan
        grouped = detail.groupby(panel_cols, sort=False, dropna=False) if all(isinstance(col, str) for col in panel_cols) else [(None, detail)]
        for _, group in grouped:
            finite_idx = group.index[np.isfinite(values.loc[group.index].to_numpy(dtype=float))]
            n = len(finite_idx)
            if n == 0:
                continue
            k = max(1, min(num_bins, n))
            ordered_idx = values.loc[finite_idx].sort_values(kind="mergesort").index.to_numpy()
            bin_ids = np.floor(np.arange(n, dtype=float) * k / n).astype(int)
            bin_ids = np.minimum(bin_ids, k - 1)
            centers = np.full(n, np.nan, dtype=float)
            for bin_id in range(k):
                positions = np.flatnonzero(bin_ids == bin_id)
                if positions.size:
                    centers[positions] = float(values.loc[ordered_idx[positions]].mean())
            detail.loc[ordered_idx, bin_id_col] = bin_ids
            detail.loc[ordered_idx, bin_col] = centers
    else:
        bin_width = float(bin_width)
        if bin_width > 0:
            detail[bin_col] = np.round(values / bin_width) * bin_width
        else:
            detail[bin_col] = values
        detail[bin_id_col] = detail[bin_col]

    if signed and "abs_current_minus_previous_same_node_value" in detail.columns:
        abs_values = pd.to_numeric(
            detail["abs_current_minus_previous_same_node_value"], errors="coerce"
        )
        bin_width = float(bin_width)
        if bin_width > 0:
            detail["abs_delta_bin"] = np.round(abs_values / bin_width) * bin_width
        else:
            detail["abs_delta_bin"] = abs_values
        detail["abs_delta_bin_id"] = detail["abs_delta_bin"]
    return detail, bin_col, bin_id_col


def enrich_paid_latents_with_metadata(paid_latent_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if paid_latent_df.empty:
        return paid_latent_df.copy()
    meta = metadata.reset_index().rename(columns={"index": "trial_index"})
    return paid_latent_df.merge(meta, on="trial_index", how="left")


def compute_current_vs_previous_latent_change(
    paid_latents: pd.DataFrame,
    bin_width: float,
    num_bins: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "condition_index",
        "sample_set",
        "trial_index",
        "paid_observation_index",
        "observed_node",
        "observed_reward",
        "z_mu",
        "z_logvar",
    }
    if paid_latents.empty or not required.issubset(paid_latents.columns):
        return pd.DataFrame(), pd.DataFrame()
    trial_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "condition_index",
        "sample_set",
        "trial_index",
    ]
    df = paid_latents.copy()
    df["observed_node"] = pd.to_numeric(df["observed_node"], errors="coerce")
    df["observed_reward"] = pd.to_numeric(df["observed_reward"], errors="coerce")
    df["paid_observation_index"] = pd.to_numeric(df["paid_observation_index"], errors="coerce")
    df = df[np.isfinite(df["observed_node"]) & np.isfinite(df["paid_observation_index"])].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["observed_node"] = df["observed_node"].astype(int)
    df = df.sort_values(trial_cols + ["paid_observation_index"]).reset_index(drop=True)

    by_trial = df.groupby(trial_cols, sort=False, dropna=False)
    df["_trial_observation_position"] = by_trial.cumcount()
    df["previous_observation_z_mu"] = by_trial["z_mu"].shift(1)
    df["previous_observation_z_logvar"] = by_trial["z_logvar"].shift(1)
    by_trial_node = df.groupby(trial_cols + ["observed_node"], sort=False, dropna=False)
    df["previous_same_node_observed_reward"] = by_trial_node["observed_reward"].shift(1)
    df["previous_same_node_paid_observation_index"] = by_trial_node[
        "paid_observation_index"
    ].shift(1)
    df["previous_same_node_observation_position"] = by_trial_node[
        "_trial_observation_position"
    ].shift(1)

    first_node_positions = (
        df.groupby(trial_cols + ["observed_node"], sort=False, dropna=False)[
            "_trial_observation_position"
        ]
        .min()
        .reset_index()
        .sort_values(trial_cols + ["_trial_observation_position", "observed_node"])
    )
    first_node_positions["_node_first_rank"] = first_node_positions.groupby(
        trial_cols, sort=False, dropna=False
    ).cumcount()
    first_seen = first_node_positions[
        first_node_positions["_node_first_rank"] == 0
    ][trial_cols + ["observed_node", "_trial_observation_position"]].rename(
        columns={
            "observed_node": "_first_seen_node",
            "_trial_observation_position": "_first_seen_position",
        }
    )
    second_seen = first_node_positions[
        first_node_positions["_node_first_rank"] == 1
    ][trial_cols + ["observed_node", "_trial_observation_position"]].rename(
        columns={
            "observed_node": "_second_seen_node",
            "_trial_observation_position": "_second_seen_position",
        }
    )
    df = df.merge(first_seen, on=trial_cols, how="left")
    df = df.merge(second_seen, on=trial_cols, how="left")
    df["_earliest_other_node_position"] = np.where(
        df["_first_seen_node"] != df["observed_node"],
        df["_first_seen_position"],
        df["_second_seen_position"],
    )
    df["other_node_observed_before_previous_same_node"] = (
        np.isfinite(
            pd.to_numeric(df["_earliest_other_node_position"], errors="coerce")
        )
        & np.isfinite(
            pd.to_numeric(
                df["previous_same_node_observation_position"], errors="coerce"
            )
        )
        & (
            pd.to_numeric(df["_earliest_other_node_position"], errors="coerce")
            < pd.to_numeric(
                df["previous_same_node_observation_position"], errors="coerce"
            )
        )
    )

    valid = (
        df["previous_observation_z_mu"].notna()
        & df["previous_observation_z_logvar"].notna()
        & np.isfinite(pd.to_numeric(df["previous_same_node_observed_reward"], errors="coerce"))
        & df["other_node_observed_before_previous_same_node"]
    )
    detail = df.loc[valid].copy()
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail["current_minus_previous_same_node_value"] = (
        pd.to_numeric(detail["observed_reward"], errors="coerce")
        - pd.to_numeric(detail["previous_same_node_observed_reward"], errors="coerce")
    )
    detail["abs_current_minus_previous_same_node_value"] = np.abs(
        detail["current_minus_previous_same_node_value"]
    )
    detail, delta_col, delta_id_col = assign_observed_difference_bins(
        detail,
        bin_width,
        num_bins,
        include_seed=False,
    )

    current_mu = np.stack(detail["z_mu"].to_numpy()).astype(float)
    current_logvar = np.stack(detail["z_logvar"].to_numpy()).astype(float)
    previous_mu = np.stack(detail["previous_observation_z_mu"].to_numpy()).astype(float)
    previous_logvar = np.stack(detail["previous_observation_z_logvar"].to_numpy()).astype(float)
    var_current = np.exp(np.clip(current_logvar, -20.0, 20.0))
    var_previous = np.exp(np.clip(previous_logvar, -20.0, 20.0))
    diff2 = (current_mu - previous_mu) ** 2
    kl_current_previous = 0.5 * (
        previous_logvar
        - current_logvar
        + (var_current + diff2) / (var_previous + 1e-12)
        - 1.0
    )
    kl_previous_current = 0.5 * (
        current_logvar
        - previous_logvar
        + (var_previous + diff2) / (var_current + 1e-12)
        - 1.0
    )
    detail["latent_sym_kl_current_vs_previous_observation"] = np.nanmean(
        0.5 * (kl_current_previous + kl_previous_current),
        axis=1,
    )
    mu_delta = current_mu - previous_mu
    detail["z0_mu_delta_current_vs_previous_observation"] = (
        mu_delta[:, 0] if mu_delta.shape[1] > 0 else np.nan
    )
    detail["z1_mu_delta_current_vs_previous_observation"] = (
        mu_delta[:, 1] if mu_delta.shape[1] > 1 else np.nan
    )
    if mu_delta.shape[1] > 1:
        detail["mean_z01_mu_delta_current_vs_previous_observation"] = np.nanmean(mu_delta[:, :2], axis=1)
        detail["mean_z01_mu_displacement_current_vs_previous_observation"] = np.nanmean(
            np.abs(mu_delta[:, :2]),
            axis=1,
        )
    else:
        detail["mean_z01_mu_delta_current_vs_previous_observation"] = np.nan
        detail["mean_z01_mu_displacement_current_vs_previous_observation"] = np.nan
    detail["z0_mu_displacement_current_vs_previous_observation"] = (
        np.abs(mu_delta[:, 0]) if mu_delta.shape[1] > 0 else np.nan
    )
    detail["z1_mu_displacement_current_vs_previous_observation"] = (
        np.abs(mu_delta[:, 1]) if mu_delta.shape[1] > 1 else np.nan
    )
    keep_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "condition_index",
        "original_condition_index",
        "sample_set",
        "paid_timestep",
        "paid_observation_index",
        "observed_node",
        "previous_same_node_paid_observation_index",
        "other_node_observed_before_previous_same_node",
        "current_minus_previous_same_node_value",
        "delta_bin",
        "delta_bin_id",
        "abs_current_minus_previous_same_node_value",
        "abs_delta_bin",
        "abs_delta_bin_id",
        "latent_sym_kl_current_vs_previous_observation",
        "z0_mu_delta_current_vs_previous_observation",
        "z1_mu_delta_current_vs_previous_observation",
        "mean_z01_mu_delta_current_vs_previous_observation",
        "z0_mu_displacement_current_vs_previous_observation",
        "z1_mu_displacement_current_vs_previous_observation",
        "mean_z01_mu_displacement_current_vs_previous_observation",
    ]
    detail = detail[keep_cols].rename(columns={"paid_timestep": "timestep"})
    per_stream = (
        detail.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "seed",
                "timestep",
                delta_id_col,
                delta_col,
                "original_condition_index",
                "sample_set",
            ],
            dropna=False,
        )
        .agg(
            latent_sym_kl_current_vs_previous_observation=(
                "latent_sym_kl_current_vs_previous_observation",
                "mean",
            ),
            z0_mu_delta_current_vs_previous_observation=(
                "z0_mu_delta_current_vs_previous_observation",
                "mean",
            ),
            z1_mu_delta_current_vs_previous_observation=(
                "z1_mu_delta_current_vs_previous_observation",
                "mean",
            ),
            mean_z01_mu_delta_current_vs_previous_observation=(
                "mean_z01_mu_delta_current_vs_previous_observation",
                "mean",
            ),
            z0_mu_displacement_current_vs_previous_observation=(
                "z0_mu_displacement_current_vs_previous_observation",
                "mean",
            ),
            z1_mu_displacement_current_vs_previous_observation=(
                "z1_mu_displacement_current_vs_previous_observation",
                "mean",
            ),
            mean_z01_mu_displacement_current_vs_previous_observation=(
                "mean_z01_mu_displacement_current_vs_previous_observation",
                "mean",
            ),
        )
        .reset_index()
    )
    summary = (
        per_stream.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "timestep",
                delta_id_col,
                delta_col,
            ],
            dropna=False,
        )
        .agg(
            mean_latent_sym_kl=("latent_sym_kl_current_vs_previous_observation", "mean"),
            sem_latent_sym_kl=("latent_sym_kl_current_vs_previous_observation", sem_finite),
            mean_z0_mu_delta=("z0_mu_delta_current_vs_previous_observation", "mean"),
            sem_z0_mu_delta=("z0_mu_delta_current_vs_previous_observation", sem_finite),
            mean_z1_mu_delta=("z1_mu_delta_current_vs_previous_observation", "mean"),
            sem_z1_mu_delta=("z1_mu_delta_current_vs_previous_observation", sem_finite),
            mean_z01_mu_delta=("mean_z01_mu_delta_current_vs_previous_observation", "mean"),
            sem_z01_mu_delta=("mean_z01_mu_delta_current_vs_previous_observation", sem_finite),
            mean_z0_mu_displacement=("z0_mu_displacement_current_vs_previous_observation", "mean"),
            sem_z0_mu_displacement=("z0_mu_displacement_current_vs_previous_observation", sem_finite),
            mean_z1_mu_displacement=("z1_mu_displacement_current_vs_previous_observation", "mean"),
            sem_z1_mu_displacement=("z1_mu_displacement_current_vs_previous_observation", sem_finite),
            mean_z01_mu_displacement=("mean_z01_mu_displacement_current_vs_previous_observation", "mean"),
            sem_z01_mu_displacement=("mean_z01_mu_displacement_current_vs_previous_observation", sem_finite),
            n_streams=("latent_sym_kl_current_vs_previous_observation", "count"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return detail, summary


def summarize_current_vs_previous_latent_change_detail(
    detail: pd.DataFrame,
    bin_width: float,
    num_bins: int = 8,
    include_seed: bool = False,
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    detail = detail.copy()
    if "other_node_observed_before_previous_same_node" in detail.columns:
        flag = detail["other_node_observed_before_previous_same_node"]
        if flag.dtype == object:
            flag = flag.astype(str).str.lower().isin({"true", "1", "t", "yes"})
        else:
            flag = flag.astype(bool)
        detail = detail[flag].copy()
        if detail.empty:
            return pd.DataFrame()
    else:
        print(
            "Current-vs-previous detail CSV lacks the "
            "other_node_observed_before_previous_same_node filter flag; "
            "re-run without --plot-only to apply the stricter inclusion rule.",
            flush=True,
        )
    if "current_minus_previous_same_node_value" not in detail.columns and "abs_current_minus_previous_same_node_value" in detail.columns:
        print(
            "Current-vs-previous detail CSV has only absolute value differences; "
            "falling back to an absolute x-axis. Re-run the diagnostic without "
            "--plot-only to save signed current_minus_previous_same_node_value.",
            flush=True,
        )
    detail, delta_col, delta_id_col = assign_observed_difference_bins(
        detail,
        bin_width,
        num_bins,
        include_seed=include_seed,
    )
    if delta_col not in detail.columns:
        return pd.DataFrame()

    agg_spec = {
        "latent_sym_kl_current_vs_previous_observation": (
            "latent_sym_kl_current_vs_previous_observation",
            "mean",
        )
    }
    for source_col in [
        "z0_mu_delta_current_vs_previous_observation",
        "z1_mu_delta_current_vs_previous_observation",
        "mean_z01_mu_delta_current_vs_previous_observation",
        "z0_mu_displacement_current_vs_previous_observation",
        "z1_mu_displacement_current_vs_previous_observation",
        "mean_z01_mu_displacement_current_vs_previous_observation",
    ]:
        if source_col in detail.columns:
            agg_spec[source_col] = (source_col, "mean")
    per_stream = (
        detail.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "seed",
                "timestep",
                delta_id_col,
                delta_col,
                "original_condition_index",
                "sample_set",
            ],
            dropna=False,
        )
        .agg(**agg_spec)
        .reset_index()
    )

    summary_aggs = {
        "mean_latent_sym_kl": ("latent_sym_kl_current_vs_previous_observation", "mean"),
        "sem_latent_sym_kl": ("latent_sym_kl_current_vs_previous_observation", sem_finite),
        "n_streams": ("latent_sym_kl_current_vs_previous_observation", "count"),
        "n_seeds": ("seed", "nunique"),
    }
    optional_summary_cols = {
        "z0_mu_delta_current_vs_previous_observation": ("mean_z0_mu_delta", "sem_z0_mu_delta"),
        "z1_mu_delta_current_vs_previous_observation": ("mean_z1_mu_delta", "sem_z1_mu_delta"),
        "mean_z01_mu_delta_current_vs_previous_observation": (
            "mean_z01_mu_delta",
            "sem_z01_mu_delta",
        ),
        "z0_mu_displacement_current_vs_previous_observation": (
            "mean_z0_mu_displacement",
            "sem_z0_mu_displacement",
        ),
        "z1_mu_displacement_current_vs_previous_observation": (
            "mean_z1_mu_displacement",
            "sem_z1_mu_displacement",
        ),
        "mean_z01_mu_displacement_current_vs_previous_observation": (
            "mean_z01_mu_displacement",
            "sem_z01_mu_displacement",
        ),
    }
    for source_col, (mean_col, sem_col) in optional_summary_cols.items():
        if source_col in per_stream.columns:
            summary_aggs[mean_col] = (source_col, "mean")
            summary_aggs[sem_col] = (source_col, sem_finite)
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "timestep",
        delta_id_col,
        delta_col,
    ]
    if include_seed:
        group_cols.insert(6, "seed")
    return (
        per_stream.groupby(
            group_cols,
            dropna=False,
        )
        .agg(**summary_aggs)
        .reset_index()
    )


def stable_group_seed(values: tuple, base_seed: int = 0) -> int:
    text = "|".join(str(value) for value in values)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return (int(digest, 16) + int(base_seed)) % (2**32 - 1)


def compute_sigma_pairwise_last_paid_kl(
    last_paid: pd.DataFrame,
    trials_per_sigma: int = 5,
    random_seed: int = 1729,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    required = {
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "seed",
        "original_condition_index",
        "sample_set",
        "sigma",
        "z_mu",
        "z_logvar",
    }
    if last_paid.empty or not required.issubset(last_paid.columns):
        return pd.DataFrame(), pd.DataFrame()
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "seed",
        "original_condition_index",
    ]
    trials_per_sigma = int(trials_per_sigma)
    for group_key, group in last_paid.groupby(group_cols, sort=False, dropna=False):
        by_sigma = {}
        for sigma, sigma_group in group.groupby("sigma", sort=False, dropna=False):
            sigma_value = float(sigma)
            sigma_group = sigma_group.sort_values("sample_set").reset_index(drop=True)
            if trials_per_sigma > 0 and len(sigma_group) > trials_per_sigma:
                seed = stable_group_seed(tuple(group_key) + (sigma_value,), random_seed)
                selected_idx = np.random.default_rng(seed).choice(
                    len(sigma_group),
                    size=trials_per_sigma,
                    replace=False,
                )
                sigma_group = sigma_group.iloc[np.sort(selected_idx)].reset_index(drop=True)
            by_sigma[sigma_value] = sigma_group
        sigmas = sorted(by_sigma)
        for sigma_a in sigmas:
            for sigma_b in sigmas:
                group_a = by_sigma[sigma_a]
                group_b = by_sigma[sigma_b]
                for idx_a, row_a in group_a.iterrows():
                    for idx_b, row_b in group_b.iterrows():
                        if sigma_a == sigma_b and int(idx_b) <= int(idx_a):
                            continue
                        rows.append(
                            {
                                "family": row_a["family"],
                                "parameter_name": row_a["parameter_name"],
                                "parameter_value": float(row_a["parameter_value"]),
                                "beta": float(row_a["beta"]),
                                "opportunity": float(row_a["opportunity"]),
                                "seed": int(row_a["seed"]),
                                "original_condition_index": int(row_a["original_condition_index"]),
                                "sample_set_a": int(row_a["sample_set"]),
                                "sample_set_b": int(row_b["sample_set"]),
                                "n_selected_sigma_a": int(len(group_a)),
                                "n_selected_sigma_b": int(len(group_b)),
                                "sigma_a": float(sigma_a),
                                "sigma_b": float(sigma_b),
                                "last_paid_sigma_pair_sym_kl": symmetric_diag_gaussian_kl_pair(
                                    np.asarray(row_a["z_mu"], dtype=float),
                                    np.asarray(row_a["z_logvar"], dtype=float),
                                    np.asarray(row_b["z_mu"], dtype=float),
                                    np.asarray(row_b["z_logvar"], dtype=float),
                                ),
                                "last_paid_sigma_pair_z_mu_mae": float(
                                    np.nanmean(
                                        np.abs(
                                            np.asarray(row_a["z_mu"], dtype=float)
                                            - np.asarray(row_b["z_mu"], dtype=float)
                                        )
                                    )
                                ),
                            }
                        )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary = (
        pairwise.groupby(
            ["family", "parameter_name", "parameter_value", "beta", "opportunity", "sigma_a", "sigma_b"],
            dropna=False,
        )
        .agg(
            mean_last_paid_sigma_pair_sym_kl=("last_paid_sigma_pair_sym_kl", "mean"),
            sem_last_paid_sigma_pair_sym_kl=("last_paid_sigma_pair_sym_kl", sem_finite),
            mean_last_paid_sigma_pair_z_mu_mae=("last_paid_sigma_pair_z_mu_mae", "mean"),
            sem_last_paid_sigma_pair_z_mu_mae=("last_paid_sigma_pair_z_mu_mae", sem_finite),
            n_pairs=("last_paid_sigma_pair_sym_kl", "count"),
            mean_selected_sigma_a=("n_selected_sigma_a", "mean"),
            mean_selected_sigma_b=("n_selected_sigma_b", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return pairwise, summary


LATENT_PAIRWISE_METRIC_COLS = [
    "pairwise_sym_kl",
    "pairwise_z_mu_mae",
    "pairwise_z_sigma_mae",
    "mean_z_sigma",
    "pairwise_prior_norm_sym_kl",
    "pairwise_prior_norm_z_mu_mae",
    "pairwise_prior_norm_z_sigma_mae",
    "mean_prior_norm_z_sigma",
]

SUMMARY_METRIC_NAMES = {
    "pairwise_sym_kl": ("mean_pairwise_sym_kl", "sem_pairwise_sym_kl"),
    "pairwise_z_mu_mae": ("mean_pairwise_z_mu_mae", "sem_pairwise_z_mu_mae"),
    "pairwise_z_sigma_mae": ("mean_pairwise_z_sigma_mae", "sem_pairwise_z_sigma_mae"),
    "mean_z_sigma": ("mean_z_sigma", "sem_z_sigma"),
    "pairwise_prior_norm_sym_kl": (
        "mean_pairwise_prior_norm_sym_kl",
        "sem_pairwise_prior_norm_sym_kl",
    ),
    "pairwise_prior_norm_z_mu_mae": (
        "mean_pairwise_prior_norm_z_mu_mae",
        "sem_pairwise_prior_norm_z_mu_mae",
    ),
    "pairwise_prior_norm_z_sigma_mae": (
        "mean_pairwise_prior_norm_z_sigma_mae",
        "sem_pairwise_prior_norm_z_sigma_mae",
    ),
    "mean_prior_norm_z_sigma": (
        "mean_prior_norm_z_sigma",
        "sem_prior_norm_z_sigma",
    ),
}


def mean_metric_aggs(source_cols: list[str] = LATENT_PAIRWISE_METRIC_COLS) -> dict:
    return {
        SUMMARY_METRIC_NAMES[col][0]: (col, "mean")
        for col in source_cols
    }


def sem_metric_aggs(mean_cols: list[str]) -> dict:
    out = {}
    for mean_col in mean_cols:
        sem_col = "sem_" + mean_col[len("mean_"):] if mean_col.startswith("mean_") else f"sem_{mean_col}"
        out[sem_col] = (mean_col, sem_finite)
    return out


def latent_group_metrics(mu: np.ndarray, logvar: np.ndarray) -> dict[str, float]:
    clipped_logvar = np.clip(logvar, -20.0, 20.0)
    sigma = np.exp(0.5 * clipped_logvar)
    metrics = {
        "mean_z_sigma": float(np.nanmean(sigma)) if sigma.size else np.nan,
        "pairwise_z_mu_mae": np.nan,
        "pairwise_z_sigma_mae": np.nan,
        "pairwise_sym_kl": np.nan,
    }
    if mu.shape[0] >= 2:
        metrics["pairwise_z_mu_mae"] = pairwise_mean_abs_difference(mu)
        metrics["pairwise_z_sigma_mae"] = pairwise_mean_abs_difference(sigma)
        metrics["pairwise_sym_kl"] = symmetric_diag_gaussian_kl(mu, logvar)
    return metrics


def prior_normalized_latent_group_metrics(
    mu: np.ndarray,
    logvar: np.ndarray,
    prior_mu: np.ndarray,
    prior_logvar: np.ndarray,
) -> dict[str, float]:
    clipped_logvar = np.clip(logvar, -20.0, 20.0)
    clipped_prior_logvar = np.clip(prior_logvar, -20.0, 20.0)
    sigma = np.exp(0.5 * clipped_logvar)
    prior_sigma = np.exp(0.5 * clipped_prior_logvar)
    prior_sigma = np.maximum(prior_sigma, 1e-8)
    norm_mu = (mu - prior_mu) / prior_sigma
    norm_sigma = sigma / prior_sigma
    norm_logvar = 2.0 * np.log(np.maximum(norm_sigma, 1e-12))
    metrics = {
        "mean_prior_norm_z_sigma": float(np.nanmean(norm_sigma)) if norm_sigma.size else np.nan,
        "pairwise_prior_norm_z_mu_mae": np.nan,
        "pairwise_prior_norm_z_sigma_mae": np.nan,
        "pairwise_prior_norm_sym_kl": np.nan,
    }
    if mu.shape[0] >= 2:
        metrics["pairwise_prior_norm_z_mu_mae"] = pairwise_mean_abs_difference(norm_mu)
        metrics["pairwise_prior_norm_z_sigma_mae"] = pairwise_mean_abs_difference(norm_sigma)
        metrics["pairwise_prior_norm_sym_kl"] = symmetric_diag_gaussian_kl(norm_mu, norm_logvar)
    return metrics


def latent_metrics_for_frame(valid: pd.DataFrame) -> dict[str, float]:
    if len(valid) == 0:
        return {
            "pairwise_sym_kl": np.nan,
            "pairwise_z_mu_mae": np.nan,
            "pairwise_z_sigma_mae": np.nan,
            "mean_z_sigma": np.nan,
            "pairwise_prior_norm_sym_kl": np.nan,
            "pairwise_prior_norm_z_mu_mae": np.nan,
            "pairwise_prior_norm_z_sigma_mae": np.nan,
            "mean_prior_norm_z_sigma": np.nan,
        }
    mu = np.stack(valid["z_mu"].to_numpy())
    logvar = np.stack(valid["z_logvar"].to_numpy())
    metrics = latent_group_metrics(mu, logvar)
    if {"prior_mu", "prior_logvar"}.issubset(valid.columns) and valid["prior_mu"].notna().all():
        prior_mu = np.stack(valid["prior_mu"].to_numpy())
        prior_logvar = np.stack(valid["prior_logvar"].to_numpy())
        metrics.update(prior_normalized_latent_group_metrics(mu, logvar, prior_mu, prior_logvar))
    else:
        metrics.update({
            "pairwise_prior_norm_sym_kl": np.nan,
            "pairwise_prior_norm_z_mu_mae": np.nan,
            "pairwise_prior_norm_z_sigma_mae": np.nan,
            "mean_prior_norm_z_sigma": np.nan,
        })
    return metrics


def reward_columns(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in frame.columns if col.startswith("reward_node_")]
    return sorted(cols, key=lambda col: int(col.rsplit("_", 1)[-1]))


def add_first_observed_reward_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    reward_cols = reward_columns(out)
    if not reward_cols or "first_observed_node" not in out:
        out["first_observed_reward"] = np.nan
        out["mean_other_node_reward"] = np.nan
        return out
    rewards = out[reward_cols].to_numpy(dtype=float)
    first_node = pd.to_numeric(out["first_observed_node"], errors="coerce").to_numpy(dtype=float)
    first_idx = first_node.astype(float) - 1.0
    valid = np.isfinite(first_idx) & (first_idx >= 0) & (first_idx < len(reward_cols))
    first_reward = np.full(len(out), np.nan, dtype=float)
    valid_rows = np.where(valid)[0]
    if len(valid_rows):
        first_reward[valid_rows] = rewards[valid_rows, first_idx[valid_rows].astype(int)]
    if len(reward_cols) > 1:
        mean_other = (np.nansum(rewards, axis=1) - first_reward) / float(len(reward_cols) - 1)
        mean_other[~valid] = np.nan
    else:
        mean_other = np.full(len(out), np.nan, dtype=float)
    out["first_observed_reward"] = first_reward
    out["mean_other_node_reward"] = mean_other
    return out


def ensure_reduced_condition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"first_observed_reward", "mean_other_node_reward"}.issubset(out.columns):
        return out
    if {"node1_reward", "node2_reward"}.issubset(out.columns):
        out["first_observed_reward"] = out["node1_reward"]
        out["mean_other_node_reward"] = out["node2_reward"]
    return out


def ensure_latent_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in LATENT_PAIRWISE_METRIC_COLS:
        if col not in out:
            out[col] = np.nan
    return out


def compute_pairwise_kl(latent_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([metadata.reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)
    merged = add_first_observed_reward_summary(merged)
    rows = []
    group_cols = ["first_observed_reward", "mean_other_node_reward"]
    for (first_reward, mean_other), group in merged.groupby(group_cols, dropna=False):
        valid = group[group["valid_last_paid"].astype(bool)]
        metrics = latent_metrics_for_frame(valid)
        rows.append(
            {
                "first_observed_reward": first_reward,
                "mean_other_node_reward": mean_other,
                "n_valid": int(len(valid)),
                "n_reward_combinations": int(group["condition_index"].nunique())
                if "condition_index" in group
                else np.nan,
                **metrics,
                "mean_observations": float(group["observations"].mean()),
                "mean_last_paid_timestep": float(valid["last_paid_timestep"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_kl_by_paid_timestep(
    paid_latent_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if paid_latent_df.empty:
        return pd.DataFrame()
    merged = paid_latent_df.merge(
        metadata.reset_index().rename(columns={"index": "trial_index"}),
        on="trial_index",
        how="left",
    )
    merged = add_first_observed_reward_summary(merged)
    rows = []
    group_cols = [
        "first_observed_reward",
        "mean_other_node_reward",
        "paid_observation_index",
    ]
    for (first_reward, mean_other, paid_idx), group in merged.groupby(group_cols, dropna=False):
        valid = group[np.isfinite(group["paid_timestep"])]
        metrics = latent_metrics_for_frame(valid)
        rows.append(
            {
                "first_observed_reward": first_reward,
                "mean_other_node_reward": mean_other,
                "paid_observation_index": int(paid_idx),
                "n_valid": int(len(valid)),
                "n_reward_combinations": int(group["condition_index"].nunique())
                if "condition_index" in group
                else np.nan,
                **metrics,
                "mean_observations": float(valid["observations"].mean()) if len(valid) else np.nan,
                "mean_paid_timestep": float(valid["paid_timestep"].mean()) if len(valid) else np.nan,
                "mean_n_paid_latents": float(valid["n_paid_latents"].mean()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_successive_timestep_kl(
    paid_latent_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if paid_latent_df.empty:
        return pd.DataFrame()
    merged = paid_latent_df.merge(
        metadata.reset_index().rename(columns={"index": "trial_index"}),
        on="trial_index",
        how="left",
    )
    merged = add_first_observed_reward_summary(merged)
    rows = []
    for trial_index, group in merged.groupby("trial_index", sort=False):
        valid = group[np.isfinite(group["paid_timestep"])].sort_values("paid_observation_index")
        if len(valid) < 2:
            continue
        records = valid.to_dict("records")
        for prev, current in zip(records[:-1], records[1:]):
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "condition_index": current.get("condition_index", np.nan),
                    "sample_set": current.get("sample_set", np.nan),
                    "first_observed_reward": current.get("first_observed_reward", np.nan),
                    "mean_other_node_reward": current.get("mean_other_node_reward", np.nan),
                    "previous_paid_observation_index": int(prev["paid_observation_index"]),
                    "paid_observation_index": int(current["paid_observation_index"]),
                    "previous_paid_timestep": float(prev["paid_timestep"]),
                    "paid_timestep": float(current["paid_timestep"]),
                    "n_paid_latents": int(current["n_paid_latents"]),
                    "observations": int(current["observations"]),
                    "successive_sym_kl": symmetric_diag_gaussian_kl_pair(
                        prev["z_mu"],
                        prev["z_logvar"],
                        current["z_mu"],
                        current["z_logvar"],
                    ),
                    "successive_z_mu_mae": float(
                        np.nanmean(
                            np.abs(
                                np.asarray(current["z_mu"], dtype=float)
                                - np.asarray(prev["z_mu"], dtype=float)
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    pairwise_df = ensure_latent_metric_columns(pairwise_df)
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
    ]
    per_seed = (
        pairwise_df.groupby(group_cols)
        .agg(
            **mean_metric_aggs(),
            median_pairwise_sym_kl=("pairwise_sym_kl", "median"),
            mean_valid_sample_sets=("n_valid", "mean"),
            n_reward_pairs=("pairwise_sym_kl", lambda x: int(np.isfinite(x).sum())),
            mean_observations=("mean_observations", "mean"),
            mean_last_paid_timestep=("mean_last_paid_timestep", "mean"),
        )
        .reset_index()
    )
    summary_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
    ]
    summary = (
        per_seed.groupby(summary_cols)
        .agg(
            **{
                mean_col: (mean_col, "mean")
                for mean_col, _ in SUMMARY_METRIC_NAMES.values()
            },
            **sem_metric_aggs([mean_col for mean_col, _ in SUMMARY_METRIC_NAMES.values()]),
            median_pairwise_sym_kl=("median_pairwise_sym_kl", "mean"),
            mean_valid_sample_sets=("mean_valid_sample_sets", "mean"),
            n_seeds=("seed", "nunique"),
            mean_reward_pairs=("n_reward_pairs", "mean"),
            mean_observations=("mean_observations", "mean"),
            mean_last_paid_timestep=("mean_last_paid_timestep", "mean"),
        )
        .reset_index()
    )
    return per_seed, summary


def summarize_paid_timestep_pairwise(
    paid_pairwise_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if paid_pairwise_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    paid_pairwise_df = ensure_reduced_condition_columns(paid_pairwise_df)
    paid_pairwise_df = ensure_latent_metric_columns(paid_pairwise_df)
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "paid_observation_index",
    ]
    by_timestep_seed = (
        paid_pairwise_df.groupby(group_cols)
        .agg(
            **mean_metric_aggs(),
            mean_valid_sample_sets=("n_valid", "mean"),
            n_reward_pairs=("pairwise_sym_kl", lambda x: int(np.isfinite(x).sum())),
            mean_paid_timestep=("mean_paid_timestep", "mean"),
            mean_n_paid_latents=("mean_n_paid_latents", "mean"),
        )
        .reset_index()
    )
    by_timestep = (
        by_timestep_seed.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "paid_observation_index",
            ]
        )
        .agg(
            **{
                mean_col: (mean_col, "mean")
                for mean_col, _ in SUMMARY_METRIC_NAMES.values()
            },
            **sem_metric_aggs([mean_col for mean_col, _ in SUMMARY_METRIC_NAMES.values()]),
            mean_valid_sample_sets=("mean_valid_sample_sets", "mean"),
            mean_reward_pairs=("n_reward_pairs", "mean"),
            mean_paid_timestep=("mean_paid_timestep", "mean"),
            mean_n_paid_latents=("mean_n_paid_latents", "mean"),
        )
        .reset_index()
    )

    # For the across-paid-latent plot, give each paid timestep equal weight.
    # First average reward-pair/sample-set KLs within a paid timestep, then
    # average those timestep means within a seed/condition.
    across_paid_seed = (
        by_timestep_seed.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "seed",
            ]
        )
        .agg(
            **{
                f"{mean_col}_across_paid_timesteps": (mean_col, "mean")
                for mean_col, _ in SUMMARY_METRIC_NAMES.values()
            },
            n_paid_timestep_bins=("mean_pairwise_sym_kl", lambda x: int(np.isfinite(x).sum())),
            mean_valid_sample_sets=("mean_valid_sample_sets", "mean"),
            n_reward_pairs=(
                "n_reward_pairs",
                "mean",
            ),
        )
        .reset_index()
    )
    across_paid = (
        across_paid_seed.groupby(
            ["family", "parameter_name", "parameter_value", "beta", "opportunity", "sigma"]
        )
        .agg(
            **{
                f"{mean_col}_across_paid_timesteps": (
                    f"{mean_col}_across_paid_timesteps",
                    "mean",
                )
                for mean_col, _ in SUMMARY_METRIC_NAMES.values()
            },
            **sem_metric_aggs([
                f"{mean_col}_across_paid_timesteps"
                for mean_col, _ in SUMMARY_METRIC_NAMES.values()
            ]),
            mean_paid_timestep_bins=("n_paid_timestep_bins", "mean"),
            mean_valid_sample_sets=("mean_valid_sample_sets", "mean"),
            mean_reward_pairs=("n_reward_pairs", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return by_timestep, across_paid_seed, across_paid


def summarize_successive_timestep_kl(
    successive_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if successive_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    successive_df = successive_df.copy()
    if "successive_z_mu_mae" not in successive_df.columns:
        successive_df["successive_z_mu_mae"] = np.nan
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "paid_observation_index",
    ]
    by_timestep_seed = (
        successive_df.groupby(group_cols)
        .agg(
            mean_successive_sym_kl=("successive_sym_kl", "mean"),
            mean_successive_z_mu_mae=("successive_z_mu_mae", "mean"),
            n_transition_rows=("successive_sym_kl", lambda x: int(np.isfinite(x).sum())),
            n_sample_sets=("sample_set", "nunique"),
            n_reward_conditions=("condition_index", "nunique"),
            mean_previous_paid_timestep=("previous_paid_timestep", "mean"),
            mean_paid_timestep=("paid_timestep", "mean"),
            mean_n_paid_latents=("n_paid_latents", "mean"),
        )
        .reset_index()
    )
    by_timestep = (
        by_timestep_seed.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "paid_observation_index",
            ]
        )
        .agg(
            mean_successive_sym_kl=("mean_successive_sym_kl", "mean"),
            sem_successive_sym_kl=(
                "mean_successive_sym_kl",
                lambda x: (
                    float(np.nanstd(x, ddof=1) / math.sqrt(np.isfinite(x).sum()))
                    if np.isfinite(x).sum() > 1
                    else np.nan
                ),
            ),
            mean_successive_z_mu_mae=("mean_successive_z_mu_mae", "mean"),
            sem_successive_z_mu_mae=(
                "mean_successive_z_mu_mae",
                lambda x: (
                    float(np.nanstd(x, ddof=1) / math.sqrt(np.isfinite(x).sum()))
                    if np.isfinite(x).sum() > 1
                    else np.nan
                ),
            ),
            mean_transition_rows=("n_transition_rows", "mean"),
            mean_sample_sets=("n_sample_sets", "mean"),
            mean_reward_conditions=("n_reward_conditions", "mean"),
            mean_previous_paid_timestep=("mean_previous_paid_timestep", "mean"),
            mean_paid_timestep=("mean_paid_timestep", "mean"),
            mean_n_paid_latents=("mean_n_paid_latents", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    across_seed = (
        by_timestep_seed.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "seed",
            ]
        )
        .agg(
            mean_successive_sym_kl_across_timesteps=("mean_successive_sym_kl", "mean"),
            mean_successive_z_mu_mae_across_timesteps=("mean_successive_z_mu_mae", "mean"),
            n_transition_timestep_bins=("mean_successive_sym_kl", lambda x: int(np.isfinite(x).sum())),
            mean_transition_rows=("n_transition_rows", "mean"),
            mean_sample_sets=("n_sample_sets", "mean"),
            mean_reward_conditions=("n_reward_conditions", "mean"),
        )
        .reset_index()
    )
    across = (
        across_seed.groupby(
            ["family", "parameter_name", "parameter_value", "beta", "opportunity", "sigma"]
        )
        .agg(
            mean_successive_sym_kl_across_timesteps=(
                "mean_successive_sym_kl_across_timesteps",
                "mean",
            ),
            sem_successive_sym_kl_across_timesteps=(
                "mean_successive_sym_kl_across_timesteps",
                lambda x: (
                    float(np.nanstd(x, ddof=1) / math.sqrt(np.isfinite(x).sum()))
                    if np.isfinite(x).sum() > 1
                    else np.nan
                ),
            ),
            mean_successive_z_mu_mae_across_timesteps=(
                "mean_successive_z_mu_mae_across_timesteps",
                "mean",
            ),
            sem_successive_z_mu_mae_across_timesteps=(
                "mean_successive_z_mu_mae_across_timesteps",
                lambda x: (
                    float(np.nanstd(x, ddof=1) / math.sqrt(np.isfinite(x).sum()))
                    if np.isfinite(x).sum() > 1
                    else np.nan
                ),
            ),
            mean_transition_timestep_bins=("n_transition_timestep_bins", "mean"),
            mean_transition_rows=("mean_transition_rows", "mean"),
            mean_sample_sets=("mean_sample_sets", "mean"),
            mean_reward_conditions=("mean_reward_conditions", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return by_timestep, across_seed, across


def color_ramp(hex_colors: list[str], n: int):
    if n <= 0:
        return []
    anchors = np.asarray([matplotlib.colors.to_rgba(color) for color in hex_colors], dtype=float)
    if n == 1:
        return [tuple(anchors[0])]
    anchor_x = np.linspace(0.0, 1.0, len(anchors))
    xs = np.linspace(0.0, 1.0, n)
    return [
        tuple(float(np.interp(x, anchor_x, anchors[:, channel])) for channel in range(4))
        for x in xs
    ]


def color_values(n: int, cmap_name):
    if isinstance(cmap_name, (list, tuple)):
        return color_ramp(list(cmap_name), n)
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(0.65)]
    return [cmap(v) for v in np.linspace(0.25, 0.9, n)]


SAMPLE_COUNT_COLUMNS = (
    "mean_valid_sample_sets",
    "mean_sample_sets",
    "n_valid",
    "n_sample_sets",
    "n_streams",
    "n_pairs",
    "mean_transition_rows",
    "n_transition_rows",
)


def filter_plot_min_samples(frame: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    if frame.empty or int(min_samples) <= 1:
        return frame.copy()
    for col in SAMPLE_COUNT_COLUMNS:
        if col in frame.columns:
            counts = pd.to_numeric(frame[col], errors="coerce")
            return frame[counts >= int(min_samples)].copy()
    return frame.copy()


LATENT_METRIC_PLOT_SPECS = [
    {
        "prefix": "pairwise_z_mu_mae",
        "pairwise_col": "pairwise_z_mu_mae",
        "summary_mean": "mean_pairwise_z_mu_mae",
        "summary_sem": "sem_pairwise_z_mu_mae",
        "across_mean": "mean_pairwise_z_mu_mae_across_paid_timesteps",
        "across_sem": "sem_pairwise_z_mu_mae_across_paid_timesteps",
        "last_ylabel": "Pairwise z_mu MAE\\nlast paid latent",
        "timestep_ylabel": "Pairwise z_mu MAE",
        "across_ylabel": "Pairwise z_mu MAE\\nmean across paid latents",
        "grid_ylabel": "z_mu\\nMAE",
        "grid_title": "Pairwise z_mu MAE by first observed reward and mean other reward",
    },
    {
        "prefix": "pairwise_prior_norm_z_mu_mae",
        "pairwise_col": "pairwise_prior_norm_z_mu_mae",
        "summary_mean": "mean_pairwise_prior_norm_z_mu_mae",
        "summary_sem": "sem_pairwise_prior_norm_z_mu_mae",
        "across_mean": "mean_pairwise_prior_norm_z_mu_mae_across_paid_timesteps",
        "across_sem": "sem_pairwise_prior_norm_z_mu_mae_across_paid_timesteps",
        "last_ylabel": "Pairwise prior-normalized\\nz_mu MAE\\nlast paid latent",
        "timestep_ylabel": "Pairwise prior-normalized\\nz_mu MAE",
        "across_ylabel": "Pairwise prior-normalized\\nz_mu MAE\\nmean across paid latents",
        "grid_ylabel": "prior-norm\\nz_mu MAE",
        "grid_title": "Pairwise prior-normalized z_mu MAE by first observed reward and mean other reward",
    },
    {
        "prefix": "pairwise_z_sigma_mae",
        "pairwise_col": "pairwise_z_sigma_mae",
        "summary_mean": "mean_pairwise_z_sigma_mae",
        "summary_sem": "sem_pairwise_z_sigma_mae",
        "across_mean": "mean_pairwise_z_sigma_mae_across_paid_timesteps",
        "across_sem": "sem_pairwise_z_sigma_mae_across_paid_timesteps",
        "last_ylabel": "Pairwise z_sigma MAE\\nlast paid latent",
        "timestep_ylabel": "Pairwise z_sigma MAE",
        "across_ylabel": "Pairwise z_sigma MAE\\nmean across paid latents",
        "grid_ylabel": "z_sigma\\nMAE",
        "grid_title": "Pairwise z_sigma MAE by first observed reward and mean other reward",
    },
    {
        "prefix": "mean_z_sigma",
        "pairwise_col": "mean_z_sigma",
        "summary_mean": "mean_z_sigma",
        "summary_sem": "sem_z_sigma",
        "across_mean": "mean_z_sigma_across_paid_timesteps",
        "across_sem": "sem_z_sigma_across_paid_timesteps",
        "last_ylabel": "Mean z_sigma\\nlast paid latent",
        "timestep_ylabel": "Mean z_sigma",
        "across_ylabel": "Mean z_sigma\\nacross paid latents",
        "grid_ylabel": "mean\\nz_sigma",
        "grid_title": "Mean z_sigma by first observed reward and mean other reward",
    },
]


def plot_summary(
    summary: pd.DataFrame,
    outpath: Path,
    *,
    mean_col: str = "mean_pairwise_sym_kl",
    sem_col: str = "sem_pairwise_sym_kl",
    ylabel: str = "Pairwise KL\\nlast paid latent\\n(log scale)",
    log_y: bool = True,
    min_samples: int = 1,
) -> None:
    if summary.empty or mean_col not in summary or sem_col not in summary:
        return
    summary = filter_plot_min_samples(summary, min_samples)
    if summary.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * PANEL_WIDTH_IN + 1.3, PANEL_HEIGHT_IN + 0.55),
        sharey=True,
    )
    y_lower, y_upper = metric_limits(summary, mean_col, sem_col, log_y)
    families = [
        ("vary_beta", "Vary beta", FAMILY_COLOR_RAMPS["vary_beta"]),
        ("vary_opportunity", "Vary opportunity", FAMILY_COLOR_RAMPS["vary_opportunity"]),
    ]
    for ax, (family, title, cmap_name) in zip(axes, families):
        data = summary[summary["family"] == family].copy()
        if data.empty:
            ax.set_title(title)
            continue
        params = sorted(data["parameter_value"].unique())
        colors = color_values(len(params), cmap_name)
        for color, param in zip(colors, params):
            sub = data[data["parameter_value"] == param].sort_values("sigma")
            label = f"{data['parameter_name'].iloc[0]}={param:g}"
            y = metric_y_values(sub[mean_col].to_numpy(), y_lower, log_y)
            yerr = metric_yerr(y, sub[sem_col].to_numpy(), y_lower, log_y)
            ax.errorbar(
                sub["sigma"],
                y,
                yerr=yerr,
                marker="o",
                linewidth=0.9,
                markersize=2.5,
                capsize=1.5,
                color=color,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Observation noise sigma")
        set_metric_axis(ax, y_lower, y_upper, log_y)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout(w_pad=0.7)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_across_paid_summary(
    summary: pd.DataFrame,
    outpath: Path,
    *,
    mean_col: str = "mean_pairwise_sym_kl_across_paid_timesteps",
    sem_col: str = "sem_pairwise_sym_kl_across_paid_timesteps",
    ylabel: str = "Pairwise KL\\nmean across paid latents\\n(log scale)",
    log_y: bool = True,
    min_samples: int = 1,
) -> None:
    if summary.empty or mean_col not in summary or sem_col not in summary:
        return
    summary = filter_plot_min_samples(summary, min_samples)
    if summary.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * PANEL_WIDTH_IN + 1.3, PANEL_HEIGHT_IN + 0.55),
        sharey=True,
    )
    y_lower, y_upper = metric_limits(summary, mean_col, sem_col, log_y)
    families = [
        ("vary_beta", "Vary beta", FAMILY_COLOR_RAMPS["vary_beta"]),
        ("vary_opportunity", "Vary opportunity", FAMILY_COLOR_RAMPS["vary_opportunity"]),
    ]
    for ax, (family, title, cmap_name) in zip(axes, families):
        data = summary[summary["family"] == family].copy()
        if data.empty:
            ax.set_title(title)
            continue
        params = sorted(data["parameter_value"].unique())
        colors = color_values(len(params), cmap_name)
        for color, param in zip(colors, params):
            sub = data[data["parameter_value"] == param].sort_values("sigma")
            label = f"{data['parameter_name'].iloc[0]}={param:g}"
            y = metric_y_values(sub[mean_col].to_numpy(), y_lower, log_y)
            ax.errorbar(
                sub["sigma"],
                y,
                yerr=metric_yerr(y, sub[sem_col].to_numpy(), y_lower, log_y),
                marker="o",
                linewidth=0.9,
                markersize=2.5,
                capsize=1.5,
                color=color,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Observation noise sigma")
        set_metric_axis(ax, y_lower, y_upper, log_y)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout(w_pad=0.7)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_paid_timestep_sigma_rows(
    by_timestep: pd.DataFrame,
    outpath: Path,
    *,
    mean_col: str = "mean_pairwise_sym_kl",
    sem_col: str = "sem_pairwise_sym_kl",
    ylabel: str = "Pairwise KL\\n(log scale)",
    log_y: bool = True,
    min_samples: int = 1,
) -> None:
    if by_timestep.empty:
        return
    if mean_col not in by_timestep or sem_col not in by_timestep:
        return
    data = filter_plot_min_samples(by_timestep, min_samples)
    data = data[np.isfinite(data[mean_col])].copy()
    if data.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    sigmas = sorted(data["sigma"].unique())
    n_rows = len(sigmas)
    y_lower, y_upper = metric_limits(data, mean_col, sem_col, log_y)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(2.15 * PANEL_WIDTH_IN + 1.45, max(n_rows, 1) * PANEL_HEIGHT_IN + 0.55),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    family_specs = [
        ("vary_beta", "beta", FAMILY_COLOR_RAMPS["vary_beta"], "o", "-"),
        ("vary_opportunity", "opp", FAMILY_COLOR_RAMPS["vary_opportunity"], "^", "--"),
    ]
    handles = []
    labels = []
    for row_idx, sigma in enumerate(sigmas):
        ax = axes[row_idx]
        sigma_data = data[data["sigma"] == sigma]
        for family, short_name, cmap_name, marker, linestyle in family_specs:
            fam_data = sigma_data[sigma_data["family"] == family]
            if fam_data.empty:
                continue
            params = sorted(fam_data["parameter_value"].unique())
            colors = color_values(len(params), cmap_name)
            for color, param in zip(colors, params):
                sub = fam_data[fam_data["parameter_value"] == param].sort_values(
                    "paid_observation_index"
                )
                label = f"{short_name}={param:g}"
                y = metric_y_values(sub[mean_col].to_numpy(), y_lower, log_y)
                line = ax.errorbar(
                    sub["paid_observation_index"],
                    y,
                    yerr=metric_yerr(y, sub[sem_col].to_numpy(), y_lower, log_y),
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=0.9,
                    markersize=2.5,
                    capsize=1.5,
                    color=color,
                    label=label,
                )
                if row_idx == 0 and label not in labels:
                    handles.append(line)
                    labels.append(label)
        ax.set_title(f"sigma={sigma:g}", loc="left")
        set_metric_axis(ax, y_lower, y_upper, log_y)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Paid observation index")
    axes[0].set_ylabel(ylabel)
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.79, 0.5))
    fig.tight_layout(rect=(0, 0, 0.78, 1), h_pad=0.75)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def summarize_by_reward_pair(
    pairwise: pd.DataFrame,
    *,
    value_col: str = "pairwise_sym_kl",
    mean_col: str = "mean_pairwise_sym_kl",
    sem_col: str = "sem_pairwise_sym_kl",
) -> pd.DataFrame:
    pairwise = ensure_reduced_condition_columns(pairwise)
    pairwise = ensure_latent_metric_columns(pairwise)
    if value_col not in pairwise:
        return pd.DataFrame()
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "first_observed_reward",
        "mean_other_node_reward",
    ]
    return (
        pairwise.groupby(group_cols)
        .agg(
            **{
                mean_col: (value_col, "mean"),
                sem_col: (
                    value_col,
                    sem_finite,
                ),
            },
            mean_valid_sample_sets=("n_valid", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )


def summarize_all_reward_grid_metrics(pairwise: pd.DataFrame) -> pd.DataFrame:
    pairwise = ensure_reduced_condition_columns(pairwise)
    pairwise = ensure_latent_metric_columns(pairwise)
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "first_observed_reward",
        "mean_other_node_reward",
    ]
    return (
        pairwise.groupby(group_cols)
        .agg(
            **mean_metric_aggs(),
            **{
                sem_col: (source_col, sem_finite)
                for source_col, (_, sem_col) in SUMMARY_METRIC_NAMES.items()
            },
            mean_valid_sample_sets=("n_valid", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )


def plot_reward_pair_grid(
    pairwise: pd.DataFrame,
    outpath: Path,
    *,
    value_col: str = "pairwise_sym_kl",
    mean_col: str = "mean_pairwise_sym_kl",
    sem_col: str = "sem_pairwise_sym_kl",
    ylabel_suffix: str = "KL\\nlog",
    title: str = "Pairwise KL by first observed reward and mean other reward (log y)",
    log_y: bool = True,
    min_samples: int = 1,
) -> None:
    grid = summarize_by_reward_pair(
        pairwise,
        value_col=value_col,
        mean_col=mean_col,
        sem_col=sem_col,
    )
    if grid.empty:
        return
    grid = filter_plot_min_samples(grid, min_samples)
    grid = grid[np.isfinite(grid[mean_col])].copy()
    if grid.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    first_reward_values = sorted(grid["first_observed_reward"].unique(), reverse=True)
    mean_other_values = sorted(grid["mean_other_node_reward"].unique())
    n_rows = len(first_reward_values)
    n_cols = len(mean_other_values)
    legend_width = 1.65
    fig = plt.figure(
        figsize=(n_cols * PANEL_WIDTH_IN + legend_width, n_rows * PANEL_HEIGHT_IN + 0.65)
    )
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [1.28],
        wspace=0.26,
        hspace=0.30,
        left=0.06,
        right=0.985,
        bottom=0.055,
        top=0.945,
    )
    family_styles = {
        "vary_beta": {
            "cmap": FAMILY_COLOR_RAMPS["vary_beta"],
            "marker": "o",
            "linestyle": "-",
            "label": "beta",
        },
        "vary_opportunity": {
            "cmap": FAMILY_COLOR_RAMPS["vary_opportunity"],
            "marker": "^",
            "linestyle": "--",
            "label": "opp",
        },
    }
    color_lookup: dict[tuple[str, float], tuple[float, float, float, float]] = {}
    for family, style in family_styles.items():
        vals = sorted(grid.loc[grid["family"] == family, "parameter_value"].unique())
        for value, color in zip(vals, color_values(len(vals), style["cmap"])):
            color_lookup[(family, float(value))] = color

    legend_handles = []
    legend_labels = []
    for row_i, first_reward in enumerate(first_reward_values):
        for col_i, mean_other in enumerate(mean_other_values):
            ax = fig.add_subplot(gs[row_i, col_i])
            panel = grid[
                (grid["first_observed_reward"] == first_reward)
                & (grid["mean_other_node_reward"] == mean_other)
            ].copy()
            if panel.empty:
                ax.axis("off")
                continue
            y_lower, y_upper = metric_limits(panel, mean_col, sem_col, log_y)
            for (family, param), sub in panel.groupby(["family", "parameter_value"]):
                sub = sub.sort_values("sigma")
                style = family_styles.get(family, family_styles["vary_beta"])
                y = metric_y_values(sub[mean_col].to_numpy(), y_lower, log_y)
                line = ax.errorbar(
                    sub["sigma"],
                    y,
                    yerr=metric_yerr(y, sub[sem_col].to_numpy(), y_lower, log_y),
                    color=color_lookup.get((family, float(param)), "black"),
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=0.75,
                    markersize=1.8,
                    capsize=1.0,
                )[0]
                label = f"{style['label']}={float(param):g}"
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)
            if row_i == 0:
                ax.set_title(f"mean other={mean_other:g}", pad=2)
            if col_i == 0:
                ax.set_ylabel(f"first R={first_reward:g}\n{ylabel_suffix}")
            else:
                ax.set_yticklabels([])
            if row_i == n_rows - 1:
                ax.set_xlabel("sigma")
            else:
                ax.set_xticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            set_metric_axis(ax, y_lower, y_upper, log_y)
            ax.tick_params(length=2, pad=1)
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="center left",
        title="Parameter",
        title_fontsize=PLOT_FONT_SIZE_PT,
    )
    fig.suptitle(title, y=0.992)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_current_vs_previous_latent_change(
    summary: pd.DataFrame,
    outpath: Path,
    *,
    min_samples: int = 1,
) -> None:
    if summary.empty:
        return
    summary = filter_plot_min_samples(summary, min_samples)
    if summary.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    sigmas = sorted(summary["sigma"].unique())
    timesteps = sorted(summary["timestep"].unique())
    x_col = "delta_bin" if "delta_bin" in summary.columns else "abs_delta_bin"
    x_label = (
        "current - previous\nsame-node value"
        if x_col == "delta_bin"
        else "|current - previous\nsame-node value|"
    )
    x_limits = symmetric_observed_difference_xlim(summary, x_col)
    fig, axes = plt.subplots(
        len(timesteps),
        len(sigmas),
        figsize=(max(1, len(sigmas)) * PANEL_WIDTH_IN + 1.6, max(1, len(timesteps)) * PANEL_HEIGHT_IN + 0.7),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    y_lower, y_upper = dynamic_log_kl_limits(
        summary["mean_latent_sym_kl"].to_numpy(dtype=float),
        summary["sem_latent_sym_kl"].to_numpy(dtype=float),
    )
    color_lookup: dict[tuple[str, float], object] = {}
    for family, fam_data in summary.groupby("family"):
        params = sorted(float(x) for x in fam_data["parameter_value"].unique())
        color_lookup.update(
            {
                (family, value): color
                for value, color in family_colors_for_values(family, params).items()
            }
        )
    handles, labels = [], []
    for row_i, timestep in enumerate(timesteps):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = summary[(summary["timestep"] == timestep) & (summary["sigma"] == sigma)]
            if panel.empty:
                ax.axis("off")
                continue
            for (family, parameter_value), sub in panel.groupby(["family", "parameter_value"]):
                sub = sub.sort_values(x_col)
                linestyle = "-" if family == "vary_beta" else "--"
                marker = "o" if family == "vary_beta" else "^"
                y = metric_y_values(sub["mean_latent_sym_kl"].to_numpy(), y_lower, log_y=True)
                line = ax.errorbar(
                    sub[x_col],
                    y,
                    yerr=metric_yerr(y, sub["sem_latent_sym_kl"].to_numpy(), y_lower, log_y=True),
                    color=color_lookup.get((family, float(parameter_value)), "black"),
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=0.9,
                    markersize=2.2,
                    capsize=1.5,
                    label=family_param_label(sub.iloc[0]),
                )
                label = family_param_label(sub.iloc[0])
                if label not in labels:
                    handles.append(line)
                    labels.append(label)
            if row_i == 0:
                ax.set_title(f"sigma={sigma:g}")
            if col_i == 0:
                ax.set_ylabel(f"t={timestep}\nKL\n(log)")
            if row_i == len(timesteps) - 1:
                ax.set_xlabel(x_label)
            set_log_kl_axis(ax, y_lower, y_upper)
            ax.axvline(0.0, color="#999999", linewidth=0.5, zorder=0)
            ax.set_xlim(*x_limits)
            ax.tick_params(length=2, pad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=False)
    fig.tight_layout(rect=(0, 0, 0.82, 1), h_pad=0.65, w_pad=0.55)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_current_vs_previous_mu_displacement(
    summary: pd.DataFrame,
    outpath: Path,
    *,
    min_samples: int = 1,
    component: str = "mean",
) -> None:
    if summary.empty:
        return
    summary = summary.copy()
    if "mean_z01_mu_delta" not in summary.columns and {"mean_z0_mu_delta", "mean_z1_mu_delta"}.issubset(summary.columns):
        summary["mean_z01_mu_delta"] = 0.5 * (
            pd.to_numeric(summary["mean_z0_mu_delta"], errors="coerce")
            + pd.to_numeric(summary["mean_z1_mu_delta"], errors="coerce")
        )
        if {"sem_z0_mu_delta", "sem_z1_mu_delta"}.issubset(summary.columns):
            summary["sem_z01_mu_delta"] = 0.5 * np.sqrt(
                pd.to_numeric(summary["sem_z0_mu_delta"], errors="coerce") ** 2
                + pd.to_numeric(summary["sem_z1_mu_delta"], errors="coerce") ** 2
            )
    if "mean_z01_mu_displacement" not in summary.columns and {
        "mean_z0_mu_displacement",
        "mean_z1_mu_displacement",
    }.issubset(summary.columns):
        summary["mean_z01_mu_displacement"] = 0.5 * (
            pd.to_numeric(summary["mean_z0_mu_displacement"], errors="coerce")
            + pd.to_numeric(summary["mean_z1_mu_displacement"], errors="coerce")
        )
        if {"sem_z0_mu_displacement", "sem_z1_mu_displacement"}.issubset(summary.columns):
            summary["sem_z01_mu_displacement"] = 0.5 * np.sqrt(
                pd.to_numeric(summary["sem_z0_mu_displacement"], errors="coerce") ** 2
                + pd.to_numeric(summary["sem_z1_mu_displacement"], errors="coerce") ** 2
            )

    component_specs = {
        "mean": {
            "signed": ("mean_z01_mu_delta", "sem_z01_mu_delta", "mean delta mu"),
            "absolute": ("mean_z01_mu_displacement", "sem_z01_mu_displacement", "mean |delta mu|"),
        },
        "z0": {
            "signed": ("mean_z0_mu_delta", "sem_z0_mu_delta", "z0 delta mu"),
            "absolute": ("mean_z0_mu_displacement", "sem_z0_mu_displacement", "z0 |delta mu|"),
        },
        "z1": {
            "signed": ("mean_z1_mu_delta", "sem_z1_mu_delta", "z1 delta mu"),
            "absolute": ("mean_z1_mu_displacement", "sem_z1_mu_displacement", "z1 |delta mu|"),
        },
    }
    if component not in component_specs:
        raise ValueError(f"Unknown mu displacement component: {component}")
    signed_mean_col, signed_sem_col, signed_ylabel = component_specs[component]["signed"]
    abs_mean_col, abs_sem_col, abs_ylabel = component_specs[component]["absolute"]
    use_signed_mu = {signed_mean_col, signed_sem_col}.issubset(summary.columns)
    required = (
        {signed_mean_col, signed_sem_col}
        if use_signed_mu
        else {abs_mean_col, abs_sem_col}
    )
    if not required.issubset(summary.columns):
        return
    summary = filter_plot_min_samples(summary, min_samples)
    if summary.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    sigmas = sorted(summary["sigma"].unique())
    timesteps = sorted(summary["timestep"].unique())
    x_col = "delta_bin" if "delta_bin" in summary.columns else "abs_delta_bin"
    x_label = (
        "current - previous\nsame-node value"
        if x_col == "delta_bin"
        else "|current - previous\nsame-node value|"
    )
    x_limits = symmetric_observed_difference_xlim(summary, x_col)
    if use_signed_mu:
        metrics = [
            (component, signed_mean_col, signed_sem_col, "o", "-"),
        ]
    else:
        metrics = [
            (component, abs_mean_col, abs_sem_col, "o", "-"),
        ]
    fig, axes = plt.subplots(
        len(timesteps),
        len(sigmas),
        figsize=(max(1, len(sigmas)) * PANEL_WIDTH_IN + 1.8, max(1, len(timesteps)) * PANEL_HEIGHT_IN + 0.7),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    all_values = np.concatenate(
        [summary[col].to_numpy(dtype=float) for _, col, _, _, _ in metrics if col in summary]
    )
    finite = all_values[np.isfinite(all_values)]
    if finite.size:
        y_min = float(np.nanmin(finite))
        y_max = float(np.nanmax(finite))
    else:
        y_min, y_max = (-1.0, 1.0) if use_signed_mu else (0.0, 1.0)
    if use_signed_mu:
        if y_max <= y_min:
            y_min -= 0.5
            y_max += 0.5
        pad = 0.08 * max(y_max - y_min, 1e-6)
        y_min -= pad
        y_max += pad
    else:
        y_min = 0.0
        y_max = max(y_max * 1.08, 1e-6)
    color_lookup: dict[tuple[str, float], object] = {}
    for family, fam_data in summary.groupby("family"):
        params = sorted(float(x) for x in fam_data["parameter_value"].unique())
        color_lookup.update(
            {
                (family, value): color
                for value, color in family_colors_for_values(family, params).items()
            }
        )
    handles, labels = [], []
    for row_i, timestep in enumerate(timesteps):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = summary[(summary["timestep"] == timestep) & (summary["sigma"] == sigma)]
            if panel.empty:
                ax.axis("off")
                continue
            for (family, parameter_value), sub in panel.groupby(["family", "parameter_value"]):
                sub = sub.sort_values(x_col)
                base_color = color_lookup.get((family, float(parameter_value)), "black")
                family_line = "-" if family == "vary_beta" else "--"
                for dim_label, mean_col, sem_col, marker, dim_line in metrics:
                    label = family_param_label(sub.iloc[0])
                    line = ax.errorbar(
                        sub[x_col],
                        sub[mean_col],
                        yerr=sub[sem_col],
                        color=base_color,
                        linestyle=family_line if dim_label == "z0" else dim_line,
                        marker=marker,
                        linewidth=0.9,
                        markersize=2.0,
                        capsize=1.3,
                        label=label,
                        alpha=0.95,
                    )
                    if label not in labels:
                        handles.append(line)
                        labels.append(label)
            if row_i == 0:
                ax.set_title(f"sigma={sigma:g}")
            if col_i == 0:
                ax.set_ylabel(f"t={timestep}\n{signed_ylabel if use_signed_mu else abs_ylabel}")
            if row_i == len(timesteps) - 1:
                ax.set_xlabel(x_label)
            if use_signed_mu:
                ax.axhline(0.0, color="#999999", linewidth=0.5, zorder=0)
            ax.axvline(0.0, color="#999999", linewidth=0.5, zorder=0)
            ax.set_xlim(*x_limits)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(length=2, pad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=False)
    fig.tight_layout(rect=(0, 0, 0.80, 1), h_pad=0.65, w_pad=0.55)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_current_vs_previous_mu_components_by_seed(
    summary_by_seed: pd.DataFrame,
    outdir: Path,
    *,
    min_samples: int = 1,
) -> None:
    if summary_by_seed.empty or "seed" not in summary_by_seed.columns:
        return
    for seed in sorted(pd.to_numeric(summary_by_seed["seed"], errors="coerce").dropna().unique()):
        seed_int = int(seed)
        seed_summary = summary_by_seed[
            pd.to_numeric(summary_by_seed["seed"], errors="coerce") == seed
        ].copy()
        if seed_summary.empty:
            continue
        plot_current_vs_previous_mu_displacement(
            seed_summary,
            outdir / f"latent_current_vs_previous_observation_z0_mu_delta_by_timestep_sigma_seed_{seed_int}.png",
            min_samples=min_samples,
            component="z0",
        )
        plot_current_vs_previous_mu_displacement(
            seed_summary,
            outdir / f"latent_current_vs_previous_observation_z1_mu_delta_by_timestep_sigma_seed_{seed_int}.png",
            min_samples=min_samples,
            component="z1",
        )


def plot_sigma_pair_heatmaps(
    summary: pd.DataFrame,
    outdir: Path,
    *,
    min_samples: int = 1,
) -> None:
    if summary.empty:
        return
    summary = filter_plot_min_samples(summary, min_samples)
    if summary.empty:
        return
    metric_specs = [
        {
            "mean_col": "mean_last_paid_sigma_pair_sym_kl",
            "prefix": "last_paid_latent_sigma_pairwise_kl_heatmap",
            "title": "Last-paid latent KL",
            "cbar": "mean symmetric KL\n(log scale)",
            "log": True,
        },
        {
            "mean_col": "mean_last_paid_sigma_pair_z_mu_mae",
            "prefix": "last_paid_latent_sigma_pairwise_z_mu_mae_heatmap",
            "title": "Last-paid z_mu MAE",
            "cbar": "mean pairwise\nz_mu MAE",
            "log": False,
        },
    ]
    for spec in metric_specs:
        mean_col = spec["mean_col"]
        if mean_col not in summary.columns:
            continue
        values = pd.to_numeric(summary[mean_col], errors="coerce").to_numpy(dtype=float)
        if bool(spec["log"]):
            positive = values[np.isfinite(values) & (values > 0.0)]
            vmin = float(np.nanmin(positive)) if positive.size else LOG_KL_EPS
            vmax = float(np.nanmax(positive)) if positive.size else 1.0
            vmin = max(vmin, LOG_KL_EPS)
            vmax = max(vmax, vmin * 10.0)
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            text_threshold = math.sqrt(vmin * vmax)
        else:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            vmin = min(0.0, float(np.nanmin(finite)))
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            text_threshold = vmin + 0.55 * (vmax - vmin)

        for (family, parameter_value), data in summary.groupby(["family", "parameter_value"]):
            sigmas = sorted(set(data["sigma_a"]).union(set(data["sigma_b"])))
            matrix = np.full((len(sigmas), len(sigmas)), np.nan, dtype=float)
            for _, row in data.iterrows():
                i = sigmas.index(float(row["sigma_a"]))
                j = sigmas.index(float(row["sigma_b"]))
                matrix[i, j] = float(row[mean_col])
            if bool(spec["log"]):
                plot_matrix = np.where(np.isfinite(matrix) & (matrix > 0.0), matrix, vmin)
            else:
                plot_matrix = matrix.copy()
            hidden_triangle = np.tril(np.ones_like(matrix, dtype=bool), k=0)
            plot_matrix = np.ma.masked_where(hidden_triangle | ~np.isfinite(matrix), plot_matrix)
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad(color="white")
            fig, ax = plt.subplots(figsize=(2.55, 2.25))
            im = ax.imshow(plot_matrix, origin="lower", norm=norm, cmap=cmap)
            ax.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas], fontsize=PLOT_FONT_SIZE_PT)
            ax.set_yticks(range(len(sigmas)), [f"{s:g}" for s in sigmas], fontsize=PLOT_FONT_SIZE_PT)
            ax.set_xlabel("sigma b")
            ax.set_ylabel("sigma a")
            ax.set_title(f"{spec['title']}\n{family_param_label(data.iloc[0])}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=PLOT_FONT_SIZE_PT)
            cbar.set_label(spec["cbar"], fontsize=PLOT_FONT_SIZE_PT)
            for i in range(len(sigmas)):
                for j in range(len(sigmas)):
                    if i >= j:
                        continue
                    if np.isfinite(matrix[i, j]):
                        display_value = max(float(matrix[i, j]), vmin) if bool(spec["log"]) else float(matrix[i, j])
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2g}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if display_value > text_threshold else "black",
                        )
            fig.tight_layout()
            param_name = "beta" if family == "vary_beta" else "opp" if family == "vary_opportunity" else "param"
            outpath = outdir / (
                f"{spec['prefix']}_"
                f"{family}_{param_name}_{value_token(parameter_value)}.png"
            )
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)


def model_density_label(row: pd.Series) -> str:
    family = str(row.get("family", ""))
    parameter_value = float(row["parameter_value"]) if pd.notna(row.get("parameter_value", np.nan)) else np.nan
    seed = int(row["seed"]) if pd.notna(row.get("seed", np.nan)) else -1
    if family == "vary_beta":
        top = f"beta {parameter_value:g}"
    elif family == "vary_opportunity":
        top = f"opp {parameter_value:g}"
    else:
        top = f"{row.get('parameter_name', 'param')} {parameter_value:g}"
    return f"{top}\nseed {seed}"


def model_density_sort_key(row: pd.Series) -> str:
    family_order = {"vary_beta": 0, "vary_opportunity": 1}
    family = str(row.get("family", ""))
    order = family_order.get(family, 9)
    parameter_value = float(row["parameter_value"]) if pd.notna(row.get("parameter_value", np.nan)) else np.nan
    seed = int(row["seed"]) if pd.notna(row.get("seed", np.nan)) else -1
    return f"{order:02d}_{parameter_value:020.8f}_{seed:06d}_{family}"


def param_density_folder_name(row: pd.Series) -> str:
    family = str(row.get("family", "param"))
    parameter_value = float(row["parameter_value"]) if pd.notna(row.get("parameter_value", np.nan)) else np.nan
    if family == "vary_beta":
        return f"vary_beta_beta_{value_token(parameter_value)}"
    if family == "vary_opportunity":
        return f"vary_opportunity_opp_{value_token(parameter_value)}"
    parameter_name = str(row.get("parameter_name", "param"))
    return f"{family}_{parameter_name}_{value_token(parameter_value)}"


def build_paid_latent_density_frame(
    paid_latents: pd.DataFrame,
    path_map: np.ndarray | None = None,
) -> pd.DataFrame:
    required = {
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "paid_timestep",
        "paid_observation_index",
        "observed_node",
        "observed_reward",
        "actual_observed_reward",
        "z_mu",
        "z_logvar",
        "prior_mu",
        "prior_logvar",
    }
    if paid_latents.empty or not required.issubset(paid_latents.columns):
        return pd.DataFrame()
    rows = []
    reward_cols_all = reward_columns(paid_latents)
    path_map_arr = np.asarray(path_map, dtype=float) if path_map is not None else None
    for _, row in paid_latents.iterrows():
        try:
            z_mu = np.asarray(row["z_mu"], dtype=float)
            z_logvar = np.asarray(row["z_logvar"], dtype=float)
            prior_mu = np.asarray(row["prior_mu"], dtype=float)
            prior_logvar = np.asarray(row["prior_logvar"], dtype=float)
        except (TypeError, ValueError):
            continue
        if z_mu.ndim != 1 or z_mu.size < 2:
            continue
        z_sigma = np.exp(0.5 * np.clip(z_logvar, -20.0, 20.0))
        prior_sigma = np.exp(0.5 * np.clip(prior_logvar, -20.0, 20.0))
        if not (
            z_sigma.shape == z_mu.shape
            and prior_mu.shape == z_mu.shape
            and prior_sigma.shape == z_mu.shape
        ):
            continue
        if not (
            np.all(np.isfinite(z_mu))
            and np.all(np.isfinite(z_sigma))
            and np.all(np.isfinite(prior_mu))
            and np.all(np.isfinite(prior_sigma))
        ):
            continue
        observed_node = int(row["observed_node"])
        reward_cols = reward_cols_all
        other_actual_reward = np.nan
        observed_path_index = observed_node
        observed_path_reward = float(row["actual_observed_reward"])
        mean_other_path_reward = np.nan
        if reward_cols and 1 <= observed_node <= len(reward_cols):
            rewards = np.asarray([float(row[col]) for col in reward_cols], dtype=float)
            other_rewards = np.delete(rewards, observed_node - 1)
            other_actual_reward = float(np.nanmean(other_rewards)) if other_rewards.size else np.nan
            if path_map_arr is not None and path_map_arr.ndim == 2 and path_map_arr.shape[1] == rewards.size:
                path_values = path_map_arr @ rewards
                path_hits = np.flatnonzero(path_map_arr[:, observed_node - 1] > 0.0)
                if path_hits.size:
                    observed_path_index = int(path_hits[0]) + 1
                    observed_path_reward = float(path_values[int(path_hits[0])])
                    other_path_values = np.delete(path_values, int(path_hits[0]))
                    mean_other_path_reward = (
                        float(np.nanmean(other_path_values)) if other_path_values.size else np.nan
                    )
            else:
                mean_other_path_reward = other_actual_reward
        prior_sigma = np.maximum(prior_sigma, 1e-8)
        norm_mu = (z_mu - prior_mu) / prior_sigma
        norm_sigma = z_sigma / prior_sigma
        out = {
            "family": row["family"],
            "parameter_name": row["parameter_name"],
            "parameter_value": float(row["parameter_value"]),
            "beta": float(row["beta"]),
            "opportunity": float(row["opportunity"]),
            "sigma": float(row["sigma"]),
            "seed": int(row["seed"]),
            "model_label": model_density_label(row),
            "model_sort": model_density_sort_key(row),
            "timestep": int(row["paid_timestep"]),
            "paid_observation_index": int(row["paid_observation_index"]),
            "observed_node": observed_node,
            "observed_reward": float(row["observed_reward"]),
            "actual_node_reward": float(row["actual_observed_reward"]),
            "other_node_actual_reward": other_actual_reward,
            "observed_path_index": int(observed_path_index),
            "actual_path_reward": float(observed_path_reward),
            "mean_other_path_reward": mean_other_path_reward,
            "observed_minus_mean_other_path_reward": (
                float(observed_path_reward - mean_other_path_reward)
                if np.isfinite(mean_other_path_reward)
                else np.nan
            ),
            "force_first_observe_node": int(row.get("force_first_observe_node", 0) or 0),
            "force_round_robin_observations": bool(row.get("force_round_robin_observations", False)),
            "z_mu_0": float(z_mu[0]),
            "z_mu_1": float(z_mu[1]),
            "z_sigma_0": float(z_sigma[0]),
            "z_sigma_1": float(z_sigma[1]),
            "prior_mu_0": float(prior_mu[0]),
            "prior_mu_1": float(prior_mu[1]),
            "prior_sigma_0": float(prior_sigma[0]),
            "prior_sigma_1": float(prior_sigma[1]),
            "prior_normalized_z_mu_0": float(norm_mu[0]),
            "prior_normalized_z_mu_1": float(norm_mu[1]),
            "prior_normalized_z_sigma_0": float(norm_sigma[0]),
            "prior_normalized_z_sigma_1": float(norm_sigma[1]),
        }
        for i, value in enumerate(norm_mu):
            out[f"prior_normalized_mu_{i}"] = float(value)
        for i, value in enumerate(norm_sigma):
            out[f"prior_normalized_sigma_{i}"] = float(value)
        rows.append(out)
    return pd.DataFrame(rows)


def prepare_density_path_reward_columns(density_rows: pd.DataFrame) -> pd.DataFrame:
    rows = density_rows.copy()
    if rows.empty:
        return rows
    if "actual_path_reward" not in rows.columns:
        rows["actual_path_reward"] = pd.to_numeric(
            rows.get("actual_node_reward", pd.Series(np.nan, index=rows.index)),
            errors="coerce",
        )
    if "mean_other_path_reward" not in rows.columns:
        rows["mean_other_path_reward"] = pd.to_numeric(
            rows.get("other_node_actual_reward", pd.Series(np.nan, index=rows.index)),
            errors="coerce",
        )
    if "observed_minus_mean_other_path_reward" not in rows.columns:
        rows["observed_minus_mean_other_path_reward"] = (
            pd.to_numeric(rows["actual_path_reward"], errors="coerce")
            - pd.to_numeric(rows["mean_other_path_reward"], errors="coerce")
        )
    if "observed_path_index" not in rows.columns:
        rows["observed_path_index"] = pd.to_numeric(
            rows.get("observed_node", pd.Series(np.nan, index=rows.index)),
            errors="coerce",
        )
    return rows


def latent_mu_trace_covariance_summaries(
    density_rows: pd.DataFrame,
    *,
    min_samples: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    required = {
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "paid_observation_index",
        "prior_normalized_z_mu_0",
        "prior_normalized_z_mu_1",
    }
    if density_rows.empty or not required.issubset(density_rows.columns):
        return pd.DataFrame(), {}
    rows = prepare_density_path_reward_columns(density_rows)
    for col in [
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "paid_observation_index",
        "observed_path_index",
        "actual_path_reward",
        "mean_other_path_reward",
        "observed_minus_mean_other_path_reward",
        "prior_normalized_z_mu_0",
        "prior_normalized_z_mu_1",
    ]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "family",
            "parameter_name",
            "parameter_value",
            "beta",
            "opportunity",
            "sigma",
            "seed",
            "paid_observation_index",
            "observed_path_index",
            "actual_path_reward",
            "mean_other_path_reward",
            "observed_minus_mean_other_path_reward",
            "prior_normalized_z_mu_0",
            "prior_normalized_z_mu_1",
        ]
    )
    if rows.empty:
        return pd.DataFrame(), {}

    cell_group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "sigma",
        "seed",
        "paid_observation_index",
        "observed_path_index",
        "actual_path_reward",
        "mean_other_path_reward",
        "observed_minus_mean_other_path_reward",
    ]
    cells = (
        rows.groupby(cell_group_cols, dropna=False)
        .agg(
            var_z0_mu_norm=("prior_normalized_z_mu_0", lambda x: float(np.nanvar(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan),
            var_z1_mu_norm=("prior_normalized_z_mu_1", lambda x: float(np.nanvar(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan),
            n_states=("prior_normalized_z_mu_0", lambda x: int(np.isfinite(x).sum())),
        )
        .reset_index()
    )
    cells["trace_cov_z_mu_norm"] = (
        pd.to_numeric(cells["var_z0_mu_norm"], errors="coerce")
        + pd.to_numeric(cells["var_z1_mu_norm"], errors="coerce")
    )
    cells = cells[np.isfinite(cells["trace_cov_z_mu_norm"])].copy()
    if int(min_samples) > 1:
        cells = cells[pd.to_numeric(cells["n_states"], errors="coerce") >= int(min_samples)].copy()
    if cells.empty:
        return cells, {}

    base_cols = ["family", "parameter_name", "parameter_value", "beta", "opportunity", "sigma"]
    seed_cols = base_cols + ["seed"]
    x_specs = {
        "paid_observation_index": {
            "x_col": "paid_observation_index",
            "extra_condition_cols": [
                "observed_path_index",
                "actual_path_reward",
                "mean_other_path_reward",
            ],
        },
        "actual_path_reward": {
            "x_col": "actual_path_reward",
            "extra_condition_cols": [
                "paid_observation_index",
                "observed_path_index",
                "mean_other_path_reward",
            ],
        },
        "observed_minus_mean_other_path_reward": {
            "x_col": "observed_minus_mean_other_path_reward",
            "extra_condition_cols": [
                "paid_observation_index",
                "observed_path_index",
                "actual_path_reward",
                "mean_other_path_reward",
            ],
        },
    }
    summaries: dict[str, pd.DataFrame] = {}
    for name, spec in x_specs.items():
        x_col = spec["x_col"]
        seed_summary = (
            cells.groupby(seed_cols + [x_col], dropna=False)
            .agg(
                seed_trace_cov_z_mu_norm=("trace_cov_z_mu_norm", "mean"),
                n_cells=("trace_cov_z_mu_norm", lambda x: int(np.isfinite(x).sum())),
                mean_n_states=("n_states", "mean"),
            )
            .reset_index()
        )
        summary = (
            seed_summary.groupby(base_cols + [x_col], dropna=False)
            .agg(
                mean_trace_cov_z_mu_norm=("seed_trace_cov_z_mu_norm", "mean"),
                sem_trace_cov_z_mu_norm=("seed_trace_cov_z_mu_norm", sem_finite),
                n_seeds=("seed", "nunique"),
                mean_cells=("n_cells", "mean"),
                mean_n_states=("mean_n_states", "mean"),
            )
            .reset_index()
        )
        summaries[name] = summary
    return cells, summaries


def plot_latent_mu_trace_covariance_sigma_rows(
    summary: pd.DataFrame,
    outpath: Path,
    *,
    x_col: str,
    x_label: str,
    min_samples: int = 1,
    symmetric_x: bool = False,
) -> None:
    if summary.empty or x_col not in summary.columns:
        return
    data = summary.copy()
    data["mean_trace_cov_z_mu_norm"] = pd.to_numeric(
        data["mean_trace_cov_z_mu_norm"], errors="coerce"
    )
    data["sem_trace_cov_z_mu_norm"] = pd.to_numeric(
        data["sem_trace_cov_z_mu_norm"], errors="coerce"
    )
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[x_col, "mean_trace_cov_z_mu_norm"]
    )
    if data.empty:
        return
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.dpi": 300,
        }
    )
    sigmas = sorted(pd.to_numeric(data["sigma"], errors="coerce").dropna().unique())
    if not sigmas:
        return
    y_lower, y_upper = dynamic_linear_limits(
        data["mean_trace_cov_z_mu_norm"].to_numpy(dtype=float),
        data["sem_trace_cov_z_mu_norm"].to_numpy(dtype=float),
    )
    y_lower = min(0.0, y_lower)
    if y_upper <= y_lower:
        y_upper = y_lower + 1.0
    family_specs = [
        ("vary_beta", "Vary beta", "beta", FAMILY_COLOR_RAMPS["vary_beta"], "o", "-"),
        ("vary_opportunity", "Vary opportunity", "opp", FAMILY_COLOR_RAMPS["vary_opportunity"], "^", "--"),
    ]
    fig, axes = plt.subplots(
        len(sigmas),
        len(family_specs),
        figsize=(
            len(family_specs) * 1.55 * PANEL_WIDTH_IN + 1.55,
            max(len(sigmas), 1) * PANEL_HEIGHT_IN + 0.65,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    handles, labels = [], []
    for row_idx, sigma in enumerate(sigmas):
        sigma_data = data[np.isclose(pd.to_numeric(data["sigma"], errors="coerce"), sigma)].copy()
        for col_idx, (family, family_title, short_name, cmap_name, marker, linestyle) in enumerate(family_specs):
            ax = axes[row_idx, col_idx]
            fam_data = sigma_data[sigma_data["family"] == family].copy()
            if fam_data.empty:
                if row_idx == 0:
                    ax.set_title(family_title)
                if col_idx == 0:
                    ax.set_ylabel(f"sigma={sigma:g}\ntrace Cov\nVar(z0)+Var(z1)")
                ax.set_ylim(y_lower, y_upper)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(length=2, pad=1)
                continue
            params = sorted(pd.to_numeric(fam_data["parameter_value"], errors="coerce").dropna().unique())
            colors = color_values(len(params), cmap_name)
            for color, param in zip(colors, params):
                sub = fam_data[
                    np.isclose(pd.to_numeric(fam_data["parameter_value"], errors="coerce"), param)
                ].sort_values(x_col)
                if sub.empty:
                    continue
                label = f"{short_name}={float(param):g}"
                line = ax.errorbar(
                    sub[x_col],
                    sub["mean_trace_cov_z_mu_norm"],
                    yerr=sub["sem_trace_cov_z_mu_norm"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=0.9,
                    markersize=2.5,
                    capsize=1.5,
                    color=color,
                    label=label,
                )
                if row_idx == 0 and label not in labels:
                    handles.append(line)
                    labels.append(label)
            if row_idx == 0:
                ax.set_title(family_title)
            if col_idx == 0:
                ax.set_ylabel(f"sigma={sigma:g}\ntrace Cov\nVar(z0)+Var(z1)")
            ax.set_ylim(y_lower, y_upper)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=2, pad=1)
            if symmetric_x:
                ax.set_xlim(*symmetric_observed_difference_xlim(data, x_col))
                ax.axvline(0.0, color="#999999", linewidth=0.45, zorder=0)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    if handles:
        fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.84, 0.5))
    fig.tight_layout(rect=(0, 0, 0.83, 1), h_pad=0.75, w_pad=0.65)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_latent_mu_trace_covariance_outputs(
    density_rows: pd.DataFrame,
    outdir: Path,
    *,
    min_samples: int,
) -> None:
    cells, summaries = latent_mu_trace_covariance_summaries(
        density_rows,
        min_samples=int(min_samples),
    )
    if cells.empty or not summaries:
        print("Latent z_mu trace covariance: no eligible cells to plot.", flush=True)
        return
    cells.to_csv(outdir / "latent_z_mu_trace_covariance_cells.csv", index=False)
    plot_specs = {
        "paid_observation_index": (
            "paid observation index",
            "latent_z_mu_trace_covariance_vs_paid_observation_index_sigma_rows_beta_opp_overlay.png",
            False,
        ),
        "actual_path_reward": (
            "actual reward of observed path",
            "latent_z_mu_trace_covariance_vs_observed_path_reward_sigma_rows_beta_opp_overlay.png",
            False,
        ),
        "observed_minus_mean_other_path_reward": (
            "observed path reward - mean other path reward",
            "latent_z_mu_trace_covariance_vs_observed_minus_mean_other_path_reward_sigma_rows_beta_opp_overlay.png",
            True,
        ),
    }
    for key, summary in summaries.items():
        summary.to_csv(outdir / f"latent_z_mu_trace_covariance_by_{key}.csv", index=False)
        x_label, filename, symmetric_x = plot_specs[key]
        plot_latent_mu_trace_covariance_sigma_rows(
            summary,
            outdir / filename,
            x_col=key,
            x_label=x_label,
            min_samples=min_samples,
            symmetric_x=symmetric_x,
        )


def observed_value_contour_groups(
    df: pd.DataFrame,
    *,
    color_col: str,
    min_density_samples: int,
    max_groups: int = 8,
) -> list[tuple[float, pd.DataFrame]]:
    if df.empty or color_col not in df.columns:
        return []
    work = df.copy()
    work[color_col] = pd.to_numeric(work[color_col], errors="coerce")
    work = work[np.isfinite(work[color_col])].copy()
    if work.empty:
        return []
    min_n = max(1, int(min_density_samples))
    values = np.asarray(sorted(work[color_col].dropna().unique()), dtype=float)
    groups: list[tuple[float, pd.DataFrame]] = []
    if len(values) <= max_groups:
        for value in values:
            sub = work[np.isclose(work[color_col], value)].copy()
            if len(sub) >= min_n:
                groups.append((float(value), sub))
        return groups
    n_bins = min(int(max_groups), max(1, len(work) // min_n))
    if n_bins <= 1:
        return [(float(np.nanmean(work[color_col])), work)] if len(work) >= min_n else []
    try:
        bins = pd.qcut(work[color_col], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(work[color_col], bins=n_bins, duplicates="drop")
    for _, sub in work.groupby(bins, observed=True):
        if len(sub) >= min_n:
            groups.append((float(np.nanmean(sub[color_col])), sub.copy()))
    return groups


def scatter_density_points(
    ax,
    df: pd.DataFrame,
    *,
    color_col: str,
    cmap,
    norm,
    max_points: int,
    seed: int,
) -> None:
    work = df[["z_mu_0", "z_mu_1", color_col]].copy()
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return
    if max_points > 0 and len(work) > max_points:
        work = work.sample(max_points, random_state=seed)
    ax.scatter(
        work["z_mu_0"],
        work["z_mu_1"],
        c=work[color_col],
        cmap=cmap,
        norm=norm,
        s=5,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )


def draw_per_trial_posterior_contours(
    ax,
    df: pd.DataFrame,
    *,
    color_col: str,
    cmap,
    norm,
    max_points: int,
    seed: int,
    mass: float = POSTERIOR_TRIAL_CONTOUR_MASS,
) -> None:
    required = ["z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1", color_col]
    if df.empty or any(col not in df.columns for col in required):
        return
    work = df[required].copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["z_sigma_0"] > 0.0) & (work["z_sigma_1"] > 0.0)].copy()
    if work.empty:
        return
    if max_points > 0 and len(work) > max_points:
        work = work.sample(max_points, random_state=seed)
    clipped_mass = min(max(float(mass), 1e-6), 1.0 - 1e-6)
    radius = math.sqrt(-2.0 * math.log(1.0 - clipped_mass))
    for row in work.itertuples(index=False):
        mu0 = float(getattr(row, "z_mu_0"))
        mu1 = float(getattr(row, "z_mu_1"))
        sigma0 = float(getattr(row, "z_sigma_0"))
        sigma1 = float(getattr(row, "z_sigma_1"))
        value = float(getattr(row, color_col))
        ellipse = Ellipse(
            (mu0, mu1),
            width=2.0 * radius * sigma0,
            height=2.0 * radius * sigma1,
            angle=0.0,
            facecolor="none",
            edgecolor=cmap(norm(value)),
            linewidth=0.28,
            alpha=0.22,
            zorder=1,
        )
        ax.add_patch(ellipse)


def zero_centered_axis_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    if frame.empty:
        return (-5.0, 5.0), (-5.0, 5.0)
    x = pd.to_numeric(frame.get("z_mu_0", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame.get("z_mu_1", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if finite.size == 0:
        max_abs = 5.0
    else:
        max_abs = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            max_abs = 5.0
    max_abs = max(max_abs * 1.08, 1.0)
    return (-max_abs, max_abs), (-max_abs, max_abs)


def plot_model_reward_density_grid(
    frame: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    density_uses_posterior: bool,
    min_samples: int,
    max_points: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    required = {"model_label", "model_sort", "actual_node_reward", "z_mu_0", "z_mu_1", "observed_reward"}
    if frame.empty or not required.issubset(frame.columns):
        return
    plot_df = frame.copy()
    for col in ["actual_node_reward", "z_mu_0", "z_mu_1", "observed_reward"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["model_label", "model_sort", "actual_node_reward", "z_mu_0", "z_mu_1", "observed_reward"]
    )
    if plot_df.empty:
        return
    rewards = sorted(float(x) for x in plot_df["actual_node_reward"].dropna().unique())
    model_key = (
        plot_df[["model_label", "model_sort"]]
        .drop_duplicates()
        .sort_values("model_sort")
    )
    models = model_key["model_label"].astype(str).tolist()
    if not rewards or not models:
        return
    color_values = plot_df["observed_reward"].to_numpy(dtype=float)
    color_values = color_values[np.isfinite(color_values)]
    if color_values.size == 0:
        return
    color_min = float(np.nanmin(color_values))
    color_max = float(np.nanmax(color_values))
    if math.isclose(color_min, color_max):
        color_min -= 0.5
        color_max += 0.5
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=color_min, vmax=color_max)
    n_rows = len(models)
    n_cols = len(rewards)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * PANEL_WIDTH_IN + 1.15, n_rows * PANEL_HEIGHT_IN + 0.95),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    x_grid = np.linspace(float(xlim[0]), float(xlim[1]), 120)
    y_grid = np.linspace(float(ylim[0]), float(ylim[1]), 120)
    density_fn = gpga.base.posterior_z_mixture_density if density_uses_posterior else gpga.base.empirical_mu_kde_density
    tick_values = [float(xlim[0]), 0.0, float(xlim[1])]
    for row_i, model in enumerate(models):
        for col_i, reward in enumerate(rewards):
            ax = axes[row_i, col_i]
            panel = plot_df[
                (plot_df["model_label"].astype(str) == str(model))
                & np.isclose(plot_df["actual_node_reward"], reward)
            ].copy()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks(tick_values)
            ax.set_yticks([float(ylim[0]), 0.0, float(ylim[1])])
            if row_i == 0:
                ax.set_title(f"R={reward:g}", fontsize=PLOT_FONT_SIZE_PT)
            if col_i == 0:
                ax.set_ylabel(str(model), fontsize=PLOT_FONT_SIZE_PT)
            if panel.empty or len(panel) < max(1, int(min_samples)):
                ax.text(0.5, 0.5, f"n={len(panel)}", transform=ax.transAxes, ha="center", va="center", fontsize=6)
            else:
                if density_uses_posterior:
                    draw_per_trial_posterior_contours(
                        ax,
                        panel,
                        color_col="observed_reward",
                        cmap=cmap,
                        norm=norm,
                        max_points=max_points,
                        seed=stable_group_seed((title, row_i, reward, "posterior_contours")),
                    )
                else:
                    for value, group in observed_value_contour_groups(
                        panel,
                        color_col="observed_reward",
                        min_density_samples=min_samples,
                    ):
                        density = density_fn(
                            group,
                            x_grid,
                            y_grid,
                            max_points=max_points,
                            seed=stable_group_seed((title, row_i, reward, value)),
                            min_samples=min_samples,
                        )
                        if density is None:
                            continue
                        levels = gpga.base.positive_contour_levels(density, masses=(0.50, 0.90))
                        if levels is None:
                            continue
                        ax.contour(
                            x_grid,
                            y_grid,
                            density,
                            levels=levels,
                            colors=[cmap(norm(float(value)))],
                            linewidths=0.65,
                            alpha=0.85,
                        )
                scatter_density_points(
                    ax,
                    panel,
                    color_col="observed_reward",
                    cmap=cmap,
                    norm=norm,
                    max_points=max_points,
                    seed=stable_group_seed((title, row_i, reward, "scatter")),
                )
            ax.axhline(0.0, color="#bdbdbd", linewidth=0.35, zorder=0)
            ax.axvline(0.0, color="#bdbdbd", linewidth=0.35, zorder=0)
            ax.tick_params(length=2, pad=1, labelsize=PLOT_FONT_SIZE_PT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=PLOT_FONT_SIZE_PT, y=0.995)
    fig.supxlabel(x_label, fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(y_label, fontsize=PLOT_FONT_SIZE_PT, x=0.005)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.tight_layout(rect=(0.02, 0.04, 0.88, 0.96), h_pad=0.35, w_pad=0.25)
    fig_height = float(fig.get_size_inches()[1])
    cbar_height = min(0.82, PANEL_HEIGHT_IN / max(fig_height, 1e-6))
    cbar_y = 0.5 - 0.5 * cbar_height
    cbar_ax = fig.add_axes([0.91, cbar_y, 0.018, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=PLOT_FONT_SIZE_PT)
    cbar.set_label("observed value", fontsize=PLOT_FONT_SIZE_PT)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_other_vs_observed_reward_density_grid(
    frame: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    density_uses_posterior: bool,
    min_samples: int,
    max_points: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    required = {
        "actual_node_reward",
        "other_node_actual_reward",
        "z_mu_0",
        "z_mu_1",
        "observed_reward",
    }
    if frame.empty or not required.issubset(frame.columns):
        return
    plot_df = frame.copy()
    for col in ["actual_node_reward", "other_node_actual_reward", "z_mu_0", "z_mu_1", "observed_reward"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual_node_reward", "other_node_actual_reward", "z_mu_0", "z_mu_1", "observed_reward"]
    )
    if plot_df.empty:
        return
    observed_rewards = sorted(float(x) for x in plot_df["actual_node_reward"].dropna().unique())
    other_rewards = sorted(
        (float(x) for x in plot_df["other_node_actual_reward"].dropna().unique()),
        reverse=True,
    )
    if not observed_rewards or not other_rewards:
        return
    color_values = plot_df["observed_reward"].to_numpy(dtype=float)
    color_values = color_values[np.isfinite(color_values)]
    if color_values.size == 0:
        return
    color_min = float(np.nanmin(color_values))
    color_max = float(np.nanmax(color_values))
    if math.isclose(color_min, color_max):
        color_min -= 0.5
        color_max += 0.5
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=color_min, vmax=color_max)
    n_rows = len(other_rewards)
    n_cols = len(observed_rewards)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * PANEL_WIDTH_IN + 1.15, n_rows * PANEL_HEIGHT_IN + 0.95),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    x_grid = np.linspace(float(xlim[0]), float(xlim[1]), 120)
    y_grid = np.linspace(float(ylim[0]), float(ylim[1]), 120)
    density_fn = gpga.base.posterior_z_mixture_density if density_uses_posterior else gpga.base.empirical_mu_kde_density
    tick_values = [float(xlim[0]), 0.0, float(xlim[1])]
    for row_i, other_reward in enumerate(other_rewards):
        for col_i, observed_reward in enumerate(observed_rewards):
            ax = axes[row_i, col_i]
            panel = plot_df[
                np.isclose(plot_df["other_node_actual_reward"], other_reward)
                & np.isclose(plot_df["actual_node_reward"], observed_reward)
            ].copy()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks(tick_values)
            ax.set_yticks([float(ylim[0]), 0.0, float(ylim[1])])
            if row_i == 0:
                ax.set_title(f"R(obs)={observed_reward:g}", fontsize=PLOT_FONT_SIZE_PT)
            if col_i == 0:
                ax.set_ylabel(f"R(other)={other_reward:g}", fontsize=PLOT_FONT_SIZE_PT)
            if panel.empty or len(panel) < max(1, int(min_samples)):
                ax.text(0.5, 0.5, f"n={len(panel)}", transform=ax.transAxes, ha="center", va="center", fontsize=6)
            else:
                if density_uses_posterior:
                    draw_per_trial_posterior_contours(
                        ax,
                        panel,
                        color_col="observed_reward",
                        cmap=cmap,
                        norm=norm,
                        max_points=max_points,
                        seed=stable_group_seed((title, row_i, col_i, "posterior_contours")),
                    )
                else:
                    for value, group in observed_value_contour_groups(
                        panel,
                        color_col="observed_reward",
                        min_density_samples=min_samples,
                    ):
                        density = density_fn(
                            group,
                            x_grid,
                            y_grid,
                            max_points=max_points,
                            seed=stable_group_seed((title, row_i, col_i, value)),
                            min_samples=min_samples,
                        )
                        if density is None:
                            continue
                        levels = gpga.base.positive_contour_levels(density, masses=(0.50, 0.90))
                        if levels is None:
                            continue
                        ax.contour(
                            x_grid,
                            y_grid,
                            density,
                            levels=levels,
                            colors=[cmap(norm(float(value)))],
                            linewidths=0.65,
                            alpha=0.85,
                        )
                scatter_density_points(
                    ax,
                    panel,
                    color_col="observed_reward",
                    cmap=cmap,
                    norm=norm,
                    max_points=max_points,
                    seed=stable_group_seed((title, row_i, col_i, "scatter")),
                )
            ax.axhline(0.0, color="#bdbdbd", linewidth=0.35, zorder=0)
            ax.axvline(0.0, color="#bdbdbd", linewidth=0.35, zorder=0)
            ax.tick_params(length=2, pad=1, labelsize=PLOT_FONT_SIZE_PT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=PLOT_FONT_SIZE_PT, y=0.995)
    fig.supxlabel(x_label, fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(y_label, fontsize=PLOT_FONT_SIZE_PT, x=0.005)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.tight_layout(rect=(0.02, 0.04, 0.88, 0.96), h_pad=0.35, w_pad=0.25)
    fig_height = float(fig.get_size_inches()[1])
    cbar_height = min(0.82, PANEL_HEIGHT_IN / max(fig_height, 1e-6))
    cbar_y = 0.5 - 0.5 * cbar_height
    cbar_ax = fig.add_axes([0.91, cbar_y, 0.018, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=PLOT_FONT_SIZE_PT)
    cbar.set_label("observed value", fontsize=PLOT_FONT_SIZE_PT)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_sigma_pga_scores(
    frame: pd.DataFrame,
    max_states: int,
    max_iters: int,
    *,
    cache_dir: Path | None = None,
    reuse_cache: bool = False,
) -> pd.DataFrame:
    mu_cols = sorted(
        [col for col in frame.columns if col.startswith("prior_normalized_mu_")],
        key=lambda col: int(col.rsplit("_", 1)[-1]),
    )
    sigma_cols = sorted(
        [col for col in frame.columns if col.startswith("prior_normalized_sigma_")],
        key=lambda col: int(col.rsplit("_", 1)[-1]),
    )
    if frame.empty or len(mu_cols) < 2 or len(mu_cols) != len(sigma_cols):
        return pd.DataFrame()
    scores_cache_path = cache_dir / "aggregate_pga_scores.csv" if cache_dir is not None else None
    fit_cache_path = cache_dir / "aggregate_pga_fit.npz" if cache_dir is not None else None
    cache_label = str(cache_dir) if cache_dir is not None else "no-cache-dir"
    print(
        f"PGA[{cache_label}]: requested for {len(frame)} row(s); "
        f"reuse_cache={bool(reuse_cache)}; max_states={int(max_states)}; max_iters={int(max_iters)}",
        flush=True,
    )
    if reuse_cache and scores_cache_path is not None and scores_cache_path.exists():
        print(f"PGA[{cache_label}]: loading cached score table {scores_cache_path}", flush=True)
        cached = pd.read_csv(scores_cache_path)
        if {"z_mu_0", "z_mu_1"}.issubset(cached.columns):
            print(f"PGA[{cache_label}]: loaded {len(cached)} cached score row(s)", flush=True)
            return cached
        print(
            f"PGA[{cache_label}]: cached score table is missing z_mu_0/z_mu_1; refitting",
            flush=True,
        )

    work = frame.copy().reset_index(drop=True)
    values = work[mu_cols + sigma_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.notna().all(axis=1)
    work = work.loc[valid].reset_index(drop=True)
    print(f"PGA[{cache_label}]: {len(work)} row(s) remain after finite mu/sigma filtering", flush=True)
    if len(work) < 2:
        print(f"PGA[{cache_label}]: fewer than 2 rows; skipping PGA", flush=True)
        return pd.DataFrame()
    mu = work[mu_cols].to_numpy(dtype=float)
    sigma = work[sigma_cols].to_numpy(dtype=float)
    sigma = np.maximum(sigma, 1e-8)
    if reuse_cache and fit_cache_path is not None and fit_cache_path.exists():
        print(f"PGA[{cache_label}]: loading cached fit {fit_cache_path}", flush=True)
        pga = gpga.ProductGaussianPGA.load(fit_cache_path)
    else:
        fit_idx = np.arange(len(work), dtype=int)
        if int(max_states) > 0 and len(fit_idx) > int(max_states):
            rng = np.random.default_rng(20260706)
            fit_idx = np.sort(rng.choice(fit_idx, size=int(max_states), replace=False))
        print(
            f"PGA[{cache_label}]: fitting PGA on {len(fit_idx)} sampled state(s) "
            f"from {len(work)} available row(s)",
            flush=True,
        )
        pga = gpga.ProductGaussianPGA(
            n_components=2,
            max_iters=int(max_iters),
            progress_label=f"PGA[{cache_label}]",
            progress_every=max(1, min(10, int(max_iters))),
        )
        pga.fit(mu[fit_idx], sigma[fit_idx])
        if fit_cache_path is not None:
            fit_cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"PGA[{cache_label}]: saving fit cache {fit_cache_path}", flush=True)
            pga.save(fit_cache_path)
    print(f"PGA[{cache_label}]: transforming {len(work)} state(s) to scores", flush=True)
    scores = pga.transform(mu, sigma)
    if scores.shape[1] < 2:
        print(f"PGA[{cache_label}]: transformed scores have fewer than 2 components; skipping", flush=True)
        return pd.DataFrame()
    work["z_mu_0"] = scores[:, 0]
    work["z_mu_1"] = scores[:, 1]
    # PGA scores are point coordinates; remove posterior sigmas so the shared
    # plotter uses empirical KDE instead of Gaussian posterior mixtures.
    for col in ["z_sigma_0", "z_sigma_1"]:
        if col in work.columns:
            work = work.drop(columns=[col])
    if scores_cache_path is not None:
        scores_cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"PGA[{cache_label}]: saving score cache {scores_cache_path}", flush=True)
        work.to_csv(scores_cache_path, index=False)
    print(f"PGA[{cache_label}]: done; score_rows={len(work)}", flush=True)
    return work


def plot_aggregate_latent_density_outputs(
    density_rows: pd.DataFrame,
    outdir: Path,
    *,
    min_samples: int,
    max_points: int,
    pga_max_states: int,
    pga_max_iters: int,
    reuse_pga_fits: bool = False,
    keep_pooled_plots: bool = False,
    fit_pga: bool = True,
    write_normalized_z_plots: bool = True,
) -> None:
    if density_rows.empty:
        return
    if not bool(fit_pga) and not bool(write_normalized_z_plots):
        print(
            "Aggregate latent density: skipping because both PGA fitting/plots and "
            "normalized-z plots are disabled.",
            flush=True,
        )
        return
    root = outdir / "aggregate_latent_density_by_sigma"
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"Aggregate latent density: starting with {len(density_rows)} paid latent row(s); "
        f"reuse_pga_fits={bool(reuse_pga_fits)}; "
        f"keep_pooled_plots={bool(keep_pooled_plots)}; "
        f"fit_pga={bool(fit_pga)}; "
        f"write_normalized_z_plots={bool(write_normalized_z_plots)}; "
        f"output_root={root}",
        flush=True,
    )
    if "other_node_actual_reward" not in density_rows.columns:
        print(
            "Aggregate latent density: paid_latent_density_rows.csv does not contain "
            "other_node_actual_reward, so the beta/opp-specific row-by-other-reward "
            "density plots cannot be generated from this cache. Regenerate once "
            "without --plot-only to rebuild the density cache.",
            flush=True,
        )
    if bool(fit_pga) and not reuse_pga_fits:
        print(
            "Aggregate latent density: --reuse-aggregate-pga-fits is not set, "
            "so PGA will be refit even if previous cache files exist.",
            flush=True,
        )
    if not bool(fit_pga):
        print(
            "Aggregate latent density: --no-pga-fitting is set; skipping PGA "
            "fit/load and PGA score plots.",
            flush=True,
        )
    if not bool(write_normalized_z_plots):
        print(
            "Aggregate latent density: normalized-z plots are disabled for this "
            "non-default tree.",
            flush=True,
        )
    for sigma in sorted(pd.to_numeric(density_rows["sigma"], errors="coerce").dropna().unique()):
        sigma_rows = density_rows[np.isclose(pd.to_numeric(density_rows["sigma"], errors="coerce"), sigma)].copy()
        if sigma_rows.empty:
            continue
        sigma_dir = root / f"sigma_{value_token(sigma)}"
        sigma_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Aggregate latent density: sigma={sigma:g}; rows={len(sigma_rows)}; dir={sigma_dir}",
            flush=True,
        )

        norm_rows = sigma_rows.copy()
        norm_rows["z_mu_0"] = pd.to_numeric(norm_rows["prior_normalized_z_mu_0"], errors="coerce")
        norm_rows["z_mu_1"] = pd.to_numeric(norm_rows["prior_normalized_z_mu_1"], errors="coerce")
        norm_rows["z_sigma_0"] = pd.to_numeric(norm_rows["prior_normalized_z_sigma_0"], errors="coerce")
        norm_rows["z_sigma_1"] = pd.to_numeric(norm_rows["prior_normalized_z_sigma_1"], errors="coerce")
        if bool(fit_pga):
            pga_rows = fit_sigma_pga_scores(
                sigma_rows,
                max_states=pga_max_states,
                max_iters=pga_max_iters,
                cache_dir=sigma_dir,
                reuse_cache=reuse_pga_fits,
            )
        else:
            pga_rows = pd.DataFrame()
        pga_xlim, pga_ylim = zero_centered_axis_limits(pga_rows)

        timesteps = sorted(pd.to_numeric(sigma_rows["timestep"], errors="coerce").dropna().astype(int).unique())
        nodes = sorted(pd.to_numeric(sigma_rows["observed_node"], errors="coerce").dropna().astype(int).unique())
        print(
            f"Aggregate latent density: sigma={sigma:g}; plotting {len(timesteps)} timestep(s) "
            f"x {len(nodes)} observed-node value(s)",
            flush=True,
        )

        def write_density_set(
            *,
            target_dir: Path,
            norm_source: pd.DataFrame,
            pga_source: pd.DataFrame,
            title_suffix: str,
        ) -> None:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Aggregate latent density: writing density set {target_dir} "
                f"with norm_rows={len(norm_source)}, pga_rows={len(pga_source)}",
                flush=True,
            )
            for timestep in timesteps:
                for node in nodes:
                    selector = (
                        np.isclose(pd.to_numeric(norm_source["timestep"], errors="coerce"), timestep)
                        & np.isclose(pd.to_numeric(norm_source["observed_node"], errors="coerce"), node)
                    )
                    if write_normalized_z_plots:
                        plot_model_reward_density_grid(
                            norm_source[selector].copy(),
                            target_dir / f"prior_normalized_z_t{timestep}_observed_node_{node}.png",
                            title=(
                                f"Prior-normalized z, sigma={sigma:g}, t={timestep}, "
                                f"observed node={node}{title_suffix}"
                            ),
                            x_label="(z_0 - prior_mu_0) / prior_sigma_0",
                            y_label="(z_1 - prior_mu_1) / prior_sigma_1",
                            density_uses_posterior=True,
                            min_samples=min_samples,
                            max_points=max_points,
                            xlim=(-5.0, 5.0),
                            ylim=(-5.0, 5.0),
                        )
                    if not pga_source.empty:
                        pga_selector = (
                            np.isclose(pd.to_numeric(pga_source["timestep"], errors="coerce"), timestep)
                            & np.isclose(pd.to_numeric(pga_source["observed_node"], errors="coerce"), node)
                        )
                        plot_model_reward_density_grid(
                            pga_source[pga_selector].copy(),
                            target_dir / f"pga_scores_t{timestep}_observed_node_{node}.png",
                            title=(
                                f"Prior-normalized PGA scores, sigma={sigma:g}, t={timestep}, "
                                f"observed node={node}{title_suffix}"
                            ),
                            x_label="PGA score 0",
                            y_label="PGA score 1",
                            density_uses_posterior=False,
                            min_samples=min_samples,
                            max_points=max_points,
                            xlim=pga_xlim,
                            ylim=pga_ylim,
                        )

        def write_param_reward_pair_density_set(
            *,
            target_dir: Path,
            norm_source: pd.DataFrame,
            pga_source: pd.DataFrame,
            title_suffix: str,
        ) -> None:
            if norm_source.empty or "other_node_actual_reward" not in norm_source.columns:
                return
            target_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Aggregate latent density: writing reward-pair density set {target_dir} "
                f"with norm_rows={len(norm_source)}, pga_rows={len(pga_source)}",
                flush=True,
            )
            for timestep in timesteps:
                for node in nodes:
                    selector = (
                        np.isclose(pd.to_numeric(norm_source["timestep"], errors="coerce"), timestep)
                        & np.isclose(pd.to_numeric(norm_source["observed_node"], errors="coerce"), node)
                    )
                    if write_normalized_z_plots:
                        plot_other_vs_observed_reward_density_grid(
                            norm_source[selector].copy(),
                            target_dir / f"prior_normalized_z_t{timestep}_observed_node_{node}.png",
                            title=(
                                f"Prior-normalized z, sigma={sigma:g}, t={timestep}, "
                                f"observed node={node}{title_suffix}"
                            ),
                            x_label="(z_0 - prior_mu_0) / prior_sigma_0",
                            y_label="(z_1 - prior_mu_1) / prior_sigma_1",
                            density_uses_posterior=True,
                            min_samples=min_samples,
                            max_points=max_points,
                            xlim=(-5.0, 5.0),
                            ylim=(-5.0, 5.0),
                        )
                    if not pga_source.empty and "other_node_actual_reward" in pga_source.columns:
                        pga_selector = (
                            np.isclose(pd.to_numeric(pga_source["timestep"], errors="coerce"), timestep)
                            & np.isclose(pd.to_numeric(pga_source["observed_node"], errors="coerce"), node)
                        )
                        plot_other_vs_observed_reward_density_grid(
                            pga_source[pga_selector].copy(),
                            target_dir / f"pga_scores_t{timestep}_observed_node_{node}.png",
                            title=(
                                f"Prior-normalized PGA scores, sigma={sigma:g}, t={timestep}, "
                                f"observed node={node}{title_suffix}"
                            ),
                            x_label="PGA score 0",
                            y_label="PGA score 1",
                            density_uses_posterior=False,
                            min_samples=min_samples,
                            max_points=max_points,
                            xlim=pga_xlim,
                            ylim=pga_ylim,
                        )

        if keep_pooled_plots:
            write_density_set(
                target_dir=sigma_dir,
                norm_source=norm_rows,
                pga_source=pga_rows,
                title_suffix="",
            )
        else:
            print(
                "Aggregate latent density: skipping pooled all-seed plots. "
                "Use --keep-pooled-aggregate-density-plots to also write them.",
                flush=True,
            )

        seeds = sorted(pd.to_numeric(sigma_rows["seed"], errors="coerce").dropna().astype(int).unique())
        for seed in seeds:
            print(f"Aggregate latent density: sigma={sigma:g}; seed={seed}", flush=True)
            seed_norm_rows = norm_rows[
                np.isclose(pd.to_numeric(norm_rows["seed"], errors="coerce"), seed)
            ].copy()
            seed_pga_rows = (
                pga_rows[np.isclose(pd.to_numeric(pga_rows["seed"], errors="coerce"), seed)].copy()
                if not pga_rows.empty
                else pd.DataFrame()
            )
            if keep_pooled_plots:
                write_density_set(
                    target_dir=sigma_dir / f"seed_{value_token(seed)}",
                    norm_source=seed_norm_rows,
                    pga_source=seed_pga_rows,
                    title_suffix=f", seed={seed}",
                )
            elif "other_node_actual_reward" not in seed_norm_rows.columns:
                print(
                    "Aggregate latent density: not writing pooled seed-level plots by default, "
                    "and this cached density table cannot write grouped beta/opp plots because "
                    "other_node_actual_reward is missing.",
                    flush=True,
                )
            if not seed_norm_rows.empty:
                for (_, _), param_norm_rows in seed_norm_rows.groupby(
                    ["family", "parameter_value"],
                    sort=False,
                    dropna=False,
                ):
                    if param_norm_rows.empty:
                        continue
                    print(
                        f"Aggregate latent density: sigma={sigma:g}; seed={seed}; "
                        f"{family_param_label(param_norm_rows.iloc[0])}",
                        flush=True,
                    )
                    param_dir = sigma_dir / f"seed_{value_token(seed)}" / param_density_folder_name(param_norm_rows.iloc[0])
                    if not seed_pga_rows.empty:
                        param_pga_rows = seed_pga_rows[
                            (seed_pga_rows["family"].astype(str) == str(param_norm_rows["family"].iloc[0]))
                            & np.isclose(
                                pd.to_numeric(seed_pga_rows["parameter_value"], errors="coerce"),
                                float(param_norm_rows["parameter_value"].iloc[0]),
                            )
                        ].copy()
                    else:
                        param_pga_rows = pd.DataFrame()
                    write_param_reward_pair_density_set(
                        target_dir=param_dir,
                        norm_source=param_norm_rows.copy(),
                        pga_source=param_pga_rows,
                        title_suffix=f", seed={seed}, {family_param_label(param_norm_rows.iloc[0])}",
                    )


def write_sample_set_trace_analysis_outputs(
    paid_latents: pd.DataFrame,
    outdir: Path,
    *,
    diff_bin_width: float,
    diff_num_bins: int,
    sigma_pair_trials_per_sigma: int,
    min_samples: int = 1,
) -> None:
    if paid_latents.empty:
        return
    last_paid = paid_latents[
        pd.to_numeric(paid_latents["paid_observation_index"], errors="coerce")
        == pd.to_numeric(paid_latents["n_paid_latents"], errors="coerce")
    ].copy()
    if not last_paid.empty:
        print(
            f"Building sigma-pair heatmap diagnostics from {len(last_paid)} last-paid latent row(s).",
            flush=True,
        )
        sigma_pairwise, sigma_summary = compute_sigma_pairwise_last_paid_kl(
            last_paid,
            trials_per_sigma=int(sigma_pair_trials_per_sigma),
        )
        sigma_pairwise.to_csv(outdir / "last_paid_latent_sigma_pairwise_kl_detail.csv", index=False)
        sigma_summary.to_csv(outdir / "last_paid_latent_sigma_pairwise_kl_summary.csv", index=False)
        plot_sigma_pair_heatmaps(sigma_summary, outdir, min_samples=min_samples)


def plot_saved_sample_set_trace_analysis_outputs(
    outdir: Path,
    *,
    min_samples: int = 1,
    diff_bin_width: float = 1.0,
    diff_num_bins: int = 8,
) -> bool:
    plotted_any = False
    sigma_summary_path = outdir / "last_paid_latent_sigma_pairwise_kl_summary.csv"
    if sigma_summary_path.exists():
        sigma_summary = pd.read_csv(sigma_summary_path)
        plot_sigma_pair_heatmaps(sigma_summary, outdir, min_samples=min_samples)
        plotted_any = True
    return plotted_any


def plot_all_metric_variants(
    pairwise: pd.DataFrame,
    summary: pd.DataFrame,
    paid_by_timestep: pd.DataFrame,
    paid_summary: pd.DataFrame,
    outdir: Path,
    successive_by_timestep: pd.DataFrame | None = None,
    successive_summary: pd.DataFrame | None = None,
    min_samples: int = 1,
) -> None:
    if len(paid_by_timestep):
        plot_paid_timestep_sigma_rows(
            paid_by_timestep,
            outdir / "pairwise_kl_by_paid_timestep_sigma_rows_beta_opp_overlay.png",
            min_samples=min_samples,
        )
        if "mean_pairwise_z_mu_mae" in paid_by_timestep.columns:
            plot_paid_timestep_sigma_rows(
                paid_by_timestep,
                outdir / "pairwise_z_mu_mae_by_paid_timestep_sigma_rows_beta_opp_overlay.png",
                mean_col="mean_pairwise_z_mu_mae",
                sem_col="sem_pairwise_z_mu_mae",
                ylabel="Pairwise z_mu MAE",
                log_y=False,
                min_samples=min_samples,
            )
    if successive_by_timestep is not None and len(successive_by_timestep):
        plot_paid_timestep_sigma_rows(
            successive_by_timestep,
            outdir / "successive_kl_vs_paid_timestep_sigma_rows_beta_opp_overlay.png",
            mean_col="mean_successive_sym_kl",
            sem_col="sem_successive_sym_kl",
            ylabel="KL between current\\nand previous paid latent\\n(log scale)",
            log_y=True,
            min_samples=min_samples,
        )
        if "mean_successive_z_mu_mae" in successive_by_timestep.columns:
            plot_paid_timestep_sigma_rows(
                successive_by_timestep,
                outdir / "successive_z_mu_mae_vs_paid_timestep_sigma_rows_beta_opp_overlay.png",
                mean_col="mean_successive_z_mu_mae",
                sem_col="sem_successive_z_mu_mae",
                ylabel="z_mu MAE current\\nvs previous paid latent",
                log_y=False,
                min_samples=min_samples,
            )
    if len(paid_summary) and "mean_z_sigma_across_paid_timesteps" in paid_summary.columns:
        plot_across_paid_summary(
            paid_summary,
            outdir / "mean_z_sigma_mean_across_paid_latents_by_sigma.png",
            mean_col="mean_z_sigma_across_paid_timesteps",
            sem_col="sem_z_sigma_across_paid_timesteps",
            ylabel="Mean z_sigma\\nacross paid latents",
            log_y=False,
            min_samples=min_samples,
        )


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = jp.build_task(args.tree_size, args.tree_type, args.input_type)
    config = make_config(args, seed=seed, beta=beta, opportunity=opportunity, sigma=sigma)
    model_name = jp.model_name_for(config, task)
    weights_path = Path(config.model_dir) / f"{model_name}.msgpack"
    progress_label = (
        f"{family} {parameter_name}={parameter_value:g} beta={beta:g} "
        f"opp={opportunity:g} sigma={sigma:g} seed={seed}"
    )
    print(f"{progress_label}: loading model {weights_path}", flush=True)
    model, params = jp.load_state_for_sim(config, task)
    rewards, streams, metadata = build_reward_combination_trials(
        np.asarray(task.reward_values, dtype=float),
        num_nodes=int(task.num_nodes),
        sigma=float(sigma),
        n_sample_sets=int(args.n_sample_sets),
        max_observations=int(args.max_observations_before_stop),
        seed=int(seed + round(1000 * sigma) + round(17 * beta) + round(31 * opportunity)),
        n_reward_combinations=int(args.n_reward_combinations),
        reward_combination_seed=int(seed),
    )
    n_conditions = int(metadata["condition_index"].nunique()) if "condition_index" in metadata.columns else 0
    print(
        f"{progress_label}: generated {len(rewards)} total trial(s) "
        f"from {n_conditions} reward condition(s) x {int(args.n_sample_sets)} sample set(s); "
        f"nodes={int(task.num_nodes)}; streams_shape={tuple(streams.shape)}",
        flush=True,
    )
    latent_df, paid_latent_df = rollout_with_streams(
        model,
        params,
        config,
        task,
        rewards,
        streams,
        seed_offset=int(round(10_000 * sigma) + round(beta) + round(10_000 * opportunity)),
        force_round_robin_observations=bool(args.force_round_robin_observations),
        force_first_observe_node=int(args.force_first_observe_node),
        progress_label=progress_label,
    )
    print(f"{progress_label}: enriching metadata and computing pairwise summaries", flush=True)
    paid_latents_with_metadata = enrich_paid_latents_with_metadata(paid_latent_df, metadata)
    pairwise = compute_pairwise_kl(latent_df, metadata)
    paid_pairwise = compute_pairwise_kl_by_paid_timestep(paid_latent_df, metadata)
    successive = compute_successive_timestep_kl(paid_latent_df, metadata)
    print(
        f"{progress_label}: pairwise_rows={len(pairwise)}, "
        f"paid_pairwise_rows={len(paid_pairwise)}, successive_rows={len(successive)}, "
        f"paid_latent_rows={len(paid_latents_with_metadata)}",
        flush=True,
    )
    for col, value in [
        ("family", family),
        ("parameter_name", parameter_name),
        ("parameter_value", parameter_value),
        ("beta", beta),
        ("opportunity", opportunity),
        ("sigma", sigma),
        ("seed", seed),
        ("force_round_robin_observations", bool(args.force_round_robin_observations)),
        ("force_first_observe_node", int(args.force_first_observe_node)),
    ]:
        pairwise[col] = value
        if len(paid_pairwise):
            paid_pairwise[col] = value
        if len(successive):
            successive[col] = value
        if len(paid_latents_with_metadata):
            paid_latents_with_metadata[col] = value
    return pairwise, paid_pairwise, successive, paid_latents_with_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", nargs="?", default="default")
    parser.add_argument("--preset-file", default=str(preset_file_default()))
    parser.add_argument("--vary-beta-values", "--betas", dest="vary_beta_values", default=None)
    parser.add_argument("--vary-opportunity-values", "--opportunity-costs", dest="vary_opportunity_values", default=None)
    parser.add_argument("--sigmas", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--rnn-units", "--rnn-dims", dest="rnn_units", default=None)
    parser.add_argument("--latent-dim", "--latent-dims", dest="latent_dim", default=None)
    parser.add_argument("--lambda-value", "--lambdas", dest="lambda_value", default=None)
    parser.add_argument("--alpha", "--alphas", dest="alpha", default=None)
    parser.add_argument("--n-sample-sets", type=int, default=50)
    parser.add_argument(
        "--n-reward-combinations",
        "--max-reward-combinations",
        type=int,
        default=0,
        help=(
            "Randomly sample this many actual node-reward combinations before "
            "expanding each into --n-sample-sets streams. The default 0 keeps "
            "the exhaustive grid over all reward combinations."
        ),
    )
    parser.add_argument("--max-observations-before-stop", type=int, default=None)
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--sampled-lambda-critic", choices=["q", "value"], default="q")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help=(
            "Minimum sample sets required for a plotted point/cell. "
            "This only filters plots; CSV summaries are still saved unfiltered."
        ),
    )
    parser.add_argument(
        "--latent-density-max-points",
        type=int,
        default=1500,
        help="Maximum points/states drawn per aggregate latent-density panel.",
    )
    parser.add_argument(
        "--aggregate-pga-max-states",
        type=int,
        default=8000,
        help="Maximum paid latent rows used to fit each sigma-specific aggregate PGA.",
    )
    parser.add_argument(
        "--aggregate-pga-max-iters",
        type=int,
        default=100,
        help="Maximum Karcher-mean iterations for aggregate prior-normalized PGA fits.",
    )
    parser.add_argument(
        "--reuse-aggregate-pga-fits",
        action="store_true",
        help=(
            "Reuse per-sigma aggregate PGA cache files in the output directory "
            "instead of refitting them. If only the fitted .npz exists, scores "
            "are recomputed from that fit and cached."
        ),
    )
    parser.add_argument(
        "--no-pga-fitting",
        action="store_true",
        help=(
            "Skip aggregate PGA fit/load and PGA score density plots. This still "
            "allows normalized-z aggregate plots for the 2-node default task."
        ),
    )
    parser.add_argument(
        "--skip-aggregate-latent-density",
        action="store_true",
        help=(
            "Skip the aggregate_latent_density_by_sigma output tree entirely. "
            "Other diagnostics that use paid_latent_density_rows.csv, such as "
            "trace-covariance spread plots, are still generated."
        ),
    )
    parser.add_argument(
        "--keep-pooled-aggregate-density-plots",
        action="store_true",
        help=(
            "Also write the older pooled aggregate latent-density plots directly "
            "under each sigma/ and sigma/seed_* folder. By default, only the "
            "beta/opp-specific reward-grid folders are generated."
        ),
    )
    parser.add_argument(
        "--analysis-diff-bin-width",
        type=float,
        default=1.0,
        help=(
            "Fixed bin width for current - previous same-node observation value "
            "when --analysis-diff-num-bins is 0."
        ),
    )
    parser.add_argument(
        "--analysis-diff-num-bins",
        type=int,
        default=8,
        help=(
            "Number of equal-count bins for observed-difference latent-change plots. "
            "Bins are computed within each sigma x timestep panel; use 0 to return "
            "to fixed-width bins from --analysis-diff-bin-width."
        ),
    )
    parser.add_argument(
        "--sigma-pair-trials-per-sigma",
        type=int,
        default=5,
        help=(
            "For sigma-pair heatmaps, randomly select up to this many last-paid "
            "trials per sigma within each family/parameter/seed/reward-combo group "
            "before computing cross-sigma latent KL. Use 0 to use all available "
            "last-paid trials per sigma."
        ),
    )
    parser.add_argument(
        "--force-round-robin-observations",
        action="store_true",
        help=(
            "Force a matched revisit schedule: node 1, node 2, ..., all nodes, "
            "then repeat with the next sample from each node until max observations."
        ),
    )
    parser.add_argument(
        "--force-first-observe-node",
        type=int,
        default=0,
        help=(
            "Force only the first action to observe this 1-indexed node, then "
            "let the model policy choose all later actions. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing CSVs in outdir and regenerate PNGs without simulations.",
    )
    parser.add_argument(
        "--no-combo-subdir",
        action="store_true",
        help=(
            "Use --outdir directly instead of appending the tree/beta/opportunity "
            "combination subfolder. This is useful for regenerating old flat outputs."
        ),
    )
    return load_default_preset(parser.parse_args())


def main() -> None:
    args = parse_args()
    outdir = resolve_output_dir(args)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {outdir}", flush=True)
    print(f"Minimum sample sets per plotted point/cell: {int(args.min_samples)}", flush=True)
    if args.plot_only:
        pairwise_path = outdir / "pairwise_kl_by_first_observed_mean_other.csv"
        if not pairwise_path.exists():
            pairwise_path = outdir / "pairwise_kl_by_reward_pair.csv"
        summary_path = outdir / "pairwise_kl_summary.csv"
        paid_timestep_reward_pair_path = (
            outdir / "pairwise_kl_by_paid_timestep_first_observed_mean_other.csv"
        )
        if not paid_timestep_reward_pair_path.exists():
            paid_timestep_reward_pair_path = outdir / "pairwise_kl_by_paid_timestep_reward_pair.csv"
        paid_summary_path = outdir / "pairwise_kl_across_paid_timesteps_summary.csv"
        successive_path = outdir / "successive_timestep_kl.csv"
        successive_by_timestep_path = outdir / "successive_timestep_kl_by_timestep.csv"
        successive_summary_path = outdir / "successive_timestep_kl_across_timesteps_summary.csv"
        if not pairwise_path.exists():
            raise FileNotFoundError(f"Missing existing pairwise CSV: {pairwise_path}")
        print(f"Plot-only: reading pairwise CSV {pairwise_path}", flush=True)
        pairwise = pd.read_csv(pairwise_path)
        print(f"Plot-only: pairwise rows={len(pairwise)}", flush=True)
        summary = pd.read_csv(summary_path) if summary_path.exists() else summarize(pairwise)[1]
        print(
            f"Plot-only: summary rows={len(summary)} from "
            f"{summary_path if summary_path.exists() else 'fresh summarize(pairwise)'}",
            flush=True,
        )
        if any(spec["pairwise_col"] in pairwise.columns for spec in LATENT_METRIC_PLOT_SPECS) and any(
            spec["summary_mean"] not in summary.columns for spec in LATENT_METRIC_PLOT_SPECS
        ):
            _, summary = summarize(pairwise)
            summary.to_csv(summary_path, index=False)
        paid_by_timestep = pd.DataFrame()
        paid_summary = pd.DataFrame()
        if paid_timestep_reward_pair_path.exists():
            print(f"Plot-only: reading paid-timestep CSV {paid_timestep_reward_pair_path}", flush=True)
            paid_timestep_reward_pair = pd.read_csv(paid_timestep_reward_pair_path)
            print(f"Plot-only: paid-timestep pairwise rows={len(paid_timestep_reward_pair)}", flush=True)
            by_timestep, paid_seed, paid_summary = summarize_paid_timestep_pairwise(
                paid_timestep_reward_pair
            )
            paid_by_timestep = by_timestep
            by_timestep.to_csv(outdir / "pairwise_kl_by_paid_timestep.csv", index=False)
            paid_seed.to_csv(outdir / "pairwise_kl_across_paid_timesteps_by_seed.csv", index=False)
            paid_summary.to_csv(outdir / "pairwise_kl_across_paid_timesteps_summary.csv", index=False)
        elif paid_summary_path.exists():
            print(f"Plot-only: reading paid summary CSV {paid_summary_path}", flush=True)
            paid_summary = pd.read_csv(paid_summary_path)
        else:
            print(
                "No per-paid-timestep CSV found; old last-paid-only outputs cannot "
                "be used to calculate timestep-wise KL.",
                flush=True,
            )
        successive_by_timestep = pd.DataFrame()
        successive_summary = pd.DataFrame()
        if successive_path.exists():
            print(f"Plot-only: reading successive timestep CSV {successive_path}", flush=True)
            successive = pd.read_csv(successive_path)
            print(f"Plot-only: successive rows={len(successive)}", flush=True)
            successive_by_timestep, successive_seed, successive_summary = summarize_successive_timestep_kl(
                successive
            )
            successive_by_timestep.to_csv(outdir / "successive_timestep_kl_by_timestep.csv", index=False)
            successive_seed.to_csv(outdir / "successive_timestep_kl_across_timesteps_by_seed.csv", index=False)
            successive_summary.to_csv(successive_summary_path, index=False)
        elif successive_by_timestep_path.exists() or successive_summary_path.exists():
            if successive_by_timestep_path.exists():
                successive_by_timestep = pd.read_csv(successive_by_timestep_path)
            if successive_summary_path.exists():
                successive_summary = pd.read_csv(successive_summary_path)
        else:
            print(
                "No successive_timestep_kl.csv found; adjacent-timestep KL requires "
                "a fresh diagnostic run with raw paid latents.",
                flush=True,
            )
        plot_all_metric_variants(
            pairwise,
            summary,
            paid_by_timestep,
            paid_summary,
            outdir,
            successive_by_timestep,
            successive_summary,
            min_samples=int(args.min_samples),
        )
        if not plot_saved_sample_set_trace_analysis_outputs(
            outdir,
            min_samples=int(args.min_samples),
            diff_bin_width=float(args.analysis_diff_bin_width),
            diff_num_bins=int(args.analysis_diff_num_bins),
        ):
            print(
                "No sigma-pair heatmap summaries found. The plots "
                "last_paid_latent_sigma_pairwise_*_heatmap_* require "
                "last_paid_latent_sigma_pairwise_kl_summary.csv, which is only "
                "written by fresh diagnostic runs after this change.",
                flush=True,
            )
        density_path = outdir / "paid_latent_density_rows.csv"
        if density_path.exists():
            print(f"Plot-only: reading paid latent density CSV {density_path}", flush=True)
            density_rows = pd.read_csv(density_path)
            print(f"Plot-only: paid latent density rows={len(density_rows)}", flush=True)
            print("Plot-only: writing latent z_mu trace covariance spread plots.", flush=True)
            write_latent_mu_trace_covariance_outputs(
                density_rows,
                outdir,
                min_samples=int(args.min_samples),
            )
            if bool(args.skip_aggregate_latent_density):
                print(
                    "Plot-only: --skip-aggregate-latent-density is set; "
                    "not writing aggregate_latent_density_by_sigma plots.",
                    flush=True,
                )
            else:
                plot_aggregate_latent_density_outputs(
                    density_rows,
                    outdir,
                    min_samples=int(args.min_samples),
                    max_points=int(args.latent_density_max_points),
                    pga_max_states=int(args.aggregate_pga_max_states),
                    pga_max_iters=int(args.aggregate_pga_max_iters),
                    reuse_pga_fits=bool(args.reuse_aggregate_pga_fits),
                    keep_pooled_plots=bool(args.keep_pooled_aggregate_density_plots),
                    fit_pga=not bool(args.no_pga_fitting),
                    write_normalized_z_plots=should_write_normalized_z_aggregate_plots(args),
                )
        else:
            print(
                "No paid_latent_density_rows.csv found; aggregate normalized-z/PGA "
                "density plots require a fresh diagnostic run after this change.",
                flush=True,
            )
        print(f"Regenerated plots from existing CSVs in {outdir}", flush=True)
        return

    pairwise_parts = []
    paid_pairwise_parts = []
    successive_parts = []
    paid_latent_parts = []
    for family, parameter_name, parameter_value, beta, opportunity in args.parameter_combos:
        for sigma in args.sigmas:
            for seed in args.seeds:
                print(
                    f"Running {family}: beta={beta:g}, opp={opportunity:g}, "
                    f"sigma={sigma:g}, seed={seed}",
                    flush=True,
                )
                pairwise, paid_pairwise, successive, paid_latents = run_one(
                    args,
                    family=family,
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    beta=beta,
                    opportunity=opportunity,
                    sigma=sigma,
                    seed=seed,
                )
                pairwise_parts.append(pairwise)
                if len(paid_pairwise):
                    paid_pairwise_parts.append(paid_pairwise)
                if len(successive):
                    successive_parts.append(successive)
                if len(paid_latents):
                    paid_latent_parts.append(paid_latents)
    pairwise = pd.concat(pairwise_parts, ignore_index=True) if pairwise_parts else pd.DataFrame()
    paid_pairwise = (
        pd.concat(paid_pairwise_parts, ignore_index=True)
        if paid_pairwise_parts
        else pd.DataFrame()
    )
    successive = (
        pd.concat(successive_parts, ignore_index=True)
        if successive_parts
        else pd.DataFrame()
    )
    paid_latents = (
        pd.concat(paid_latent_parts, ignore_index=True)
        if paid_latent_parts
        else pd.DataFrame()
    )
    per_seed, summary = summarize(pairwise) if len(pairwise) else (pd.DataFrame(), pd.DataFrame())
    print("Summarizing paid-timestep and successive-latent diagnostics.", flush=True)
    paid_by_timestep, paid_seed, paid_summary = (
        summarize_paid_timestep_pairwise(paid_pairwise)
        if len(paid_pairwise)
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    successive_by_timestep, successive_seed, successive_summary = (
        summarize_successive_timestep_kl(successive)
        if len(successive)
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    pairwise.to_csv(outdir / "pairwise_kl_by_first_observed_mean_other.csv", index=False)
    per_seed.to_csv(outdir / "pairwise_kl_by_seed.csv", index=False)
    summary.to_csv(outdir / "pairwise_kl_summary.csv", index=False)
    paid_pairwise.to_csv(outdir / "pairwise_kl_by_paid_timestep_first_observed_mean_other.csv", index=False)
    paid_by_timestep.to_csv(outdir / "pairwise_kl_by_paid_timestep.csv", index=False)
    paid_seed.to_csv(outdir / "pairwise_kl_across_paid_timesteps_by_seed.csv", index=False)
    paid_summary.to_csv(outdir / "pairwise_kl_across_paid_timesteps_summary.csv", index=False)
    successive.to_csv(outdir / "successive_timestep_kl.csv", index=False)
    successive_by_timestep.to_csv(outdir / "successive_timestep_kl_by_timestep.csv", index=False)
    successive_seed.to_csv(outdir / "successive_timestep_kl_across_timesteps_by_seed.csv", index=False)
    successive_summary.to_csv(outdir / "successive_timestep_kl_across_timesteps_summary.csv", index=False)
    print("Writing sigma-pair heatmap diagnostics.", flush=True)
    write_sample_set_trace_analysis_outputs(
        paid_latents,
        outdir,
        diff_bin_width=float(args.analysis_diff_bin_width),
        diff_num_bins=int(args.analysis_diff_num_bins),
        sigma_pair_trials_per_sigma=int(args.sigma_pair_trials_per_sigma),
        min_samples=int(args.min_samples),
    )
    density_task = jp.build_task(args.tree_size, args.tree_type, args.input_type)
    density_rows = build_paid_latent_density_frame(paid_latents, path_map=density_task.path_map)
    if not density_rows.empty:
        density_rows.to_csv(outdir / "paid_latent_density_rows.csv", index=False)
        print("Writing latent z_mu trace covariance spread plots.", flush=True)
        write_latent_mu_trace_covariance_outputs(
            density_rows,
            outdir,
            min_samples=int(args.min_samples),
        )
        if bool(args.skip_aggregate_latent_density):
            print(
                "--skip-aggregate-latent-density is set; not writing "
                "aggregate_latent_density_by_sigma plots.",
                flush=True,
            )
        else:
            print(
                "Writing aggregate latent density plots "
                f"(normalized_z={should_write_normalized_z_aggregate_plots(args)}, "
                f"pga={not bool(args.no_pga_fitting)}).",
                flush=True,
            )
            plot_aggregate_latent_density_outputs(
                density_rows,
                outdir,
                min_samples=int(args.min_samples),
                max_points=int(args.latent_density_max_points),
                pga_max_states=int(args.aggregate_pga_max_states),
                pga_max_iters=int(args.aggregate_pga_max_iters),
                reuse_pga_fits=bool(args.reuse_aggregate_pga_fits),
                keep_pooled_plots=bool(args.keep_pooled_aggregate_density_plots),
                fit_pga=not bool(args.no_pga_fitting),
                write_normalized_z_plots=should_write_normalized_z_aggregate_plots(args),
            )
    if len(summary):
        print("Writing standard pairwise diagnostic plots.", flush=True)
        plot_all_metric_variants(
            pairwise,
            summary,
            paid_by_timestep,
            paid_summary,
            outdir,
            successive_by_timestep,
            successive_summary,
            min_samples=int(args.min_samples),
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.5g}"), flush=True)
    print(f"Saved outputs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
