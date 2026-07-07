#!/usr/bin/env python3
"""Simulation-only latent perturbation test for revisit models.

For each checkpoint, this script runs paired baseline and intervention
simulations on the same sampled reward matrix. In the intervention condition,
the latent used for decoding at selected timesteps is replaced by a sample from
the learned prior, i.e. z = prior_mu + prior_sigma * epsilon. A larger reward
drop for later perturbation timesteps supports the idea that late precision is
more behaviorally important.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import jax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_jax import planning as jp  # noqa: E402


PLOT_FONT_SIZE_PT = 7
PANEL_WIDTH_IN = 15 / 25.4
PANEL_HEIGHT_IN = 33 / 25.4
PANEL_GAP_IN = 0.13
LEFT_MARGIN_IN = 0.62
BOTTOM_MARGIN_IN = 0.46
TOP_MARGIN_IN = 0.20
RIGHT_MARGIN_IN = 0.12
LEGEND_WIDTH_IN = 1.05


def parse_list(raw: str, typ=float):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            out.extend(parse_list(item, typ=typ))
        return out
    return [typ(x.strip()) for x in str(raw).replace(",", " ").split() if x.strip()]


def parse_csv_list(raw: str | None, typ=float):
    if raw is None:
        return None
    vals = [typ(x.strip()) for x in str(raw).split(",") if x.strip()]
    return vals if vals else None


def normalize_tree_name(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "2n": "default",
        "2n_default": "default",
        "default2": "default",
        "3n": "bandit3",
        "3n_bandit3": "bandit3",
        "4n": "disjoint2x2",
        "4n_disjoint2x2": "disjoint2x2",
        "6n": "disjoint3x2",
        "6n_disjoint3x2": "disjoint3x2",
    }
    return aliases.get(key, key)


def default_preset_file() -> Path:
    return ROOT / "analyses" / "exp_binary" / "revisit_plot_presets.csv"


def preset_value(row: pd.Series, name: str, default: str | None = None) -> str:
    if name not in row or pd.isna(row[name]):
        if default is None:
            raise KeyError(f"Missing preset column {name!r}")
        return default
    value = str(row[name]).strip()
    if not value and default is not None:
        return default
    return value


def apply_revisit_preset_args(args: argparse.Namespace) -> argparse.Namespace:
    preset_file = Path(args.preset_file)
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_file}")
    preset = pd.read_csv(preset_file)
    preset["tree"] = preset["tree"].map(normalize_tree_name)
    tree_name = normalize_tree_name(args.tree)
    beta_rows = preset[(preset["tree"] == tree_name) & (preset["vary"] == "beta")]
    opp_rows = preset[(preset["tree"] == tree_name) & (preset["vary"] == "opportunity")]
    if beta_rows.empty or opp_rows.empty:
        raise ValueError(f"Need both beta and opportunity rows for tree={tree_name} in {preset_file}")
    beta_row = beta_rows.iloc[0]
    opp_row = opp_rows.iloc[0]
    shared = beta_row
    args.tree_size = int(preset_value(shared, "tree_size"))
    args.tree_type = preset_value(shared, "tree_config", "default") or "default"
    args.input_type = preset_value(shared, "input_type", "uniform")
    args.expansion_decision_version = preset_value(shared, "expansion_decision_version", "lstm")
    args.model_variant = preset_value(shared, "model_variant", "vae")
    args.lambda_values = parse_csv_list(preset_value(shared, "lambda_arg", "100.0"), float)
    args.alphas = parse_csv_list(preset_value(shared, "alpha_arg", "0.0"), float)
    args.seeds = (
        parse_list(args.seeds, int)
        if args.seeds is not None
        else parse_csv_list(preset_value(shared, "seed_arg"), int)
    )
    args.sigmas = parse_csv_list(args.sigmas, float) or parse_csv_list(preset_value(shared, "sigma_arg", "0"), float)
    args.rnn_dims = parse_csv_list(args.rnn_dims, int) or parse_csv_list(preset_value(shared, "rnn_units_arg"), int)
    args.latent_dims = parse_csv_list(args.latent_dims, int) or parse_csv_list(preset_value(shared, "latent_dim_arg"), int)
    args.max_observations_before_stop = int(
        args.max_observations_before_stop
        if args.max_observations_before_stop is not None
        else preset_value(shared, "max_observations_arg")
    )
    args.allow_node_revisit = True
    beta_values = parse_csv_list(args.betas, float) or parse_csv_list(preset_value(beta_row, "beta_arg"), float)
    opportunity_values = (
        parse_csv_list(args.opportunity_costs, float)
        or parse_csv_list(preset_value(opp_row, "opportunity_arg"), float)
    )
    beta_family_opps = parse_csv_list(preset_value(beta_row, "opportunity_arg", "0.0"), float)
    opportunity_family_betas = parse_csv_list(preset_value(opp_row, "beta_arg", "1000.0"), float)
    combos = []
    for beta in beta_values or []:
        for opp in beta_family_opps or []:
            combos.append(("vary_beta", "beta", float(beta), float(beta), float(opp)))
    for beta in opportunity_family_betas or []:
        for opp in opportunity_values or []:
            combos.append(("vary_opportunity", "opportunity", float(opp), float(beta), float(opp)))
    args.parameter_combos = combos
    if args.outdir is None:
        output_root = Path(args.output_root or preset_value(shared, "results_dir", "results"))
        args.outdir = str(output_root / "revisit_hypothesis1_perturbation" / f"{tree_name}_beta_vs_opportunity")
    print(f"Using revisit plot preset: tree={tree_name} from {preset_file}", flush=True)
    return args


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
        }
    )


def figure_size(n_cols: int = 1, n_rows: int = 1, legend: bool = False) -> tuple[float, float]:
    width = (
        LEFT_MARGIN_IN
        + RIGHT_MARGIN_IN
        + n_cols * PANEL_WIDTH_IN
        + max(0, n_cols - 1) * PANEL_GAP_IN
        + (LEGEND_WIDTH_IN if legend else 0.0)
    )
    height = (
        BOTTOM_MARGIN_IN
        + TOP_MARGIN_IN
        + n_rows * PANEL_HEIGHT_IN
        + max(0, n_rows - 1) * PANEL_GAP_IN
    )
    return max(width, 2.2), max(height, 2.0)


def color_ramp(hex_colors: list[str], n: int) -> list[tuple[float, float, float, float]]:
    """Match the R revisit comparison color ramps for beta/opp series."""
    if n <= 0:
        return []
    if n == 1:
        return [mcolors.to_rgba(hex_colors[0])]
    cmap = mcolors.LinearSegmentedColormap.from_list("revisit_parameter_ramp", hex_colors, N=n)
    return [cmap(i / (n - 1)) for i in range(n)]


def make_config(
    args: argparse.Namespace,
    seed: int,
    beta: float,
    lambda_: float,
    alpha: float,
    opportunity: float,
    sigma: float,
    rnn_dim: int,
    latent_dim: int,
) -> jp.RunConfig:
    num_steps = int(args.num_steps or ((args.max_observations_before_stop + 1) if args.allow_node_revisit else args.tree_size))
    return jp.RunConfig(
        lambda_=float(lambda_),
        alpha=float(alpha),
        beta=float(beta),
        model_dir=str(args.checkpoint_root),
        epochs=120,
        input_type=str(args.input_type),
        seed=int(seed),
        tree_size=int(args.tree_size),
        train_mode="sim",
        tree_type=str(args.tree_type),
        opportunity_cost=float(opportunity),
        expansion_decision_version=jp.normalize_expansion_decision_version(args.expansion_decision_version),
        model_variant=jp.normalize_model_variant(args.model_variant),
        rnn_units=int(rnn_dim),
        latent_dim=int(latent_dim),
        sim_dir=str(args.outdir),
        n_sim_trials=int(args.n_trials),
        num_envs=int(args.n_trials),
        num_steps=num_steps,
        update_epochs=int(args.update_epochs),
        ppo_minibatches=1,
        steps_per_epoch=max(int(args.n_trials) * num_steps, 1),
        return_target_rollouts=int(args.return_target_rollouts),
        return_target_mode=jp.normalize_return_target_mode(args.return_target_mode),
        sampled_lambda_critic=str(args.sampled_lambda_critic),
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=int(args.target_critic_update_interval),
        target_critic_tau=float(args.target_critic_tau),
        backend=args.backend,
        jit_training=True,
        profile_update_components=False,
        profile_update_components_every=1,
        enable_reconstruction=False,
        enable_probe=False,
        allow_node_revisit=bool(args.allow_node_revisit),
        max_observations_before_stop=int(args.max_observations_before_stop),
        observation_sigma=max(float(sigma), 0.0),
        kl_start_multiplier=max(float(args.kl_start_multiplier), 0.0),
        kl_annealing_epochs=max(int(args.kl_annealing_epochs), 0),
        node_coverage_aux_coef=0.0,
        node_coverage_aux_epochs=0,
    )


def clone_with_perturb(
    model: jp.PlanningVAE,
    mode: str,
    timestep: int,
    scale: float,
) -> jp.PlanningVAE:
    return jp.PlanningVAE(
        rnn_units=model.rnn_units,
        latent_dim=model.latent_dim,
        time_steps=model.time_steps,
        num_paths=model.num_paths,
        path_map=model.path_map,
        reward_values=model.reward_values,
        reward_norm_value=model.reward_norm_value,
        expansion_decision_version=model.expansion_decision_version,
        use_autoencoder=model.use_autoencoder,
        enable_reconstruction=model.enable_reconstruction,
        enable_probe=model.enable_probe,
        allow_node_revisit=model.allow_node_revisit,
        max_observations_before_stop=model.max_observations_before_stop,
        opportunity_cost=model.opportunity_cost,
        observation_sigma=model.observation_sigma,
        lambda_=model.lambda_,
        alpha=model.alpha,
        beta=model.beta,
        reward_feature_dim_override=model.reward_feature_dim_override,
        include_visited_lstm_input=model.include_visited_lstm_input,
        latent_perturb_mode=mode,
        latent_perturb_timestep=int(timestep),
        latent_perturb_scale=float(scale),
    )


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.where(np.isfinite(probs), probs, 0.0)
    sums = probs.sum(axis=1)
    valid = sums > 0
    out = np.full(probs.shape[0], np.nan, dtype=float)
    if np.any(valid):
        p = probs[valid] / sums[valid, None]
        out[valid] = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=1)
    return out


def sample_terminal(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(probabilities.shape[0], dtype=int)
    for i, row in enumerate(probabilities):
        probs = np.where(np.isfinite(row), row, 0.0)
        total = float(probs.sum())
        if total <= 0:
            probs = np.ones_like(probs) / max(len(probs), 1)
        else:
            probs = probs / total
        out[i] = int(rng.choice(len(probs), p=probs))
    return out


def run_condition(
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    reset_rewards,
    condition: str,
    rng_seed_offset: int,
) -> pd.DataFrame:
    rng = jax.random.PRNGKey(config.seed + 100_000 + rng_seed_offset)
    np_rng = np.random.default_rng(config.seed + 200_000 + rng_seed_offset)
    reward_feature_dim = jp.reward_feature_dim_for_sigma(config.observation_sigma)
    checkpoint_dim = int(model.reward_feature_dim_override)
    if checkpoint_dim > 0:
        reward_feature_dim = checkpoint_dim
    carry = jp.initial_carry(
        config.n_sim_trials,
        task,
        config.rnn_units,
        reward_feature_dim,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    carry = jp.reset_done_envs(carry, reset_rewards)
    sched = jp.ScheduleValues(1.0, 1.0 / config.beta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3)
    transitions = []
    for _ in range(config.num_steps):
        rng, step_rng = jax.random.split(rng)
        carry, trans = model.apply(
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
        transitions.append(jax.device_get(trans))

    rewards = np.asarray(reset_rewards)
    path_map = np.asarray(task.path_map, dtype=float)
    path_rewards = rewards @ path_map.T
    num_trials = rewards.shape[0]
    num_paths = path_rewards.shape[1]
    chosen_path = np.full(num_trials, -1, dtype=int)
    stop_timestep = np.full(num_trials, config.num_steps, dtype=int)
    terminal_probs_last = np.ones((num_trials, num_paths), dtype=float) / num_paths
    terminal_entropy_at_decision = np.full(num_trials, np.nan, dtype=float)
    total_kl = np.zeros(num_trials, dtype=float)
    kl_by_timestep = np.zeros((num_trials, config.num_steps), dtype=float)
    observed_by_timestep = np.zeros((num_trials, config.num_steps), dtype=bool)
    stop_found = np.zeros(num_trials, dtype=bool)

    for t, trans in enumerate(transitions, start=1):
        is_stop = np.asarray(trans.is_stop, dtype=float) > 0.5
        node_index = np.asarray(trans.node_index, dtype=int)
        terminal_path_index = np.asarray(trans.terminal_path_index, dtype=int)
        terminal_probs = np.asarray(trans.action_output, dtype=float)
        paid_kl = np.asarray(trans.paid_kl, dtype=float)
        terminal_probs_last = terminal_probs
        total_kl += np.where(np.isfinite(paid_kl), paid_kl, 0.0)
        kl_by_timestep[:, t - 1] = np.where(np.isfinite(paid_kl), paid_kl, 0.0)
        observed_by_timestep[:, t - 1] = node_index >= 0
        newly_stopped = is_stop & (~stop_found)
        if np.any(newly_stopped):
            chosen_path[newly_stopped] = terminal_path_index[newly_stopped]
            stop_timestep[newly_stopped] = t
            terminal_entropy_at_decision[newly_stopped] = entropy(terminal_probs[newly_stopped])
        stop_found |= is_stop

    missing = chosen_path < 0
    if np.any(missing):
        sampled = sample_terminal(terminal_probs_last[missing], np_rng)
        chosen_path[missing] = sampled
        terminal_entropy_at_decision[missing] = entropy(terminal_probs_last[missing])

    chosen_reward = path_rewards[np.arange(num_trials), chosen_path]
    best_reward = np.max(path_rewards, axis=1)
    sorted_rewards = np.sort(path_rewards, axis=1)
    reward_gap = sorted_rewards[:, -1] - sorted_rewards[:, -2] if num_paths > 1 else np.zeros(num_trials)
    choice_accuracy = np.where(
        reward_gap > 1e-8,
        np.isclose(chosen_reward, best_reward).astype(float),
        np.nan,
    )
    final_pre_stop_kl = np.full(num_trials, np.nan, dtype=float)
    last_positive_pre_stop_kl = np.full(num_trials, np.nan, dtype=float)
    for i, stop_t in enumerate(stop_timestep):
        pre_idx = np.arange(max(int(stop_t) - 1, 0))
        if pre_idx.size:
            final_pre_stop_kl[i] = kl_by_timestep[i, pre_idx[-1]]
            positives = kl_by_timestep[i, pre_idx][kl_by_timestep[i, pre_idx] > 0]
            if positives.size:
                last_positive_pre_stop_kl[i] = positives[-1]

    rows = pd.DataFrame({
        "condition": condition,
        "graph": np.arange(num_trials),
        "chosen_path": chosen_path,
        "chosen_reward": chosen_reward,
        "normalized_chosen_reward": chosen_reward / float(task.reward_norm),
        "best_reward": best_reward,
        "reward_gap": reward_gap,
        "choice_accuracy": choice_accuracy,
        "terminal_choice_entropy": terminal_entropy_at_decision,
        "stop_timestep": stop_timestep,
        "total_kl": total_kl,
        "final_pre_stop_kl": final_pre_stop_kl,
        "last_positive_pre_stop_kl": last_positive_pre_stop_kl,
    })
    for t in range(1, config.num_steps + 1):
        rows[f"kl_d_t{t}"] = kl_by_timestep[:, t - 1]
        rows[f"observed_t{t}"] = observed_by_timestep[:, t - 1]
    return rows


def summarize_paired(all_rows: pd.DataFrame) -> pd.DataFrame:
    baseline = all_rows[all_rows["condition"] == "baseline"].set_index("graph")
    rows = []
    for condition, group in all_rows[all_rows["condition"] != "baseline"].groupby("condition"):
        joined = group.set_index("graph").join(
            baseline[
                [
                    "chosen_reward",
                    "normalized_chosen_reward",
                    "choice_accuracy",
                    "terminal_choice_entropy",
                    "stop_timestep",
                    "total_kl",
                ]
            ],
            rsuffix="_baseline",
        )
        perturb_timestep = int(condition.replace("prior_noise_t", ""))
        joined["baseline_stop_timestep"] = pd.to_numeric(joined["stop_timestep_baseline"], errors="coerce")
        joined["perturbed_stop_timestep"] = pd.to_numeric(joined["stop_timestep"], errors="coerce")
        joined["perturb_reached_in_baseline"] = joined["baseline_stop_timestep"] > perturb_timestep
        joined["chosen_reward_drop"] = joined["chosen_reward_baseline"] - joined["chosen_reward"]
        joined["normalized_reward_drop"] = (
            joined["normalized_chosen_reward_baseline"] - joined["normalized_chosen_reward"]
        )
        joined["choice_accuracy_drop"] = joined["choice_accuracy_baseline"] - joined["choice_accuracy"]
        joined["terminal_entropy_change"] = (
            joined["terminal_choice_entropy"] - joined["terminal_choice_entropy_baseline"]
        )
        joined["stop_timestep_change"] = joined["stop_timestep"] - joined["stop_timestep_baseline"]
        joined["total_kl_change"] = joined["total_kl"] - joined["total_kl_baseline"]
        for baseline_stop_timestep, piece in joined.groupby("baseline_stop_timestep", dropna=True):
            rows.append({
                "condition": condition,
                "perturb_timestep": perturb_timestep,
                "baseline_stop_timestep": int(baseline_stop_timestep),
                "n": int(len(piece)),
                "perturb_reached_in_baseline": float(piece["perturb_reached_in_baseline"].mean()),
                "chosen_reward_drop": float(piece["chosen_reward_drop"].mean()),
                "normalized_reward_drop": float(piece["normalized_reward_drop"].mean()),
                "choice_accuracy_drop": float(piece["choice_accuracy_drop"].mean()),
                "terminal_entropy_change": float(piece["terminal_entropy_change"].mean()),
                "stop_timestep_change": float(piece["stop_timestep_change"].mean()),
                "total_kl_change": float(piece["total_kl_change"].mean()),
            })
    return pd.DataFrame(rows).sort_values(["baseline_stop_timestep", "perturb_timestep"])


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def plot_summary(summary: pd.DataFrame, outdir: Path, label: str) -> None:
    if summary.empty:
        return
    setup_plot_style()
    fig_w, fig_h = figure_size(legend=True)
    stop_values = sorted(summary["baseline_stop_timestep"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - (RIGHT_MARGIN_IN + LEGEND_WIDTH_IN) / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
    )
    handles = []
    labels = []
    min_stop = min(stop_values) if stop_values else 0
    max_stop = max(stop_values) if stop_values else 1
    for stop_t in stop_values:
        piece = summary[summary["baseline_stop_timestep"] == stop_t].sort_values("perturb_timestep")
        color = cmap(0.5 if min_stop == max_stop else (stop_t - min_stop) / (max_stop - min_stop))
        handle = ax.plot(
            piece["perturb_timestep"],
            piece["normalized_reward_drop"],
            marker="o",
            linewidth=1.0,
            markersize=2.5,
            color=color,
        )[0]
        handles.append(handle)
        labels.append(f"stop {int(stop_t)}")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("perturbed\ntimestep")
    ax.set_ylabel("Norm\nreward drop")
    ax.set_title(label, pad=1.5)
    ax.tick_params(pad=1.5)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.04, 0.5),
            frameon=False,
            handlelength=1.1,
        )
    fig.savefig(outdir / "latent_perturbation_normalized_reward_drop.png", dpi=220)
    plt.close(fig)


def summarize_reward_drop(all_summary: pd.DataFrame) -> pd.DataFrame:
    if all_summary.empty:
        return pd.DataFrame()
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "parameter_label",
        "sigma",
        "baseline_stop_timestep",
        "perturb_timestep",
    ]
    return (
        all_summary
        .groupby(group_cols, as_index=False)
        .agg(
            normalized_reward_drop=("normalized_reward_drop", "mean"),
            normalized_reward_drop_sem=("normalized_reward_drop", sem),
            chosen_reward_drop=("chosen_reward_drop", "mean"),
            chosen_reward_drop_sem=("chosen_reward_drop", sem),
            n_runs=("seed", "count"),
        )
    )


def add_paid_memory_perturbation_flag(summary: pd.DataFrame) -> pd.DataFrame:
    """Flag perturbations whose latent would be paid before a later observation.

    The stop timestep includes the stop action. If baseline total t = 4, the
    trial observed at t=1,2,3 and stopped at t=4. The t=3 latent is the final
    free latent before stopping, so only perturb_timestep < total_t - 1 is kept
    for the paid-memory perturbation plot.
    """
    if summary.empty:
        return summary.copy()
    out = summary.copy()
    stop_t = pd.to_numeric(out["baseline_stop_timestep"], errors="coerce")
    perturb_t = pd.to_numeric(out["perturb_timestep"], errors="coerce")
    out["is_paid_memory_perturbation"] = perturb_t < (stop_t - 1)
    return out


def plot_grouped_reward_drop(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        return
    setup_plot_style()
    outdir.mkdir(parents=True, exist_ok=True)
    sigmas = sorted(summary["sigma"].dropna().unique())
    stop_values = sorted(summary["baseline_stop_timestep"].dropna().unique())
    if not sigmas or not stop_values:
        return
    n_cols = len(sigmas)
    n_rows = len(stop_values)
    fig_w, fig_h = figure_size(n_cols=n_cols, n_rows=n_rows, legend=True)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [max(0.8, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - RIGHT_MARGIN_IN / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
        wspace=0.50,
        hspace=0.38,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axes[row_i, col_i] = fig.add_subplot(
                gs[row_i, col_i],
                sharex=axes[0, 0] if row_i or col_i else None,
                sharey=axes[0, 0] if row_i or col_i else None,
            )
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")

    styles = {}
    family_specs = {
        # Same VAE parameter color semantics as plot_revisit_policy_diagnostics.R:
        # lower beta is darker green; higher opportunity cost is darker blue.
        "vary_beta": (["#00441b", "#238b45", "#74c476"], "o"),
        "vary_opportunity": (["#6baed6", "#2171b5", "#08306b"], "^"),
        "manual": (["#252525", "#737373", "#bdbdbd"], "s"),
    }
    for family, fdata in summary.groupby("family", dropna=False):
        ramp_colors, marker = family_specs.get(str(family), (["#252525", "#737373", "#bdbdbd"], "s"))
        vals = sorted(pd.to_numeric(fdata["parameter_value"], errors="coerce").dropna().unique())
        cols = color_ramp(ramp_colors, len(vals))
        for value, color in zip(vals, cols):
            styles[(str(family), float(value))] = (color, marker)

    legend_handles = []
    legend_labels = []
    plotted_labels = set()
    for row_i, stop_t in enumerate(stop_values):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = summary[
                np.isclose(summary["baseline_stop_timestep"], stop_t)
                & np.isclose(summary["sigma"], sigma)
            ]
            for keys, line_data in panel.groupby(
                ["family", "parameter_value", "parameter_label"],
                dropna=False,
            ):
                family, parameter_value, parameter_label = keys
                line_data = line_data.sort_values("perturb_timestep")
                color, marker = styles.get((str(family), float(parameter_value)), ("black", "o"))
                err = line_data["normalized_reward_drop_sem"].to_numpy(dtype=float)
                handle = ax.errorbar(
                    line_data["perturb_timestep"],
                    line_data["normalized_reward_drop"],
                    yerr=err,
                    marker=marker,
                    linestyle="-",
                    linewidth=0.85,
                    markersize=2.4,
                    color=color,
                    markeredgecolor="black" if marker == "^" else color,
                    markeredgewidth=0.35 if marker == "^" else 0.0,
                )
                label = str(parameter_label)
                if label not in plotted_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
                    plotted_labels.add(label)
            ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.45)
            title_bits = []
            if row_i == 0:
                title_bits.append(f"sigma {sigma:g}")
            title_bits.append(f"total t {int(stop_t)}")
            ax.set_title("\n".join(title_bits), pad=1.5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(pad=1.5)
            if row_i < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if col_i > 0:
                ax.tick_params(labelleft=False)
    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            frameon=False,
            handlelength=1.1,
        )
    fig.supxlabel("perturbed\ntimestep", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel("Norm\nreward drop", fontsize=PLOT_FONT_SIZE_PT, x=0.015)
    plot_path = outdir / "reward_drop_vs_perturbed_timestep_by_total_timestep_sigma_panels.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(f"Saved {plot_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired latent perturbation test for revisit models. With a positional "
            "<tree>, shared parameters are read from analyses/exp_binary/"
            "revisit_plot_presets.csv, matching plot_revisit_beta_opp_comparison.R."
        )
    )
    parser.add_argument("tree", nargs="?", default=None)
    parser.add_argument("--preset-file", default=str(default_preset_file()))
    parser.add_argument("--output-root", "--results-dir", default=None)
    parser.add_argument("--alphas", nargs="+", default=["0.0"])
    parser.add_argument(
        "--betas",
        "--beta-values",
        "--vary-betas",
        "--vary-beta-values",
        dest="betas",
        default=None,
    )
    parser.add_argument("--lambdas", dest="lambda_values", nargs="+", default=["100.0"])
    parser.add_argument(
        "--opportunity-costs",
        "--opportunity-values",
        "--opportunities",
        "--vary-opps",
        "--vary-opportunities",
        "--vary-opportunity-values",
        dest="opportunity_costs",
        default=None,
    )
    parser.add_argument("--sigmas", "--observation-sigmas", dest="sigmas", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--rnn-dims", default=None)
    parser.add_argument("--latent-dims", default=None)
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform")
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae")
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--allow-node-revisit", action="store_true")
    parser.add_argument("--max-observations-before-stop", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--perturb-timesteps", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--return-target-mode", default="sampled_lambda")
    parser.add_argument("--sampled-lambda-critic", choices=["q", "value"], default="q")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--return-target-rollouts", type=int, default=8)
    parser.add_argument("--target-critic-update-interval", type=int, default=100)
    parser.add_argument("--target-critic-tau", type=float, default=1.0)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument(
        "--save-per-run-plots",
        action="store_true",
        help=(
            "Also write the old one-checkpoint-at-a-time perturbation PNGs under "
            "by_run/. By default only the combined beta/opp figure is plotted."
        ),
    )
    args = parser.parse_args()
    args.parameter_combos = None
    if args.tree is not None:
        return apply_revisit_preset_args(args)

    args.alphas = parse_list(args.alphas, float)
    args.betas = parse_list(args.betas or "", float)
    args.lambda_values = parse_list(args.lambda_values, float)
    args.opportunity_costs = parse_list(args.opportunity_costs or "0.0", float)
    args.sigmas = parse_list(args.sigmas or "0.0", float)
    args.seeds = parse_list(args.seeds, int)
    args.rnn_dims = parse_list(args.rnn_dims or "16", int)
    args.latent_dims = parse_list(args.latent_dims or "2", int)
    args.max_observations_before_stop = int(args.max_observations_before_stop or 10)
    args.outdir = args.outdir or "analysis_outputs/revisit_latent_perturbation_jax"
    if not args.betas:
        raise ValueError("Explicit mode requires --betas/--vary-beta-values.")
    if not args.seeds:
        raise ValueError("Explicit mode requires --seeds.")
    args.parameter_combos = [
        ("manual", "beta", float(beta), float(beta), float(opp))
        for beta in args.betas
        for opp in args.opportunity_costs
    ]
    return args


def main() -> None:
    args = parse_args()
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    summary_parts = []
    for seed in args.seeds:
        for family, parameter_name, parameter_value, beta, opp in args.parameter_combos:
            for lambda_ in args.lambda_values:
                for alpha in args.alphas:
                    for sigma in args.sigmas:
                        for rnn_dim in args.rnn_dims:
                            for latent_dim in args.latent_dims:
                                config = make_config(args, seed, beta, lambda_, alpha, opp, sigma, rnn_dim, latent_dim)
                                task = jp.build_task(config.tree_size, config.tree_type, config.input_type)
                                model, params = jp.load_state_for_sim(config, task)
                                reward_rng = jax.random.PRNGKey(seed + 700_000)
                                reset_rewards = jp.sample_reward_matrix(
                                    reward_rng,
                                    config.n_sim_trials,
                                    task.num_nodes,
                                    task.reward_values,
                                )
                                parameter_label = (
                                    f"beta = {parameter_value:g}"
                                    if parameter_name == "beta"
                                    else f"opp = {parameter_value:g}"
                                )
                                combo_label = (
                                    f"seed_{seed}_{family}_{parameter_name}_{parameter_value:g}_"
                                    f"beta_{beta:g}_opp_{opp:g}_lambda_{lambda_:g}_sigma_{sigma:g}_"
                                    f"rnn_{rnn_dim}_latent_{latent_dim}_{task.tree_type}"
                                )
                                combo_dir = outroot / "by_run" / combo_label.replace(".", "p")
                                combo_dir.mkdir(parents=True, exist_ok=True)
                                all_rows = [
                                    run_condition(
                                        model,
                                        params,
                                        config,
                                        task,
                                        reset_rewards,
                                        "baseline",
                                        rng_seed_offset=0,
                                    )
                                ]
                                for t in args.perturb_timesteps:
                                    pmodel = clone_with_perturb(
                                        model,
                                        mode="prior_noise",
                                        timestep=int(t),
                                        scale=float(args.perturb_scale),
                                    )
                                    all_rows.append(
                                        run_condition(
                                            pmodel,
                                            params,
                                            config,
                                            task,
                                            reset_rewards,
                                            f"prior_noise_t{int(t)}",
                                            rng_seed_offset=0,
                                        )
                                    )
                                combo_rows = pd.concat(all_rows, ignore_index=True)
                                combo_rows.insert(0, "seed", seed)
                                combo_rows.insert(1, "family", family)
                                combo_rows.insert(2, "parameter_name", parameter_name)
                                combo_rows.insert(3, "parameter_value", parameter_value)
                                combo_rows.insert(4, "parameter_label", parameter_label)
                                combo_rows.insert(5, "beta", beta)
                                combo_rows.insert(6, "lambda", lambda_)
                                combo_rows.insert(7, "alpha", alpha)
                                combo_rows.insert(8, "opportunity_cost", opp)
                                combo_rows.insert(9, "sigma", sigma)
                                combo_rows.to_csv(combo_dir / "latent_perturbation_trial_metrics.csv", index=False)
                                summary = summarize_paired(combo_rows)
                                for col, val in [
                                    ("seed", seed),
                                    ("family", family),
                                    ("parameter_name", parameter_name),
                                    ("parameter_value", parameter_value),
                                    ("parameter_label", parameter_label),
                                    ("beta", beta),
                                    ("lambda", lambda_),
                                    ("alpha", alpha),
                                    ("opportunity_cost", opp),
                                    ("sigma", sigma),
                                    ("rnn_dim", rnn_dim),
                                    ("latent_dim", latent_dim),
                                    ("tree_type", task.tree_type),
                                ]:
                                    summary[col] = val
                                summary.to_csv(combo_dir / "latent_perturbation_summary.csv", index=False)
                                if args.save_per_run_plots:
                                    plot_summary(summary, combo_dir, combo_label)
                                summary_parts.append(summary)
                                print(f"Saved perturbation results to {combo_dir}", flush=True)
    if summary_parts:
        all_summary = pd.concat(summary_parts, ignore_index=True)
        all_summary.to_csv(outroot / "latent_perturbation_summary_all.csv", index=False)
        grouped = (
            all_summary
            .groupby([
                "family",
                "parameter_name",
                "parameter_value",
                "parameter_label",
                "beta",
                "opportunity_cost",
                "sigma",
                "baseline_stop_timestep",
                "perturb_timestep",
            ], as_index=False)
            .agg(
                normalized_reward_drop=("normalized_reward_drop", "mean"),
                choice_accuracy_drop=("choice_accuracy_drop", "mean"),
                terminal_entropy_change=("terminal_entropy_change", "mean"),
                stop_timestep_change=("stop_timestep_change", "mean"),
                n_runs=("seed", "count"),
            )
        )
        grouped.to_csv(outroot / "latent_perturbation_summary_grouped.csv", index=False)
        reward_drop_summary = add_paid_memory_perturbation_flag(summarize_reward_drop(all_summary))
        reward_drop_summary.to_csv(outroot / "reward_drop_summary_all_perturbations.csv", index=False)
        reward_drop_summary_plotted = reward_drop_summary[
            reward_drop_summary["is_paid_memory_perturbation"].astype(bool)
        ].copy()
        reward_drop_summary_plotted.to_csv(outroot / "reward_drop_summary.csv", index=False)
        plot_grouped_reward_drop(reward_drop_summary_plotted, outroot)


if __name__ == "__main__":
    main()
