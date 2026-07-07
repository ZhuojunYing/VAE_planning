#!/usr/bin/env python3
"""Hypothesis-1 diagnostics for revisit simulations.

This script uses saved JAX revisit simulation CSVs. It does not load model
checkpoints or re-run simulations.

Main outputs:
  * trial_metrics.csv
  * timestep_kl_outcomes.csv
  * kl_at_timestep_vs_{accuracy,entropy,reward}_conditioned.png
  * final_vs_total_kl_{accuracy,entropy,reward}.png
  * final_vs_total_kl_correlations.csv
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_jax import planning as jp  # noqa: E402


PLOT_FONT_SIZE_PT = 7
PANEL_WIDTH_IN = 15 / 25.4
PANEL_HEIGHT_IN = 33 / 25.4
CONDITIONED_PANEL_HEIGHT_IN = 18 / 25.4
PANEL_GAP_IN = 0.10
LEFT_MARGIN_IN = 0.62
BOTTOM_MARGIN_IN = 0.48
TOP_MARGIN_IN = 0.28
RIGHT_MARGIN_IN = 0.08
LEGEND_WIDTH_IN = 1.05


FILENAME_RE = re.compile(
    r"^lambda_(?P<lambda>[^_]+)_alpha_(?P<alpha>[^_]+)_beta_(?P<beta>[^_]+)_"
    r"opportunity_(?P<opportunity>.+?)_expansion_(?P<expansion>[^_]+)_"
    r"variant_(?P<variant>[^_]+)_seed_(?P<seed>\d+)_"
    r"(?P<tree>\d+n(?:_[^_]+)?)_rnn_(?P<rnn>\d+)_latent_(?P<latent>\d+)"
    r"(?P<extras>.*)_(?P<input>uniform|binary)\.csv$"
)


@dataclass(frozen=True)
class SimFile:
    path: Path
    lambda_: float
    alpha: float
    beta: float
    opportunity: float
    seed: int
    tree_label: str
    rnn: int
    latent: int
    sigma: float
    maxobs: int | None
    critic: str
    input_type: str
    family: str = "simulation"


def parse_float_list(raw: str | None, default: list[float] | None = None) -> list[float] | None:
    if raw is None:
        return default
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    return vals if vals else default


def parse_int_list(raw: str | None, default: list[int] | None = None) -> list[int] | None:
    if raw is None:
        return default
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    return vals if vals else default


def unique_floats(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        if not float_in(value, out):
            out.append(float(value))
    return out


def float_in(value: float, allowed: list[float] | None, tol: float = 1e-8) -> bool:
    if allowed is None:
        return True
    return any(abs(float(value) - float(x)) <= tol for x in allowed)


def int_in(value: int, allowed: list[int] | None) -> bool:
    return allowed is None or int(value) in set(int(x) for x in allowed)


def pair_key(beta: float, opportunity: float) -> tuple[float, float]:
    return (round(float(beta), 10), round(float(opportunity), 10))


def parse_file(path: Path) -> SimFile | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    extras = m.group("extras") or ""
    sigma = 0.0
    sigma_m = re.search(r"_obs_sigma_([^_]+)", extras)
    if sigma_m:
        sigma = float(sigma_m.group(1))
    maxobs = None
    maxobs_m = re.search(r"_revisit_maxobs_(\d+)", extras)
    if maxobs_m:
        maxobs = int(maxobs_m.group(1))
    critic = "value" if "_vcritic" in extras else "q"
    return SimFile(
        path=path,
        lambda_=float(m.group("lambda")),
        alpha=float(m.group("alpha")),
        beta=float(m.group("beta")),
        opportunity=float(m.group("opportunity")),
        seed=int(m.group("seed")),
        tree_label=m.group("tree"),
        rnn=int(m.group("rnn")),
        latent=int(m.group("latent")),
        sigma=sigma,
        maxobs=maxobs,
        critic=critic,
        input_type=m.group("input"),
    )


def tree_label_for(tree_size: int, tree_type: str) -> str:
    task = jp.build_task(tree_size, tree_type, "uniform")
    return f"{tree_size}n{task.tree_name_suffix}"


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
    return Path(__file__).resolve().with_name("revisit_plot_presets.csv")


def preset_value(row: pd.Series, name: str, default: str | None = None) -> str:
    if name not in row or pd.isna(row[name]):
        if default is None:
            raise KeyError(f"Missing required preset column {name!r}")
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
    args.lambdas = parse_float_list(preset_value(shared, "lambda_arg", "100.0"))
    args.alphas = parse_float_list(preset_value(shared, "alpha_arg", "0.0"))
    args.seeds = parse_int_list(args.seeds) if args.seeds is not None else parse_int_list(preset_value(shared, "seed_arg"))
    args.sigmas = parse_float_list(args.sigmas) if args.sigmas is not None else parse_float_list(preset_value(shared, "sigma_arg", "0"))
    args.rnn_dims = parse_int_list(args.rnn_dims) if args.rnn_dims is not None else parse_int_list(preset_value(shared, "rnn_units_arg"))
    args.latent_dims = (
        parse_int_list(args.latent_dims)
        if args.latent_dims is not None
        else parse_int_list(preset_value(shared, "latent_dim_arg"))
    )
    args.max_observations_before_stop = (
        int(args.max_observations_before_stop)
        if args.max_observations_before_stop is not None
        else int(preset_value(shared, "max_observations_arg"))
    )

    beta_values = parse_float_list(args.betas, default=None) or parse_float_list(preset_value(beta_row, "beta_arg"))
    opportunity_values = (
        parse_float_list(args.opportunity_costs, default=None)
        or parse_float_list(preset_value(opp_row, "opportunity_arg"))
    )
    beta_family_opportunities = parse_float_list(preset_value(beta_row, "opportunity_arg", "0.0"))
    opportunity_family_betas = parse_float_list(preset_value(opp_row, "beta_arg", "1000.0"))
    requested_pairs: dict[tuple[float, float], str] = {}
    for beta in beta_values or []:
        for opp in beta_family_opportunities or []:
            requested_pairs[pair_key(beta, opp)] = "vary_beta"
    for beta in opportunity_family_betas or []:
        for opp in opportunity_values or []:
            requested_pairs[pair_key(beta, opp)] = "vary_opportunity"
    args.requested_pair_families = requested_pairs
    args.betas = unique_floats([key[0] for key in requested_pairs])
    args.opportunity_costs = unique_floats([key[1] for key in requested_pairs])

    source = preset_value(shared, "simulation_source_arg", "jax").lower()
    preset_input_dir = preset_value(shared, "input_dir", "outputs/simulations")
    if args.input_dir is None:
        args.input_dir = (
            str(Path(preset_input_dir).parent / "jax_simulations")
            if source == "jax" and Path(preset_input_dir).name == "simulations"
            else preset_input_dir
        )
    if args.outdir is None:
        output_root = Path(args.output_root or preset_value(shared, "results_dir", "results"))
        args.outdir = str(output_root / "revisit_hypothesis1" / f"{tree_name}_beta_vs_opportunity")
    print(
        f"Using revisit plot preset: tree={tree_name} from {preset_file}",
        flush=True,
    )
    return args


def find_files(args: argparse.Namespace) -> list[SimFile]:
    target_tree = tree_label_for(args.tree_size, args.tree_type)
    files = []
    for path in sorted(Path(args.input_dir).glob("*.csv")):
        info = parse_file(path)
        if info is None:
            continue
        if info.tree_label != target_tree:
            continue
        if info.input_type != args.input_type:
            continue
        if args.requested_pair_families is not None:
            family = args.requested_pair_families.get(pair_key(info.beta, info.opportunity))
            if family is None:
                continue
            info = replace(info, family=family)
        if not float_in(info.lambda_, args.lambdas):
            continue
        if not float_in(info.alpha, args.alphas):
            continue
        if not float_in(info.beta, args.betas):
            continue
        if not float_in(info.opportunity, args.opportunity_costs):
            continue
        if not float_in(info.sigma, args.sigmas):
            continue
        if not int_in(info.seed, args.seeds):
            continue
        if not int_in(info.rnn, args.rnn_dims):
            continue
        if not int_in(info.latent, args.latent_dims):
            continue
        if args.max_observations_before_stop is not None and info.maxobs != args.max_observations_before_stop:
            continue
        if args.sampled_lambda_critic != "any" and info.critic != args.sampled_lambda_critic:
            continue
        files.append(info)
    return files


def timestep_columns(cols: list[str], prefix: str) -> list[int]:
    out = []
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for col in cols:
        m = pat.match(col)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def read_reduced_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    keep = {
        "graph",
        "chosen_path",
        "V",
        "MI",
        "opportunity_cost",
        "observation_sigma",
        "node",
        "actual_reward",
    }
    patterns = (
        r"^stop_t\d+$",
        r"^kl_d_t\d+$",
        r"^expanded_node_t\d+$",
        r"^terminal_choice_prob_path\d+_t\d+$",
        r"^action_output_path\d+_t\d+$",
    )
    for col in header:
        if col in keep or any(re.match(p, col) for p in patterns):
            keep.add(col)
    return pd.read_csv(path, usecols=[c for c in header if c in keep])


def normalize_chosen_path(chosen: pd.Series, num_paths: int) -> pd.Series:
    vals = pd.to_numeric(chosen, errors="coerce")
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        return vals
    if finite.min() >= 1 and finite.max() <= num_paths:
        return vals - 1
    return vals


def terminal_entropy(prob_values: np.ndarray) -> np.ndarray:
    probs = np.where(np.isfinite(prob_values), prob_values, 0.0)
    sums = probs.sum(axis=1)
    valid = sums > 0
    out = np.full(probs.shape[0], np.nan, dtype=float)
    if np.any(valid):
        p = probs[valid] / sums[valid, None]
        out[valid] = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=1)
    return out


def stop_timestep(trial_df: pd.DataFrame, stop_ts: list[int]) -> np.ndarray:
    stop_mat = np.column_stack([
        trial_df.get(f"stop_t{t}", pd.Series(False, index=trial_df.index)).fillna(False).astype(bool).to_numpy()
        for t in stop_ts
    ])
    any_stop = stop_mat.any(axis=1)
    first_idx = stop_mat.argmax(axis=1)
    max_t = max(stop_ts) if stop_ts else 0
    return np.where(any_stop, np.asarray(stop_ts, dtype=int)[first_idx], max_t)


def parameter_fields(info: SimFile) -> tuple[str, float, str]:
    if info.family == "vary_beta":
        return "beta", float(info.beta), f"beta = {info.beta:g}"
    if info.family == "vary_opportunity":
        return "opportunity", float(info.opportunity), f"opp = {info.opportunity:g}"
    return "beta_opp", float(info.beta), f"beta = {info.beta:g}, opp = {info.opportunity:g}"


def build_metrics_for_file(info: SimFile, task: jp.TaskSpec, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_reduced_csv(info.path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    trial = df.drop_duplicates("graph").copy().reset_index(drop=True)
    rewards = (
        df[["graph", "node", "actual_reward"]]
        .drop_duplicates(["graph", "node"])
        .pivot(index="graph", columns="node", values="actual_reward")
        .sort_index(axis=1)
    )
    reward_graphs = rewards.index.to_numpy()
    trial = trial.set_index("graph").loc[reward_graphs].reset_index()
    reward_mat = rewards.to_numpy(dtype=float)
    path_map = np.asarray(task.path_map, dtype=float)
    path_rewards = reward_mat @ path_map.T
    num_paths = path_rewards.shape[1]
    chosen_zero = normalize_chosen_path(trial["chosen_path"], num_paths).to_numpy(dtype=float)
    chosen_idx = np.where(np.isfinite(chosen_zero), chosen_zero.astype(int), -1)
    valid_chosen = (chosen_idx >= 0) & (chosen_idx < num_paths)
    chosen_reward = np.full(len(trial), np.nan, dtype=float)
    chosen_reward[valid_chosen] = path_rewards[np.where(valid_chosen)[0], chosen_idx[valid_chosen]]
    sorted_rewards = np.sort(path_rewards, axis=1)
    best_reward = sorted_rewards[:, -1]
    second_reward = sorted_rewards[:, -2] if num_paths > 1 else sorted_rewards[:, -1]
    reward_gap = best_reward - second_reward
    best_mask = np.isclose(path_rewards, best_reward[:, None])
    accuracy = np.full(len(trial), np.nan, dtype=float)
    valid_acc = valid_chosen & (reward_gap > 1e-8)
    accuracy[valid_acc] = best_mask[np.where(valid_acc)[0], chosen_idx[valid_acc]].astype(float)

    stop_ts = timestep_columns(trial.columns.tolist(), "stop_t")
    kl_ts = timestep_columns(trial.columns.tolist(), "kl_d_t")
    prob_ts = sorted(set(
        int(m.group(2))
        for col in trial.columns
        for m in [re.match(r"^terminal_choice_prob_path(\d+)_t(\d+)$", col)]
        if m
    ))
    max_t = max(kl_ts or stop_ts or prob_ts)
    total_stop_t = stop_timestep(trial, stop_ts or list(range(1, max_t + 1)))
    entropy_at_decision = np.full(len(trial), np.nan, dtype=float)
    for t in range(1, max_t + 1):
        rows = total_stop_t == t
        prob_cols = [f"terminal_choice_prob_path{p}_t{t}" for p in range(1, num_paths + 1)]
        if rows.any() and all(c in trial.columns for c in prob_cols):
            entropy_at_decision[rows] = terminal_entropy(trial.loc[rows, prob_cols].to_numpy(dtype=float))

    total_kl = pd.to_numeric(trial.get("MI", np.nan), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(total_kl).any():
        total_kl = np.zeros(len(trial), dtype=float)
        for t in kl_ts:
            total_kl += pd.to_numeric(trial[f"kl_d_t{t}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    final_pre_stop_kl = np.full(len(trial), np.nan, dtype=float)
    last_positive_pre_stop_kl = np.full(len(trial), np.nan, dtype=float)
    for i, stop_t in enumerate(total_stop_t):
        pre_ts = [t for t in kl_ts if t < stop_t]
        if pre_ts:
            final_pre_stop_kl[i] = pd.to_numeric(trial.loc[i, f"kl_d_t{pre_ts[-1]}"], errors="coerce")
            vals = np.asarray([
                pd.to_numeric(trial.loc[i, f"kl_d_t{t}"], errors="coerce")
                for t in pre_ts
            ], dtype=float)
            positive = vals[np.isfinite(vals) & (vals > 0)]
            if positive.size:
                last_positive_pre_stop_kl[i] = positive[-1]

    gap_bin_width = float(args.reward_gap_bin_width)
    if gap_bin_width > 0:
        reward_gap_bin = np.round(reward_gap / gap_bin_width) * gap_bin_width
    else:
        reward_gap_bin = reward_gap

    parameter_name, parameter_value, parameter_label = parameter_fields(info)
    base = pd.DataFrame({
        "source_file": info.path.name,
        "graph": trial["graph"].to_numpy(),
        "seed": info.seed,
        "family": info.family,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "parameter_label": parameter_label,
        "beta": info.beta,
        "lambda": info.lambda_,
        "alpha": info.alpha,
        "opportunity_cost": info.opportunity,
        "sigma": info.sigma,
        "rnn_dim": info.rnn,
        "latent_dim": info.latent,
        "critic": info.critic,
        "total_stop_timestep": total_stop_t,
        "reward_gap": reward_gap,
        "reward_gap_bin": reward_gap_bin,
        "chosen_reward": chosen_reward,
        "normalized_chosen_reward": chosen_reward / float(task.reward_norm),
        "choice_accuracy": accuracy,
        "terminal_choice_entropy": entropy_at_decision,
        "total_kl": total_kl,
        "final_pre_stop_kl": final_pre_stop_kl,
        "last_positive_pre_stop_kl": last_positive_pre_stop_kl,
    })

    rows = []
    for t in kl_ts:
        kl = pd.to_numeric(trial[f"kl_d_t{t}"], errors="coerce").to_numpy(dtype=float)
        expanded_col = f"expanded_node_t{t}"
        if expanded_col in trial.columns:
            observed = np.isfinite(pd.to_numeric(trial[expanded_col], errors="coerce").to_numpy(dtype=float))
        else:
            observed = np.ones(len(trial), dtype=bool)
        keep = (t < total_stop_t) & observed & np.isfinite(kl)
        if not np.any(keep):
            continue
        piece = base.loc[keep].copy()
        piece["timestep"] = t
        piece["kl_paid_at_timestep"] = kl[keep]
        rows.append(piece)
    timestep = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return base, timestep


def sem(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def bin_by_kl(df: pd.DataFrame, kl_col: str, group_cols: list[str], metric: str, bins: int, min_samples: int) -> pd.DataFrame:
    rows = []
    for key, group in df.dropna(subset=[kl_col, metric]).groupby(group_cols, dropna=False):
        work = group.copy()
        finite = np.isfinite(work[kl_col]) & np.isfinite(work[metric])
        work = work.loc[finite]
        if len(work) < min_samples:
            continue
        try:
            work["kl_bin"] = pd.qcut(work[kl_col], q=min(bins, max(1, len(work) // min_samples)), duplicates="drop")
        except ValueError:
            work["kl_bin"] = pd.cut(work[kl_col], bins=1)
        for _bin, bdf in work.groupby("kl_bin", observed=True):
            if len(bdf) < min_samples:
                continue
            record = {col: val for col, val in zip(group_cols, key if isinstance(key, tuple) else (key,))}
            record.update({
                "kl_mean": float(bdf[kl_col].mean()),
                "metric_mean": float(bdf[metric].mean()),
                "metric_sem_by_seed": sem(bdf.groupby("seed")[metric].mean()),
                "n": int(len(bdf)),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def short_metric_label(metric: str) -> str:
    return {
        "choice_accuracy": "P(best path)",
        "terminal_choice_entropy": "Choice\nentropy",
        "normalized_chosen_reward": "Norm\nreward",
    }.get(metric, metric)


def apply_small_plot_style() -> None:
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


def small_multipanel_size(
    n_cols: int,
    n_rows: int,
    legend_width: float = 0.0,
    panel_height: float = PANEL_HEIGHT_IN,
) -> tuple[float, float]:
    width = (
        LEFT_MARGIN_IN
        + RIGHT_MARGIN_IN
        + legend_width
        + n_cols * PANEL_WIDTH_IN
        + max(0, n_cols - 1) * PANEL_GAP_IN
    )
    height = (
        BOTTOM_MARGIN_IN
        + TOP_MARGIN_IN
        + n_rows * panel_height
        + max(0, n_rows - 1) * PANEL_GAP_IN
    )
    return max(width, 2.2), max(height, 2.0)


def plot_conditioned(summary: pd.DataFrame, metric: str, out_path: Path, max_stop_times: int, max_gap_bins: int):
    if summary.empty:
        return
    apply_small_plot_style()
    stop_values = sorted(summary["total_stop_timestep"].dropna().unique())[:max_stop_times]
    gap_values = sorted(summary["reward_gap_bin"].dropna().unique())[:max_gap_bins]
    data = summary[
        summary["total_stop_timestep"].isin(stop_values)
        & summary["reward_gap_bin"].isin(gap_values)
    ].copy()
    if data.empty:
        return
    n_rows = max(len(stop_values), 1)
    n_cols = max(len(gap_values), 1)
    fig_w, fig_h = small_multipanel_size(
        n_cols,
        n_rows,
        legend_width=LEGEND_WIDTH_IN,
        panel_height=CONDITIONED_PANEL_HEIGHT_IN,
    )
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [max(0.65, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
        wspace=0.45,
        hspace=0.35,
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - RIGHT_MARGIN_IN / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
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
    cmap = plt.get_cmap("viridis")
    timesteps = sorted(data["timestep"].dropna().unique())
    t_min, t_max = min(timesteps), max(timesteps)
    for row_i, stop_t in enumerate(stop_values):
        for col_i, gap in enumerate(gap_values):
            ax = axes[row_i, col_i]
            panel = data[(data["total_stop_timestep"] == stop_t) & (np.isclose(data["reward_gap_bin"], gap))]
            for t, tdf in panel.groupby("timestep"):
                color = cmap(0.5 if t_min == t_max else (float(t) - t_min) / (t_max - t_min))
                yerr = tdf["metric_sem_by_seed"].to_numpy(dtype=float) if "metric_sem_by_seed" in tdf else None
                ax.errorbar(
                    tdf["kl_mean"],
                    tdf["metric_mean"],
                    yerr=yerr,
                    marker="o",
                    linestyle="-",
                    linewidth=0.8,
                    markersize=2.5,
                    color=color,
                    alpha=0.9,
                )
            ax.set_title(f"stop {int(stop_t)}\ngap {gap:g}", pad=1.5)
            ax.tick_params(pad=1.5)
            if row_i < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if col_i > 0:
                ax.tick_params(labelleft=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_min, t_max))
    # Keep the timestep scale outside the plot panels, but compact. Letting it
    # span all rows makes the scale visually dominate the small-panel figures.
    cbar_height = min(0.34, max(0.16, 1.0 / max(3.0 * n_rows, 1.0)))
    cbar_y = 0.5 - cbar_height / 2.0
    cax = legend_ax.inset_axes([0.18, cbar_y, 0.16, cbar_height])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("timestep")
    cbar.ax.tick_params(pad=1.5)
    fig.supxlabel("KL paid at\ntimestep", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(short_metric_label(metric), fontsize=PLOT_FONT_SIZE_PT, x=0.015)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_predictor_comparison(
    trial_metrics: pd.DataFrame,
    metric: str,
    out_path: Path,
    min_samples: int,
    bins: int,
    max_gap_bins: int,
):
    apply_small_plot_style()
    pieces = []
    for pred in ["final_pre_stop_kl", "last_positive_pre_stop_kl", "total_kl"]:
        if pred not in trial_metrics:
            continue
        tmp = trial_metrics.dropna(subset=[pred, metric]).copy()
        if len(tmp) < min_samples:
            continue
        tmp["predictor"] = pred
        tmp["predictor_value"] = tmp[pred]
        pieces.append(tmp)
    if not pieces:
        return pd.DataFrame()
    data = pd.concat(pieces, ignore_index=True)
    summary = bin_by_kl(
        data,
        "predictor_value",
        ["predictor", "sigma", "reward_gap_bin"],
        metric,
        bins=bins,
        min_samples=min_samples,
    )
    if summary.empty:
        return summary
    labels = {
        "final_pre_stop_kl": "final pre-stop KL",
        "last_positive_pre_stop_kl": "last positive pre-stop KL",
        "total_kl": "total KL",
    }
    sigmas = sorted(summary["sigma"].dropna().unique())
    gap_values = sorted(summary["reward_gap_bin"].dropna().unique())[:max_gap_bins]
    summary = summary[summary["reward_gap_bin"].isin(gap_values)].copy()
    if summary.empty:
        return summary
    n_cols = max(1, len(sigmas))
    n_rows = max(1, len(gap_values))
    fig_w, fig_h = small_multipanel_size(n_cols, n_rows, legend_width=LEGEND_WIDTH_IN)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [max(0.65, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
        wspace=0.45,
        hspace=0.35,
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - RIGHT_MARGIN_IN / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
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
    colors = {"final_pre_stop_kl": "#1f77b4", "last_positive_pre_stop_kl": "#2ca02c", "total_kl": "#d62728"}
    for row_i, gap in enumerate(gap_values):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = summary[
                np.isclose(summary["sigma"], sigma)
                & np.isclose(summary["reward_gap_bin"], gap)
            ]
            for pred, pdf in panel.groupby("predictor"):
                yerr = pdf["metric_sem_by_seed"].to_numpy(dtype=float) if "metric_sem_by_seed" in pdf else None
                ax.errorbar(
                    pdf["kl_mean"],
                    pdf["metric_mean"],
                    yerr=yerr,
                    marker="o",
                    linestyle="-",
                    linewidth=0.9,
                    markersize=2.5,
                    color=colors.get(pred, "black"),
                    label=labels.get(pred, pred),
                )
            title_parts = []
            if row_i == 0:
                title_parts.append(f"sigma {sigma:g}")
            title_parts.append(f"gap {gap:g}")
            ax.set_title("\n".join(title_parts), pad=1.5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(pad=1.5)
            if row_i < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if col_i > 0:
                ax.tick_params(labelleft=False)
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels_out, loc="center left", frameon=False, handlelength=1.1)
    fig.supxlabel("KL predictor", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(short_metric_label(metric), fontsize=PLOT_FONT_SIZE_PT, x=0.015)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return summary


def correlation_table(trial_metrics: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    predictors = ["final_pre_stop_kl", "last_positive_pre_stop_kl", "total_kl"]
    group_cols = ["beta", "opportunity_cost", "sigma", "reward_gap_bin"]
    for keys, group in trial_metrics.groupby(group_cols, dropna=False):
        for pred in predictors:
            for metric in metrics:
                work = group[[pred, metric]].dropna()
                if len(work) < 10:
                    corr = np.nan
                else:
                    corr = float(work[pred].corr(work[metric], method="spearman"))
                rows.append({
                    "beta": keys[0],
                    "opportunity_cost": keys[1],
                    "sigma": keys[2],
                    "reward_gap_bin": keys[3],
                    "predictor": pred,
                    "metric": metric,
                    "spearman": corr,
                    "n": int(len(work)),
                })
    return pd.DataFrame(rows)


def file_num_label(value: float) -> str:
    if not np.isfinite(float(value)):
        return "na"
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def subset_output_stub(family: str, parameter_name: str, parameter_value: float) -> str:
    family_stub = str(family).replace("vary_", "")
    name_stub = str(parameter_name).replace("opportunity", "opp")
    return f"{family_stub}_{name_stub}_{file_num_label(parameter_value)}"


def iter_parameter_subsets(
    trial_metrics: pd.DataFrame,
    timestep_metrics: pd.DataFrame,
) -> list[tuple[str, Path, pd.DataFrame, pd.DataFrame]]:
    required = {"family", "parameter_name", "parameter_value"}
    if trial_metrics.empty or timestep_metrics.empty or not required.issubset(trial_metrics.columns):
        return []
    keys = (
        trial_metrics[["family", "parameter_name", "parameter_value", "parameter_label"]]
        .drop_duplicates()
        .sort_values(["family", "parameter_value"])
    )
    subsets = []
    for row in keys.itertuples(index=False):
        family = str(row.family)
        parameter_name = str(row.parameter_name)
        parameter_value = float(row.parameter_value)
        parameter_label = str(row.parameter_label)
        trial_keep = (
            (trial_metrics["family"] == family)
            & (trial_metrics["parameter_name"] == parameter_name)
            & np.isclose(pd.to_numeric(trial_metrics["parameter_value"], errors="coerce"), parameter_value)
        )
        timestep_keep = (
            (timestep_metrics["family"] == family)
            & (timestep_metrics["parameter_name"] == parameter_name)
            & np.isclose(pd.to_numeric(timestep_metrics["parameter_value"], errors="coerce"), parameter_value)
        )
        stub = subset_output_stub(family, parameter_name, parameter_value)
        subsets.append((parameter_label, Path(stub), trial_metrics.loc[trial_keep].copy(), timestep_metrics.loc[timestep_keep].copy()))
    return subsets


def write_plots_for_subset(
    trial_metrics: pd.DataFrame,
    timestep_metrics: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    label: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = ["choice_accuracy", "terminal_choice_entropy", "normalized_chosen_reward"]
    for metric in metrics:
        summary = bin_by_kl(
            timestep_metrics,
            "kl_paid_at_timestep",
            ["timestep", "total_stop_timestep", "reward_gap_bin"],
            metric,
            bins=args.kl_bins,
            min_samples=args.min_samples,
        )
        if not summary.empty:
            summary.insert(0, "subset", label)
        summary.to_csv(outdir / f"kl_at_timestep_vs_{metric}_conditioned_summary.csv", index=False)
        plot_conditioned(
            summary,
            metric,
            outdir / f"kl_at_timestep_vs_{metric}_conditioned.png",
            max_stop_times=args.max_stop_times,
            max_gap_bins=args.max_gap_bins,
        )
        pred_summary = plot_predictor_comparison(
            trial_metrics,
            metric,
            outdir / f"final_vs_total_kl_{metric}.png",
            min_samples=args.min_samples,
            bins=args.kl_bins,
            max_gap_bins=args.max_gap_bins,
        )
        if not pred_summary.empty:
            pred_summary.insert(0, "subset", label)
            pred_summary.to_csv(outdir / f"final_vs_total_kl_{metric}_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hypothesis-1 revisit diagnostics. With a positional <tree>, inputs "
            "match plot_revisit_beta_opp_comparison.R and shared params are read "
            "from revisit_plot_presets.csv."
        )
    )
    parser.add_argument("tree", nargs="?", default=None)
    parser.add_argument("--preset-file", default=str(default_preset_file()))
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-root", "--results-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--tree-size", type=int, default=None)
    parser.add_argument("--tree-type", default=None)
    parser.add_argument("--input-type", default=None)
    parser.add_argument("--lambdas", type=str, default="100.0")
    parser.add_argument("--alphas", type=str, default="0.0")
    parser.add_argument(
        "--betas",
        "--beta-values",
        "--vary-betas",
        "--vary-beta-values",
        dest="betas",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--opportunity-costs",
        "--opportunity-values",
        "--opportunities",
        "--vary-opps",
        "--vary-opportunities",
        "--vary-opportunity-values",
        dest="opportunity_costs",
        type=str,
        default=None,
    )
    parser.add_argument("--sigmas", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--rnn-dims", type=str, default=None)
    parser.add_argument("--latent-dims", type=str, default=None)
    parser.add_argument("--max-observations-before-stop", type=int, default=None)
    parser.add_argument(
        "--sampled-lambda-critic",
        "--critic",
        "--critic-type",
        choices=["any", "q", "value"],
        default="q",
    )
    parser.add_argument("--reward-gap-bin-width", type=float, default=1.0)
    parser.add_argument("--kl-bins", type=int, default=6)
    parser.add_argument("--min-samples", "--min-sampes", "--min-n", dest="min_samples", type=int, default=25)
    parser.add_argument("--max-stop-times", type=int, default=8)
    parser.add_argument("--max-gap-bins", type=int, default=8)
    args = parser.parse_args()
    args.requested_pair_families = None
    if args.tree is not None:
        return apply_revisit_preset_args(args)

    args.input_dir = args.input_dir or "outputs/jax_simulations"
    args.outdir = args.outdir or str(Path(args.output_root or "results") / "revisit_hypothesis1")
    args.tree_size = 2 if args.tree_size is None else int(args.tree_size)
    args.tree_type = args.tree_type or "default"
    args.input_type = args.input_type or "uniform"
    args.lambdas = parse_float_list(args.lambdas)
    args.alphas = parse_float_list(args.alphas)
    args.betas = parse_float_list(args.betas)
    args.opportunity_costs = parse_float_list(args.opportunity_costs)
    args.sigmas = parse_float_list(args.sigmas)
    args.seeds = parse_int_list(args.seeds)
    args.rnn_dims = parse_int_list(args.rnn_dims)
    args.latent_dims = parse_int_list(args.latent_dims)
    return args


def main() -> None:
    args = parse_args()
    task = jp.build_task(args.tree_size, args.tree_type, args.input_type)
    files = find_files(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not files:
        raise SystemExit("No matching simulation CSVs found.")
    print(f"Found {len(files)} simulation CSV file(s).")

    trial_parts = []
    timestep_parts = []
    for i, info in enumerate(files, start=1):
        trial, timestep = build_metrics_for_file(info, task, args)
        if not trial.empty:
            trial_parts.append(trial)
        if not timestep.empty:
            timestep_parts.append(timestep)
        if i % 25 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)} CSVs", flush=True)
    trial_metrics = pd.concat(trial_parts, ignore_index=True) if trial_parts else pd.DataFrame()
    timestep_metrics = pd.concat(timestep_parts, ignore_index=True) if timestep_parts else pd.DataFrame()
    trial_metrics.to_csv(outdir / "trial_metrics.csv", index=False)
    timestep_metrics.to_csv(outdir / "timestep_kl_outcomes.csv", index=False)

    metrics = ["choice_accuracy", "terminal_choice_entropy", "normalized_chosen_reward"]
    subsets = iter_parameter_subsets(trial_metrics, timestep_metrics)
    if subsets:
        print(f"Writing separate plot sets for {len(subsets)} varied parameter value(s).", flush=True)
        for label, stub, trial_subset, timestep_subset in subsets:
            subset_dir = outdir / "by_parameter" / stub
            write_plots_for_subset(trial_subset, timestep_subset, subset_dir, args, label)
    else:
        write_plots_for_subset(trial_metrics, timestep_metrics, outdir, args, "all")

    corr = correlation_table(trial_metrics, metrics)
    corr.to_csv(outdir / "final_vs_total_kl_correlations.csv", index=False)
    print(f"Saved hypothesis-1 KL timing diagnostics to {outdir}")


if __name__ == "__main__":
    main()
