#!/usr/bin/env python
"""JAX checkpoint version of analyze_latent_angle_planning.py.

This script uses the current step-batched JAX/Flax VAE model in
model_jax/planning.py and the JAX .msgpack checkpoints, then reuses the
original latent geometry analysis and plotting functions.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated.*",
    category=FutureWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_DIR = REPO_ROOT / "model"
MODEL_JAX_DIR = REPO_ROOT / "model_jax"
for path in (str(REPO_ROOT), str(MODEL_DIR), str(MODEL_JAX_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
from flax import serialization

import analyze_latent_angle_planning as base
import helper
try:
    from model_jax import planning as jp
except ModuleNotFoundError:
    import planning as jp
from latent_angle_utils import (
    make_model_config,
    sample_rewards,
    safe_json_dumps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument(
        "--scalers",
        "--betas",
        dest="scaler_values",
        nargs="+",
        type=float,
        required=True,
        help="Reward/action/critic weight scalers. --betas is accepted as a legacy alias.",
    )
    parser.add_argument(
        "--lambdas",
        dest="lambda_values",
        nargs="+",
        type=float,
        required=True,
        help="Direct KL/information-cost scalers.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--rnn-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[32])
    parser.add_argument("--opportunity-costs", "--opportunity-cost", dest="opportunity_costs", nargs="+", type=float, default=[0.0])
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", default="analysis_outputs/latent_angle_planning_jax")
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform", choices=["uniform", "binary"])
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae", choices=["vae", "rnn", "jax_ppo", "jax_vae"])
    parser.add_argument("--device", default="cpu", help="Accepted for CLI symmetry; JAX analysis uses CPU.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-within-path-n", type=int, default=50)
    parser.add_argument("--min-reward-group-n", type=int, default=10)
    parser.add_argument("--latent-density-grid-n", type=int, default=150)
    parser.add_argument("--analysis-seed-offset", type=int, default=100_000)
    parser.add_argument(
        "--also-plot-combined",
        action="store_true",
        help="Also write the old pooled plots in --outdir. Per seed/scaler/opportunity plots are always written.",
    )
    return parser.parse_args()


def normalize_variant_for_file(variant: str) -> str:
    key = str(variant).strip().lower()
    if key in ("jax", "jax_vae"):
        return "vae"
    return key


def _extract_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
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


def _close(a: Optional[float], b: float, tol: float = 1e-8) -> bool:
    return a is not None and abs(float(a) - float(b)) <= tol


def find_jax_checkpoint(
    root: str,
    *,
    lambda_value: float,
    alpha: float,
    beta: float,
    seed: int,
    tree_size: int,
    tree_type: str,
    rnn_dim: int,
    latent_dim: int,
    opportunity_cost: float,
    expansion_decision_version: str,
    model_variant: str,
    input_type: str = "uniform",
) -> Tuple[Optional[Path], List[str]]:
    root_path = Path(root)
    if not root_path.exists():
        return None, [f"checkpoint root does not exist: {root}"]
    task = jp.build_task(tree_size, tree_type, input_type)
    expected_tree = f"{tree_size}n{task.tree_name_suffix}"
    model_variant = normalize_variant_for_file(model_variant)
    matches = []
    for path in root_path.rglob("*.msgpack"):
        name = path.name
        found_lambda = _extract_float(r"lambda_([0-9eE.+-]+)", name)
        found_scaler = _extract_float(r"scaler_([0-9eE.+-]+)", name)
        found_legacy_beta = _extract_float(r"beta_([0-9eE.+-]+)", name)
        new_match = _close(found_lambda, lambda_value) and _close(found_scaler, beta)
        current_beta_match = _close(found_lambda, lambda_value) and _close(found_legacy_beta, beta)
        reversed_legacy_match = _close(found_lambda, beta) and _close(found_legacy_beta, lambda_value)
        if not (
            (new_match or current_beta_match or reversed_legacy_match)
            and _close(_extract_float(r"alpha_([0-9eE.+-]+)", name), alpha)
            and _extract_int(r"seed_([0-9]+)", name) == seed
            and _close(_extract_float(r"opportunity_([0-9eE.+-]+)", name), opportunity_cost)
        ):
            continue
        if expected_tree not in name:
            continue
        if _extract_int(r"rnn_([0-9]+)", name) != int(rnn_dim):
            continue
        if _extract_int(r"latent_([0-9]+)", name) != int(latent_dim):
            continue
        if f"expansion_{expansion_decision_version}" not in name:
            continue
        if f"variant_{model_variant}" not in name:
            continue
        matches.append((0 if (new_match or current_beta_match) else 1, path))
    if not matches:
        return None, [f"no .msgpack checkpoint matched tree={expected_tree} variant={model_variant}"]
    matches.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
    return matches[0][1], [f"matched {matches[0][1].name}"]


def build_jax_model_and_params(
    checkpoint_path: Path,
    *,
    alpha: float,
    beta: float,
    lambda_value: float,
    opportunity_cost: float,
    tree_size: int,
    tree_type: str,
    input_type: str,
    expansion_decision_version: str,
    model_variant: str,
    rnn_dim: int,
    latent_dim: int,
):
    task = jp.build_task(tree_size, tree_type, input_type)
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
        expansion_decision_version=expansion_decision_version,
        use_autoencoder=(normalize_variant_for_file(model_variant) != "rnn"),
        enable_reconstruction=False,
        enable_probe=False,
        allow_node_revisit=False,
        max_observations_before_stop=int(task.num_nodes),
        opportunity_cost=float(opportunity_cost),
        observation_sigma=0.0,
        lambda_=float(lambda_value),
        alpha=float(alpha),
        beta=float(beta),
        include_visited_lstm_input=jp.use_visited_lstm_input_for_task(task),
    )
    dummy = jp.initial_carry(
        1,
        task,
        int(rnn_dim),
        1,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    schedule = jp.ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(1.0 / float(beta), dtype=jnp.float32),
        current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
        node_coverage_aux_coef=jnp.asarray(0.0, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )
    template = model.init(
        jax.random.PRNGKey(0),
        dummy,
        jax.random.PRNGKey(1),
        schedule,
        None,
        False,
        False,
        False,
    )["params"]
    params = serialization.from_bytes(template, checkpoint_path.read_bytes())
    return model, params, task


def jax_output_tuple(
    transitions: List[jp.StepTransition],
    *,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    terminal_rng: np.random.Generator,
) -> Tuple:
    outputs = [None] * base.BASE_OUTPUT_COUNT
    if len(transitions) == 0:
        raise ValueError("No JAX transitions were produced.")
    z_mu = np.stack([np.asarray(trans.z_mu) for trans in transitions], axis=1)
    z_logvar = np.stack([np.asarray(trans.z_logvar) for trans in transitions], axis=1)
    prior_mu = np.stack([np.asarray(trans.prior_mu) for trans in transitions], axis=1)
    prior_logvar = np.stack([np.asarray(trans.prior_logvar) for trans in transitions], axis=1)
    node_selections = np.stack([np.asarray(trans.node_index) for trans in transitions], axis=1).astype(int)
    stop_decisions = np.stack([np.asarray(trans.is_stop) for trans in transitions], axis=1)[:, :, None].astype(bool)
    observed_masks = np.stack([np.asarray(trans.observed_after) for trans in transitions], axis=1).astype(bool)
    action_outputs = np.stack([np.asarray(trans.action_output) for trans in transitions], axis=1)
    paid_kl = np.stack([np.asarray(trans.paid_kl) for trans in transitions], axis=1)[:, :, None]
    observed_kl = np.stack([np.asarray(trans.observed_kl) for trans in transitions], axis=1)[:, :, None]
    terminal_path_indices = np.stack(
        [np.asarray(trans.terminal_path_index) for trans in transitions],
        axis=1,
    ).astype(int)
    path_rewards = np.asarray(rewards, dtype=float) @ np.asarray(task.path_map, dtype=float).T
    selected_paths = np.full(z_mu.shape[0], -1, dtype=int)
    for trial_i in range(z_mu.shape[0]):
        stop_steps = np.where(stop_decisions[trial_i, :, 0])[0]
        if len(stop_steps) > 0:
            path_i = int(terminal_path_indices[trial_i, stop_steps[0]])
            if 0 <= path_i < task.num_paths:
                selected_paths[trial_i] = path_i
        if selected_paths[trial_i] < 0:
            probs = action_outputs[trial_i, -1]
            probs = np.asarray(probs, dtype=float)
            probs = np.where(np.isfinite(probs), probs, 0.0)
            total = probs.sum()
            if total <= 0:
                probs = np.ones(task.num_paths, dtype=float) / task.num_paths
            else:
                probs = probs / total
            selected_paths[trial_i] = int(terminal_rng.choice(task.num_paths, p=probs))
    outputs[0] = np.zeros((z_mu.shape[0], z_mu.shape[1], task.num_nodes, 9), dtype=np.float32)
    outputs[1] = action_outputs
    outputs[2] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[3] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[4] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[5] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[6] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[7] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[8] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[9] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[10] = np.sum(paid_kl, axis=(1, 2))
    outputs[11] = z_mu
    outputs[12] = node_selections
    outputs[13] = stop_decisions
    outputs[14] = observed_masks
    outputs[15] = action_outputs
    outputs[16] = paid_kl
    outputs[17] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[18] = np.zeros((z_mu.shape[0], z_mu.shape[1]), dtype=np.float32)
    outputs[19] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[20] = np.mean(stop_decisions, axis=1)
    outputs[21] = 1.0 - outputs[20]
    outputs[22] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[23] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[24] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs[28] = selected_paths
    outputs[29] = observed_kl
    outputs[30] = np.zeros((z_mu.shape[0], z_mu.shape[1], 1), dtype=np.float32)
    outputs[31] = np.zeros((z_mu.shape[0], z_mu.shape[1], 1), dtype=np.float32)
    for idx, value in enumerate(outputs):
        if value is None:
            outputs[idx] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs.extend([z_logvar, prior_mu, prior_logvar])
    return tuple(outputs)


def run_jax_model_trials(
    model: jp.PlanningVAE,
    params,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    batch_size: int,
    seed: int,
    beta: float,
) -> Tuple:
    batches = []
    for batch_i, start in enumerate(range(0, rewards.shape[0], batch_size)):
        batch_rewards = rewards[start:start + batch_size]
        carry = jp.initial_carry(
            batch_rewards.shape[0],
            task,
            model.rnn_units,
            1,
            jp.visited_lstm_feature_dim_for_task(task),
        )
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        schedule = jp.ScheduleValues(
            current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
            current_beta=jnp.asarray(1.0 / float(beta), dtype=jnp.float32),
            current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
            expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
            expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
            node_coverage_aux_coef=jnp.asarray(0.0, dtype=jnp.float32),
            forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
            ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
        )
        rng = jax.random.PRNGKey(seed + batch_i)
        transitions = []
        for _ in range(task.num_nodes):
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
            transitions.append(jax.device_get(trans))
        terminal_rng = np.random.default_rng(seed + 50_000 + batch_i)
        batches.append(jax_output_tuple(
            transitions,
            task=task,
            rewards=batch_rewards,
            terminal_rng=terminal_rng,
        ))
    merged = []
    n_outputs = len(batches[0])
    for output_i in range(n_outputs):
        parts = [batch[output_i] for batch in batches]
        first = np.asarray(parts[0])
        if first.ndim > 0 and first.shape[0] == min(batch_size, rewards.shape[0]):
            merged.append(np.concatenate(parts, axis=0))
        else:
            merged.append(first)
    return tuple(merged)


def file_token(value) -> str:
    text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def per_combo_group_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "seed",
        "beta",
        "opportunity_cost",
        "lambda_value",
        "alpha",
        "rnn_dim",
        "latent_dim",
        "tree_size",
        "tree_type",
        "input_type",
        "expansion_decision_version",
        "model_variant",
    ]
    return [col for col in preferred if col in df.columns]


def per_combo_output_dir(outdir: Path, group_name: dict) -> Path:
    parts = [
        f"seed_{file_token(group_name.get('seed', 'NA'))}",
        f"beta_{file_token(group_name.get('beta', 'NA'))}",
        f"opp_{file_token(group_name.get('opportunity_cost', 'NA'))}",
    ]
    if "lambda_value" in group_name:
        parts.append(f"lambda_{file_token(group_name.get('lambda_value'))}")
    if "rnn_dim" in group_name:
        parts.append(f"rnn_{file_token(group_name.get('rnn_dim'))}")
    if "latent_dim" in group_name:
        parts.append(f"latent_{file_token(group_name.get('latent_dim'))}")
    if "tree_type" in group_name:
        parts.append(f"tree_{file_token(group_name.get('tree_type'))}")
    return outdir / "by_seed_beta_opportunity" / "_".join(parts)


def grouped_failures(failures: List[dict], group_df: pd.DataFrame) -> List[dict]:
    if not failures or "model_id" not in group_df.columns:
        return []
    model_ids = set(group_df["model_id"].dropna().astype(str).unique())
    return [
        failure for failure in failures
        if str(failure.get("model_id", "")) in model_ids
    ]


def group_trial_count(group_df: pd.DataFrame, fallback: int) -> int:
    if "trial_uid" in group_df.columns:
        count = int(group_df["trial_uid"].nunique())
        if count > 0:
            return count
    if "trial_id" in group_df.columns:
        count = int(group_df["trial_id"].nunique())
        if count > 0:
            return count
    return int(fallback)


def run_plotting_pipeline(
    df: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    failures: List[dict],
    *,
    label: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    base.plot_latent_2d_density_reward_outputs(
        df,
        outdir,
        args.latent_density_grid_n,
        args.cv_folds,
    )
    base.run_reward_encoding_analyses(
        df,
        outdir,
        args.cv_folds,
        failures,
        group_trial_count(df, args.n_trials),
        make_plots=False,
    )
    base.run_geometry_meaning_analyses(
        df,
        outdir,
        args.cv_folds,
        args.min_within_path_n,
        make_plots=False,
    )
    base.run_halfplane_reward_geometry_analysis(
        df,
        outdir,
        args.min_reward_group_n,
        make_plots=False,
    )
    try:
        base.run_prior_centered_geometry_reward_analysis(
            df,
            outdir,
            args.min_reward_group_n,
        )
    except Exception as exc:
        print(f"Plotting failed for {label}: {exc}")


def run_per_combo_plotting(
    data: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    failures: List[dict],
) -> pd.DataFrame:
    group_cols = per_combo_group_columns(data)
    if not group_cols:
        run_plotting_pipeline(data, outdir / "by_seed_beta_opportunity" / "all", args, failures, label="all")
        return pd.DataFrame([{"plot_outdir": str(outdir / "by_seed_beta_opportunity" / "all")}])

    index_rows = []
    grouped = data.groupby(group_cols, dropna=False)
    for values, group_df in grouped:
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(group_cols, values))
        group_outdir = per_combo_output_dir(outdir, group_name)
        label = ", ".join(f"{key}={value}" for key, value in group_name.items())
        print(f"Writing per-combination JAX latent plots for {label} -> {group_outdir}")
        run_plotting_pipeline(
            group_df.copy(),
            group_outdir,
            args,
            grouped_failures(failures, group_df),
            label=label,
        )
        index_rows.append({
            **group_name,
            "n_rows": int(len(group_df)),
            "n_trials": group_trial_count(group_df, args.n_trials),
            "plot_outdir": str(group_outdir),
            "figures_dir": str(group_outdir / "figures"),
        })
    return pd.DataFrame(index_rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        base.apply_7pt_plot_style(plt)
    except Exception:
        pass

    all_frames = []
    failures = []
    model_variant = normalize_variant_for_file(args.model_variant)
    for alpha, scaler, lambda_value, seed, rnn_dim, latent_dim, opportunity_cost in itertools.product(
        args.alphas,
        args.scaler_values,
        args.lambda_values,
        args.seeds,
        args.rnn_dims,
        args.latent_dims,
        args.opportunity_costs,
    ):
        model_id = (
            f"jax_lambda_{lambda_value}_alpha_{alpha}_scaler_{scaler}_seed_{seed}_"
            f"rnn_{rnn_dim}_latent_{latent_dim}_opp_{opportunity_cost}"
        )
        try:
            checkpoint_path, notes = find_jax_checkpoint(
                args.checkpoint_root,
                lambda_value=lambda_value,
                alpha=alpha,
                beta=scaler,
                seed=seed,
                tree_size=args.tree_size,
                tree_type=args.tree_type,
                rnn_dim=rnn_dim,
                latent_dim=latent_dim,
                opportunity_cost=opportunity_cost,
                expansion_decision_version=args.expansion_decision_version,
                model_variant=model_variant,
                input_type=args.input_type,
            )
            if checkpoint_path is None:
                failures.append({"model_id": model_id, "reason": "; ".join(notes)})
                print(f"Skipping {model_id}: {'; '.join(notes)}")
                continue
            config = make_model_config(
                helper,
                tree_size=args.tree_size,
                tree_type=args.tree_type,
                input_type=args.input_type,
                seed=seed,
                rnn_dim=rnn_dim,
                latent_dim=latent_dim,
                expansion_decision_version=args.expansion_decision_version,
                model_variant=model_variant,
                checkpoint_root=args.checkpoint_root,
            )
            model, params, task = build_jax_model_and_params(
                checkpoint_path,
                alpha=alpha,
                beta=scaler,
                lambda_value=lambda_value,
                opportunity_cost=opportunity_cost,
                tree_size=args.tree_size,
                tree_type=args.tree_type,
                input_type=args.input_type,
                expansion_decision_version=args.expansion_decision_version,
                model_variant=model_variant,
                rnn_dim=rnn_dim,
                latent_dim=latent_dim,
            )
            rewards = sample_rewards(
                args.n_trials,
                task.num_nodes,
                args.input_type,
                seed=args.analysis_seed_offset + seed,
            )
            outputs = run_jax_model_trials(
                model,
                params,
                task,
                rewards,
                args.batch_size,
                seed=args.analysis_seed_offset + seed + 10_000,
                beta=scaler,
            )
            metadata = {
                "model_id": model_id,
                "checkpoint_path": str(checkpoint_path),
                "alpha": alpha,
                "beta": scaler,
                "scaler": scaler,
                "lambda_value": lambda_value,
                "seed": seed,
                "rnn_dim": rnn_dim,
                "latent_dim": latent_dim,
                "opportunity_cost": opportunity_cost,
                "tree_size": args.tree_size,
                "tree_type": args.tree_type,
                "input_type": args.input_type,
                "expansion_decision_version": args.expansion_decision_version,
                "model_variant": model_variant,
                "model_backend": "jax",
                "model_config": safe_json_dumps({
                    "time_steps": config.time_steps,
                    "num_paths": config.num_paths,
                    "index_path_map": {
                        str(k): [int(vv) for vv in v]
                        for k, v in config.index_path_map.items()
                    },
                    "reward_norm_value": config.reward_norm_value,
                }),
            }
            frame = base.trial_timestep_dataframe(
                metadata=metadata,
                config=config,
                rewards=rewards,
                outputs=outputs,
            )
            all_frames.append(frame)
            print(f"Analyzed {model_id} from {checkpoint_path}")
        except Exception as exc:
            failures.append({"model_id": model_id, "reason": repr(exc)})
            print(f"Failed {model_id}: {exc}")

    failure_path = outdir / "latent_angle_failure_log.csv"
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    if not all_frames:
        raise SystemExit(f"No JAX models were analyzed. See {failure_path}")

    data = pd.concat(all_frames, ignore_index=True)
    data = base.add_geometry_meaning_columns(data)
    data.to_csv(outdir / "latent_angle_trial_timestep_data.csv", index=False)
    try:
        data.to_parquet(outdir / "latent_angle_trial_timestep_data.parquet", index=False)
    except Exception as exc:
        print(f"Parquet output skipped: {exc}")

    transition_data = base.build_temporal_transition_features(data)
    transition_data.to_csv(outdir / "latent_temporal_transition_features.csv", index=False)
    try:
        if len(transition_data) > 0:
            transition_data.to_parquet(outdir / "latent_temporal_transition_features.parquet", index=False)
    except Exception as exc:
        print(f"Temporal transition parquet output skipped: {exc}")

    base.run_reward_encoding_analyses(data, outdir, args.cv_folds, failures, args.n_trials, make_plots=False)
    base.run_geometry_meaning_analyses(data, outdir, args.cv_folds, args.min_within_path_n, make_plots=False)
    base.run_halfplane_reward_geometry_analysis(data, outdir, args.min_reward_group_n, make_plots=False)
    group_plot_index = run_per_combo_plotting(data, outdir, args, failures)
    group_plot_index.to_csv(outdir / "per_seed_beta_opportunity_plot_index.csv", index=False)
    if args.also_plot_combined:
        print(f"Writing combined JAX latent plots to {outdir}")
        base.plot_latent_2d_density_reward_outputs(data, outdir, args.latent_density_grid_n, args.cv_folds)
        try:
            base.run_prior_centered_geometry_reward_analysis(data, outdir, args.min_reward_group_n)
        except Exception as exc:
            print(f"Combined plotting failed: {exc}")
    print(f"Saved JAX latent angle analysis outputs to {outdir}")


if __name__ == "__main__":
    main()
