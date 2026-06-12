#!/usr/bin/env python
"""JAX checkpoint version of analyze_latent_angle_planning.py.

This script uses the JAX/Flax VAE model in model/jax_planning.py and the JAX
.msgpack checkpoints, then reuses the original latent geometry analysis and
plotting functions.
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
for path in (str(REPO_ROOT), str(MODEL_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
from flax import serialization

import analyze_latent_angle_planning as base
import helper
import jax_planning as jp
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
) -> Tuple[Optional[Path], List[str]]:
    root_path = Path(root)
    if not root_path.exists():
        return None, [f"checkpoint root does not exist: {root}"]
    task = jp.build_task(tree_size, tree_type, "uniform")
    expected_tree = jp.tree_label(task)
    model_variant = normalize_variant_for_file(model_variant)
    matches = []
    for path in root_path.rglob("*.msgpack"):
        name = path.name
        found_lambda = _extract_float(r"lambda_([0-9eE.+-]+)", name)
        found_scaler = _extract_float(r"scaler_([0-9eE.+-]+)", name)
        found_legacy_beta = _extract_float(r"beta_([0-9eE.+-]+)", name)
        new_match = _close(found_lambda, lambda_value) and _close(found_scaler, beta)
        legacy_match = _close(found_lambda, beta) and _close(found_legacy_beta, lambda_value)
        if not (
            (new_match or legacy_match)
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
        matches.append((0 if new_match else 1, path))
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
    model = jp.JaxVariationalRNN(
        task=task,
        rnn_units=int(rnn_dim),
        latent_dim=int(latent_dim),
        alpha=float(alpha),
        kl_scaler=float(lambda_value),
        weight_scaler=float(beta),
        opportunity_cost=float(opportunity_cost),
        input_type=input_type,
        expansion_decision_version=expansion_decision_version,
        use_autoencoder=(normalize_variant_for_file(model_variant) != "rnn"),
        min_observations_before_stop=1 if expansion_decision_version == "decoder" else 0,
    )
    dummy = jnp.zeros((1, task.num_nodes, 1), dtype=jnp.float32)
    template = model.init(jax.random.PRNGKey(0), dummy, jax.random.PRNGKey(1), training=False, compute_losses=False)
    params = serialization.from_bytes(template, checkpoint_path.read_bytes())
    return model, params, task


def jax_output_tuple(out: jp.VAEForwardResult) -> Tuple:
    outputs = [None] * base.BASE_OUTPUT_COUNT
    outputs[0] = np.asarray(out.category_outputs)
    outputs[1] = np.asarray(out.action_output)
    outputs[2] = np.asarray(out.total_loss)
    outputs[3] = np.asarray(out.first_decoder_loss)
    outputs[4] = np.asarray(out.second_decoder_loss)
    outputs[5] = np.asarray(out.action_head_loss)
    outputs[6] = np.asarray(out.critic_loss)
    outputs[7] = np.asarray(out.information_loss)
    outputs[8] = np.asarray(out.action_loss)
    outputs[9] = np.asarray(out.reconstruction_loss)
    outputs[10] = np.asarray(out.information_cost)
    outputs[11] = np.asarray(out.z_means)
    outputs[12] = np.asarray(out.node_selections)
    outputs[13] = np.asarray(out.stop_decisions)
    outputs[14] = np.asarray(out.observed_masks)
    outputs[15] = np.asarray(out.action_outputs_sequence)
    outputs[16] = np.asarray(out.kl_d_sequence)
    outputs[17] = np.asarray(out.expansion_head_loss)
    outputs[18] = np.asarray(out.expansion_log_probs)
    outputs[19] = np.asarray(out.expansion_loss)
    outputs[20] = np.asarray(out.expansion_stop_rate)
    outputs[21] = np.asarray(out.expansion_continue_rate)
    outputs[22] = np.asarray(out.opportunity_loss)
    outputs[23] = np.asarray(out.lstm_probe_loss)
    outputs[24] = np.asarray(out.lstm_probe_accuracy)
    outputs[28] = np.asarray(out.terminal_path_output)
    outputs[29] = np.asarray(out.observation_kl_d_sequence)
    outputs[30] = np.asarray(out.lstm_state_sequence)
    outputs[31] = np.asarray(out.decoder_state_sequence)
    z_mu = np.asarray(out.z_means)
    for idx, value in enumerate(outputs):
        if value is None:
            outputs[idx] = np.zeros((z_mu.shape[0],), dtype=np.float32)
    outputs.extend([
        np.asarray(out.z_logvars),
        np.asarray(out.prior_means),
        np.asarray(out.prior_logvars),
    ])
    return tuple(outputs)


def run_jax_model_trials(model, params, rewards: np.ndarray, batch_size: int, seed: int) -> Tuple:
    batches = []
    for batch_i, start in enumerate(range(0, rewards.shape[0], batch_size)):
        batch = jnp.asarray(rewards[start:start + batch_size, :, None], dtype=jnp.float32)
        out = model.apply(
            params,
            batch,
            jax.random.PRNGKey(seed + batch_i),
            training=False,
            compute_losses=False,
        )
        batches.append(jax_output_tuple(out))
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
                rewards,
                args.batch_size,
                seed=args.analysis_seed_offset + seed + 10_000,
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

    base.plot_latent_2d_density_reward_outputs(data, outdir, args.latent_density_grid_n, args.cv_folds)
    base.run_reward_encoding_analyses(data, outdir, args.cv_folds, failures, args.n_trials, make_plots=False)
    base.run_geometry_meaning_analyses(data, outdir, args.cv_folds, args.min_within_path_n, make_plots=False)
    base.run_halfplane_reward_geometry_analysis(data, outdir, args.min_reward_group_n, make_plots=False)
    try:
        base.run_prior_centered_geometry_reward_analysis(data, outdir, args.min_reward_group_n)
    except Exception as exc:
        print(f"Plotting failed: {exc}")
    print(f"Saved JAX latent angle analysis outputs to {outdir}")


if __name__ == "__main__":
    main()
