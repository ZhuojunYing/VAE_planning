#!/usr/bin/env python3
"""Fast simplex temporal evidence weights for evidence-accumulation models.

This is the Python/NNLS counterpart of
analyses/exp_binary/plot_evidence_accumulation_simplex_weights.R.

For the primary raw-logit target it fits the equivalent model

    q_hat = bias + X @ c,  c_t >= 0

and recovers

    gain = sum(c)
    simplex_weight_t = c_t / gain
    effective_coefficient_t = c_t

This avoids repeated softmax/gain BFGS fits while preserving the same
nonnegative simplex decomposition.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.optimize import lsq_linear
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script needs scipy. Please run it in vae_env or install scipy.") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_csv_values(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x for x in re.split(r"[, \t\n]+", str(value).strip()) if x]


def as_num(value) -> float:
    try:
        return float(str(value).replace("p", "."))
    except Exception:
        return float("nan")


def numeric_values(value: str | None) -> list[float]:
    return [x for x in (as_num(v) for v in parse_csv_values(value)) if math.isfinite(x)]


def num_label(value) -> str:
    x = as_num(value)
    if not math.isfinite(x):
        return str(value)
    return f"{x:.7g}"


def value_token(value) -> str:
    return re.sub(r"(^p|p$)", "", re.sub(r"[^A-Za-z0-9]+", "p", num_label(value)))


def parameter_equal(values, target: float, tol: float = 1e-6) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.abs(arr - float(target)) <= tol


def values_match_any(values, targets: Iterable[float], unknown_ok: bool = False) -> np.ndarray:
    targets = list(targets)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not targets:
        return np.ones(arr.shape[0], dtype=bool)
    keep = np.zeros(arr.shape[0], dtype=bool)
    for target in targets:
        keep |= np.isfinite(arr) & (np.abs(arr - float(target)) <= 1e-6)
    if unknown_ok:
        keep |= ~np.isfinite(arr)
    return keep


def string_match_any(values, targets: Iterable[str], unknown_ok: bool = False) -> np.ndarray:
    targets = list(targets)
    text = pd.Series(values, dtype="object").astype(str)
    known = text.notna() & (text != "") & (text != "nan")
    if not targets:
        return np.ones(len(text), dtype=bool)
    keep = known & text.isin(targets)
    if unknown_ok:
        keep |= ~known
    return keep.to_numpy(dtype=bool)


def parse_bool(value, default=None):
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return default
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")


def option_present(argv: list[str], names: Iterable[str]) -> bool:
    names = tuple(names)
    return any(arg in names or any(arg.startswith(name + "=") for name in names) for arg in argv)


def read_presets(path: Path, task: str) -> tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    presets = pd.read_csv(path, dtype=str).fillna("")
    rows = presets[presets["task"].str.strip() == task].copy()
    memory = rows[rows["vary"].str.strip().isin(["memory_lambda", "memory", "beta"])]
    opp = rows[rows["vary"].str.strip() == "opportunity"]
    if memory.empty or opp.empty:
        raise ValueError(f"Need memory_lambda and opportunity rows for task={task} in {path}")
    return memory.iloc[0], opp.iloc[0]


def preset_value(row: pd.Series, column: str, default=None):
    if column not in row.index:
        return default
    value = str(row[column]).strip()
    return default if value == "" or value.lower() == "nan" else value


def join_csv_unique(*values) -> str | None:
    out: list[str] = []
    for value in values:
        for item in parse_csv_values(value):
            if item not in out:
                out.append(item)
    return ",".join(out) if out else None


def resolve_fixed_duration_input_dir(memory_row: pd.Series) -> str:
    direct = preset_value(
        memory_row,
        "fixed_duration_input_dir",
        preset_value(memory_row, "integration_input_dir", preset_value(memory_row, "duration_input_dir", None)),
    )
    if direct:
        return direct
    base = preset_value(memory_row, "input_dir", "outputs/jax_simulations_evi")
    candidates = [base + "_fixed_duration"]
    if base.endswith("_evi"):
        candidates.append(base[:-4] + "_evi_fixed_duration")
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "outputs/jax_simulations_evi_fixed_duration"


@dataclass
class Metadata:
    source_file: str
    source_format: str
    checkpoint: str
    loss_scale: float
    lambda_value: float
    memory_lambda: float
    choice_at_end_only: bool
    duration_mode: str
    alpha: float
    beta: float
    opportunity_cost: float
    correct_reward: float
    pay_kl_on_stop: bool
    seed: float
    observation_noise_std: float
    max_observations_before_stop: float
    rnn_units: float
    latent_dim: float
    input_type: str


def field(base: str, prefix: str) -> str | None:
    match = re.search(rf"(?:^|_){re.escape(prefix)}_([^_]+)", base)
    return match.group(1) if match else None


def parse_metadata(path: Path) -> Metadata:
    base = path.name
    checkpoint_match = re.search(r"_checkpoint_([^_]+)", base)
    checkpoint = checkpoint_match.group(1) if checkpoint_match else "final"
    loss_scale = as_num(field(base, "loss_scale"))
    legacy_lambda = as_num(field(base, "lambda"))
    if not math.isfinite(loss_scale):
        loss_scale = legacy_lambda
    beta = as_num(field(base, "beta"))
    memory_lambda = as_num(field(base, "memorylambda"))
    if not math.isfinite(memory_lambda) and math.isfinite(beta):
        memory_lambda = 1.0 / beta if beta != 0 else float("inf")
    opp = as_num(field(base, "opportunity"))
    if not math.isfinite(opp):
        opp = as_num(field(base, "opp"))
    input_type_match = re.search(r"_([^_]+)_wide\.csv$", base) or re.search(r"_([^_]+)\.csv$", base)
    input_type = input_type_match.group(1) if input_type_match else ""
    if "_policy_duration_" in base:
        duration_mode = "policy"
    elif "_fixed_duration_" in base:
        duration_mode = "fixed"
    else:
        duration_mode = "training_sim"
    return Metadata(
        source_file=str(path),
        source_format="duration_wide" if base.endswith("_wide.csv") else "training_sim",
        checkpoint=checkpoint,
        loss_scale=loss_scale,
        lambda_value=loss_scale,
        memory_lambda=memory_lambda,
        choice_at_end_only=bool(re.search(r"_observer_endchoice(?:_|$)", base)),
        duration_mode=duration_mode,
        alpha=as_num(field(base, "alpha")),
        beta=beta,
        opportunity_cost=opp,
        correct_reward=as_num(field(base, "correctreward")),
        pay_kl_on_stop=bool(re.search(r"_stop_paid(?:_|$)", base)),
        seed=as_num(field(base, "seed")),
        observation_noise_std=as_num(field(base, "obsstd")),
        max_observations_before_stop=as_num(field(base, "maxobs")),
        rnn_units=as_num(field(base, "rnn")),
        latent_dim=as_num(field(base, "latent")),
        input_type=input_type,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast evidence-accumulation simplex weights via nonnegative least squares.")
    parser.add_argument("task", nargs="?", default="evidence")
    parser.add_argument("--preset-file", default="analyses/exp_binary/evidence_accumulation_plot_presets.csv")
    parser.add_argument("--input-dir", default="outputs/jax_simulations_evi_fixed_duration")
    parser.add_argument("--use-training-simulations", "--use-regular-simulations", "--regular-simulations", action="store_true")
    parser.add_argument("--output-root", "--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vary-memory-lambda-values", "--memory-lambda-values", "--memory-lambdas", dest="memory_lambda_values", default=None)
    parser.add_argument("--vary-beta-values", "--beta-values", "--betas", dest="beta_values", default=None)
    parser.add_argument("--vary-opportunity-values", "--opportunity-values", "--opportunities", "--opportunity-costs", dest="opportunity_values", default=None)
    parser.add_argument("--fixed-opp", "--fixed-opportunity", "--fixed-opportunity-cost", dest="fixed_opp", default=None)
    parser.add_argument("--fixed-memory-lambda", "--fixed-memory-lambda-value", dest="fixed_memory_lambda", default=None)
    parser.add_argument("--fixed-beta", dest="fixed_beta", default=None)
    parser.add_argument("--fixed-coherence", "--coherence", dest="fixed_coherence", default=None)
    parser.add_argument("--pool-coherence", action="store_true")
    parser.add_argument("--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std", dest="obsstd", default=None)
    parser.add_argument("--simple-fixed-obsstd", "--simple-obsstd", "--fixed-std-simple", "--fixed-obsstd-simple", dest="simple_obsstd", default=None)
    parser.add_argument("--simple-coherence-values", "--simple-coherences", dest="simple_coherences", default=None)
    parser.add_argument("--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value", dest="loss_scale", default=None)
    parser.add_argument("--correct-reward", "--reward-scale", "--terminal-correct-reward", dest="correct_reward", default=None)
    parser.add_argument("--input-type", default=None)
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--pay-kl-on-stop", "--stop-paid", dest="pay_kl_on_stop", action="store_true", default=None)
    stop.add_argument("--no-pay-kl-on-stop", "--no-stop-paid", "--legacy-no-stop-paid", dest="pay_kl_on_stop", action="store_false")
    obs = parser.add_mutually_exclusive_group()
    obs.add_argument("--observer-only", "--choice-at-end-only", "--observer-end-choice", dest="observer_only", action="store_true", default=None)
    obs.add_argument("--non-observer", "--self-timed", "--policy-duration", dest="observer_only", action="store_false")
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--checkpoints", "--fixed-checkpoint", dest="checkpoints", default=None)
    parser.add_argument("--rnn-units", "--rnn-dims", "--rnn-dim", dest="rnn_units", default=None)
    parser.add_argument("--latent-dim", "--latent-dims", dest="latent_dim", default=None)
    parser.add_argument("--max-observations", "--max-observations-before-stop", "--maxobs", dest="max_observations", default="10")
    parser.add_argument("--comparison-mode", default="both", choices=["both", "memory_lambda", "memory", "beta", "opportunity", "checkpoint"])
    parser.add_argument("--target-type", default="auto", choices=["auto", "logit", "probability", "sampled_choice"])
    parser.add_argument("--min-trials-per-fit", type=int, default=1000)
    parser.add_argument("--num-random-starts", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--num-cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--skip-z-mu-simplex",
        action="store_true",
        help="Only fit final choice-logit simplex weights; skip per-timestep z_mu outcome fits.",
    )
    parser.add_argument(
        "--z-mu-simplex-dims",
        default=None,
        help="Comma/space separated latent dimensions to use as z_mu outcomes. Default: all z_mu dims present.",
    )
    parser.add_argument("--panel-mm", type=float, default=40.0, help="Approximate width/height of each data panel in mm.")
    parser.add_argument("--font-size", type=float, default=7.0, help="Base plot font size in points.")
    return parser


def apply_preset_defaults(args: argparse.Namespace, argv: list[str]) -> tuple[pd.Series, pd.Series]:
    memory_row, opp_row = read_presets(Path(args.preset_file), args.task)
    if not option_present(argv, ["--input-dir"]):
        args.input_dir = preset_value(memory_row, "input_dir", "outputs/jax_simulations_evi") if args.use_training_simulations else resolve_fixed_duration_input_dir(memory_row)
    if not option_present(argv, ["--output-root", "--results-dir"]):
        args.output_root = preset_value(memory_row, "results_dir", args.output_root)
    if not option_present(argv, ["--input-type"]):
        args.input_type = preset_value(memory_row, "input_type", args.input_type)
    if not option_present(argv, ["--vary-memory-lambda-values", "--memory-lambda-values", "--memory-lambdas", "--vary-beta-values", "--beta-values", "--betas"]):
        args.memory_lambda_values = preset_value(memory_row, "memory_lambda_arg", preset_value(memory_row, "beta_arg", args.memory_lambda_values))
    if not option_present(argv, ["--vary-opportunity-values", "--opportunity-values", "--opportunities", "--opportunity-costs"]):
        args.opportunity_values = preset_value(opp_row, "opportunity_arg", args.opportunity_values)
    if not option_present(argv, ["--fixed-opp", "--fixed-opportunity", "--fixed-opportunity-cost"]):
        args.fixed_opp = preset_value(memory_row, "opportunity_arg", args.fixed_opp)
    if not option_present(argv, ["--fixed-memory-lambda", "--fixed-memory-lambda-value", "--fixed-beta"]):
        args.fixed_memory_lambda = preset_value(opp_row, "memory_lambda_arg", preset_value(opp_row, "beta_arg", args.fixed_memory_lambda))
    if args.simple_obsstd and not option_present(argv, ["--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std"]):
        args.obsstd = args.simple_obsstd
    elif not option_present(argv, ["--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std"]):
        args.obsstd = preset_value(memory_row, "observation_noise_std_arg", args.obsstd)
    if not option_present(argv, ["--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value"]):
        args.loss_scale = preset_value(memory_row, "loss_scale_arg", preset_value(memory_row, "lambda_arg", args.loss_scale))
    if not option_present(argv, ["--alpha"]):
        args.alpha = preset_value(memory_row, "alpha_arg", args.alpha)
    if not option_present(argv, ["--seeds"]):
        if args.comparison_mode in {"memory_lambda", "memory", "beta"}:
            args.seeds = preset_value(memory_row, "seed_arg", None)
        elif args.comparison_mode == "opportunity":
            args.seeds = preset_value(opp_row, "seed_arg", None)
        else:
            args.seeds = join_csv_unique(preset_value(memory_row, "seed_arg", None), preset_value(opp_row, "seed_arg", None))
    if not option_present(argv, ["--rnn-units", "--rnn-dims", "--rnn-dim"]):
        args.rnn_units = preset_value(memory_row, "rnn_units_arg", args.rnn_units)
    if not option_present(argv, ["--latent-dim", "--latent-dims"]):
        args.latent_dim = preset_value(memory_row, "latent_dim_arg", args.latent_dim)
    if not option_present(argv, ["--max-observations", "--max-observations-before-stop", "--maxobs"]):
        args.max_observations = preset_value(memory_row, "max_observations_arg", args.max_observations)
    if not option_present(argv, ["--correct-reward", "--reward-scale", "--terminal-correct-reward"]):
        args.correct_reward = preset_value(memory_row, "correct_reward_arg", args.correct_reward)
    if not option_present(argv, ["--pay-kl-on-stop", "--stop-paid", "--no-pay-kl-on-stop", "--no-stop-paid", "--legacy-no-stop-paid"]):
        args.pay_kl_on_stop = parse_bool(preset_value(memory_row, "pay_kl_on_stop_arg", None), args.pay_kl_on_stop)
    if not option_present(argv, ["--observer-only", "--choice-at-end-only", "--observer-end-choice", "--non-observer", "--self-timed", "--policy-duration"]):
        args.observer_only = parse_bool(preset_value(memory_row, "observer_only_arg", None), args.observer_only)
    return memory_row, opp_row


def seed_values_for_family(args, memory_row, opp_row, family: str, explicit_seeds: bool) -> list[float]:
    if explicit_seeds:
        return numeric_values(args.seeds)
    if family == "beta":
        return numeric_values(preset_value(memory_row, "seed_arg", None))
    if family == "opportunity":
        return numeric_values(preset_value(opp_row, "seed_arg", None))
    return numeric_values(args.seeds)


def list_input_files(args) -> list[Path]:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    pattern = "*_evidence.csv" if args.use_training_simulations else "*_wide.csv"
    files = sorted(input_dir.glob(pattern))
    if not args.use_training_simulations:
        files = [p for p in files if re.search(r"_(fixed|policy)_duration_\d+_checkpoint_.*_wide\.csv$", p.name)]
    if not files:
        kind = "regular evidence simulation" if args.use_training_simulations else "duration-controlled wide"
        raise FileNotFoundError(f"No {kind} CSVs found in {input_dir}.")
    return files


def metadata_frame(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        meta = parse_metadata(path)
        row = meta.__dict__.copy()
        row["lambda"] = row.pop("lambda_value")
        info = path.stat()
        row["file_size"] = info.st_size
        row["file_mtime"] = info.st_mtime
        rows.append(row)
    return pd.DataFrame(rows)


def prefilter_manifest(manifest: pd.DataFrame, args, memory_row, opp_row, explicit_seeds: bool) -> pd.DataFrame:
    keep = manifest["file_size"].to_numpy(dtype=float) > 0
    for col, raw in [
        ("loss_scale", args.loss_scale),
        ("alpha", args.alpha),
        ("seed", args.seeds),
        ("rnn_units", args.rnn_units),
        ("latent_dim", args.latent_dim),
        ("max_observations_before_stop", args.max_observations),
        ("observation_noise_std", args.obsstd),
        ("correct_reward", args.correct_reward),
    ]:
        vals = numeric_values(raw)
        if vals:
            keep &= values_match_any(manifest[col], vals, unknown_ok=True)
    inputs = parse_csv_values(args.input_type)
    if inputs:
        keep &= string_match_any(manifest["input_type"], inputs, unknown_ok=True)
    if args.pay_kl_on_stop is not None:
        keep &= manifest["pay_kl_on_stop"].isna().to_numpy() | (manifest["pay_kl_on_stop"].astype(bool).to_numpy() == bool(args.pay_kl_on_stop))
    if args.observer_only is not None:
        keep &= manifest["choice_at_end_only"].isna().to_numpy() | (manifest["choice_at_end_only"].astype(bool).to_numpy() == bool(args.observer_only))
    checkpoints = parse_csv_values(args.checkpoints)
    if checkpoints:
        keep &= string_match_any(manifest["checkpoint"], checkpoints, unknown_ok=True)

    memory_values = numeric_values(args.memory_lambda_values) or numeric_values(args.beta_values)
    opp_values = numeric_values(args.opportunity_values)
    fixed_opp = numeric_values(args.fixed_opp)
    fixed_memory = numeric_values(args.fixed_memory_lambda) or numeric_values(args.fixed_beta)
    family_filters = []
    if memory_values:
        filt = values_match_any(manifest["memory_lambda"], memory_values, unknown_ok=True)
        if fixed_opp:
            filt &= values_match_any(manifest["opportunity_cost"], fixed_opp, unknown_ok=True)
        seed_vals = seed_values_for_family(args, memory_row, opp_row, "beta", explicit_seeds)
        if seed_vals:
            filt &= values_match_any(manifest["seed"], seed_vals, unknown_ok=True)
        family_filters.append(filt)
    if opp_values:
        filt = values_match_any(manifest["opportunity_cost"], opp_values, unknown_ok=True)
        if fixed_memory:
            filt &= values_match_any(manifest["memory_lambda"], fixed_memory, unknown_ok=True)
        seed_vals = seed_values_for_family(args, memory_row, opp_row, "opportunity", explicit_seeds)
        if seed_vals:
            filt &= values_match_any(manifest["seed"], seed_vals, unknown_ok=True)
        family_filters.append(filt)
    if family_filters:
        keep &= np.logical_or.reduce(family_filters)
    return manifest.loc[keep].copy()


def fill_metadata_columns(dat: pd.DataFrame, meta: pd.Series) -> pd.DataFrame:
    for col, value in meta.items():
        if col in {"source_file", "file_size", "file_mtime"}:
            continue
        if col not in dat.columns or pd.to_numeric(dat[col], errors="coerce").notna().sum() == 0:
            dat[col] = value
    if "lambda" not in dat.columns or pd.to_numeric(dat["lambda"], errors="coerce").notna().sum() == 0:
        dat["lambda"] = dat["loss_scale"]
    if "trial_id" not in dat.columns:
        dat["trial_id"] = dat["graph"] if "graph" in dat.columns else np.arange(len(dat))
    if "run_id" not in dat.columns:
        dat["run_id"] = Path(str(meta["source_file"])).stem
    if "checkpoint" not in dat.columns:
        dat["checkpoint"] = meta["checkpoint"]
    if "training_step" not in dat.columns:
        dat["training_step"] = -1
    return dat


def load_trials(manifest: pd.DataFrame, max_obs: int) -> pd.DataFrame:
    obs_cols = [f"observation_{i}" for i in range(1, max_obs + 1)]
    sample_cols = [f"evidence_sample_t{i}" for i in range(1, max_obs + 1)]
    target_cols = []
    for t in range(1, max_obs + 1):
        target_cols.extend([
            f"raw_logit_choose_a_t{t}",
            f"raw_logit_choose_b_t{t}",
            f"choice_logit_t{t}",
            f"p_choose_b_given_terminal_t{t}",
            f"policy_choose_a_t{t}",
            f"policy_choose_b_t{t}",
            f"valid_t{t}",
        ])
    base_cols = [
        "graph",
        "trial_id",
        "run_id",
        "seed",
        "checkpoint",
        "training_step",
        "loss_scale",
        "lambda",
        "memory_lambda",
        "choice_at_end_only",
        "duration_mode",
        "alpha",
        "beta",
        "opportunity_cost",
        "coherence",
        "signed_coherence",
        "observation_noise_std",
        "correct_choice",
        "terminal_action",
        "choose_right",
        "correct",
        "num_observations",
        "correct_reward",
        "pay_kl_on_stop",
        "input_type",
    ]
    loaded = []
    for _, meta in manifest.iterrows():
        path = Path(meta["source_file"])
        cols = list(pd.read_csv(path, nrows=0).columns)
        z_mu_cols = [c for c in cols if re.match(r"^z_mu_\d+_t\d+$", c)]
        usecols = [c for c in base_cols + obs_cols + sample_cols + target_cols + z_mu_cols if c in cols]
        has_obs = all(c in cols for c in obs_cols)
        has_samples = all(c in cols for c in sample_cols)
        if not has_obs and not has_samples:
            print(f"Skipping {path.name}: no complete observation/evidence_sample columns.", file=sys.stderr)
            continue
        dat = pd.read_csv(path, usecols=usecols)
        if not has_obs and has_samples:
            for i in range(1, max_obs + 1):
                dat[f"observation_{i}"] = dat[f"evidence_sample_t{i}"]
        dat = fill_metadata_columns(dat, meta)
        dat["source_file"] = str(path)
        loaded.append(dat)
    if not loaded:
        raise RuntimeError("No usable trial rows were loaded.")
    return pd.concat(loaded, ignore_index=True, sort=False)


def filter_trials(dat: pd.DataFrame, args, memory_row, opp_row, explicit_seeds: bool, max_obs: int) -> pd.DataFrame:
    for col, raw, label in [
        ("loss_scale", args.loss_scale, "loss_scale"),
        ("alpha", args.alpha, "alpha"),
        ("seed", args.seeds, "seed"),
        ("rnn_units", args.rnn_units, "rnn_units"),
        ("latent_dim", args.latent_dim, "latent_dim"),
        ("correct_reward", args.correct_reward, "correct_reward"),
    ]:
        vals = numeric_values(raw)
        if vals and col in dat.columns:
            dat = dat.loc[values_match_any(dat[col], vals)].copy()
            print(f"Filter {label}={','.join(map(num_label, vals))} kept {len(dat)} trial(s).")
    if args.input_type and "input_type" in dat.columns:
        inputs = parse_csv_values(args.input_type)
        if inputs:
            dat = dat.loc[string_match_any(dat["input_type"], inputs)].copy()
            print(f"Filter input_type={','.join(inputs)} kept {len(dat)} trial(s).")
    if args.pay_kl_on_stop is not None:
        dat = dat.loc[dat["pay_kl_on_stop"].astype(bool) == bool(args.pay_kl_on_stop)].copy()
    if args.observer_only is not None:
        dat = dat.loc[dat["choice_at_end_only"].astype(bool) == bool(args.observer_only)].copy()
    if args.checkpoints:
        checkpoints = parse_csv_values(args.checkpoints)
        dat = dat.loc[dat["checkpoint"].astype(str).isin(checkpoints)].copy()
    dat["coherence_magnitude"] = pd.to_numeric(dat["coherence"], errors="coerce").abs()
    simple_cohs = numeric_values(args.simple_coherences)
    if simple_cohs:
        available = np.sort(dat["coherence_magnitude"].dropna().unique())
        snapped = []
        for requested in simple_cohs:
            if len(available) == 0:
                continue
            nearest = available[np.argmin(np.abs(available - requested))]
            if abs(nearest - requested) <= 1e-5:
                snapped.append(float(nearest))
            else:
                print(f"Warning: requested coherence {requested} unavailable; available={available}", file=sys.stderr)
        if not snapped:
            raise RuntimeError("No requested --simple-coherence-values were found.")
        dat = dat.loc[values_match_any(dat["coherence_magnitude"], snapped)].copy()
        print(f"Using simple coherence magnitudes {','.join(map(num_label, snapped))}; {len(dat)} trial(s) remain.")
    elif not args.pool_coherence:
        selected = as_num(args.fixed_coherence) if args.fixed_coherence else float(dat.loc[dat["coherence_magnitude"] > 0, "coherence_magnitude"].mode().iloc[0])
        dat = dat.loc[parameter_equal(dat["coherence_magnitude"], selected)].copy()
        print(f"Using fixed coherence magnitude {num_label(selected)}; {len(dat)} trial(s) remain.")

    memory_values = numeric_values(args.memory_lambda_values) or numeric_values(args.beta_values)
    if not memory_values:
        fixed_opp = as_num(args.fixed_opp if args.fixed_opp is not None else 0)
        memory_values = sorted(dat.loc[parameter_equal(dat["opportunity_cost"], fixed_opp), "memory_lambda"].dropna().unique())
    opp_values = numeric_values(args.opportunity_values)
    fixed_memory = as_num(args.fixed_memory_lambda if args.fixed_memory_lambda is not None else args.fixed_beta)
    if not math.isfinite(fixed_memory):
        fixed_memory = float(dat["memory_lambda"].mode().iloc[0])
    if not opp_values:
        opp_values = sorted(dat.loc[parameter_equal(dat["memory_lambda"], fixed_memory), "opportunity_cost"].dropna().unique())
    fixed_opp = as_num(args.fixed_opp if args.fixed_opp is not None else 0)

    keep = np.zeros(len(dat), dtype=bool)
    if args.comparison_mode in {"both", "memory_lambda", "memory", "beta"}:
        seed_vals = seed_values_for_family(args, memory_row, opp_row, "beta", explicit_seeds)
        mem_keep = values_match_any(dat["memory_lambda"], memory_values) & parameter_equal(dat["opportunity_cost"], fixed_opp)
        if seed_vals:
            mem_keep &= values_match_any(dat["seed"], seed_vals)
        keep |= mem_keep
    if args.comparison_mode in {"both", "opportunity"}:
        seed_vals = seed_values_for_family(args, memory_row, opp_row, "opportunity", explicit_seeds)
        opp_keep = values_match_any(dat["opportunity_cost"], opp_values) & parameter_equal(dat["memory_lambda"], fixed_memory)
        if seed_vals:
            opp_keep &= values_match_any(dat["seed"], seed_vals)
        keep |= opp_keep
    dat = dat.loc[keep].copy()

    obs_cols = [f"observation_{i}" for i in range(1, max_obs + 1)]
    obs = dat[obs_cols].apply(pd.to_numeric, errors="coerce")
    if "choice_at_end_only" in dat.columns:
        observer_mask = dat["choice_at_end_only"].astype(bool).to_numpy()
    else:
        observer_mask = np.full(len(dat), bool(args.observer_only), dtype=bool)
    if "num_observations" in dat.columns:
        duration = pd.to_numeric(dat["num_observations"], errors="coerce").to_numpy(dtype=float)
    else:
        duration = np.full(len(dat), float(max_obs))
    duration = np.where(observer_mask, float(max_obs), duration)
    duration_int = np.rint(duration).astype(float)
    valid = np.isfinite(duration_int) & (duration_int >= 1) & (duration_int <= max_obs)
    dat["target_duration"] = duration_int
    dat["target_value"] = np.nan
    dat["target_type"] = "logit"
    for duration_value in range(1, max_obs + 1):
        rows = valid & (duration_int == duration_value)
        if not np.any(rows):
            continue
        rows &= obs.iloc[:, :duration_value].notna().all(axis=1).to_numpy(dtype=bool)
        valid_col = f"valid_t{duration_value}"
        if valid_col in dat.columns:
            rows &= dat[valid_col].astype(bool).to_numpy()
        a_col = f"raw_logit_choose_a_t{duration_value}"
        b_col = f"raw_logit_choose_b_t{duration_value}"
        if a_col not in dat.columns or b_col not in dat.columns:
            raise RuntimeError(f"Raw A/B logits at timestep {duration_value} are required for duration-aware simplex fits.")
        target = (
            pd.to_numeric(dat.loc[rows, b_col], errors="coerce")
            - pd.to_numeric(dat.loc[rows, a_col], errors="coerce")
        )
        dat.loc[rows, "target_value"] = target.to_numpy(dtype=float)
    valid &= np.isfinite(pd.to_numeric(dat["target_value"], errors="coerce").to_numpy(dtype=float))
    dat = dat.loc[valid].copy()
    dat["target_duration"] = dat["target_duration"].astype(int)
    duration_counts = dat["target_duration"].value_counts().sort_index()
    duration_text = ",".join(f"{int(k)}:{int(v)}" for k, v in duration_counts.items())
    print(f"Included {len(dat)} valid trial(s) after duration-aware filtering; target_duration counts: {duration_text}.")
    return dat


def fit_nonnegative_logit(X: np.ndarray, y: np.ndarray):
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    res = lsq_linear(Xc, yc, bounds=(0, np.inf), tol=1e-10, max_iter=200)
    coef = np.maximum(res.x, 0)
    bias = y_mean - float(x_mean @ coef)
    pred = bias + X @ coef
    gain = float(coef.sum())
    weights = coef / gain if gain > 0 else np.full(X.shape[1], 1.0 / X.shape[1])
    return coef, weights, gain, bias, pred, res


def fit_unconstrained(X: np.ndarray, y: np.ndarray):
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coef
    return coef[1:], pred


def metrics(y, pred):
    err = y - pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else np.nan
    return mse, rmse, mae, r2


def fit_group(dat: pd.DataFrame, obs_cols: list[str], max_obs: int, cv_folds: int, seed: int, group_index: int):
    X = dat[obs_cols].to_numpy(dtype=float)
    y = dat["target_value"].to_numpy(dtype=float)
    coef, weights, gain, bias, pred, res = fit_nonnegative_logit(X, y)
    mse, rmse, mae, r2 = metrics(y, pred)
    uncon_coef, uncon_pred = fit_unconstrained(X, y)
    uncon_mse, uncon_rmse, _, uncon_r2 = metrics(y, uncon_pred)

    cv_r2 = np.nan
    cv_rmse = np.nan
    cv_loss = np.nan
    if cv_folds and cv_folds > 1 and len(dat) >= cv_folds:
        rng = np.random.default_rng(seed + group_index * 1009)
        fold_id = np.resize(np.arange(cv_folds), len(dat))
        rng.shuffle(fold_id)
        pred_cv = np.full(len(dat), np.nan)
        for fold in range(cv_folds):
            train = fold_id != fold
            test = fold_id == fold
            c, _, _, b, _, _ = fit_nonnegative_logit(X[train], y[train])
            pred_cv[test] = b + X[test] @ c
        ok = np.isfinite(pred_cv)
        cv_loss, cv_rmse, _, cv_r2 = metrics(y[ok], pred_cv[ok])

    base = dat.iloc[0].to_dict()
    target_duration = int(base.get("target_duration", max_obs))
    run_rows = []
    for pos in range(max_obs):
        row = {k: base.get(k) for k in [
            "run_id", "checkpoint", "training_step", "seed", "loss_scale", "memory_lambda",
            "choice_at_end_only", "alpha", "beta", "opportunity_cost", "observation_noise_std",
            "coherence_magnitude", "target_duration"
        ]}
        bump_contrast = np.nan
        effective_bump_contrast = np.nan
        if max_obs >= 8:
            bump_contrast = float(weights[3:7].mean() - weights[[0, 1, max_obs - 2, max_obs - 1]].mean())
            effective_bump_contrast = float(coef[3:7].mean() - coef[[0, 1, max_obs - 2, max_obs - 1]].mean())
        row.update(
            target_type="logit",
            observation_position=pos + 1,
            simplex_weight=float(weights[pos]),
            gain=gain,
            bias=bias,
            effective_coefficient=float(coef[pos]),
            unconstrained_coefficient=float(uncon_coef[pos]),
            n_trials=len(dat),
            objective_value=mse,
            converged=bool(res.success),
            optimizer_code=int(res.status),
            optimizer_message=str(res.message),
            num_iterations=int(res.nit),
            num_random_starts=1,
            full_data_R2=r2,
            full_data_RMSE=rmse,
            full_data_MAE=mae,
            full_data_loss=mse,
            cross_validated_R2=cv_r2,
            cross_validated_RMSE=cv_rmse,
            cross_validated_loss=cv_loss,
            unconstrained_R2=uncon_r2,
            unconstrained_RMSE=uncon_rmse,
            negative_unconstrained_coefficients=int(np.sum(uncon_coef < 0)),
            bump_contrast=bump_contrast,
            effective_bump_contrast=effective_bump_contrast,
            skipped_reason=np.nan,
        )
        run_rows.append(row)
    metric = run_rows[0].copy()
    metric.pop("observation_position", None)
    metric.pop("simplex_weight", None)
    metric.pop("effective_coefficient", None)
    metric.pop("unconstrained_coefficient", None)
    return run_rows, metric


def available_z_mu_dims(columns: Iterable[str], requested: str | None = None) -> list[int]:
    present = sorted({
        int(match.group(1))
        for col in columns
        for match in [re.match(r"^z_mu_(\d+)_t\d+$", str(col))]
        if match
    })
    if not present:
        return []
    requested_dims = [int(x) for x in numeric_values(requested)] if requested else []
    if not requested_dims:
        return present
    missing = sorted(set(requested_dims) - set(present))
    if missing:
        print(f"Warning: requested z_mu dim(s) not present and will be skipped: {missing}", file=sys.stderr)
    return [d for d in requested_dims if d in present]


def select_oriented_fit(X: np.ndarray, y_raw: np.ndarray):
    candidates = []
    for sign in (1.0, -1.0):
        y = sign * y_raw
        coef, weights, gain, bias, pred, res = fit_nonnegative_logit(X, y)
        mse, rmse, mae, r2 = metrics(y, pred)
        candidates.append((mse, sign, coef, weights, gain, bias, pred, res, rmse, mae, r2))
    return min(candidates, key=lambda item: item[0])


def fit_group_z_mu(
    dat: pd.DataFrame,
    max_obs: int,
    dims: list[int],
    cv_folds: int,
    seed: int,
    group_index: int,
    min_trials: int,
):
    all_obs_cols = [f"observation_{i}" for i in range(1, max_obs + 1)]
    base = dat.iloc[0].to_dict()
    target_duration = int(base.get("target_duration", max_obs))
    run_rows = []
    metric_rows = []
    for dim in dims:
        for target_step in range(1, target_duration + 1):
            target_col = f"z_mu_{dim}_t{target_step}"
            if target_col not in dat.columns:
                continue
            obs_cols = all_obs_cols[:target_step]
            fit_dat = dat[obs_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
            valid = fit_dat.notna().all(axis=1).to_numpy(dtype=bool)
            n_valid = int(valid.sum())
            metric_base = {k: base.get(k) for k in [
                "run_id", "checkpoint", "training_step", "seed", "loss_scale", "memory_lambda",
                "choice_at_end_only", "alpha", "beta", "opportunity_cost", "observation_noise_std",
                "coherence_magnitude", "target_duration"
            ]}
            metric_base.update(
                target_type="z_mu",
                latent_dim_index=dim,
                target_timestep=target_step,
                n_trials=n_valid,
            )
            if n_valid < min_trials:
                metric_rows.append({**metric_base, "skipped_reason": f"too_few_trials:{n_valid}"})
                continue
            X = fit_dat.loc[valid, obs_cols].to_numpy(dtype=float)
            y_raw = fit_dat.loc[valid, target_col].to_numpy(dtype=float)
            mse, sign, coef, weights, gain, bias, pred, res, rmse, mae, r2 = select_oriented_fit(X, y_raw)
            y = sign * y_raw
            uncon_coef, uncon_pred = fit_unconstrained(X, y)
            uncon_mse, uncon_rmse, _, uncon_r2 = metrics(y, uncon_pred)

            cv_r2 = np.nan
            cv_rmse = np.nan
            cv_loss = np.nan
            if cv_folds and cv_folds > 1 and n_valid >= cv_folds:
                rng = np.random.default_rng(seed + group_index * 1009 + dim * 9176 + target_step * 101)
                fold_id = np.resize(np.arange(cv_folds), n_valid)
                rng.shuffle(fold_id)
                pred_cv = np.full(n_valid, np.nan)
                for fold in range(cv_folds):
                    train = fold_id != fold
                    test = fold_id == fold
                    _, sign_cv, c, _, _, b, _, _, _, _, _ = select_oriented_fit(X[train], y_raw[train])
                    pred_cv[test] = sign_cv * (b + X[test] @ c)
                ok = np.isfinite(pred_cv)
                cv_loss, cv_rmse, _, cv_r2 = metrics(y_raw[ok], pred_cv[ok])

            for pos in range(target_step):
                row = metric_base.copy()
                row.update(
                    observation_position=pos + 1,
                    target_sign=sign,
                    simplex_weight=float(weights[pos]),
                    gain=gain,
                    bias=bias,
                    effective_coefficient=float(coef[pos]),
                    unconstrained_coefficient=float(uncon_coef[pos]),
                    objective_value=mse,
                    converged=bool(res.success),
                    optimizer_code=int(res.status),
                    optimizer_message=str(res.message),
                    num_iterations=int(res.nit),
                    num_random_starts=1,
                    full_data_R2=r2,
                    full_data_RMSE=rmse,
                    full_data_MAE=mae,
                    full_data_loss=mse,
                    cross_validated_R2=cv_r2,
                    cross_validated_RMSE=cv_rmse,
                    cross_validated_loss=cv_loss,
                    unconstrained_R2=uncon_r2,
                    unconstrained_RMSE=uncon_rmse,
                    negative_unconstrained_coefficients=int(np.sum(uncon_coef < 0)),
                    bump_contrast=np.nan,
                    effective_bump_contrast=np.nan,
                    skipped_reason=np.nan,
                )
                run_rows.append(row)
            metric = metric_base.copy()
            metric.update(
                target_sign=sign,
                gain=gain,
                bias=bias,
                objective_value=mse,
                converged=bool(res.success),
                optimizer_code=int(res.status),
                optimizer_message=str(res.message),
                num_iterations=int(res.nit),
                num_random_starts=1,
                full_data_R2=r2,
                full_data_RMSE=rmse,
                full_data_MAE=mae,
                full_data_loss=mse,
                cross_validated_R2=cv_r2,
                cross_validated_RMSE=cv_rmse,
                cross_validated_loss=cv_loss,
                unconstrained_R2=uncon_r2,
                unconstrained_RMSE=uncon_rmse,
                negative_unconstrained_coefficients=int(np.sum(uncon_coef < 0)),
                skipped_reason=np.nan,
            )
            metric_rows.append(metric)
    return run_rows, metric_rows


def add_family_rows(dat: pd.DataFrame, args, memory_row, opp_row, explicit_seeds: bool) -> pd.DataFrame:
    rows = []
    fixed_opp = as_num(args.fixed_opp if args.fixed_opp is not None else 0)
    fixed_memory = as_num(args.fixed_memory_lambda if args.fixed_memory_lambda is not None else args.fixed_beta)
    memory_values = numeric_values(args.memory_lambda_values) or numeric_values(args.beta_values)
    opportunity_values = numeric_values(args.opportunity_values)
    if args.comparison_mode in {"both", "memory_lambda", "memory", "beta"}:
        keep = parameter_equal(dat["opportunity_cost"], fixed_opp)
        if memory_values:
            keep &= values_match_any(dat["memory_lambda"], memory_values)
        seeds = seed_values_for_family(args, memory_row, opp_row, "beta", explicit_seeds)
        if seeds:
            keep &= values_match_any(dat["seed"], seeds)
        tmp = dat.loc[keep].copy()
        if not tmp.empty:
            tmp["family"] = "beta"
            tmp["parameter_value"] = pd.to_numeric(tmp["memory_lambda"], errors="coerce")
            rows.append(tmp)
    if args.comparison_mode in {"both", "opportunity"}:
        keep = parameter_equal(dat["memory_lambda"], fixed_memory)
        if opportunity_values:
            keep &= values_match_any(dat["opportunity_cost"], opportunity_values)
        seeds = seed_values_for_family(args, memory_row, opp_row, "opportunity", explicit_seeds)
        if seeds:
            keep &= values_match_any(dat["seed"], seeds)
        tmp = dat.loc[keep].copy()
        if not tmp.empty:
            tmp["family"] = "opportunity"
            tmp["parameter_value"] = pd.to_numeric(tmp["opportunity_cost"], errors="coerce")
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else dat.iloc[0:0].copy()


def sem(series):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def summarize(run_level_family: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "family", "parameter_value", "checkpoint", "training_step", "memory_lambda", "beta",
        "opportunity_cost", "coherence_magnitude", "observation_noise_std", "target_duration", "target_type",
        "observation_position",
    ]
    rows = []
    for keys, grp in run_level_family.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            n_runs=grp["run_id"].nunique(),
            mean_simplex_weight=grp["simplex_weight"].mean(),
            sd_simplex_weight=grp["simplex_weight"].std(ddof=1),
            se_simplex_weight=sem(grp["simplex_weight"]),
            mean_gain=grp["gain"].mean(),
            sd_gain=grp["gain"].std(ddof=1),
            se_gain=sem(grp["gain"]),
            mean_effective_coefficient=grp["effective_coefficient"].mean(),
            sd_effective_coefficient=grp["effective_coefficient"].std(ddof=1),
            se_effective_coefficient=sem(grp["effective_coefficient"]),
            mean_unconstrained_coefficient=grp["unconstrained_coefficient"].mean(),
            sd_unconstrained_coefficient=grp["unconstrained_coefficient"].std(ddof=1),
            se_unconstrained_coefficient=sem(grp["unconstrained_coefficient"]),
            mean_cross_validated_R2=grp["cross_validated_R2"].mean(),
            se_cross_validated_R2=sem(grp["cross_validated_R2"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_z_mu(run_level_family: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "family", "parameter_value", "checkpoint", "training_step", "memory_lambda", "beta",
        "opportunity_cost", "coherence_magnitude", "observation_noise_std", "target_duration", "target_type",
        "latent_dim_index", "target_timestep", "observation_position",
    ]
    rows = []
    if run_level_family.empty:
        return pd.DataFrame()
    for keys, grp in run_level_family.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            n_runs=grp["run_id"].nunique(),
            n_fits=len(grp),
            mean_target_sign=grp["target_sign"].mean() if "target_sign" in grp.columns else np.nan,
            mean_simplex_weight=grp["simplex_weight"].mean(),
            sd_simplex_weight=grp["simplex_weight"].std(ddof=1),
            se_simplex_weight=sem(grp["simplex_weight"]),
            mean_gain=grp["gain"].mean(),
            sd_gain=grp["gain"].std(ddof=1),
            se_gain=sem(grp["gain"]),
            mean_effective_coefficient=grp["effective_coefficient"].mean(),
            sd_effective_coefficient=grp["effective_coefficient"].std(ddof=1),
            se_effective_coefficient=sem(grp["effective_coefficient"]),
            mean_unconstrained_coefficient=grp["unconstrained_coefficient"].mean(),
            sd_unconstrained_coefficient=grp["unconstrained_coefficient"].std(ddof=1),
            se_unconstrained_coefficient=sem(grp["unconstrained_coefficient"]),
            mean_full_data_R2=grp["full_data_R2"].mean(),
            se_full_data_R2=sem(grp["full_data_R2"]),
            mean_cross_validated_R2=grp["cross_validated_R2"].mean(),
            se_cross_validated_R2=sem(grp["cross_validated_R2"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_z_mu_lag(run_level_family: pd.DataFrame) -> pd.DataFrame:
    if run_level_family.empty:
        return pd.DataFrame()
    dat = run_level_family.copy()
    dat["timestep_until_current"] = (
        pd.to_numeric(dat["target_timestep"], errors="coerce")
        - pd.to_numeric(dat["observation_position"], errors="coerce")
    )
    dat["timestep_relative_to_current"] = -dat["timestep_until_current"]
    group_cols = [
        "family", "parameter_value", "checkpoint", "training_step", "memory_lambda", "beta",
        "opportunity_cost", "coherence_magnitude", "observation_noise_std", "target_duration", "target_type",
        "latent_dim_index", "timestep_until_current", "timestep_relative_to_current",
    ]
    rows = []
    for keys, grp in dat.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            n_runs=grp["run_id"].nunique(),
            n_fits=len(grp),
            min_target_timestep=pd.to_numeric(grp["target_timestep"], errors="coerce").min(),
            max_target_timestep=pd.to_numeric(grp["target_timestep"], errors="coerce").max(),
            mean_simplex_weight=grp["simplex_weight"].mean(),
            sd_simplex_weight=grp["simplex_weight"].std(ddof=1),
            se_simplex_weight=sem(grp["simplex_weight"]),
            mean_effective_coefficient=grp["effective_coefficient"].mean(),
            sd_effective_coefficient=grp["effective_coefficient"].std(ddof=1),
            se_effective_coefficient=sem(grp["effective_coefficient"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_z_mu_by_target_timestep_coherence_collapsed(run_level_family: pd.DataFrame) -> pd.DataFrame:
    if run_level_family.empty:
        return pd.DataFrame()
    dat = run_level_family.copy()
    dat["timestep_relative_to_current"] = (
        pd.to_numeric(dat["observation_position"], errors="coerce")
        - pd.to_numeric(dat["target_timestep"], errors="coerce")
    )
    group_cols = [
        "family", "parameter_value", "checkpoint", "training_step", "memory_lambda", "beta",
        "opportunity_cost", "observation_noise_std", "target_duration", "target_type", "latent_dim_index",
        "target_timestep", "observation_position", "timestep_relative_to_current",
    ]
    rows = []
    for keys, grp in dat.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            n_runs=grp["run_id"].nunique(),
            n_fits=len(grp),
            n_coherence_values=pd.to_numeric(grp["coherence_magnitude"], errors="coerce").nunique(),
            mean_simplex_weight=grp["simplex_weight"].mean(),
            sd_simplex_weight=grp["simplex_weight"].std(ddof=1),
            se_simplex_weight=sem(grp["simplex_weight"]),
            mean_effective_coefficient=grp["effective_coefficient"].mean(),
            sd_effective_coefficient=grp["effective_coefficient"].std(ddof=1),
            se_effective_coefficient=sem(grp["effective_coefficient"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_coherence_collapsed(run_level_family: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "family", "parameter_value", "checkpoint", "training_step", "memory_lambda", "beta",
        "opportunity_cost", "observation_noise_std", "target_duration", "target_type", "observation_position",
    ]
    rows = []
    for keys, grp in run_level_family.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            n_runs=grp["run_id"].nunique(),
            n_fits=len(grp),
            mean_simplex_weight=grp["simplex_weight"].mean(),
            sd_simplex_weight=grp["simplex_weight"].std(ddof=1),
            se_simplex_weight=sem(grp["simplex_weight"]),
            mean_effective_coefficient=grp["effective_coefficient"].mean(),
            sd_effective_coefficient=grp["effective_coefficient"].std(ddof=1),
            se_effective_coefficient=sem(grp["effective_coefficient"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def color_map(family: str, params: list[float]) -> dict[float, str]:
    params = sorted({float(p) for p in params if math.isfinite(float(p))})
    if not params:
        return {}
    colors = ["#74c476", "#238b45", "#00441b"] if family == "beta" else ["#6baed6", "#2171b5", "#08306b"]
    cmap = LinearSegmentedColormap.from_list(f"{family}_palette", colors, N=max(len(params), 2))
    return {p: cmap(i / max(len(params) - 1, 1)) for i, p in enumerate(params)}


def safe_ylim(values, fallback=(0, 1)):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return fallback
    lo, hi = float(vals.min()), float(vals.max())
    if abs(hi - lo) < 1e-12:
        lo -= 0.5
        hi += 0.5
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def duration_suffix(duration_value: float, max_obs: int, duration_levels: Iterable[float]) -> str:
    levels = [float(x) for x in duration_levels if math.isfinite(float(x))]
    if len(levels) <= 1 and levels and int(round(levels[0])) == int(max_obs):
        return ""
    if not math.isfinite(float(duration_value)):
        return ""
    return f"_totalobs_{int(round(float(duration_value)))}"


def plot_coherence_collapsed_profiles(
    collapsed: pd.DataFrame,
    output_dir: Path,
    args,
    max_obs: int,
    obsstd_levels: list[float],
):
    if collapsed.empty:
        return
    panel_in = float(args.panel_mm) / 25.4
    profile_specs = [
        ("relative_temporal_weights", "mean_simplex_weight", "se_simplex_weight", "Relative\ntemporal weight", 1 / max_obs, (0, 0.2)),
        ("effective_evidence_coefficients", "mean_effective_coefficient", "se_effective_coefficient", "Effective\nevidence coefficient", None, (0, 1)),
    ]
    duration_levels = sorted(pd.to_numeric(collapsed.get("target_duration", pd.Series([max_obs])), errors="coerce").dropna().unique())
    for obsstd in obsstd_levels:
        odat = collapsed[np.isclose(pd.to_numeric(collapsed["observation_noise_std"], errors="coerce"), obsstd)]
        for duration_value in duration_levels:
            sdat = odat[np.isclose(pd.to_numeric(odat["target_duration"], errors="coerce"), duration_value)]
            families = [f for f in ["beta", "opportunity"] if f in set(sdat["family"])]
            if not families:
                continue
            fit_duration = int(round(float(duration_value)))
            for slug, y_col, se_col, ylab, ref, fallback in profile_specs:
                vals = pd.to_numeric(sdat[y_col], errors="coerce")
                se = pd.to_numeric(sdat[se_col], errors="coerce")
                ylim = safe_ylim(pd.concat([vals, vals - se, vals + se]), fallback)
                fig, axes = plt.subplots(1, len(families), figsize=(0.45 + len(families) * panel_in, 0.42 + panel_in), squeeze=False)
                for c, family in enumerate(families):
                    ax = axes[0, c]
                    params = sorted(pd.to_numeric(sdat.loc[sdat["family"] == family, "parameter_value"], errors="coerce").dropna().unique())
                    colors = color_map(family, params)
                    ax.set_ylim(*ylim)
                    pdat = sdat[sdat["family"] == family]
                    for param in params:
                        line = pdat[np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param)].sort_values("observation_position")
                        if line.empty:
                            continue
                        x = line["observation_position"].to_numpy(dtype=float)
                        y = line[y_col].to_numpy(dtype=float)
                        ax.plot(x, y, color=colors[param], lw=1.1, marker="o" if family == "beta" else "^", ms=2.5, label=num_label(param))
                        if line["n_fits"].max() > 1:
                            err = line[se_col].to_numpy(dtype=float)
                            ax.errorbar(x, y, yerr=err, color=colors[param], lw=0, elinewidth=0.6, capsize=1.5)
                    ax.set_xlim(0.5, fit_duration + 0.5)
                    ax.set_xticks(range(1, fit_duration + 1))
                    if ref is not None:
                        ax.axhline(1.0 / fit_duration if slug == "relative_temporal_weights" else ref, color="0.75", ls="--", lw=0.7)
                    ax.set_xlabel("Observation\nposition")
                    ax.set_title("Varying memory lambda" if family == "beta" else "Varying opportunity")
                    ax.set_ylabel(ylab if c == 0 else "")
                suffix = f"{duration_suffix(duration_value, max_obs, duration_levels)}_obsstd_{value_token(obsstd)}"
                out = output_dir / f"evidence_accumulation_simplex_{slug}{suffix}.png"
                fig.tight_layout(pad=0.6)
                fig.savefig(out, dpi=300, facecolor="white")
                plt.close(fig)
                print(f"Saved {out}")


def plot_outputs(summary: pd.DataFrame, metrics_family: pd.DataFrame, output_dir: Path, args, max_obs: int, run_level_family: pd.DataFrame | None = None):
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("relative_temporal_weights", "profile", "mean_simplex_weight", "se_simplex_weight", "Relative\ntemporal weight", 1 / max_obs, (0, 0.2)),
        ("effective_evidence_coefficients", "profile", "mean_effective_coefficient", "se_effective_coefficient", "Effective\nevidence coefficient", None, (0, 1)),
        ("overall_evidence_gain", "point", "gain", None, "Overall\nevidence gain", None, (0, 1)),
        ("cross_validated_fit_quality", "point", "cross_validated_R2", None, "CV R2", None, (0, 1)),
    ]
    obsstd_levels = sorted(pd.to_numeric(summary["observation_noise_std"], errors="coerce").dropna().unique())
    duration_levels = sorted(pd.to_numeric(summary.get("target_duration", pd.Series([max_obs])), errors="coerce").dropna().unique())
    panel_in = float(args.panel_mm) / 25.4
    plt.rcParams.update({"font.size": float(args.font_size)})
    for obsstd in obsstd_levels:
        odat = summary[np.isclose(pd.to_numeric(summary["observation_noise_std"], errors="coerce"), obsstd)]
        omet = metrics_family[np.isclose(pd.to_numeric(metrics_family["observation_noise_std"], errors="coerce"), obsstd)]
        for duration_value in duration_levels:
            sdat = odat[np.isclose(pd.to_numeric(odat["target_duration"], errors="coerce"), duration_value)]
            mdat = omet[np.isclose(pd.to_numeric(omet["target_duration"], errors="coerce"), duration_value)]
            cohs = sorted(pd.to_numeric(sdat["coherence_magnitude"], errors="coerce").dropna().unique())
            families = [f for f in ["beta", "opportunity"] if f in set(sdat["family"]).union(set(mdat["family"]))]
            if not cohs or not families:
                continue
            fit_duration = int(round(float(duration_value)))
            for slug, kind, y_col, se_col, ylab, ref, fallback in specs:
                if kind == "profile":
                    vals = pd.to_numeric(sdat[y_col], errors="coerce")
                    if se_col:
                        se = pd.to_numeric(sdat[se_col], errors="coerce")
                        ylim = safe_ylim(pd.concat([vals, vals - se, vals + se]), fallback)
                    else:
                        ylim = safe_ylim(vals, fallback)
                else:
                    vals = pd.to_numeric(mdat[y_col], errors="coerce")
                    ylim = safe_ylim(vals, fallback)
                fig, axes = plt.subplots(len(cohs), len(families), figsize=(0.45 + len(families) * panel_in, 0.42 + len(cohs) * panel_in), squeeze=False)
                for r, coh in enumerate(cohs):
                    for c, family in enumerate(families):
                        ax = axes[r, c]
                        params = sorted(pd.to_numeric(sdat.loc[sdat["family"] == family, "parameter_value"], errors="coerce").dropna().unique())
                        if kind == "point":
                            params = sorted(pd.to_numeric(mdat.loc[mdat["family"] == family, "parameter_value"], errors="coerce").dropna().unique())
                        colors = color_map(family, params)
                        ax.set_ylim(*ylim)
                        if kind == "profile":
                            pdat = sdat[(sdat["family"] == family) & np.isclose(pd.to_numeric(sdat["coherence_magnitude"], errors="coerce"), coh)]
                            for param in params:
                                line = pdat[np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param)].sort_values("observation_position")
                                if line.empty:
                                    continue
                                x = line["observation_position"].to_numpy(dtype=float)
                                y = line[y_col].to_numpy(dtype=float)
                                ax.plot(x, y, color=colors[param], lw=1.1, marker="o" if family == "beta" else "^", ms=2.5, label=num_label(param))
                                if se_col and line["n_runs"].max() > 1:
                                    err = line[se_col].to_numpy(dtype=float)
                                    ax.errorbar(x, y, yerr=err, color=colors[param], lw=0, elinewidth=0.6, capsize=1.5)
                            ax.set_xlim(0.5, fit_duration + 0.5)
                            ax.set_xticks(range(1, fit_duration + 1))
                            if ref is not None:
                                ax.axhline(1.0 / fit_duration if slug == "relative_temporal_weights" else ref, color="0.75", ls="--", lw=0.7)
                            ax.set_xlabel("Observation\nposition")
                        else:
                            pdat = mdat[(mdat["family"] == family) & np.isclose(pd.to_numeric(mdat["coherence_magnitude"], errors="coerce"), coh)]
                            x_positions = np.arange(1, len(params) + 1)
                            for i, param in enumerate(params, start=1):
                                vals = pd.to_numeric(pdat.loc[np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param), y_col], errors="coerce").dropna()
                                if vals.empty:
                                    continue
                                y = float(vals.mean())
                                yerr = sem(vals)
                                ax.errorbar([i], [y], yerr=None if not math.isfinite(yerr) else [yerr], color=colors[param], marker="o" if family == "beta" else "^", lw=0, elinewidth=0.6, capsize=1.5)
                            ax.set_xlim(0.5, max(len(params), 1) + 0.5)
                            ax.set_xticks(x_positions)
                            ax.set_xticklabels([num_label(p) for p in params])
                        if r == 0:
                            title = "Varying memory lambda" if family == "beta" else "Varying opportunity"
                            ax.set_title(title)
                        if c == 0:
                            ax.set_ylabel(f"coh {num_label(coh)}\n{ylab}")
                        else:
                            ax.set_ylabel("")
                suffix = f"{duration_suffix(duration_value, max_obs, duration_levels)}_obsstd_{value_token(obsstd)}"
                by_coherence = kind == "profile" and len(cohs) > 1
                out_slug = f"{slug}_by_coherence" if by_coherence else slug
                out = output_dir / f"evidence_accumulation_simplex_{out_slug}{suffix}.png"
                fig.tight_layout(pad=0.6)
                fig.savefig(out, dpi=300, facecolor="white")
                plt.close(fig)
                print(f"Saved {out}")
    if run_level_family is not None and not run_level_family.empty:
        collapsed = summarize_coherence_collapsed(run_level_family)
        plot_coherence_collapsed_profiles(collapsed, output_dir, args, max_obs, obsstd_levels)


def plot_z_mu_outputs(summary: pd.DataFrame, output_dir: Path, args, max_obs: int):
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_in = float(args.panel_mm) / 25.4
    plt.rcParams.update({"font.size": float(args.font_size)})
    specs = [
        ("relative_temporal_weights", "mean_simplex_weight", "se_simplex_weight", "z_mu relative\ntemporal weight", (0, 1)),
        ("effective_evidence_coefficients", "mean_effective_coefficient", "se_effective_coefficient", "z_mu effective\nevidence coefficient", (0, 1)),
    ]
    obsstd_levels = sorted(pd.to_numeric(summary["observation_noise_std"], errors="coerce").dropna().unique())
    dims = sorted(pd.to_numeric(summary["latent_dim_index"], errors="coerce").dropna().unique())
    target_steps = sorted(pd.to_numeric(summary["target_timestep"], errors="coerce").dropna().unique())
    for obsstd in obsstd_levels:
        odat = summary[np.isclose(pd.to_numeric(summary["observation_noise_std"], errors="coerce"), obsstd)]
        for dim in dims:
            ddat = odat[np.isclose(pd.to_numeric(odat["latent_dim_index"], errors="coerce"), dim)]
            if ddat.empty:
                continue
            for target_step in target_steps:
                tdat = ddat[np.isclose(pd.to_numeric(ddat["target_timestep"], errors="coerce"), target_step)]
                if tdat.empty:
                    continue
                cohs = sorted(pd.to_numeric(tdat["coherence_magnitude"], errors="coerce").dropna().unique())
                families = [f for f in ["beta", "opportunity"] if f in set(tdat["family"])]
                if not cohs or not families:
                    continue
                for slug, y_col, se_col, ylab, fallback in specs:
                    vals = pd.to_numeric(tdat[y_col], errors="coerce")
                    se = pd.to_numeric(tdat[se_col], errors="coerce")
                    ylim = safe_ylim(pd.concat([vals, vals - se, vals + se]), fallback)
                    fig, axes = plt.subplots(
                        len(cohs),
                        len(families),
                        figsize=(0.45 + len(families) * panel_in, 0.42 + len(cohs) * panel_in),
                        squeeze=False,
                    )
                    for r, coh in enumerate(cohs):
                        for c, family in enumerate(families):
                            ax = axes[r, c]
                            ax.set_ylim(*ylim)
                            pdat = tdat[
                                (tdat["family"] == family)
                                & np.isclose(pd.to_numeric(tdat["coherence_magnitude"], errors="coerce"), coh)
                            ]
                            params = sorted(pd.to_numeric(pdat["parameter_value"], errors="coerce").dropna().unique())
                            colors = color_map(family, params)
                            for param in params:
                                line = pdat[
                                    np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param)
                                ].sort_values("observation_position")
                                if line.empty:
                                    continue
                                x = line["observation_position"].to_numpy(dtype=float)
                                y = line[y_col].to_numpy(dtype=float)
                                ax.plot(
                                    x,
                                    y,
                                    color=colors[param],
                                    lw=1.1,
                                    marker="o" if family == "beta" else "^",
                                    ms=2.5,
                                    label=num_label(param),
                                )
                                if line["n_runs"].max() > 1:
                                    err = line[se_col].to_numpy(dtype=float)
                                    ax.errorbar(x, y, yerr=err, color=colors[param], lw=0, elinewidth=0.6, capsize=1.5)
                            ax.set_xlim(0.5, max_obs + 0.5)
                            ax.set_xticks(range(1, max_obs + 1))
                            if slug == "relative_temporal_weights":
                                ax.axhline(1.0 / float(target_step), color="0.75", ls="--", lw=0.7)
                            ax.set_xlabel("Observation\nposition")
                            if r == 0:
                                ax.set_title("Varying memory lambda" if family == "beta" else "Varying opportunity")
                            if c == 0:
                                ax.set_ylabel(f"coh {num_label(coh)}\n{ylab}")
                            else:
                                ax.set_ylabel("")
                    suffix = f"_obsstd_{value_token(obsstd)}"
                    out = output_dir / (
                        f"evidence_accumulation_simplex_z_mu_dim{int(dim)}_t{int(target_step)}_"
                        f"{slug}_by_coherence{suffix}.png"
                    )
                    fig.tight_layout(pad=0.6)
                    fig.savefig(out, dpi=300, facecolor="white")
                    plt.close(fig)
                    print(f"Saved {out}")


def plot_z_mu_lag_outputs(summary: pd.DataFrame, output_dir: Path, args, max_obs: int):
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_in = float(args.panel_mm) / 25.4
    plt.rcParams.update({"font.size": float(args.font_size)})
    specs = [
        (
            "relative_temporal_weights",
            "mean_simplex_weight",
            "se_simplex_weight",
            "z_mu relative\ntemporal weight",
            (0, 1),
        ),
        (
            "effective_evidence_coefficients",
            "mean_effective_coefficient",
            "se_effective_coefficient",
            "z_mu effective\nevidence coefficient",
            (0, 1),
        ),
    ]
    obsstd_levels = sorted(pd.to_numeric(summary["observation_noise_std"], errors="coerce").dropna().unique())
    dims = sorted(pd.to_numeric(summary["latent_dim_index"], errors="coerce").dropna().unique())
    for obsstd in obsstd_levels:
        odat = summary[np.isclose(pd.to_numeric(summary["observation_noise_std"], errors="coerce"), obsstd)]
        for dim in dims:
            ddat = odat[np.isclose(pd.to_numeric(odat["latent_dim_index"], errors="coerce"), dim)]
            if ddat.empty:
                continue
            cohs = sorted(pd.to_numeric(ddat["coherence_magnitude"], errors="coerce").dropna().unique())
            families = [f for f in ["beta", "opportunity"] if f in set(ddat["family"])]
            if not cohs or not families:
                continue
            for slug, y_col, se_col, ylab, fallback in specs:
                vals = pd.to_numeric(ddat[y_col], errors="coerce")
                se = pd.to_numeric(ddat[se_col], errors="coerce")
                ylim = safe_ylim(pd.concat([vals, vals - se, vals + se]), fallback)
                fig, axes = plt.subplots(
                    len(cohs),
                    len(families),
                    figsize=(0.45 + len(families) * panel_in, 0.42 + len(cohs) * panel_in),
                    squeeze=False,
                )
                for r, coh in enumerate(cohs):
                    for c, family in enumerate(families):
                        ax = axes[r, c]
                        ax.set_ylim(*ylim)
                        pdat = ddat[
                            (ddat["family"] == family)
                            & np.isclose(pd.to_numeric(ddat["coherence_magnitude"], errors="coerce"), coh)
                        ]
                        params = sorted(pd.to_numeric(pdat["parameter_value"], errors="coerce").dropna().unique())
                        colors = color_map(family, params)
                        for param in params:
                            line = pdat[
                                np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param)
                            ].sort_values("timestep_relative_to_current")
                            if line.empty:
                                continue
                            x = line["timestep_relative_to_current"].to_numpy(dtype=float)
                            y = line[y_col].to_numpy(dtype=float)
                            ax.plot(
                                x,
                                y,
                                color=colors[param],
                                lw=1.1,
                                marker="o" if family == "beta" else "^",
                                ms=2.5,
                                label=num_label(param),
                            )
                            if line["n_fits"].max() > 1:
                                err = line[se_col].to_numpy(dtype=float)
                                ax.errorbar(x, y, yerr=err, color=colors[param], lw=0, elinewidth=0.6, capsize=1.5)
                        ax.set_xlim(-(max_obs - 1) - 0.5, 0.5)
                        ax.set_xticks(range(-(max_obs - 1), 1))
                        ax.set_xlabel("Timestep relative\nto current z_mu")
                        if r == 0:
                            ax.set_title("Varying memory lambda" if family == "beta" else "Varying opportunity")
                        if c == 0:
                            ax.set_ylabel(f"coh {num_label(coh)}\n{ylab}")
                        else:
                            ax.set_ylabel("")
                suffix = f"_obsstd_{value_token(obsstd)}"
                out = output_dir / (
                    f"evidence_accumulation_simplex_z_mu_dim{int(dim)}_"
                    f"{slug}_by_lag_until_current_by_coherence{suffix}.png"
                )
                fig.tight_layout(pad=0.6)
                fig.savefig(out, dpi=300, facecolor="white")
                plt.close(fig)
                print(f"Saved {out}")


def plot_z_mu_target_timestep_coherence_collapsed_outputs(summary: pd.DataFrame, output_dir: Path, args, max_obs: int):
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_in = float(args.panel_mm) / 25.4
    plt.rcParams.update({"font.size": float(args.font_size)})
    specs = [
        (
            "relative_temporal_weights",
            "mean_simplex_weight",
            "se_simplex_weight",
            "z_mu relative\ntemporal weight",
            (0, 1),
        ),
        (
            "effective_evidence_coefficients",
            "mean_effective_coefficient",
            "se_effective_coefficient",
            "z_mu effective\nevidence coefficient",
            (0, 1),
        ),
    ]
    obsstd_levels = sorted(pd.to_numeric(summary["observation_noise_std"], errors="coerce").dropna().unique())
    dims = sorted(pd.to_numeric(summary["latent_dim_index"], errors="coerce").dropna().unique())
    duration_levels = sorted(pd.to_numeric(summary.get("target_duration", pd.Series([max_obs])), errors="coerce").dropna().unique())
    for obsstd in obsstd_levels:
        odat = summary[np.isclose(pd.to_numeric(summary["observation_noise_std"], errors="coerce"), obsstd)]
        for dim in dims:
            odim = odat[np.isclose(pd.to_numeric(odat["latent_dim_index"], errors="coerce"), dim)]
            for duration_value in duration_levels:
                ddat = odim[np.isclose(pd.to_numeric(odim["target_duration"], errors="coerce"), duration_value)]
                if ddat.empty:
                    continue
                fit_duration = int(round(float(duration_value)))
                target_steps = sorted(pd.to_numeric(ddat["target_timestep"], errors="coerce").dropna().unique())
                families = [f for f in ["beta", "opportunity"] if f in set(ddat["family"])]
                if not target_steps or not families:
                    continue
                for slug, y_col, se_col, ylab, fallback in specs:
                    vals = pd.to_numeric(ddat[y_col], errors="coerce")
                    se = pd.to_numeric(ddat[se_col], errors="coerce")
                    ylim = safe_ylim(pd.concat([vals, vals - se, vals + se]), fallback)
                    fig, axes = plt.subplots(
                        len(target_steps),
                        len(families),
                        figsize=(0.45 + len(families) * panel_in, 0.42 + len(target_steps) * panel_in),
                        squeeze=False,
                    )
                    for r, target_step in enumerate(target_steps):
                        for c, family in enumerate(families):
                            ax = axes[r, c]
                            ax.set_ylim(*ylim)
                            pdat = ddat[
                                (ddat["family"] == family)
                                & np.isclose(pd.to_numeric(ddat["target_timestep"], errors="coerce"), target_step)
                            ]
                            params = sorted(pd.to_numeric(pdat["parameter_value"], errors="coerce").dropna().unique())
                            colors = color_map(family, params)
                            for param in params:
                                line = pdat[
                                    np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param)
                                ].sort_values("observation_position")
                                if line.empty:
                                    continue
                                x = line["observation_position"].to_numpy(dtype=float)
                                y = line[y_col].to_numpy(dtype=float)
                                ax.plot(
                                    x,
                                    y,
                                    color=colors[param],
                                    lw=1.1,
                                    marker="o" if family == "beta" else "^",
                                    ms=2.5,
                                    label=num_label(param),
                                )
                                if line["n_fits"].max() > 1:
                                    err = line[se_col].to_numpy(dtype=float)
                                    ax.errorbar(x, y, yerr=err, color=colors[param], lw=0, elinewidth=0.6, capsize=1.5)
                            ax.set_xlim(0.5, fit_duration + 0.5)
                            ax.set_xticks(range(1, fit_duration + 1))
                            if slug == "relative_temporal_weights":
                                ax.axhline(1.0 / float(target_step), color="0.75", ls="--", lw=0.7)
                            ax.set_xlabel("Observation\ntimestep")
                            if r == 0:
                                ax.set_title("Varying memory lambda" if family == "beta" else "Varying opportunity")
                            if c == 0:
                                ax.set_ylabel(f"t {int(target_step)}\n{ylab}")
                            else:
                                ax.set_ylabel("")
                    suffix = f"{duration_suffix(duration_value, max_obs, duration_levels)}_obsstd_{value_token(obsstd)}"
                    out = output_dir / (
                        f"evidence_accumulation_simplex_z_mu_dim{int(dim)}_"
                        f"{slug}_by_target_timestep_coherence_collapsed{suffix}.png"
                    )
                    fig.tight_layout(pad=0.6)
                    fig.savefig(out, dpi=300, facecolor="white")
                    plt.close(fig)
                    print(f"Saved {out}")


def plot_z_mu_target_timestep_by_coherence_outputs(summary: pd.DataFrame, output_dir: Path, args, max_obs: int):
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    dat = summary.copy()
    panel_in = float(args.panel_mm) / 25.4
    plt.rcParams.update({"font.size": float(args.font_size)})
    specs = [
        (
            "relative_temporal_weights",
            "mean_simplex_weight",
            "se_simplex_weight",
            "z_mu relative\ntemporal weight",
            (0, 1),
        ),
        (
            "effective_evidence_coefficients",
            "mean_effective_coefficient",
            "se_effective_coefficient",
            "z_mu effective\nevidence coefficient",
            (0, 1),
        ),
    ]
    obsstd_levels = sorted(pd.to_numeric(dat["observation_noise_std"], errors="coerce").dropna().unique())
    dims = sorted(pd.to_numeric(dat["latent_dim_index"], errors="coerce").dropna().unique())
    duration_levels = sorted(pd.to_numeric(dat.get("target_duration", pd.Series([max_obs])), errors="coerce").dropna().unique())
    for obsstd in obsstd_levels:
        odat = dat[np.isclose(pd.to_numeric(dat["observation_noise_std"], errors="coerce"), obsstd)]
        for dim in dims:
            odim = odat[np.isclose(pd.to_numeric(odat["latent_dim_index"], errors="coerce"), dim)]
            for duration_value in duration_levels:
                ddat = odim[np.isclose(pd.to_numeric(odim["target_duration"], errors="coerce"), duration_value)]
                if ddat.empty:
                    continue
                fit_duration = int(round(float(duration_value)))
                target_steps = sorted(pd.to_numeric(ddat["target_timestep"], errors="coerce").dropna().unique())
                cohs = sorted(pd.to_numeric(ddat["coherence_magnitude"], errors="coerce").dropna().unique())
                families = [f for f in ["beta", "opportunity"] if f in set(ddat["family"])]
                if not target_steps or not cohs or not families:
                    continue
                for coh in cohs:
                    cdat = ddat[np.isclose(pd.to_numeric(ddat["coherence_magnitude"], errors="coerce"), coh)]
                    if cdat.empty:
                        continue
                    for slug, y_col, se_col, ylab, fallback in specs:
                        vals = pd.to_numeric(cdat[y_col], errors="coerce")
                        se = pd.to_numeric(cdat[se_col], errors="coerce")
                        ylim = safe_ylim(pd.concat([vals, vals - se, vals + se]), fallback)
                        fig, axes = plt.subplots(
                            len(target_steps),
                            len(families),
                            figsize=(0.45 + len(families) * panel_in, 0.42 + len(target_steps) * panel_in),
                            squeeze=False,
                        )
                        for r, target_step in enumerate(target_steps):
                            for c, family in enumerate(families):
                                ax = axes[r, c]
                                ax.set_ylim(*ylim)
                                pdat = cdat[
                                    (cdat["family"] == family)
                                    & np.isclose(pd.to_numeric(cdat["target_timestep"], errors="coerce"), target_step)
                                ]
                                params = sorted(pd.to_numeric(pdat["parameter_value"], errors="coerce").dropna().unique())
                                colors = color_map(family, params)
                                for param in params:
                                    line = pdat[
                                        np.isclose(pd.to_numeric(pdat["parameter_value"], errors="coerce"), param)
                                    ].sort_values("observation_position")
                                    if line.empty:
                                        continue
                                    x = line["observation_position"].to_numpy(dtype=float)
                                    y = line[y_col].to_numpy(dtype=float)
                                    ax.plot(
                                        x,
                                        y,
                                        color=colors[param],
                                        lw=1.1,
                                        marker="o" if family == "beta" else "^",
                                        ms=2.5,
                                        label=num_label(param),
                                    )
                                    if line["n_runs"].max() > 1:
                                        err = line[se_col].to_numpy(dtype=float)
                                        ax.errorbar(x, y, yerr=err, color=colors[param], lw=0, elinewidth=0.6, capsize=1.5)
                                ax.set_xlim(0.5, fit_duration + 0.5)
                                ax.set_xticks(range(1, fit_duration + 1))
                                if slug == "relative_temporal_weights":
                                    ax.axhline(1.0 / float(target_step), color="0.75", ls="--", lw=0.7)
                                ax.set_xlabel("Observation\ntimestep")
                                if r == 0:
                                    ax.set_title("Varying memory lambda" if family == "beta" else "Varying opportunity")
                                if c == 0:
                                    ax.set_ylabel(f"t {int(target_step)}\n{ylab}")
                                else:
                                    ax.set_ylabel("")
                        suffix = f"{duration_suffix(duration_value, max_obs, duration_levels)}_coherence_{value_token(coh)}_obsstd_{value_token(obsstd)}"
                        out = output_dir / (
                            f"evidence_accumulation_simplex_z_mu_dim{int(dim)}_"
                            f"{slug}_by_target_timestep{suffix}.png"
                        )
                        fig.tight_layout(pad=0.6)
                        fig.savefig(out, dpi=300, facecolor="white")
                        plt.close(fig)
                        print(f"Saved {out}")


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.task != "evidence":
        raise SystemExit(f"Only evidence task is supported, not {args.task}")
    if args.target_type not in {"auto", "logit"}:
        raise SystemExit(
            "Fast NNLS simplex fitting currently supports the raw-logit target only. "
            "Use --target-type logit or omit --target-type."
        )
    explicit_seeds = option_present(argv, ["--seeds"])
    memory_row, opp_row = apply_preset_defaults(args, argv)
    max_obs = int(as_num(args.max_observations))
    print(f"Using evidence accumulation preset: task={args.task} from {args.preset_file}")
    files = list_input_files(args)
    manifest = metadata_frame(files)
    selected = prefilter_manifest(manifest, args, memory_row, opp_row, explicit_seeds)
    print(f"Filename/metadata prefilter kept {len(selected)}/{len(manifest)} wide CSV file(s) before reading rows.")
    if selected.empty:
        raise SystemExit("No evidence CSVs remain after filename/metadata filters.")
    trials = load_trials(selected, max_obs)
    print(f"Loaded {selected.shape[0]} file(s) from {args.input_dir}.")
    print(f"Loaded {len(trials)} trial row(s).")
    trials = filter_trials(trials, args, memory_row, opp_row, explicit_seeds, max_obs)
    if trials.empty:
        raise SystemExit("No trials remain after filters.")
    if "training_step" not in trials.columns:
        trials["training_step"] = -1
    missing_step = pd.to_numeric(trials["training_step"], errors="coerce").isna()
    if missing_step.any():
        print(f"Replacing {int(missing_step.sum())} missing training_step value(s) with -1 for grouping; checkpoint labels are retained.")
        trials.loc[missing_step, "training_step"] = -1

    output_base = Path(args.output_dir) if args.output_dir else Path(args.output_root) / "evidence_accumulation_simplex_weights"
    if args.simple_obsstd or args.simple_coherences:
        observer_values = trials["choice_at_end_only"].astype(bool).unique()
        if len(observer_values) == 1 and observer_values[0]:
            output_dir = output_base / "observer_only_simple"
        elif len(observer_values) == 1 and not observer_values[0]:
            output_dir = output_base / "policy_timed_simple"
        else:
            output_dir = output_base / "observer_mixed_simple"
    else:
        observer_values = trials["choice_at_end_only"].astype(bool).unique()
        observer_folder = "observer_only" if len(observer_values) == 1 and observer_values[0] else "policy_timed" if len(observer_values) == 1 else "observer_mixed"
        coh_folder = "coherence_pooled" if args.pool_coherence else f"coherence_{value_token(trials['coherence_magnitude'].iloc[0])}"
        output_dir = output_base / coh_folder / observer_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving simplex outputs to {output_dir}.")

    group_cols = [
        "run_id", "checkpoint", "training_step", "seed", "loss_scale", "memory_lambda",
        "choice_at_end_only", "alpha", "beta", "opportunity_cost", "observation_noise_std",
        "coherence_magnitude", "target_duration",
    ]
    z_mu_dims = [] if args.skip_z_mu_simplex else available_z_mu_dims(trials.columns, args.z_mu_simplex_dims)
    if z_mu_dims:
        print(f"Also fitting z_mu simplex outcomes for latent dimension(s): {','.join(map(str, z_mu_dims))}.")
    elif not args.skip_z_mu_simplex:
        print("No z_mu_*_t* columns found; skipping z_mu simplex outcome fits.")
    run_rows = []
    metric_rows = []
    z_mu_run_rows = []
    z_mu_metric_rows = []
    groups = list(trials.groupby(group_cols, dropna=False))
    print(f"Fitting {len(groups)} independent run/condition group(s).")
    for idx, (keys, grp) in enumerate(groups, start=1):
        first = grp.iloc[0]
        print(f"Fitting {idx}/{len(groups)}: run_id={first['run_id']}, seed={num_label(first['seed'])}, memory_lambda={num_label(first['memory_lambda'])}, opp={num_label(first['opportunity_cost'])}, obsstd={num_label(first['observation_noise_std'])}, n={len(grp)}")
        if len(grp) < int(args.min_trials_per_fit):
            metric_rows.append({**{c: first[c] for c in group_cols}, "n_trials": len(grp), "skipped_reason": f"too_few_trials:{len(grp)}"})
            continue
        fit_duration = int(first["target_duration"])
        obs_cols = [f"observation_{i}" for i in range(1, fit_duration + 1)]
        rows, metric = fit_group(grp, obs_cols, fit_duration, int(args.num_cv_folds), int(args.seed), idx)
        run_rows.extend(rows)
        metric_rows.append(metric)
        if z_mu_dims:
            z_rows, z_metrics = fit_group_z_mu(
                grp,
                fit_duration,
                z_mu_dims,
                int(args.num_cv_folds),
                int(args.seed),
                idx,
                int(args.min_trials_per_fit),
            )
            z_mu_run_rows.extend(z_rows)
            z_mu_metric_rows.extend(z_metrics)
    run_level = pd.DataFrame(run_rows)
    fit_metrics = pd.DataFrame(metric_rows)
    if run_level.empty:
        fit_metrics.to_csv(output_dir / "evidence_accumulation_simplex_fit_metrics.csv", index=False)
        raise SystemExit("No successful simplex fits were produced.")

    run_level_family = add_family_rows(run_level, args, memory_row, opp_row, explicit_seeds)
    if "skipped_reason" in fit_metrics.columns:
        ok_metrics = fit_metrics[fit_metrics["skipped_reason"].isna()].copy()
    else:
        ok_metrics = fit_metrics.copy()
    metrics_family = add_family_rows(ok_metrics, args, memory_row, opp_row, explicit_seeds)
    summary = summarize(run_level_family)

    run_path = output_dir / "evidence_accumulation_simplex_weights_run_level.csv"
    summary_path = output_dir / "evidence_accumulation_simplex_weights_summary.csv"
    metrics_path = output_dir / "evidence_accumulation_simplex_fit_metrics.csv"
    run_level.to_csv(run_path, index=False)
    summary.to_csv(summary_path, index=False)
    fit_metrics.to_csv(metrics_path, index=False)
    print(f"Saved {run_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {metrics_path}")
    weight_sums = run_level.groupby(["run_id", "checkpoint", "seed", "memory_lambda", "opportunity_cost", "observation_noise_std", "coherence_magnitude", "target_duration"])["simplex_weight"].sum()
    print(f"Maximum deviation from sum(weight)=1: {np.max(np.abs(weight_sums.to_numpy() - 1)):.3g}.")
    print(f"Gain range: [{num_label(fit_metrics['gain'].min())}, {num_label(fit_metrics['gain'].max())}].")
    plot_outputs(summary, metrics_family, output_dir, args, max_obs, run_level_family=run_level_family)
    if z_mu_run_rows:
        z_mu_run_level = pd.DataFrame(z_mu_run_rows)
        z_mu_fit_metrics = pd.DataFrame(z_mu_metric_rows)
        z_mu_run_level_family = add_family_rows(z_mu_run_level, args, memory_row, opp_row, explicit_seeds)
        z_mu_summary = summarize_z_mu(z_mu_run_level_family)
        z_mu_lag_summary = summarize_z_mu_lag(z_mu_run_level_family)
        z_mu_timestep_collapsed_summary = summarize_z_mu_by_target_timestep_coherence_collapsed(z_mu_run_level_family)
        z_run_path = output_dir / "evidence_accumulation_simplex_z_mu_weights_run_level.csv"
        z_summary_path = output_dir / "evidence_accumulation_simplex_z_mu_weights_summary.csv"
        z_lag_summary_path = output_dir / "evidence_accumulation_simplex_z_mu_weights_by_lag_until_current_summary.csv"
        z_timestep_collapsed_summary_path = output_dir / "evidence_accumulation_simplex_z_mu_weights_by_target_timestep_coherence_collapsed_summary.csv"
        z_metrics_path = output_dir / "evidence_accumulation_simplex_z_mu_fit_metrics.csv"
        z_mu_run_level.to_csv(z_run_path, index=False)
        z_mu_summary.to_csv(z_summary_path, index=False)
        z_mu_lag_summary.to_csv(z_lag_summary_path, index=False)
        z_mu_timestep_collapsed_summary.to_csv(z_timestep_collapsed_summary_path, index=False)
        z_mu_fit_metrics.to_csv(z_metrics_path, index=False)
        print(f"Saved {z_run_path}")
        print(f"Saved {z_summary_path}")
        print(f"Saved {z_lag_summary_path}")
        print(f"Saved {z_timestep_collapsed_summary_path}")
        print(f"Saved {z_metrics_path}")
        plot_z_mu_target_timestep_coherence_collapsed_outputs(z_mu_timestep_collapsed_summary, output_dir, args, max_obs)
        plot_z_mu_target_timestep_by_coherence_outputs(z_mu_summary, output_dir, args, max_obs)
    print("Fast NNLS simplex model: q_hat = bias + X @ c, c >= 0; gain=sum(c); weight=c/gain.")


if __name__ == "__main__":
    main()
