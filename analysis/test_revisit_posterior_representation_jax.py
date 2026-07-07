#!/usr/bin/env python3
"""Hypothesis-2 diagnostics for revisit JAX models.

This script tests whether a revisit model represents a compact posterior belief
or keeps idiosyncratic sample/order history.

Outputs per parameter combination:
  * forced_history_states.csv: hand-constructed same-posterior histories.
  * forced_history_pairwise.csv: latent/policy distances between matched histories.
    The matched examples are regenerated for each observation sigma using
    continuous values, with empirical history variance equal to sigma^2.
  * probe_state_dataset.csv: random forced histories with posterior/sample labels.
  * latent_decoder_r2.csv: ridge decoders from latent features to posterior stats
    versus individual sample/order labels.
  * variance_decomposition_r2.csv: how much each latent/PGA coordinate is
    explained by posterior stats versus sample/order feature groups.

The command style mirrors plot_revisit_beta_opp_comparison.R:

  python analysis/test_revisit_posterior_representation_jax.py default \
    --vary-beta-values "20,60,100" \
    --vary-opportunity-values "0.06,0.2,0.4"
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_jax import planning as jp  # noqa: E402


PLOT_FONT_SIZE_PT = 7
PANEL_WIDTH_IN = 33 / 25.4
PANEL_HEIGHT_IN = 33 / 25.4
PANEL_GAP_IN = 0.20
LEFT_MARGIN_IN = 0.72
BOTTOM_MARGIN_IN = 0.50
TOP_MARGIN_IN = 0.22
RIGHT_MARGIN_IN = 0.12
LEGEND_WIDTH_IN = 1.12


DEFAULT_DEEP_PROBE_HIDDEN_DIMS = (64, 32)
DEFAULT_DEEP_PROBE_EPOCHS = 300
DEFAULT_DEEP_PROBE_LR = 1e-3
DEFAULT_DEEP_PROBE_WEIGHT_DECAY = 1e-4


def parse_csv_values(raw: str | None, typ=float) -> list:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            out.extend(parse_csv_values(item, typ=typ))
        return out
    return [typ(x.strip()) for x in str(raw).replace(",", " ").split() if x.strip()]


def parse_decoder_types(raw: str | None, *, full_h2_plots: bool = False) -> list[str]:
    if raw is None or not str(raw).strip():
        return ["linear", "quadratic"] if full_h2_plots else ["quadratic"]
    values = [str(x).strip().lower() for x in str(raw).replace(",", " ").split() if str(x).strip()]
    aliases = {"mlp": "deep", "nn": "deep", "ridge": "linear"}
    out = []
    for value in values:
        value = aliases.get(value, value)
        if value not in {"linear", "quadratic", "deep"}:
            raise ValueError(f"Unknown probe decoder type {value!r}; use linear, quadratic, or deep.")
        if value not in out:
            out.append(value)
    return out or (["linear", "quadratic"] if full_h2_plots else ["quadratic"])


def parse_hidden_dims(raw: str | None) -> tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_DEEP_PROBE_HIDDEN_DIMS
    dims = tuple(int(x) for x in str(raw).replace(",", " ").split() if str(x).strip())
    return tuple(d for d in dims if d > 0) or DEFAULT_DEEP_PROBE_HIDDEN_DIMS


def format_float_tag(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def deep_probe_cache_tag(config: dict) -> str:
    hidden = "x".join(str(int(x)) for x in config.get("hidden_dims", DEFAULT_DEEP_PROBE_HIDDEN_DIMS))
    return (
        f"deep_h{hidden}"
        f"_e{int(config.get('epochs', DEFAULT_DEEP_PROBE_EPOCHS))}"
        f"_lr{format_float_tag(float(config.get('lr', DEFAULT_DEEP_PROBE_LR)))}"
        f"_wd{format_float_tag(float(config.get('weight_decay', DEFAULT_DEEP_PROBE_WEIGHT_DECAY)))}"
    )


def parse_preset_csv_value(raw: str | None, typ=float) -> list | None:
    if raw is None:
        return None
    vals = [typ(x.strip()) for x in str(raw).split(",") if x.strip()]
    return vals if vals else None


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
        "2x2": "disjoint2x2",
        "disjoint2x2": "disjoint2x2",
        "6": "disjoint3x2",
        "6n": "disjoint3x2",
        "3x2": "disjoint3x2",
        "disjoint3x2": "disjoint3x2",
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


def load_preset_rows(args: argparse.Namespace) -> argparse.Namespace:
    preset_file = Path(args.preset_file)
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_file}")
    presets = pd.read_csv(preset_file)
    presets["tree_key"] = presets["tree"].map(normalize_tree_name)
    tree_name = normalize_tree_name(args.tree)
    beta_rows = presets[(presets["tree_key"] == tree_name) & (presets["vary"] == "beta")]
    opp_rows = presets[(presets["tree_key"] == tree_name) & (presets["vary"] == "opportunity")]
    if beta_rows.empty or opp_rows.empty:
        raise ValueError(f"Need beta and opportunity preset rows for tree={tree_name}.")

    beta_row = beta_rows.iloc[0]
    opp_row = opp_rows.iloc[0]
    shared = beta_row
    args.tree_size = int(preset_value(shared, "tree_size"))
    args.tree_type = preset_value(shared, "tree_config", "default") or "default"
    args.input_type = preset_value(shared, "input_type", "uniform")
    args.expansion_decision_version = preset_value(shared, "expansion_decision_version", "lstm")
    args.model_variant = preset_value(shared, "model_variant", "vae")
    args.lambda_values = parse_preset_csv_value(preset_value(shared, "lambda_arg", "100.0"), float)
    args.alphas = parse_preset_csv_value(preset_value(shared, "alpha_arg", "0.0"), float)
    args.seeds = (
        parse_csv_values(args.seeds, int)
        if args.seeds is not None
        else parse_preset_csv_value(preset_value(shared, "seed_arg"), int)
    )
    args.sigmas = (
        parse_preset_csv_value(args.sigmas, float)
        if args.sigmas is not None
        else parse_preset_csv_value(preset_value(shared, "sigma_arg", "0"), float)
    )
    args.rnn_dims = (
        parse_preset_csv_value(args.rnn_dims, int)
        if args.rnn_dims is not None
        else parse_preset_csv_value(preset_value(shared, "rnn_units_arg"), int)
    )
    args.latent_dims = (
        parse_preset_csv_value(args.latent_dims, int)
        if args.latent_dims is not None
        else parse_preset_csv_value(preset_value(shared, "latent_dim_arg"), int)
    )
    args.max_observations_before_stop = int(
        args.max_observations_before_stop
        if args.max_observations_before_stop is not None
        else preset_value(shared, "max_observations_arg")
    )
    args.allow_node_revisit = True

    beta_values = parse_preset_csv_value(args.betas, float) or parse_preset_csv_value(
        preset_value(beta_row, "beta_arg"), float
    )
    opportunity_values = parse_preset_csv_value(args.opportunity_costs, float) or parse_preset_csv_value(
        preset_value(opp_row, "opportunity_arg"), float
    )
    beta_family_opps = parse_preset_csv_value(preset_value(beta_row, "opportunity_arg", "0.0"), float)
    opportunity_family_betas = parse_preset_csv_value(preset_value(opp_row, "beta_arg", "1000.0"), float)
    combos = []
    for beta in beta_values or []:
        for opp in beta_family_opps or []:
            combos.append(("vary_beta", "beta", float(beta), float(beta), float(opp)))
    for beta in opportunity_family_betas or []:
        for opp in opportunity_values or []:
            combos.append(("vary_opportunity", "opportunity", float(opp), float(beta), float(opp)))
    args.parameter_combos = combos
    if args.outdir is None:
        root = Path(args.output_root or preset_value(shared, "results_dir", "results"))
        args.outdir = str(root / "revisit_hypothesis2" / f"{tree_name}_beta_vs_opportunity")
    print(f"Using revisit preset: tree={tree_name} from {preset_file}", flush=True)
    return args


def make_config(
    args: argparse.Namespace,
    *,
    seed: int,
    beta: float,
    lambda_value: float,
    alpha: float,
    opportunity: float,
    sigma: float,
    rnn_dim: int,
    latent_dim: int,
) -> jp.RunConfig:
    num_steps = int(args.num_steps or ((args.max_observations_before_stop + 1) if args.allow_node_revisit else args.tree_size))
    return jp.RunConfig(
        lambda_=float(lambda_value),
        alpha=float(alpha),
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
        rnn_units=int(rnn_dim),
        latent_dim=int(latent_dim),
        sim_dir="outputs/jax_simulations",
        n_sim_trials=int(args.probe_n_states),
        num_envs=200,
        num_steps=num_steps,
        update_epochs=int(args.update_epochs),
        ppo_minibatches=int(args.ppo_minibatches),
        steps_per_epoch=0,
        return_target_rollouts=int(args.return_target_rollouts),
        return_target_mode=args.return_target_mode,
        sampled_lambda_critic=args.sampled_lambda_critic,
        lambda_return=float(args.lambda_return),
        target_critic_update_interval=int(args.target_critic_update_interval),
        target_critic_tau=float(args.target_critic_tau),
        backend=args.backend,
        jit_training=False,
        profile_update_components=False,
        profile_update_components_every=0,
        enable_reconstruction=False,
        enable_probe=False,
        allow_node_revisit=bool(args.allow_node_revisit),
        max_observations_before_stop=int(args.max_observations_before_stop),
        observation_sigma=float(sigma),
        kl_start_multiplier=float(args.kl_start_multiplier),
        kl_annealing_epochs=int(args.kl_annealing_epochs),
        node_coverage_aux_coef=0.0,
        node_coverage_aux_epochs=0,
    )


def reward_feature_dim_for_loaded_model(model: jp.PlanningVAE, config: jp.RunConfig) -> int:
    checkpoint_dim = int(getattr(model, "reward_feature_dim_override", 0))
    return checkpoint_dim if checkpoint_dim > 0 else jp.reward_feature_dim_for_sigma(config.observation_sigma)


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


def posterior_node_stats(
    observations_by_node: list[list[float]],
    reward_values: np.ndarray,
    sigma: float,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_eff = max(abs(float(sigma)), float(sigma_floor))
    support = np.asarray(reward_values, dtype=float)
    prior = np.ones_like(support, dtype=float) / float(len(support))
    means = []
    variances = []
    for observations in observations_by_node:
        if len(observations) == 0:
            probs = prior
        else:
            obs = np.asarray(observations, dtype=float)
            logp = np.log(prior + 1e-300)
            logp = logp + np.sum(-0.5 * ((obs[:, None] - support[None, :]) / sigma_eff) ** 2, axis=0)
            logp = logp - np.max(logp)
            probs = np.exp(logp)
            probs = probs / np.sum(probs)
        mean = float(np.sum(probs * support))
        var = float(np.sum(probs * (support - mean) ** 2))
        means.append(mean)
        variances.append(var)
    return np.asarray(means, dtype=float), np.asarray(variances, dtype=float)


def posterior_features_for_history(
    history_nodes: Iterable[int],
    history_observations: Iterable[float],
    task: jp.TaskSpec,
    sigma: float,
    sigma_floor: float,
) -> dict[str, float]:
    observations_by_node = [[] for _ in range(task.num_nodes)]
    for node_i, obs in zip(history_nodes, history_observations):
        observations_by_node[int(node_i)].append(float(obs))
    node_mean, node_var = posterior_node_stats(observations_by_node, task.reward_values, sigma, sigma_floor)
    path_map = np.asarray(task.path_map, dtype=float)
    path_mean = path_map @ node_mean
    path_var = (path_map ** 2) @ node_var
    row: dict[str, float] = {}
    for node_i in range(task.num_nodes):
        row[f"posterior_mean_node_{node_i + 1}"] = float(node_mean[node_i])
        row[f"posterior_var_node_{node_i + 1}"] = float(node_var[node_i])
        row[f"count_node_{node_i + 1}"] = float(len(observations_by_node[node_i]))
    for path_i in range(task.num_paths):
        row[f"posterior_mean_path_{path_i + 1}"] = float(path_mean[path_i])
        row[f"posterior_var_path_{path_i + 1}"] = float(path_var[path_i])
    return row


def population_variance(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean((vals - np.mean(vals)) ** 2))


def history_case_templates(task: jp.TaskSpec, sigma: float) -> list[dict]:
    if task.num_nodes < 2:
        return []
    sigma = abs(float(sigma))
    target_variance = sigma ** 2
    two_sample_delta = sigma
    three_symmetric_delta = math.sqrt(1.5) * sigma
    three_asymmetric_small = math.sqrt(0.5) * sigma
    three_asymmetric_large = math.sqrt(2.0) * sigma
    cases = []
    # Same Gaussian posterior for node 1: same multiset, different order.
    # The two-observation empirical variance is sigma^2.
    cases.extend([
        {
            "case_group": "node1_same_set_order",
            "history_name": "node1_plus_delta_then_minus_delta",
            "nodes": [0, 0],
            "observations": [two_sample_delta, -two_sample_delta],
            "target_history_variance": target_variance,
        },
        {
            "case_group": "node1_same_set_order",
            "history_name": "node1_minus_delta_then_plus_delta",
            "nodes": [0, 0],
            "observations": [-two_sample_delta, two_sample_delta],
            "target_history_variance": target_variance,
        },
        # Different sample sets with the same empirical mean and variance.
        # [-sqrt(3/2)s, 0, +sqrt(3/2)s] and
        # [-sqrt(1/2)s, -sqrt(1/2)s, +sqrt(2)s] both have mean 0
        # and population variance sigma^2.
        {
            "case_group": "node1_same_mean_var_diff_samples",
            "history_name": "node1_symmetric_three_samples",
            "nodes": [0, 0, 0],
            "observations": [-three_symmetric_delta, 0.0, three_symmetric_delta],
            "target_history_variance": target_variance,
        },
        {
            "case_group": "node1_same_mean_var_diff_samples",
            "history_name": "node1_asymmetric_three_samples",
            "nodes": [0, 0, 0],
            "observations": [-three_asymmetric_small, -three_asymmetric_small, three_asymmetric_large],
            "target_history_variance": target_variance,
        },
    ])
    # Same posterior over two nodes: same node-observation pairs, different order.
    # Across the full three-observation history, empirical variance is sigma^2.
    cases.extend([
        {
            "case_group": "two_node_same_pairs_order",
            "history_name": "n1_high_n2_zero_n1_low",
            "nodes": [0, 1, 0],
            "observations": [three_symmetric_delta, 0.0, -three_symmetric_delta],
            "target_history_variance": target_variance,
        },
        {
            "case_group": "two_node_same_pairs_order",
            "history_name": "n1_low_n2_zero_n1_high",
            "nodes": [0, 1, 0],
            "observations": [-three_symmetric_delta, 0.0, three_symmetric_delta],
            "target_history_variance": target_variance,
        },
        {
            "case_group": "two_node_same_pairs_order",
            "history_name": "n2_zero_n1_high_n1_low",
            "nodes": [1, 0, 0],
            "observations": [0.0, three_symmetric_delta, -three_symmetric_delta],
            "target_history_variance": target_variance,
        },
    ])
    return cases


def force_history_batch(
    *,
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    cases: list[dict],
    posterior_sigma_floor: float,
) -> pd.DataFrame:
    if not cases:
        return pd.DataFrame()
    parts = []
    for _length, group_cases_df in pd.DataFrame({
        "case_index": list(range(len(cases))),
        "history_length": [len(case["nodes"]) for case in cases],
    }).groupby("history_length", sort=True):
        group_cases = [cases[int(i)] for i in group_cases_df["case_index"]]
        parts.append(_force_equal_length_history_batch(
            model=model,
            params=params,
            config=config,
            task=task,
            cases=group_cases,
            posterior_sigma_floor=posterior_sigma_floor,
        ))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _force_equal_length_history_batch(
    *,
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    cases: list[dict],
    posterior_sigma_floor: float,
) -> pd.DataFrame:
    batch_n = len(cases)
    reward_feature_dim = reward_feature_dim_for_loaded_model(model, config)
    reset_rewards = jnp.zeros((batch_n, task.num_nodes), dtype=jnp.float32)
    carry = jp.initial_carry(
        batch_n,
        task,
        config.rnn_units,
        reward_feature_dim,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    carry = jp.reset_done_envs(carry, reset_rewards)
    schedule = schedule_for(config.beta)
    max_len = max(len(case["nodes"]) for case in cases)
    rng = jax.random.PRNGKey(config.seed + 510_000)
    last_trans = None
    continue_probs_by_step = np.full((batch_n, max_len), np.nan, dtype=float)
    forced_action_probs_by_step = np.full((batch_n, max_len), np.nan, dtype=float)
    for step_i in range(max_len):
        forced_actions = []
        forced_observations = []
        for case in cases:
            forced_actions.append(int(case["nodes"][step_i]))
            forced_observations.append(float(case["observations"][step_i]))
        rng, step_rng = jax.random.split(rng)
        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            schedule,
            forced_action=jnp.asarray(forced_actions, dtype=jnp.int32),
            training=True,
            use_posterior_mean=True,
            compute_targets=False,
            forced_observation=jnp.asarray(forced_observations, dtype=jnp.float32),
            method=jp.PlanningVAE.__call__,
        )
        last_trans = jax.device_get(trans)
        probs_step = np.asarray(last_trans.probs, dtype=float)
        continue_probs_by_step[:, step_i] = np.nansum(probs_step[:, :task.num_nodes], axis=1)
        forced_action_probs_by_step[:, step_i] = probs_step[
            np.arange(batch_n),
            np.asarray(forced_actions, dtype=int),
        ]

    stop_action = np.full(batch_n, task.num_nodes, dtype=np.int32)
    rng, step_rng = jax.random.split(rng)
    _, decision_trans = model.apply(
        {"params": params},
        carry,
        step_rng,
        schedule,
        forced_action=jnp.asarray(stop_action, dtype=jnp.int32),
        training=True,
        use_posterior_mean=True,
        compute_targets=False,
        forced_observation=jnp.full((batch_n,), np.nan, dtype=jnp.float32),
        method=jp.PlanningVAE.__call__,
    )
    decision_trans = jax.device_get(decision_trans)
    if last_trans is None:
        return pd.DataFrame()

    z_mu = np.asarray(last_trans.z_mu, dtype=float)
    z_sigma = np.exp(0.5 * np.clip(np.asarray(last_trans.z_logvar, dtype=float), -10.0, 10.0))
    prior_mu = np.asarray(last_trans.prior_mu, dtype=float)
    prior_sigma = np.exp(0.5 * np.clip(np.asarray(last_trans.prior_logvar, dtype=float), -10.0, 10.0))
    decision_probs = np.asarray(decision_trans.probs, dtype=float)
    terminal_probs = np.asarray(decision_trans.action_output, dtype=float)
    rows = []
    for i, case in enumerate(cases):
        row = {
            "case_group": case["case_group"],
            "history_name": case["history_name"],
            "history_nodes": ",".join(str(int(x) + 1) for x in case["nodes"]),
            "history_observations": ",".join(f"{float(x):g}" for x in case["observations"]),
            "history_length": len(case["nodes"]),
            "target_history_variance": float(case.get("target_history_variance", np.nan)),
            "empirical_history_mean": float(np.mean(np.asarray(case["observations"], dtype=float))),
            "empirical_history_variance": population_variance(case["observations"]),
            "sigma_matched_history": bool(
                np.isfinite(float(case.get("target_history_variance", np.nan)))
                and abs(
                    population_variance(case["observations"])
                    - float(case.get("target_history_variance", np.nan))
                ) < 1e-8
            ),
            "informative_sigma_matched_case": bool(abs(float(config.observation_sigma)) > 1e-12),
            "terminal_entropy": entropy(terminal_probs[i]),
            "decision_entropy": entropy(decision_probs[i]),
            "policy_reach_continue_prob": float(np.nanprod(continue_probs_by_step[i])),
            "policy_exact_history_prob": float(np.nanprod(forced_action_probs_by_step[i])),
        }
        for step_i in range(max_len):
            row[f"policy_continue_prob_before_step_{step_i + 1}"] = float(continue_probs_by_step[i, step_i])
            row[f"policy_forced_action_prob_step_{step_i + 1}"] = float(forced_action_probs_by_step[i, step_i])
        row.update(posterior_features_for_history(
            case["nodes"],
            case["observations"],
            task,
            config.observation_sigma,
            posterior_sigma_floor,
        ))
        for k in range(config.latent_dim):
            row[f"z_mu_{k}"] = float(z_mu[i, k])
            row[f"z_sigma_{k}"] = float(z_sigma[i, k])
            row[f"prior_mu_{k}"] = float(prior_mu[i, k])
            row[f"prior_sigma_{k}"] = float(prior_sigma[i, k])
            row[f"prior_norm_z_mu_{k}"] = float((z_mu[i, k] - prior_mu[i, k]) / max(prior_sigma[i, k], 1e-8))
            row[f"prior_norm_z_sigma_{k}"] = float(z_sigma[i, k] / max(prior_sigma[i, k], 1e-8))
        for a in range(decision_probs.shape[1]):
            row[f"next_action_prob_{a}"] = float(decision_probs[i, a])
        for p in range(terminal_probs.shape[1]):
            row[f"terminal_prob_path_{p + 1}"] = float(terminal_probs[i, p])
        rows.append(row)
    return pd.DataFrame(rows)


def entropy(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    total = float(np.sum(p))
    if total <= 0:
        return float("nan")
    p = p / total
    return float(-np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0)))


def pairwise_forced_history_metrics(df: pd.DataFrame, latent_dim: int) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    z_cols = [f"z_mu_{k}" for k in range(latent_dim)]
    pn_cols = [f"prior_norm_z_mu_{k}" for k in range(latent_dim)]
    action_cols = [c for c in df.columns if c.startswith("next_action_prob_")]
    terminal_cols = [c for c in df.columns if c.startswith("terminal_prob_path_")]
    for group, piece in df.groupby("case_group", dropna=False):
        if len(piece) < 2:
            continue
        records = piece.to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a = records[i]
                b = records[j]
                rows.append({
                    "case_group": group,
                    "history_a": a["history_name"],
                    "history_b": b["history_name"],
                    "latent_mu_l2": l2_between_records(a, b, z_cols),
                    "prior_norm_latent_mu_l2": l2_between_records(a, b, pn_cols),
                    "next_action_prob_l1": l1_between_records(a, b, action_cols),
                    "terminal_prob_l1": l1_between_records(a, b, terminal_cols),
                })
    return pd.DataFrame(rows)


def l2_between_records(a: dict, b: dict, cols: list[str]) -> float:
    av = np.asarray([a.get(c, np.nan) for c in cols], dtype=float)
    bv = np.asarray([b.get(c, np.nan) for c in cols], dtype=float)
    ok = np.isfinite(av) & np.isfinite(bv)
    if not np.any(ok):
        return float("nan")
    return float(np.sqrt(np.sum((av[ok] - bv[ok]) ** 2)))


def l1_between_records(a: dict, b: dict, cols: list[str]) -> float:
    av = np.asarray([a.get(c, np.nan) for c in cols], dtype=float)
    bv = np.asarray([b.get(c, np.nan) for c in cols], dtype=float)
    ok = np.isfinite(av) & np.isfinite(bv)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(np.abs(av[ok] - bv[ok])))


def generate_probe_histories(
    *,
    task: jp.TaskSpec,
    rng: np.random.Generator,
    n_states: int,
    max_history_length: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history_lengths = rng.integers(1, max_history_length + 1, size=n_states)
    action_nodes = np.full((n_states, max_history_length), -1, dtype=int)
    observations = np.full((n_states, max_history_length), np.nan, dtype=float)
    true_rewards = rng.choice(np.asarray(task.reward_values, dtype=float), size=(n_states, task.num_nodes))
    for i, length in enumerate(history_lengths):
        nodes = rng.integers(0, task.num_nodes, size=int(length))
        action_nodes[i, :length] = nodes
        means = true_rewards[i, nodes]
        if abs(float(sigma)) > 1e-12:
            observations[i, :length] = means + float(sigma) * rng.normal(size=int(length))
        else:
            observations[i, :length] = means
    return action_nodes, observations, true_rewards, history_lengths


def rollout_probe_states(
    *,
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    action_nodes: np.ndarray,
    observations: np.ndarray,
    true_rewards: np.ndarray,
    history_lengths: np.ndarray,
    posterior_sigma_floor: float,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    max_len = action_nodes.shape[1]
    reward_feature_dim = reward_feature_dim_for_loaded_model(model, config)
    schedule = schedule_for(config.beta)
    path_map = np.asarray(task.path_map, dtype=float)
    for batch_i, start in enumerate(range(0, action_nodes.shape[0], batch_size)):
        end = min(start + batch_size, action_nodes.shape[0])
        batch_nodes = action_nodes[start:end]
        batch_obs = observations[start:end]
        batch_rewards = true_rewards[start:end]
        batch_lengths = history_lengths[start:end]
        n = end - start
        carry = jp.initial_carry(
            n,
            task,
            config.rnn_units,
            reward_feature_dim,
            jp.visited_lstm_feature_dim_for_task(task),
        )
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        rng = jax.random.PRNGKey(config.seed + 610_000 + batch_i)
        last_trans = None
        for step_i in range(max_len):
            active = step_i < batch_lengths
            forced_actions = np.where(active, batch_nodes[:, step_i], task.num_nodes).astype(np.int32)
            forced_obs = np.where(active, batch_obs[:, step_i], np.nan).astype(np.float32)
            rng, step_rng = jax.random.split(rng)
            carry, trans = model.apply(
                {"params": params},
                carry,
                step_rng,
                schedule,
                forced_action=jnp.asarray(forced_actions, dtype=jnp.int32),
                training=True,
                use_posterior_mean=True,
                compute_targets=False,
                forced_observation=jnp.asarray(forced_obs, dtype=jnp.float32),
                method=jp.PlanningVAE.__call__,
            )
            last_trans = jax.device_get(trans)
        if last_trans is None:
            continue
        z_mu = np.asarray(last_trans.z_mu, dtype=float)
        z_sigma = np.exp(0.5 * np.clip(np.asarray(last_trans.z_logvar, dtype=float), -10.0, 10.0))
        z_sample = np.asarray(last_trans.z_sample, dtype=float)
        prior_mu = np.asarray(last_trans.prior_mu, dtype=float)
        prior_sigma = np.exp(0.5 * np.clip(np.asarray(last_trans.prior_logvar, dtype=float), -10.0, 10.0))
        path_rewards = batch_rewards @ path_map.T
        for local_i in range(n):
            length = int(batch_lengths[local_i])
            nodes = batch_nodes[local_i, :length].astype(int)
            obs = batch_obs[local_i, :length].astype(float)
            row = {
                "state_id": int(start + local_i),
                "history_length": length,
                "last_observed_node": int(nodes[-1] + 1),
                "first_observed_node": int(nodes[0] + 1),
                "last_sample_value": float(obs[-1]),
                "first_sample_value": float(obs[0]),
                "true_best_path_value": float(np.max(path_rewards[local_i])),
            }
            for sample_i in range(max_len):
                if sample_i < length:
                    row[f"sample_{sample_i + 1}_node"] = int(nodes[sample_i] + 1)
                    row[f"sample_{sample_i + 1}_value"] = float(obs[sample_i])
                    row[f"sample_{sample_i + 1}_node_filled"] = int(nodes[sample_i] + 1)
                    row[f"sample_{sample_i + 1}_value_filled"] = float(obs[sample_i])
                    row[f"sample_{sample_i + 1}_present"] = 1.0
                else:
                    row[f"sample_{sample_i + 1}_node"] = np.nan
                    row[f"sample_{sample_i + 1}_value"] = np.nan
                    row[f"sample_{sample_i + 1}_node_filled"] = 0.0
                    row[f"sample_{sample_i + 1}_value_filled"] = 0.0
                    row[f"sample_{sample_i + 1}_present"] = 0.0
            row.update(posterior_features_for_history(
                nodes,
                obs,
                task,
                config.observation_sigma,
                posterior_sigma_floor,
            ))
            for path_i, path_value in enumerate(path_rewards[local_i]):
                row[f"actual_path_value_{path_i + 1}"] = float(path_value)
            for k in range(config.latent_dim):
                row[f"z_mu_{k}"] = float(z_mu[local_i, k])
                row[f"z_sample_{k}"] = float(z_sample[local_i, k])
                row[f"z_sigma_{k}"] = float(z_sigma[local_i, k])
                row[f"z_log_sigma_{k}"] = float(np.log(max(z_sigma[local_i, k], 1e-12)))
                row[f"prior_norm_z_mu_{k}"] = float(
                    (z_mu[local_i, k] - prior_mu[local_i, k]) / max(prior_sigma[local_i, k], 1e-8)
                )
                row[f"prior_norm_z_sample_{k}"] = float(
                    (z_sample[local_i, k] - prior_mu[local_i, k]) / max(prior_sigma[local_i, k], 1e-8)
                )
                row[f"prior_norm_z_log_sigma_{k}"] = float(
                    np.log(max(z_sigma[local_i, k], 1e-12) / max(prior_sigma[local_i, k], 1e-8))
                )
            rows.append(row)
    return pd.DataFrame(rows)


def rollout_policy_probe_states(
    *,
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    n_trials: int,
    posterior_sigma_floor: float,
    max_history_length: int,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    path_map = np.asarray(task.path_map, dtype=float)
    reward_feature_dim = reward_feature_dim_for_loaded_model(model, config)
    schedule = schedule_for(config.beta)
    reward_rng = jax.random.PRNGKey(config.seed + 720_000)
    reset_rewards_all = np.asarray(jp.sample_reward_matrix(
        reward_rng,
        int(n_trials),
        task.num_nodes,
        task.reward_values,
    ), dtype=float)
    for batch_i, start in enumerate(range(0, int(n_trials), batch_size)):
        end = min(start + batch_size, int(n_trials))
        batch_rewards = reset_rewards_all[start:end]
        n = end - start
        carry = jp.initial_carry(
            n,
            task,
            config.rnn_units,
            reward_feature_dim,
            jp.visited_lstm_feature_dim_for_task(task),
        )
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        rng = jax.random.PRNGKey(config.seed + 730_000 + batch_i)
        stopped = np.zeros(n, dtype=bool)
        history_nodes = np.full((n, max_history_length), -1, dtype=int)
        history_observations = np.full((n, max_history_length), np.nan, dtype=float)
        history_lengths = np.zeros(n, dtype=int)
        path_rewards = batch_rewards @ path_map.T
        for step_i in range(int(config.num_steps)):
            rng, step_rng = jax.random.split(rng)
            carry, trans = model.apply(
                {"params": params},
                carry,
                step_rng,
                schedule,
                forced_action=None,
                training=True,
                use_posterior_mean=False,
                compute_targets=False,
                method=jp.PlanningVAE.__call__,
            )
            trans = jax.device_get(trans)
            node_index = np.asarray(trans.node_index, dtype=int)
            is_observe = np.asarray(trans.is_observe, dtype=float) > 0.5
            is_stop = np.asarray(trans.is_stop, dtype=float) > 0.5
            sampled_observed_reward = np.asarray(trans.expanded_reward, dtype=float)
            z_mu = np.asarray(trans.z_mu, dtype=float)
            z_sigma = np.exp(0.5 * np.clip(np.asarray(trans.z_logvar, dtype=float), -10.0, 10.0))
            z_sample = np.asarray(trans.z_sample, dtype=float)
            prior_mu = np.asarray(trans.prior_mu, dtype=float)
            prior_sigma = np.exp(0.5 * np.clip(np.asarray(trans.prior_logvar, dtype=float), -10.0, 10.0))
            action_output = np.asarray(trans.action_output, dtype=float)
            include = (~stopped) & is_observe & (node_index >= 0)
            for local_i in np.where(include)[0]:
                slot = int(history_lengths[local_i])
                if slot >= int(max_history_length):
                    continue
                node_i = int(node_index[local_i])
                history_nodes[local_i, slot] = node_i
                history_observations[local_i, slot] = float(sampled_observed_reward[local_i])
                history_lengths[local_i] += 1
                length = int(history_lengths[local_i])
                nodes = history_nodes[local_i, :length].astype(int)
                obs = history_observations[local_i, :length].astype(float)
                row = {
                    "state_id": int(len(rows)),
                    "trial_id": int(start + local_i),
                    "source": "policy",
                    "timestep": int(step_i + 1),
                    "history_length": length,
                    "last_observed_node": int(nodes[-1] + 1),
                    "first_observed_node": int(nodes[0] + 1),
                    "last_sample_value": float(obs[-1]),
                    "first_sample_value": float(obs[0]),
                    "true_best_path_value": float(np.max(path_rewards[local_i])),
                    "terminal_choice_entropy": entropy(action_output[local_i]),
                }
                for sample_i in range(max_history_length):
                    if sample_i < length:
                        row[f"sample_{sample_i + 1}_node"] = int(nodes[sample_i] + 1)
                        row[f"sample_{sample_i + 1}_value"] = float(obs[sample_i])
                        row[f"sample_{sample_i + 1}_node_filled"] = int(nodes[sample_i] + 1)
                        row[f"sample_{sample_i + 1}_value_filled"] = float(obs[sample_i])
                        row[f"sample_{sample_i + 1}_present"] = 1.0
                    else:
                        row[f"sample_{sample_i + 1}_node"] = np.nan
                        row[f"sample_{sample_i + 1}_value"] = np.nan
                        row[f"sample_{sample_i + 1}_node_filled"] = 0.0
                        row[f"sample_{sample_i + 1}_value_filled"] = 0.0
                        row[f"sample_{sample_i + 1}_present"] = 0.0
                row.update(posterior_features_for_history(
                    nodes,
                    obs,
                    task,
                    config.observation_sigma,
                    posterior_sigma_floor,
                ))
                for path_i, path_value in enumerate(path_rewards[local_i]):
                    row[f"actual_path_value_{path_i + 1}"] = float(path_value)
                for k in range(config.latent_dim):
                    row[f"z_mu_{k}"] = float(z_mu[local_i, k])
                    row[f"z_sample_{k}"] = float(z_sample[local_i, k])
                    row[f"z_sigma_{k}"] = float(z_sigma[local_i, k])
                    row[f"z_log_sigma_{k}"] = float(np.log(max(z_sigma[local_i, k], 1e-12)))
                    row[f"prior_norm_z_mu_{k}"] = float(
                        (z_mu[local_i, k] - prior_mu[local_i, k]) / max(prior_sigma[local_i, k], 1e-8)
                    )
                    row[f"prior_norm_z_sample_{k}"] = float(
                        (z_sample[local_i, k] - prior_mu[local_i, k]) / max(prior_sigma[local_i, k], 1e-8)
                    )
                    row[f"prior_norm_z_log_sigma_{k}"] = float(
                        np.log(max(z_sigma[local_i, k], 1e-12) / max(prior_sigma[local_i, k], 1e-8))
                    )
                rows.append(row)
            stopped |= is_stop
    return pd.DataFrame(rows)


def standardize_train_test(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[ok]
    y = y[ok]
    if X.shape[0] < 20 or np.std(y) <= 1e-12:
        return np.empty((0, X.shape[1])), np.empty((0, X.shape[1])), np.asarray([]), np.asarray([])
    order = rng.permutation(X.shape[0])
    test_n = max(1, int(round(test_fraction * X.shape[0])))
    test_idx = order[:test_n]
    train_idx = order[test_n:]
    if train_idx.size < 5:
        return np.empty((0, X.shape[1])), np.empty((0, X.shape[1])), np.asarray([]), np.asarray([])
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std <= 1e-8] = 1.0
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std <= 1e-8:
        y_std = 1.0
    return (X_train - x_mean) / x_std, (X_test - x_mean) / x_std, (y_train - y_mean) / y_std, (y_test - y_mean) / y_std


def ridge_r2(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    ridge_alpha: float,
    test_fraction: float,
) -> tuple[float, int]:
    X_train, X_test, y_train, y_test = standardize_train_test(X, y, rng, test_fraction)
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return float("nan"), 0
    X_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test_design = np.column_stack([np.ones(X_test.shape[0]), X_test])
    penalty = np.eye(X_design.shape[1]) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y_train
    pred = X_test_design @ coef
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan"), int(y_test.shape[0])
    return float(1.0 - ss_res / ss_tot), int(y_test.shape[0])


def finite_design(df: pd.DataFrame, x_cols: list[str], y_col: str) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    if x_cols:
        X = numeric_cols(df, x_cols)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        return X[ok], y[ok]
    ok = np.isfinite(y)
    return np.empty((int(ok.sum()), 0), dtype=float), y[ok]


def split_standardize(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] < 20 or np.std(y) <= 1e-12:
        return np.empty((0, X.shape[1])), np.empty((0, X.shape[1])), np.asarray([]), np.asarray([])
    order = rng.permutation(X.shape[0])
    test_n = max(1, int(round(test_fraction * X.shape[0])))
    test_idx = order[:test_n]
    train_idx = order[test_n:]
    if train_idx.size < 5:
        return np.empty((0, X.shape[1])), np.empty((0, X.shape[1])), np.asarray([]), np.asarray([])
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    if X_train.shape[1]:
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)
        x_std[x_std <= 1e-8] = 1.0
        X_train = (X_train - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std <= 1e-8:
        y_std = 1.0
    return X_train, X_test, (y_train - y_mean) / y_std, (y_test - y_mean) / y_std


def ridge_r2_from_standardized(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ridge_alpha: float,
) -> float:
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return float("nan")
    X_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test_design = np.column_stack([np.ones(X_test.shape[0]), X_test])
    penalty = np.eye(X_design.shape[1]) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y_train
    pred = X_test_design @ coef
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mlp_predict_standardized(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_std: np.ndarray,
    *,
    rng: np.random.Generator,
    hidden_dims: tuple[int, ...] = DEFAULT_DEEP_PROBE_HIDDEN_DIMS,
    epochs: int = DEFAULT_DEEP_PROBE_EPOCHS,
    lr: float = DEFAULT_DEEP_PROBE_LR,
    weight_decay: float = DEFAULT_DEEP_PROBE_WEIGHT_DECAY,
) -> np.ndarray:
    """Train a small deterministic MLP probe on standardized data."""
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return np.asarray([])
    dims = [int(X_train.shape[1]), *[int(d) for d in hidden_dims if int(d) > 0], 1]
    params = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        scale = math.sqrt(2.0 / max(in_dim + out_dim, 1))
        params.append({
            "W": rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(float),
            "b": np.zeros(out_dim, dtype=float),
        })
    moments = [{"mW": np.zeros_like(p["W"]), "vW": np.zeros_like(p["W"]), "mb": np.zeros_like(p["b"]), "vb": np.zeros_like(p["b"])} for p in params]
    y_col = y_train_std.reshape(-1, 1).astype(float)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    n = max(int(X_train.shape[0]), 1)

    def forward(X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        preacts = []
        out = X
        for layer_i, p in enumerate(params):
            z = out @ p["W"] + p["b"]
            preacts.append(z)
            out = z if layer_i == len(params) - 1 else np.tanh(z)
            activations.append(out)
        return activations, preacts

    for step in range(1, max(int(epochs), 1) + 1):
        activations, _ = forward(X_train)
        pred = activations[-1]
        delta = (2.0 / n) * (pred - y_col)
        grads = []
        for layer_i in range(len(params) - 1, -1, -1):
            a_prev = activations[layer_i]
            dW = a_prev.T @ delta + float(weight_decay) * params[layer_i]["W"]
            db = delta.sum(axis=0)
            grads.append((dW, db))
            if layer_i > 0:
                delta = (delta @ params[layer_i]["W"].T) * (1.0 - activations[layer_i] ** 2)
        grads.reverse()
        for layer_i, ((dW, db), p, m) in enumerate(zip(grads, params, moments)):
            m["mW"] = beta1 * m["mW"] + (1.0 - beta1) * dW
            m["vW"] = beta2 * m["vW"] + (1.0 - beta2) * (dW * dW)
            m["mb"] = beta1 * m["mb"] + (1.0 - beta1) * db
            m["vb"] = beta2 * m["vb"] + (1.0 - beta2) * (db * db)
            mW_hat = m["mW"] / (1.0 - beta1 ** step)
            vW_hat = m["vW"] / (1.0 - beta2 ** step)
            mb_hat = m["mb"] / (1.0 - beta1 ** step)
            vb_hat = m["vb"] / (1.0 - beta2 ** step)
            p["W"] -= float(lr) * mW_hat / (np.sqrt(vW_hat) + eps)
            p["b"] -= float(lr) * mb_hat / (np.sqrt(vb_hat) + eps)
    return forward(X_test)[0][-1].reshape(-1)


def predict_from_standardized(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_std: np.ndarray,
    *,
    decoder_type: str,
    rng: np.random.Generator,
    ridge_alpha: float,
    deep_config: dict | None = None,
) -> np.ndarray:
    if decoder_type == "deep":
        deep_config = deep_config or {}
        return mlp_predict_standardized(
            X_train,
            X_test,
            y_train_std,
            rng=rng,
            hidden_dims=tuple(deep_config.get("hidden_dims", DEFAULT_DEEP_PROBE_HIDDEN_DIMS)),
            epochs=int(deep_config.get("epochs", DEFAULT_DEEP_PROBE_EPOCHS)),
            lr=float(deep_config.get("lr", DEFAULT_DEEP_PROBE_LR)),
            weight_decay=float(deep_config.get("weight_decay", DEFAULT_DEEP_PROBE_WEIGHT_DECAY)),
        )
    X_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test_design = np.column_stack([np.ones(X_test.shape[0]), X_test])
    penalty = np.eye(X_design.shape[1]) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y_train_std
    return X_test_design @ coef


def ridge_mse_with_baselines(
    X: np.ndarray,
    y: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_var: np.ndarray,
    *,
    rng: np.random.Generator,
    ridge_alpha: float,
    test_fraction: float,
    decoder_type: str = "linear",
    deep_config: dict | None = None,
) -> tuple[float, float, float, float, float, int]:
    ok = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
        & np.isfinite(baseline_mean)
        & np.isfinite(baseline_var)
    )
    X = X[ok]
    y = y[ok]
    baseline_mean = baseline_mean[ok]
    baseline_var = np.maximum(baseline_var[ok], 0.0)
    if X.shape[0] < 20 or np.std(y) <= 1e-12:
        return (float("nan"),) * 5 + (0,)
    order = rng.permutation(X.shape[0])
    test_n = max(1, int(round(test_fraction * X.shape[0])))
    test_idx = order[:test_n]
    train_idx = order[test_n:]
    if train_idx.size < 5:
        return (float("nan"),) * 5 + (0,)

    X_train = X[train_idx]
    X_test = X[test_idx]
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std <= 1e-8] = 1.0
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = y[train_idx]
    y_test = y[test_idx]
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std <= 1e-8:
        y_std = 1.0
    y_train_std = (y_train - y_mean) / y_std
    pred_std = predict_from_standardized(
        X_train,
        X_test,
        y_train_std,
        decoder_type=decoder_type,
        rng=rng,
        ridge_alpha=ridge_alpha,
        deep_config=deep_config,
    )
    if pred_std.size == 0:
        return (float("nan"),) * 5 + (0,)
    pred = pred_std * y_std + y_mean
    decoder_mse = float(np.mean((y_test - pred) ** 2))
    mean_baseline_mse = float(np.mean((y_test - baseline_mean[test_idx]) ** 2))
    random_mean_var_baseline_mse = float(
        np.mean(baseline_var[test_idx] + (y_test - baseline_mean[test_idx]) ** 2)
    )
    ratio_to_mean = decoder_mse / mean_baseline_mse if mean_baseline_mse > 1e-12 else float("nan")
    ratio_to_random = (
        decoder_mse / random_mean_var_baseline_mse
        if random_mean_var_baseline_mse > 1e-12
        else float("nan")
    )
    return (
        decoder_mse,
        mean_baseline_mse,
        random_mean_var_baseline_mse,
        ratio_to_mean,
        ratio_to_random,
        int(y_test.shape[0]),
    )


def ridge_mae_by_group(
    X: np.ndarray,
    y: np.ndarray,
    group_values: np.ndarray,
    total_values: np.ndarray | None = None,
    extra_group_values: dict[str, np.ndarray] | None = None,
    *,
    rng: np.random.Generator,
    ridge_alpha: float,
    test_fraction: float,
    decoder_type: str = "linear",
    deep_config: dict | None = None,
) -> pd.DataFrame:
    if total_values is None:
        total_values = np.full(len(y), np.nan, dtype=float)
    total_values = np.asarray(total_values, dtype=float)
    extra_group_values = extra_group_values or {}
    extra_arrays = {name: np.asarray(values, dtype=float) for name, values in extra_group_values.items()}
    ok = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
        & np.isfinite(group_values)
        & (np.isfinite(total_values) | np.all(~np.isfinite(total_values)))
    )
    for values in extra_arrays.values():
        ok = ok & (np.isfinite(values) | np.all(~np.isfinite(values)))
    X = X[ok]
    y = y[ok]
    group_values = group_values[ok]
    total_values = total_values[ok]
    extra_arrays = {name: values[ok] for name, values in extra_arrays.items()}
    if X.shape[0] < 20 or np.std(y) <= 1e-12:
        return pd.DataFrame()
    order = rng.permutation(X.shape[0])
    test_n = max(1, int(round(test_fraction * X.shape[0])))
    test_idx = order[:test_n]
    train_idx = order[test_n:]
    if train_idx.size < 5:
        return pd.DataFrame()

    X_train = X[train_idx]
    X_test = X[test_idx]
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std <= 1e-8] = 1.0
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = y[train_idx]
    y_test = y[test_idx]
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std <= 1e-8:
        y_std = 1.0
    y_train_std = (y_train - y_mean) / y_std
    pred_std = predict_from_standardized(
        X_train,
        X_test,
        y_train_std,
        decoder_type=decoder_type,
        rng=rng,
        ridge_alpha=ridge_alpha,
        deep_config=deep_config,
    )
    if pred_std.size == 0:
        return pd.DataFrame()
    pred = pred_std * y_std + y_mean
    abs_errors = np.abs(y_test - pred)
    sq_errors = np.square(y_test - pred)
    error_df = pd.DataFrame({
        "timestep_before_current_observation": group_values[test_idx],
        "observations_in_trial": total_values[test_idx],
        "abs_error": abs_errors,
        "sq_error": sq_errors,
    })
    for name, values in extra_arrays.items():
        error_df[name] = values[test_idx]
    rows = []
    group_cols = ["timestep_before_current_observation"]
    if np.any(np.isfinite(error_df["observations_in_trial"])):
        group_cols.append("observations_in_trial")
    for name in extra_arrays:
        if name in error_df.columns and np.any(np.isfinite(error_df[name])):
            group_cols.append(name)
    for key, piece in error_df.groupby(group_cols, dropna=False):
        if isinstance(key, tuple):
            timestep = key[0]
            total_observations = key[1] if len(key) > 1 else np.nan
            extra_values = dict(zip(group_cols[2:], key[2:]))
        else:
            timestep = key
            total_observations = np.nan
            extra_values = {}
        vals = pd.to_numeric(piece["abs_error"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        sq_vals = pd.to_numeric(piece["sq_error"], errors="coerce").to_numpy(dtype=float)
        sq_vals = sq_vals[np.isfinite(sq_vals)]
        if vals.size == 0 or sq_vals.size == 0:
            continue
        row = {
            "timestep_before_current_observation": float(timestep),
            "observations_in_trial": float(total_observations) if pd.notna(total_observations) else np.nan,
            "decoder_mae": float(np.mean(vals)),
            "decoder_mse": float(np.mean(sq_vals)),
            "n_test": int(vals.size),
        }
        for name, value in extra_values.items():
            row[name] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def partial_r2_by_arrays(
    X_base: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    ridge_alpha: float,
    test_fraction: float,
) -> tuple[float, float, float, int]:
    ok = (
        np.all(np.isfinite(X_base), axis=1)
        & np.all(np.isfinite(X_full), axis=1)
        & np.isfinite(y)
    )
    X_base = X_base[ok]
    X_full = X_full[ok]
    y = y[ok]
    if X_full.shape[0] < 20 or np.std(y) <= 1e-12:
        return float("nan"), float("nan"), float("nan"), 0
    order = rng.permutation(X_full.shape[0])
    test_n = max(1, int(round(test_fraction * X_full.shape[0])))
    test_idx = order[:test_n]
    train_idx = order[test_n:]
    if train_idx.size < 5:
        return float("nan"), float("nan"), float("nan"), 0

    def standardize_split(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_train = X[train_idx]
        X_test = X[test_idx]
        if X_train.shape[1] == 0:
            return X_train, X_test
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)
        x_std[x_std <= 1e-8] = 1.0
        return (X_train - x_mean) / x_std, (X_test - x_mean) / x_std

    y_train = y[train_idx]
    y_test = y[test_idx]
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std <= 1e-8:
        y_std = 1.0
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    Xb_train, Xb_test = standardize_split(X_base)
    Xf_train, Xf_test = standardize_split(X_full)
    r2_base = ridge_r2_from_standardized(Xb_train, Xb_test, y_train, y_test, ridge_alpha)
    r2_full = ridge_r2_from_standardized(Xf_train, Xf_test, y_train, y_test, ridge_alpha)
    return r2_base, r2_full, float(r2_full - r2_base), int(y_test.shape[0])


def numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def feature_sets(df: pd.DataFrame, task: jp.TaskSpec, max_history_length: int) -> dict[str, list[str]]:
    posterior_cols = []
    for node_i in range(task.num_nodes):
        posterior_cols.extend([f"posterior_mean_node_{node_i + 1}", f"posterior_var_node_{node_i + 1}"])
    for path_i in range(task.num_paths):
        posterior_cols.extend([f"posterior_mean_path_{path_i + 1}", f"posterior_var_path_{path_i + 1}"])
    sample_value_cols = [f"sample_{i + 1}_value_filled" for i in range(max_history_length)]
    sample_node_cols = [f"sample_{i + 1}_node_filled" for i in range(max_history_length)]
    sample_present_cols = [f"sample_{i + 1}_present" for i in range(max_history_length)]
    count_cols = [f"count_node_{i + 1}" for i in range(task.num_nodes)] + ["history_length"]
    sets = {
        "posterior_stats": [c for c in posterior_cols if c in df],
        "sample_values": [c for c in sample_value_cols if c in df],
        "sample_order_nodes": [
            c for c in sample_node_cols + sample_present_cols + ["first_observed_node", "last_observed_node"] if c in df
        ],
        "counts": [c for c in count_cols if c in df],
    }
    sets["sample_values_and_order"] = sets["sample_values"] + sets["sample_order_nodes"] + sets["counts"]
    sets["all_features"] = sorted(set(sets["posterior_stats"] + sets["sample_values_and_order"]))
    return sets


def latent_feature_representations(df: pd.DataFrame, latent_dim: int) -> dict[str, list[str]]:
    raw = [f"z_mu_{k}" for k in range(latent_dim)]
    raw_full = raw + [f"z_log_sigma_{k}" for k in range(latent_dim)]
    prior_norm = [f"prior_norm_z_mu_{k}" for k in range(latent_dim)]
    prior_norm_full = prior_norm + [f"prior_norm_z_log_sigma_{k}" for k in range(latent_dim)]
    reps = {
        "z_mu": [c for c in raw if c in df],
        "z_mu_logsigma": [c for c in raw_full if c in df],
        "prior_norm_z_mu": [c for c in prior_norm if c in df],
        "prior_norm_z_mu_logsigma": [c for c in prior_norm_full if c in df],
    }
    pga = [c for c in ["pga_score_0", "pga_score_1"] if c in df]
    if pga:
        reps["pga_scores"] = pga
    return reps


def decoder_targets(df: pd.DataFrame, task: jp.TaskSpec, max_history_length: int) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for node_i in range(min(task.num_nodes, 2)):
        targets.append((f"posterior_mean_node_{node_i + 1}", "posterior_stats"))
        targets.append((f"posterior_var_node_{node_i + 1}", "posterior_stats"))
    for path_i in range(min(task.num_paths, 2)):
        targets.append((f"posterior_mean_path_{path_i + 1}", "posterior_stats"))
        targets.append((f"posterior_var_path_{path_i + 1}", "posterior_stats"))
    for sample_i in range(max_history_length):
        targets.append((f"sample_{sample_i + 1}_value", "individual_sample_values"))
        targets.append((f"sample_{sample_i + 1}_node", "sample_order"))
    targets.extend([
        ("first_sample_value", "individual_sample_values"),
        ("last_sample_value", "individual_sample_values"),
        ("first_observed_node", "sample_order"),
        ("last_observed_node", "sample_order"),
        ("history_length", "sample_order"),
    ])
    return [(name, family) for name, family in targets if name in df]


def run_latent_decoders(
    df: pd.DataFrame,
    task: jp.TaskSpec,
    latent_dim: int,
    max_history_length: int,
    seed: int,
    ridge_alpha: float,
    test_fraction: float,
) -> pd.DataFrame:
    reps = latent_feature_representations(df, latent_dim)
    targets = decoder_targets(df, task, max_history_length)
    rows = []
    for rep_name, rep_cols in reps.items():
        if not rep_cols:
            continue
        X = numeric_cols(df, rep_cols)
        for target_name, target_family in targets:
            y = pd.to_numeric(df[target_name], errors="coerce").to_numpy(dtype=float)
            r2, n_test = ridge_r2(
                X,
                y,
                np.random.default_rng(seed + stable_hash(rep_name + target_name)),
                ridge_alpha,
                test_fraction,
            )
            rows.append({
                "representation": rep_name,
                "target": target_name,
                "target_family": target_family,
                "r2": r2,
                "n_test": n_test,
                "n_features": len(rep_cols),
            })
    return pd.DataFrame(rows)


def run_variance_decomposition(
    df: pd.DataFrame,
    task: jp.TaskSpec,
    latent_dim: int,
    max_history_length: int,
    seed: int,
    ridge_alpha: float,
    test_fraction: float,
) -> pd.DataFrame:
    feature_groups = feature_sets(df, task, max_history_length)
    coord_cols = [f"z_mu_{k}" for k in range(latent_dim)]
    coord_cols += [f"prior_norm_z_mu_{k}" for k in range(latent_dim)]
    coord_cols += [c for c in ["pga_score_0", "pga_score_1"] if c in df]
    rows = []
    for coord in coord_cols:
        if coord not in df:
            continue
        y = pd.to_numeric(df[coord], errors="coerce").to_numpy(dtype=float)
        coord_type = "pga" if coord.startswith("pga_") else ("prior_norm_latent_dim" if coord.startswith("prior_norm_") else "latent_dim")
        for group_name, cols in feature_groups.items():
            if not cols:
                continue
            X = numeric_cols(df, cols)
            r2, n_test = ridge_r2(
                X,
                y,
                np.random.default_rng(seed + stable_hash(coord + group_name)),
                ridge_alpha,
                test_fraction,
            )
            rows.append({
                "coordinate": coord,
                "coordinate_type": coord_type,
                "feature_group": group_name,
                "r2": r2,
                "n_test": n_test,
                "n_features": len(cols),
            })
    return pd.DataFrame(rows)


def regex_like_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if any(str(c).startswith(prefix) for prefix in prefixes)]


def infer_feature_sets_from_probe(df: pd.DataFrame) -> dict[str, list[str]]:
    posterior_cols = [
        c for c in df.columns
        if str(c).startswith(("posterior_mean_node_", "posterior_var_node_", "posterior_mean_path_", "posterior_var_path_"))
    ]
    sample_value_cols = [
        c for c in df.columns
        if str(c).startswith("sample_") and str(c).endswith("_value_filled")
    ]
    sample_node_cols = [
        c for c in df.columns
        if str(c).startswith("sample_") and str(c).endswith("_node_filled")
    ]
    sample_present_cols = [
        c for c in df.columns
        if str(c).startswith("sample_") and str(c).endswith("_present")
    ]
    count_cols = [c for c in df.columns if str(c).startswith("count_node_")]
    context_cols = [c for c in count_cols + ["history_length", "timestep"] if c in df.columns]
    order_cols = [
        c for c in sample_node_cols + sample_present_cols + ["first_observed_node", "last_observed_node"]
        if c in df.columns
    ]
    history_cols = list(dict.fromkeys(sample_value_cols + order_cols + context_cols))
    return {
        "posterior_stats": posterior_cols,
        "sample_values": sample_value_cols,
        "sample_order": order_cols,
        "history_context": context_cols,
        "sample_values_and_order": history_cols,
        "posterior_plus_context": list(dict.fromkeys(posterior_cols + context_cols)),
    }


def infer_coordinate_groups_from_probe(df: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "latent_mu": [c for c in df.columns if str(c).startswith("z_mu_")],
        "prior_norm_latent_mu": [c for c in df.columns if str(c).startswith("prior_norm_z_mu_")],
        "pga": [c for c in ["pga_score_0", "pga_score_1"] if c in df.columns],
    }
    return {name: cols for name, cols in groups.items() if cols}


def latent_representations_from_probe(
    df: pd.DataFrame,
    representation_mode: str = "sample",
) -> dict[str, list[str]]:
    def sort_latent_cols(prefix: str) -> list[str]:
        cols = [c for c in df.columns if str(c).startswith(prefix)]

        def key(col: str) -> int:
            try:
                return int(str(col).rsplit("_", 1)[-1])
            except ValueError:
                return 10_000

        return sorted(cols, key=key)

    z_mu = sort_latent_cols("z_mu_")
    z_log_sigma = sort_latent_cols("z_log_sigma_")
    z_sample = sort_latent_cols("z_sample_")
    prior_norm_z_mu = sort_latent_cols("prior_norm_z_mu_")
    prior_norm_z_log_sigma = sort_latent_cols("prior_norm_z_log_sigma_")
    prior_norm_z_sample = sort_latent_cols("prior_norm_z_sample_")
    reps = {
        "z_sample": z_sample,
        "prior_norm_z_sample": prior_norm_z_sample,
        "z_mu": z_mu,
        "z_mu_logsigma": z_mu + z_log_sigma,
        "prior_norm_z_mu": prior_norm_z_mu,
        "prior_norm_z_mu_logsigma": prior_norm_z_mu + prior_norm_z_log_sigma,
    }
    pga = [c for c in ["pga_score_0", "pga_score_1"] if c in df.columns]
    if pga:
        reps["pga_scores"] = pga
    reps = {name: cols for name, cols in reps.items() if cols}
    if representation_mode == "sample":
        keep = {"z_sample", "prior_norm_z_sample"}
    elif representation_mode == "minimal":
        keep = {"z_mu", "prior_norm_z_mu"}
    elif representation_mode == "mu_logsigma":
        keep = {"z_mu", "z_mu_logsigma", "prior_norm_z_mu", "prior_norm_z_mu_logsigma"}
    else:
        keep = set(reps)
    return {name: cols for name, cols in reps.items() if name in keep}


def ensure_sampled_latent_columns(df: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    if df.empty:
        return df
    if any(str(c).startswith("z_sample_") for c in df.columns):
        return df
    z_mu_cols = sorted(
        [c for c in df.columns if str(c).startswith("z_mu_")],
        key=lambda c: int(str(c).rsplit("_", 1)[-1]),
    )
    if not z_mu_cols:
        return df
    z_sigma_cols = sorted(
        [c for c in df.columns if str(c).startswith("z_sigma_")],
        key=lambda c: int(str(c).rsplit("_", 1)[-1]),
    )
    if z_sigma_cols:
        sigma = numeric_cols(df, z_sigma_cols)
    else:
        z_log_sigma_cols = sorted(
            [c for c in df.columns if str(c).startswith("z_log_sigma_")],
            key=lambda c: int(str(c).rsplit("_", 1)[-1]),
        )
        if not z_log_sigma_cols:
            return df
        sigma = np.exp(numeric_cols(df, z_log_sigma_cols))
    mu = numeric_cols(df, z_mu_cols)
    if mu.shape != sigma.shape:
        return df
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=mu.shape)
    sampled = mu + sigma * eps
    out = df.copy()
    for k in range(sampled.shape[1]):
        out[f"z_sample_{k}"] = sampled[:, k]
    prior_mu_cols = sorted(
        [c for c in df.columns if str(c).startswith("prior_mu_")],
        key=lambda c: int(str(c).rsplit("_", 1)[-1]),
    )
    prior_sigma_cols = sorted(
        [c for c in df.columns if str(c).startswith("prior_sigma_")],
        key=lambda c: int(str(c).rsplit("_", 1)[-1]),
    )
    if prior_mu_cols and prior_sigma_cols and len(prior_mu_cols) == sampled.shape[1]:
        prior_mu = numeric_cols(df, prior_mu_cols)
        prior_sigma = np.maximum(numeric_cols(df, prior_sigma_cols), 1e-8)
        prior_norm = (sampled - prior_mu) / prior_sigma
    else:
        prior_norm_mu_cols = sorted(
            [c for c in df.columns if str(c).startswith("prior_norm_z_mu_")],
            key=lambda c: int(str(c).rsplit("_", 1)[-1]),
        )
        prior_norm_log_sigma_cols = sorted(
            [c for c in df.columns if str(c).startswith("prior_norm_z_log_sigma_")],
            key=lambda c: int(str(c).rsplit("_", 1)[-1]),
        )
        if prior_norm_mu_cols and prior_norm_log_sigma_cols:
            prior_norm_mu = numeric_cols(df, prior_norm_mu_cols)
            prior_norm_sigma = np.exp(numeric_cols(df, prior_norm_log_sigma_cols))
            prior_norm = prior_norm_mu + prior_norm_sigma * eps
        else:
            prior_norm = None
    if prior_norm is not None:
        for k in range(prior_norm.shape[1]):
            out[f"prior_norm_z_sample_{k}"] = prior_norm[:, k]
    return out


def sample_sequence_indices(df: pd.DataFrame) -> list[int]:
    indices = []
    for col in df.columns:
        match = re.fullmatch(r"sample_(\d+)_value", str(col))
        if match and f"sample_{match.group(1)}_node" in df.columns:
            indices.append(int(match.group(1)))
    return sorted(indices)


def ensure_node_value_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or any(re.fullmatch(r"node_\d+_visit_\d+_value", str(c)) for c in df.columns):
        return df
    sample_indices = sample_sequence_indices(df)
    if not sample_indices:
        return df
    node_cols = [f"sample_{i}_node" for i in sample_indices]
    value_cols = [f"sample_{i}_value" for i in sample_indices]
    node_arr = df[node_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    value_arr = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(node_arr) & np.isfinite(value_arr) & (node_arr > 0)
    if not np.any(valid):
        return df
    max_node = int(np.nanmax(node_arr[valid]))
    max_visits = len(sample_indices)
    out = df.copy()
    for node_id in range(1, max_node + 1):
        visit_values = np.full((len(out), max_visits), np.nan, dtype=float)
        visit_count = np.zeros(len(out), dtype=int)
        for sample_pos in range(max_visits):
            valid_rows = np.where(valid[:, sample_pos])[0]
            if valid_rows.size == 0:
                continue
            row_idx = valid_rows[node_arr[valid_rows, sample_pos].astype(int) == node_id]
            if row_idx.size == 0:
                continue
            slots = visit_count[row_idx]
            in_range = slots < max_visits
            row_idx = row_idx[in_range]
            slots = slots[in_range]
            visit_values[row_idx, slots] = value_arr[row_idx, sample_pos]
            visit_count[row_idx] += 1
        out[f"node_{node_id}_visit_count"] = visit_count.astype(float)
        for visit_i in range(max_visits):
            values = visit_values[:, visit_i]
            out[f"node_{node_id}_visit_{visit_i + 1}_value"] = values
            out[f"node_{node_id}_visit_{visit_i + 1}_present"] = np.isfinite(values).astype(float)
    return out


def node_visit_value_columns(df: pd.DataFrame) -> list[str]:
    def key(col: str) -> tuple[int, int]:
        match = re.fullmatch(r"node_(\d+)_visit_(\d+)_value", str(col))
        if not match:
            return (10_000, 10_000)
        return int(match.group(1)), int(match.group(2))

    return sorted(
        [c for c in df.columns if re.fullmatch(r"node_\d+_visit_\d+_value", str(c))],
        key=key,
    )


def node_visit_present_columns(df: pd.DataFrame) -> list[str]:
    def key(col: str) -> tuple[int, int]:
        match = re.fullmatch(r"node_(\d+)_visit_(\d+)_present", str(col))
        if not match:
            return (10_000, 10_000)
        return int(match.group(1)), int(match.group(2))

    return sorted(
        [c for c in df.columns if re.fullmatch(r"node_\d+_visit_\d+_present", str(c))],
        key=key,
    )


def node_id_from_visit_value_col(target_name: str) -> int | None:
    match = re.fullmatch(r"node_(\d+)_visit_(\d+)_value", str(target_name))
    return int(match.group(1)) if match else None


def node_visit_index_from_visit_value_col(target_name: str) -> int | None:
    match = re.fullmatch(r"node_(\d+)_visit_(\d+)_value", str(target_name))
    return int(match.group(2)) if match else None


def conditional_decoder_targets_from_probe(
    df: pd.DataFrame,
    target_mode: str = "minimal",
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if target_mode == "minimal":
        candidates = [
            ("first_sample_value", "individual_sample_values"),
            ("last_sample_value", "individual_sample_values"),
            ("first_observed_node", "sample_order"),
            ("last_observed_node", "sample_order"),
        ]
        return [(name, family) for name, family in candidates if name in df.columns]
    for col in df.columns:
        text = str(col)
        if text.startswith("sample_") and text.endswith("_value"):
            targets.append((text, "individual_sample_values"))
        elif text in {"first_sample_value", "last_sample_value"}:
            targets.append((text, "individual_sample_values"))
        elif target_mode == "all":
            if text.startswith("sample_") and text.endswith("_node"):
                targets.append((text, "sample_order"))
            elif text in {"first_observed_node", "last_observed_node"}:
                targets.append((text, "sample_order"))
    return targets


def metadata_group_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in [
            "family",
            "parameter_name",
            "parameter_value",
            "parameter_label",
            "seed",
            "beta",
            "lambda",
            "alpha",
            "opportunity_cost",
            "sigma",
            "rnn_dim",
            "latent_dim",
            "tree_type",
        ]
        if c in df.columns
    ]


def run_conditional_variance_decomposition_from_probe(
    probe: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
) -> pd.DataFrame:
    """Partial R2 tests that explicitly condition out correlated explanations.

    For each latent coordinate, this asks two directional questions:
      * posterior_given_history: do posterior stats add explanatory power after
        the individual sample values, order, counts, and timestep are known?
      * history_given_posterior: do sample/order features add explanatory power
        after posterior stats, counts, and timestep are known?
    """
    if probe.empty:
        return pd.DataFrame()
    coord_groups = infer_coordinate_groups_from_probe(probe)
    group_cols = metadata_group_columns(probe)
    rows = []
    for key, piece in probe.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        seed = int(float(meta.get("seed", 0))) if "seed" in meta else 0
        local_features = infer_feature_sets_from_probe(piece)
        local_coords = infer_coordinate_groups_from_probe(piece)
        posterior_cols = local_features["posterior_stats"]
        history_cols = local_features["sample_values_and_order"]
        context_cols = local_features["history_context"]
        if not posterior_cols or not history_cols:
            continue
        tests = [
            (
                "posterior_given_history",
                history_cols,
                [c for c in posterior_cols if c not in history_cols],
            ),
            (
                "history_given_posterior",
                local_features["posterior_plus_context"],
                [c for c in history_cols if c not in local_features["posterior_plus_context"]],
            ),
            (
                "sample_order_given_posterior",
                local_features["posterior_plus_context"],
                [c for c in local_features["sample_order"] if c not in local_features["posterior_plus_context"]],
            ),
            (
                "sample_values_given_posterior",
                local_features["posterior_plus_context"],
                [c for c in local_features["sample_values"] if c not in local_features["posterior_plus_context"]],
            ),
        ]
        for coord_type, coords in local_coords.items():
            for coord in coords:
                y = pd.to_numeric(piece[coord], errors="coerce").to_numpy(dtype=float)
                for test_name, base_cols, add_cols in tests:
                    if not add_cols:
                        continue
                    base = numeric_cols(piece, base_cols) if base_cols else np.empty((len(piece), 0), dtype=float)
                    added = numeric_cols(piece, add_cols)
                    full = np.column_stack([base, added]) if base.shape[1] else added
                    r2_base, r2_full, delta_r2, n_test = partial_r2_by_arrays(
                        base,
                        full,
                        y,
                        rng=np.random.default_rng(seed + stable_hash(coord + test_name)),
                        ridge_alpha=ridge_alpha,
                        test_fraction=test_fraction,
                    )
                    row = dict(meta)
                    row.update({
                        "coordinate": coord,
                        "coordinate_type": coord_type,
                        "test": test_name,
                        "r2_base": r2_base,
                        "r2_full": r2_full,
                        "delta_r2": delta_r2,
                        "n_test": n_test,
                        "n_base_features": len(base_cols),
                        "n_added_features": len(add_cols),
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


def run_conditional_latent_decoders_from_probe(
    probe: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
    target_mode: str = "minimal",
    representation_mode: str = "minimal",
) -> pd.DataFrame:
    """Decode sample/order targets from latent features after posterior controls.

    This is the direct test for sample-specific information:
      target ~ posterior stats + counts + timestep + latent representation

    The reported delta R2 is the improvement from adding the latent
    representation to the posterior/context baseline.
    """
    if probe.empty:
        return pd.DataFrame()
    group_cols = metadata_group_columns(probe)
    rows = []
    for key, piece in probe.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        seed = int(float(meta.get("seed", 0))) if "seed" in meta else 0
        local_features = infer_feature_sets_from_probe(piece)
        base_cols = local_features["posterior_plus_context"]
        reps = latent_representations_from_probe(piece, representation_mode)
        targets = conditional_decoder_targets_from_probe(piece, target_mode)
        if not base_cols or not reps or not targets:
            continue
        base = numeric_cols(piece, base_cols)
        for rep_name, rep_cols in reps.items():
            latent = numeric_cols(piece, rep_cols)
            full = np.column_stack([base, latent])
            for target_name, target_family in targets:
                y = pd.to_numeric(piece[target_name], errors="coerce").to_numpy(dtype=float)
                r2_base, r2_full, delta_r2, n_test = partial_r2_by_arrays(
                    base,
                    full,
                    y,
                    rng=np.random.default_rng(seed + stable_hash(rep_name + target_name + "conditional_decoder")),
                    ridge_alpha=ridge_alpha,
                    test_fraction=test_fraction,
                )
                row = dict(meta)
                row.update({
                    "representation": rep_name,
                    "target": target_name,
                    "target_family": target_family,
                    "r2_base": r2_base,
                    "r2_full": r2_full,
                    "delta_r2": delta_r2,
                    "n_test": n_test,
                    "n_base_features": len(base_cols),
                    "n_latent_features": len(rep_cols),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def paid_observation_states(probe: pd.DataFrame) -> pd.DataFrame:
    """Return observation latents that are actually carried to another observation.

    In the LSTM VAE, the KL generated after observing reward k is paid only if
    the agent observes reward k + 1. If the next action is stop, that final
    observation latent should not be used for memory-cost decoding probes.
    """
    if probe.empty or "trial_id" not in probe.columns:
        return pd.DataFrame()
    sort_cols = [c for c in ["trial_id", "timestep", "history_length"] if c in probe.columns]
    work = probe.sort_values(sort_cols).copy()
    group_cols = metadata_group_columns(work) + ["trial_id"]
    grouped = work.groupby(group_cols, dropna=False)
    work["observation_index_in_trial"] = grouped.cumcount() + 1
    count_col = "state_id" if "state_id" in work.columns else work.columns[0]
    work["observations_in_trial"] = grouped[count_col].transform("size")
    work["latent_paid_forward"] = work["observation_index_in_trial"] < work["observations_in_trial"]
    paid = work.loc[work["latent_paid_forward"]].reset_index(drop=True)
    return ensure_node_value_history_columns(paid)


def terminal_pre_stop_states(probe: pd.DataFrame) -> pd.DataFrame:
    """Return the final observation latent before the stop/forced-stop choice.

    This is intentionally different from paid_observation_states(): the final
    observation latent is the state used to choose stop, but its KL is not paid
    if the next action is stop.
    """
    if probe.empty or "trial_id" not in probe.columns:
        return pd.DataFrame()
    sort_cols = [c for c in ["trial_id", "timestep", "history_length"] if c in probe.columns]
    work = probe.sort_values(sort_cols).copy()
    group_cols = metadata_group_columns(work) + ["trial_id"]
    grouped = work.groupby(group_cols, dropna=False)
    work["observation_index_in_trial"] = grouped.cumcount() + 1
    count_col = "state_id" if "state_id" in work.columns else work.columns[0]
    work["observations_in_trial"] = grouped[count_col].transform("size")
    work["latent_paid_forward"] = work["observation_index_in_trial"] < work["observations_in_trial"]
    idx = grouped.tail(1).index
    return ensure_node_value_history_columns(work.loc[idx].reset_index(drop=True))


def terminal_history_targets_from_probe(df: pd.DataFrame, target_mode: str = "all") -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if "history_length" in df.columns:
        targets.append(("history_length", "history_length"))
    node_value_cols = node_visit_value_columns(df)
    node_present_cols = node_visit_present_columns(df)
    if target_mode == "minimal":
        first_visit_cols = [c for c in node_value_cols if "_visit_1_value" in str(c)]
        return targets + [(name, "node_value_history") for name in first_visit_cols]
    if target_mode in {"values", "all"}:
        targets.extend((name, "node_value_history") for name in node_value_cols)
    if target_mode == "all":
        targets.extend((name, "node_visit_presence") for name in node_present_cols)
    seen = set()
    unique_targets = []
    for target in targets:
        if target[0] in seen:
            continue
        unique_targets.append(target)
        seen.add(target[0])
    return unique_targets


def quadratic_features(X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return X
    pieces = [X]
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            pieces.append((X[:, i] * X[:, j])[:, None])
    return np.column_stack(pieces)


def run_terminal_history_decoders_from_probe(
    probe: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
    target_mode: str = "all",
    representation_mode: str = "minimal",
) -> pd.DataFrame:
    """Decode observed history from latents that are paid forward."""
    terminal = paid_observation_states(probe)
    if terminal.empty:
        return pd.DataFrame()
    group_cols = metadata_group_columns(terminal)
    rows = []
    for key, piece in terminal.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        seed = int(float(meta.get("seed", 0))) if "seed" in meta else 0
        reps = latent_representations_from_probe(piece, representation_mode)
        targets = terminal_history_targets_from_probe(piece, target_mode)
        if not reps or not targets:
            continue
        for rep_name, rep_cols in reps.items():
            X_linear = numeric_cols(piece, rep_cols)
            design_mats = {
                "linear": X_linear,
                "quadratic": quadratic_features(X_linear),
            }
            for decoder_type, X in design_mats.items():
                for target_name, target_family in targets:
                    y = pd.to_numeric(piece[target_name], errors="coerce").to_numpy(dtype=float)
                    r2, n_test = ridge_r2(
                        X,
                        y,
                        np.random.default_rng(seed + stable_hash(rep_name + decoder_type + target_name + "terminal_history")),
                        ridge_alpha,
                        test_fraction,
                    )
                    row = dict(meta)
                    row.update({
                        "representation": rep_name,
                        "decoder_type": decoder_type,
                        "target": target_name,
                        "target_family": target_family,
                        "r2": r2,
                        "n_test": n_test,
                        "n_features": X.shape[1],
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


def run_terminal_history_conditional_decoders_from_probe(
    probe: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
    target_mode: str = "all",
    representation_mode: str = "sample",
) -> pd.DataFrame:
    """Decode paid-forward history only beyond posterior/context controls."""
    terminal = paid_observation_states(probe)
    if terminal.empty:
        return pd.DataFrame()
    group_cols = metadata_group_columns(terminal)
    rows = []
    for key, piece in terminal.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        seed = int(float(meta.get("seed", 0))) if "seed" in meta else 0
        feature_sets_local = infer_feature_sets_from_probe(piece)
        base_cols = feature_sets_local["posterior_plus_context"]
        reps = latent_representations_from_probe(piece, representation_mode)
        targets = [
            (target, family)
            for target, family in terminal_history_targets_from_probe(piece, target_mode)
            if family != "history_length"
        ]
        if not base_cols or not reps or not targets:
            continue
        base = numeric_cols(piece, base_cols)
        for rep_name, rep_cols in reps.items():
            X_linear = numeric_cols(piece, rep_cols)
            design_mats = {
                "linear": X_linear,
                "quadratic": quadratic_features(X_linear),
            }
            for decoder_type, latent_design in design_mats.items():
                full = np.column_stack([base, latent_design])
                for target_name, target_family in targets:
                    y = pd.to_numeric(piece[target_name], errors="coerce").to_numpy(dtype=float)
                    r2_base, r2_full, delta_r2, n_test = partial_r2_by_arrays(
                        base,
                        full,
                        y,
                        rng=np.random.default_rng(
                            seed
                            + stable_hash(rep_name + decoder_type + target_name + "terminal_history_conditional")
                        ),
                        ridge_alpha=ridge_alpha,
                        test_fraction=test_fraction,
                    )
                    row = dict(meta)
                    row.update({
                        "representation": rep_name,
                        "decoder_type": decoder_type,
                        "target": target_name,
                        "target_family": target_family,
                        "r2_base": r2_base,
                        "r2_full": r2_full,
                        "delta_r2": delta_r2,
                        "n_test": n_test,
                        "n_base_features": len(base_cols),
                        "n_latent_features": latent_design.shape[1],
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


def terminal_sample_value_targets_from_probe(df: pd.DataFrame, target_mode: str = "all") -> list[str]:
    if target_mode == "minimal":
        return [c for c in node_visit_value_columns(df) if "_visit_1_value" in str(c)]
    return node_visit_value_columns(df)


def reward_support_for_input_type(input_type: str) -> np.ndarray:
    if str(input_type).lower() == "binary":
        return np.asarray([0.0, 1.0], dtype=float)
    return np.asarray([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], dtype=float)


def observation_prior_mean_var(df: pd.DataFrame, input_type: str) -> tuple[np.ndarray, np.ndarray]:
    support = reward_support_for_input_type(input_type)
    prior_mean = float(np.mean(support))
    prior_var = float(np.mean((support - prior_mean) ** 2))
    if "sigma" in df.columns:
        sigma = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(dtype=float)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    else:
        sigma = np.zeros(len(df), dtype=float)
    return (
        np.full(len(df), prior_mean, dtype=float),
        np.full(len(df), prior_var, dtype=float) + np.square(sigma),
    )


def history_value_mean_var(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    value_cols = [
        c for c in df.columns
        if str(c).startswith("sample_") and str(c).endswith("_value")
    ]
    if not value_cols:
        n = len(df)
        return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float)
    values = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1)
    var = np.nanmean((values - mean[:, None]) ** 2, axis=1)
    return mean, var


def node_history_value_mean_var(
    df: pd.DataFrame,
    target_name: str,
    *,
    input_type: str = "uniform",
    leave_one_out: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/variance of same-node observed values, optionally excluding target visit.

    When leave_one_out is true, target node_i_visit_j_value is predicted from
    the other visits to node i. Rows with no other visit fall back to the task
    prior observation distribution, including observation noise sigma^2.
    """
    match = re.fullmatch(r"node_(\d+)_visit_(\d+)_value", str(target_name))
    node_id = int(match.group(1)) if match else None
    target_visit = int(match.group(2)) if match else None
    if node_id is None:
        return history_value_mean_var(df)
    value_cols = [
        c for c in node_visit_value_columns(df)
        if str(c).startswith(f"node_{node_id}_visit_")
    ]
    if not value_cols:
        return history_value_mean_var(df)
    values = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if leave_one_out and target_visit is not None:
        target_col = f"node_{node_id}_visit_{target_visit}_value"
        if target_col in value_cols:
            values = values.copy()
            values[:, value_cols.index(target_col)] = np.nan
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    has_other = counts > 0
    prior_mean, prior_var = observation_prior_mean_var(df, input_type)
    mean = np.divide(sums, counts, out=prior_mean.copy(), where=has_other)
    sq = np.where(finite, np.square(values - mean[:, None]), 0.0).sum(axis=1)
    var = np.divide(sq, counts, out=prior_var.copy(), where=has_other)
    return mean, var


def run_terminal_history_value_error_baselines_from_probe(
    probe: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
    target_mode: str = "all",
    representation_mode: str = "sample",
    input_type: str = "uniform",
    representation_filter: set[str] | None = None,
    decoder_type_filter: set[str] | None = None,
    deep_config: dict | None = None,
) -> pd.DataFrame:
    """Compare paid-forward sampled-latent history decoding to baselines."""
    all_paid = paid_observation_states(probe)
    final_paid = terminal_pre_stop_states(probe)
    if all_paid.empty and final_paid.empty:
        return pd.DataFrame()
    rows = []
    for latent_scope, terminal in [
        ("all_paid_observations", all_paid),
        ("final_paid_observation", final_paid),
    ]:
        if terminal.empty:
            continue
        group_cols = metadata_group_columns(terminal)
        for key, piece in terminal.groupby(group_cols, dropna=False):
            meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            seed = int(float(meta.get("seed", 0))) if "seed" in meta else 0
            strata: list[tuple[str, float, pd.DataFrame]] = [("all", np.nan, piece)]
            if "observations_in_trial" in piece.columns:
                obs_counts = pd.to_numeric(piece["observations_in_trial"], errors="coerce")
                for obs_count in sorted(obs_counts[np.isfinite(obs_counts)].unique()):
                    strata.append((
                        "by_total_observations",
                        float(obs_count),
                        piece.loc[obs_counts == obs_count],
                    ))
            for stratum_name, total_observations, stratum_piece in strata:
                reps = latent_representations_from_probe(stratum_piece, representation_mode)
                targets = terminal_sample_value_targets_from_probe(stratum_piece, target_mode)
                if not reps or not targets:
                    continue
                for rep_name, rep_cols in reps.items():
                    if representation_filter is not None and rep_name not in representation_filter:
                        continue
                    X_linear = numeric_cols(stratum_piece, rep_cols)
                    design_mats = {
                        "linear": X_linear,
                        "quadratic": quadratic_features(X_linear),
                        "deep": X_linear,
                    }
                    for decoder_type, X in design_mats.items():
                        if decoder_type_filter is not None and decoder_type not in decoder_type_filter:
                            continue
                        for target_name in targets:
                            y = pd.to_numeric(stratum_piece[target_name], errors="coerce").to_numpy(dtype=float)
                            history_mean, history_var = node_history_value_mean_var(
                                stratum_piece,
                                target_name,
                                input_type=input_type,
                                leave_one_out=True,
                            )
                            (
                                decoder_mse,
                                mean_baseline_mse,
                                random_mean_var_baseline_mse,
                                ratio_to_mean,
                                ratio_to_random,
                                n_test,
                            ) = ridge_mse_with_baselines(
                                X,
                                y,
                                history_mean,
                                history_var,
                                rng=np.random.default_rng(
                                    seed
                                    + stable_hash(
                                        rep_name
                                        + decoder_type
                                        + target_name
                                        + latent_scope
                                        + stratum_name
                                        + str(total_observations)
                                        + "terminal_history_value_error"
                                    )
                                ),
                                ridge_alpha=ridge_alpha,
                                test_fraction=test_fraction,
                                decoder_type=decoder_type,
                                deep_config=deep_config,
                            )
                            row = dict(meta)
                            row.update({
                                "representation": rep_name,
                                "decoder_type": decoder_type,
                                "target": target_name,
                                "target_family": "node_value_history",
                                "value_error_latent_scope": latent_scope,
                                "value_error_stratum": stratum_name,
                                "observations_in_trial": total_observations,
                                "decoder_mse": decoder_mse,
                                "mean_baseline_mse": mean_baseline_mse,
                                "random_mean_var_baseline_mse": random_mean_var_baseline_mse,
                                "mse_ratio_to_history_mean": ratio_to_mean,
                                "mse_ratio_to_random_mean_var": ratio_to_random,
                                "mse_diff_to_history_mean": decoder_mse - mean_baseline_mse,
                                "mse_diff_to_random_mean_var": decoder_mse - random_mean_var_baseline_mse,
                                "n_test": n_test,
                                "n_features": X.shape[1],
                            })
                            rows.append(row)
    return pd.DataFrame(rows)


def run_terminal_history_value_mae_by_timestep_from_probe(
    probe: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
    target_mode: str = "all",
    representation_mode: str = "sample",
    representation_filter: set[str] | None = None,
    decoder_type_filter: set[str] | None = None,
    deep_config: dict | None = None,
) -> pd.DataFrame:
    """Decode node-specific value histories and report MAE by observation timestep.

    Each node visit value is decoded as a separate target. Rows where a node has
    not yet been observed have NaN targets and are excluded for that node, so a
    one-node history contributes only to that observed node's MAE.
    """
    terminal = paid_observation_states(probe)
    if terminal.empty:
        return pd.DataFrame()
    group_cols = metadata_group_columns(terminal)
    rows = []
    for key, piece in terminal.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        seed = int(float(meta.get("seed", 0))) if "seed" in meta else 0
        reps = latent_representations_from_probe(piece, representation_mode)
        targets = terminal_sample_value_targets_from_probe(piece, target_mode)
        if not reps or not targets:
            continue
        if "history_length" in piece.columns:
            timestep_values = pd.to_numeric(piece["history_length"], errors="coerce").to_numpy(dtype=float)
        else:
            timestep_values = pd.to_numeric(piece["timestep"], errors="coerce").to_numpy(dtype=float)
        if "observations_in_trial" in piece.columns:
            total_observation_values = pd.to_numeric(
                piece["observations_in_trial"],
                errors="coerce",
            ).to_numpy(dtype=float)
        else:
            total_observation_values = np.full(len(piece), np.nan, dtype=float)
        extra_group_values: dict[str, np.ndarray] = {}
        if {"actual_path_value_1", "actual_path_value_2"}.issubset(piece.columns):
            path_1 = pd.to_numeric(piece["actual_path_value_1"], errors="coerce").to_numpy(dtype=float)
            path_2 = pd.to_numeric(piece["actual_path_value_2"], errors="coerce").to_numpy(dtype=float)
            extra_group_values["actual_path_value_1"] = path_1
            extra_group_values["actual_path_value_2"] = path_2
            extra_group_values["abs_actual_path_value_difference"] = np.abs(path_1 - path_2)
        for rep_name, rep_cols in reps.items():
            if representation_filter is not None and rep_name not in representation_filter:
                continue
            X_linear = numeric_cols(piece, rep_cols)
            design_mats = {
                "linear": X_linear,
                "quadratic": quadratic_features(X_linear),
                "deep": X_linear,
            }
            for decoder_type, X in design_mats.items():
                if decoder_type_filter is not None and decoder_type not in decoder_type_filter:
                    continue
                for target_name in targets:
                    node_id = node_id_from_visit_value_col(target_name)
                    visit_index = node_visit_index_from_visit_value_col(target_name)
                    if node_id is None:
                        continue
                    y = pd.to_numeric(piece[target_name], errors="coerce").to_numpy(dtype=float)
                    mae_by_timestep = ridge_mae_by_group(
                        X,
                        y,
                        timestep_values,
                        total_observation_values,
                        extra_group_values=extra_group_values,
                        rng=np.random.default_rng(
                            seed
                            + stable_hash(
                                rep_name
                                + decoder_type
                                + target_name
                                + "terminal_history_value_mae_by_timestep"
                            )
                        ),
                        ridge_alpha=ridge_alpha,
                        test_fraction=test_fraction,
                        decoder_type=decoder_type,
                        deep_config=deep_config,
                    )
                    if mae_by_timestep.empty:
                        continue
                    for _, mae_row in mae_by_timestep.iterrows():
                        row = dict(meta)
                        row.update({
                            "representation": rep_name,
                            "decoder_type": decoder_type,
                            "target": target_name,
                            "target_family": "node_value_history",
                            "target_node": int(node_id),
                            "target_visit_index": int(visit_index) if visit_index is not None else np.nan,
                            "timestep_before_current_observation": float(mae_row["timestep_before_current_observation"]),
                            "observations_in_trial": float(mae_row["observations_in_trial"]) if "observations_in_trial" in mae_row and pd.notna(mae_row["observations_in_trial"]) else np.nan,
                            "abs_actual_path_value_difference": (
                                float(mae_row["abs_actual_path_value_difference"])
                                if "abs_actual_path_value_difference" in mae_row
                                and pd.notna(mae_row["abs_actual_path_value_difference"])
                                else np.nan
                            ),
                            "actual_path_value_1": (
                                float(mae_row["actual_path_value_1"])
                                if "actual_path_value_1" in mae_row
                                and pd.notna(mae_row["actual_path_value_1"])
                                else np.nan
                            ),
                            "actual_path_value_2": (
                                float(mae_row["actual_path_value_2"])
                                if "actual_path_value_2" in mae_row
                                and pd.notna(mae_row["actual_path_value_2"])
                                else np.nan
                            ),
                            "decoder_mae": float(mae_row["decoder_mae"]),
                            "decoder_mse": (
                                float(mae_row["decoder_mse"])
                                if "decoder_mse" in mae_row and pd.notna(mae_row["decoder_mse"])
                                else np.nan
                            ),
                            "n_test": int(mae_row["n_test"]),
                            "n_features": X.shape[1],
                        })
                        rows.append(row)
    return pd.DataFrame(rows)


def rounded_signature(df: pd.DataFrame, cols: list[str], bin_width: float) -> pd.Series:
    if not cols:
        return pd.Series([""], index=df.index)
    pieces = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if bin_width > 0:
            vals = np.round(vals / bin_width) * bin_width
        vals = np.where(np.isfinite(vals), vals, np.nan)
        pieces.append(pd.Series(vals, index=df.index).map(lambda x: "nan" if pd.isna(x) else f"{float(x):.3g}"))
    return pd.concat(pieces, axis=1).astype(str).agg("|".join, axis=1)


def run_matched_posterior_bin_effects(
    probe: pd.DataFrame,
    *,
    posterior_bin_width: float = 0.5,
    min_bin_n: int = 8,
) -> pd.DataFrame:
    """Within-posterior-bin latent variation attributable to distinct histories."""
    if probe.empty:
        return pd.DataFrame()
    group_cols = metadata_group_columns(probe)
    rows = []
    for key, piece in probe.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        local_features = infer_feature_sets_from_probe(piece)
        coord_groups = infer_coordinate_groups_from_probe(piece)
        posterior_bin_cols = list(dict.fromkeys(
            local_features["posterior_stats"] + local_features["history_context"]
        ))
        history_cols = local_features["sample_values_and_order"]
        if not posterior_bin_cols or not history_cols:
            continue
        work = piece.copy()
        work["_posterior_bin"] = rounded_signature(work, posterior_bin_cols, posterior_bin_width)
        work["_history_signature"] = rounded_signature(work, history_cols, posterior_bin_width)
        for coord_type, coords in coord_groups.items():
            if not coords:
                continue
            bin_rows = []
            for posterior_bin, bin_df in work.groupby("_posterior_bin", dropna=False):
                if len(bin_df) < int(min_bin_n):
                    continue
                if bin_df["_history_signature"].nunique(dropna=False) < 2:
                    continue
                X = numeric_cols(bin_df, coords)
                ok = np.all(np.isfinite(X), axis=1)
                X = X[ok]
                valid_bin = bin_df.loc[ok].copy()
                if X.shape[0] < int(min_bin_n):
                    continue
                centroid = X.mean(axis=0, keepdims=True)
                within_ms = float(np.mean(np.sum((X - centroid) ** 2, axis=1)))
                order_centroids = []
                order_counts = []
                for _, order_df in valid_bin.groupby("_history_signature", dropna=False):
                    X_order = numeric_cols(order_df, coords)
                    ok_order = np.all(np.isfinite(X_order), axis=1)
                    X_order = X_order[ok_order]
                    if X_order.shape[0] == 0:
                        continue
                    order_centroids.append(X_order.mean(axis=0))
                    order_counts.append(X_order.shape[0])
                if len(order_centroids) < 2:
                    continue
                order_centroids = np.vstack(order_centroids)
                order_counts_arr = np.asarray(order_counts, dtype=float)
                weights = order_counts_arr / max(float(order_counts_arr.sum()), 1.0)
                centroid_of_orders = np.sum(order_centroids * weights[:, None], axis=0, keepdims=True)
                between_order_ms = float(np.sum(weights * np.sum((order_centroids - centroid_of_orders) ** 2, axis=1)))
                bin_rows.append({
                    "posterior_bin": posterior_bin,
                    "n_states": int(X.shape[0]),
                    "n_history_signatures": int(len(order_centroids)),
                    "latent_within_posterior_ms": within_ms,
                    "between_history_centroid_ms": between_order_ms,
                })
            if not bin_rows:
                continue
            bin_df = pd.DataFrame(bin_rows)
            row = dict(meta)
            row.update({
                "coordinate_type": coord_type,
                "posterior_bin_width": float(posterior_bin_width),
                "min_bin_n": int(min_bin_n),
                "n_posterior_bins": int(len(bin_df)),
                "n_states": int(bin_df["n_states"].sum()),
                "mean_n_history_signatures": float(bin_df["n_history_signatures"].mean()),
                "latent_within_posterior_ms": float(np.average(
                    bin_df["latent_within_posterior_ms"],
                    weights=bin_df["n_states"],
                )),
                "between_history_centroid_ms": float(np.average(
                    bin_df["between_history_centroid_ms"],
                    weights=bin_df["n_states"],
                )),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return int(value)


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


def figure_size(n_cols: int, n_rows: int, legend: bool = True) -> tuple[float, float]:
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
    if n <= 0:
        return []
    if n == 1:
        return [mcolors.to_rgba(hex_colors[0])]
    cmap = mcolors.LinearSegmentedColormap.from_list("hypothesis2_parameter_ramp", hex_colors, N=n)
    return [cmap(i / (n - 1)) for i in range(n)]


def family_title(family: str) -> str:
    aliases = {
        "vary_beta": "Beta varies\n(opportunity = 0)",
        "vary_opportunity": "Opportunity varies\n(beta = 1000)",
        "manual": "Manual grid",
    }
    return aliases.get(str(family), str(family))


def family_style_map(summary: pd.DataFrame) -> dict[tuple[str, float], tuple[tuple[float, float, float, float], str]]:
    styles: dict[tuple[str, float], tuple[tuple[float, float, float, float], str]] = {}
    family_specs = {
        "vary_beta": (["#00441b", "#238b45", "#74c476"], "o"),
        "vary_opportunity": (["#6baed6", "#2171b5", "#08306b"], "^"),
        "manual": (["#252525", "#737373", "#bdbdbd"], "s"),
    }
    for family, piece in summary.groupby("family", dropna=False):
        ramp, marker = family_specs.get(str(family), family_specs["manual"])
        values = sorted(pd.to_numeric(piece["parameter_value"], errors="coerce").dropna().unique())
        for value, color in zip(values, color_ramp(ramp, len(values))):
            styles[(str(family), float(value))] = (color, marker)
    return styles


def sem_series(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return float("nan")
    return float(np.std(vals, ddof=1) / math.sqrt(vals.size))


def safe_label(value) -> str:
    text = str(value).replace("_", " ")
    text = text.replace("posterior stats", "posterior\nstats")
    text = text.replace("individual sample values", "individual\nsamples")
    text = text.replace("sample order", "sample\norder")
    text = text.replace("sample values and order", "sample values\n+ order")
    text = text.replace("same mean var diff samples", "same mean/var\ndiff samples")
    text = text.replace("same set order", "same set\norder")
    text = text.replace("same pairs order", "same pairs\norder")
    text = text.replace("posterior given history", "posterior\n| history")
    text = text.replace("history given posterior", "history\n| posterior")
    text = text.replace("sample values given posterior", "sample values\n| posterior")
    text = text.replace("sample order given posterior", "sample order\n| posterior")
    text = text.replace("prior norm latent mu", "prior-norm\nlatent")
    text = text.replace("latent mu", "latent\nmu")
    return text


def read_or_collect_csv(outroot: Path, aggregate_name: str, leaf_name: str) -> pd.DataFrame:
    aggregate_path = outroot / aggregate_name
    if aggregate_path.exists():
        return pd.read_csv(aggregate_path)
    paths = sorted(path for path in outroot.glob(f"*/{leaf_name}") if path.is_file())
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(aggregate_path, index=False)
    return data


def close_to_any(series: pd.Series, values: Iterable[float], *, atol: float = 1e-8) -> np.ndarray:
    requested = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not requested:
        return np.ones(len(series), dtype=bool)
    observed = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(series), dtype=bool)
    for value in requested:
        mask |= np.isclose(observed, value, rtol=1e-7, atol=atol)
    return mask


def requested_filter_is_active(requested_filter: dict | None) -> bool:
    if not requested_filter:
        return False
    return any(bool(requested_filter.get(key)) for key in [
        "parameter_combos",
        "sigmas",
        "seeds",
        "lambda_values",
        "alphas",
        "rnn_dims",
        "latent_dims",
    ])


def requested_filter_cache_suffix(requested_filter: dict | None) -> str:
    if not requested_filter_is_active(requested_filter):
        return ""
    payload = {
        key: requested_filter.get(key)
        for key in [
            "parameter_combos",
            "sigmas",
            "seeds",
            "lambda_values",
            "alphas",
            "rnn_dims",
            "latent_dims",
        ]
    }
    return f"_filtered_{stable_hash(repr(payload))}"


def describe_requested_filter(requested_filter: dict | None) -> str:
    if not requested_filter_is_active(requested_filter):
        return "none"
    parts = []
    combos = requested_filter.get("parameter_combos") or []
    beta_values = sorted({
        float(beta)
        for family, _, _, beta, _ in combos
        if str(family) == "vary_beta"
    })
    opportunity_values = sorted({
        float(opp)
        for family, _, _, _, opp in combos
        if str(family) == "vary_opportunity"
    })
    if beta_values:
        parts.append("beta=" + ",".join(f"{value:g}" for value in beta_values))
    if opportunity_values:
        parts.append("opportunity=" + ",".join(f"{value:g}" for value in opportunity_values))
    for key, label in [
        ("sigmas", "sigma"),
        ("seeds", "seed"),
        ("lambda_values", "lambda"),
        ("alphas", "alpha"),
        ("rnn_dims", "rnn"),
        ("latent_dims", "latent"),
    ]:
        values = requested_filter.get(key) or []
        if values:
            parts.append(f"{label}=" + ",".join(f"{float(value):g}" for value in values))
    return "; ".join(parts) if parts else "active"


def filter_requested_rows(data: pd.DataFrame, requested_filter: dict | None) -> pd.DataFrame:
    """Restrict cached or freshly computed diagnostics to the requested grid."""
    if data.empty or not requested_filter_is_active(requested_filter):
        return data
    keep = np.ones(len(data), dtype=bool)

    combos = requested_filter.get("parameter_combos") or []
    if combos:
        combo_keep = np.zeros(len(data), dtype=bool)
        used_combo_column = False
        for family, parameter_name, parameter_value, beta, opportunity in combos:
            combo_mask = np.ones(len(data), dtype=bool)
            if "family" in data.columns:
                used_combo_column = True
                combo_mask &= data["family"].astype(str).to_numpy() == str(family)
            if "parameter_name" in data.columns:
                used_combo_column = True
                combo_mask &= data["parameter_name"].astype(str).to_numpy() == str(parameter_name)
            if "parameter_value" in data.columns:
                used_combo_column = True
                combo_mask &= close_to_any(data["parameter_value"], [float(parameter_value)])
            if "beta" in data.columns:
                used_combo_column = True
                combo_mask &= close_to_any(data["beta"], [float(beta)])
            if "opportunity_cost" in data.columns:
                used_combo_column = True
                combo_mask &= close_to_any(data["opportunity_cost"], [float(opportunity)])
            combo_keep |= combo_mask
        if used_combo_column:
            keep &= combo_keep

    column_filters = [
        ("sigma", requested_filter.get("sigmas") or []),
        ("seed", requested_filter.get("seeds") or []),
        ("lambda", requested_filter.get("lambda_values") or []),
        ("alpha", requested_filter.get("alphas") or []),
        ("rnn_dim", requested_filter.get("rnn_dims") or []),
        ("latent_dim", requested_filter.get("latent_dims") or []),
    ]
    for col, values in column_filters:
        if col in data.columns and values:
            keep &= close_to_any(data[col], values)
    return data.loc[keep].copy()


def summarize_seed_metric(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    seed_col: str = "seed",
    weight_col: str | None = None,
    min_weight: int = 0,
) -> pd.DataFrame:
    if data.empty or value_col not in data:
        return pd.DataFrame()
    work = data.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[np.isfinite(work[value_col])]
    use_weights = bool(weight_col and weight_col in work.columns)
    if use_weights:
        work["_summary_weight"] = pd.to_numeric(work[weight_col], errors="coerce")
        work = work[np.isfinite(work["_summary_weight"]) & (work["_summary_weight"] > 0)]
        if min_weight and min_weight > 0:
            work = work[work["_summary_weight"] >= int(min_weight)]
    if work.empty:
        return pd.DataFrame()
    seed_cols = [c for c in group_cols + [seed_col] if c in work.columns]
    rows = []
    for key, piece in work.groupby(seed_cols, dropna=False):
        row = dict(zip(seed_cols, key if isinstance(key, tuple) else (key,)))
        values = pd.to_numeric(piece[value_col], errors="coerce").to_numpy(dtype=float)
        if use_weights:
            weights = pd.to_numeric(piece["_summary_weight"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if not np.any(ok):
                continue
            row["value"] = float(np.average(values[ok], weights=weights[ok]))
            row["n_weight"] = float(np.sum(weights[ok]))
            row["n_points"] = int(np.sum(ok))
        else:
            ok = np.isfinite(values)
            if not np.any(ok):
                continue
            row["value"] = float(np.mean(values[ok]))
            row["n_points"] = int(np.sum(ok))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    per_seed = pd.DataFrame(rows)
    aggregations = {
        "mean": ("value", "mean"),
        "sem": ("value", sem_series),
        "n_seeds": (seed_col, "nunique"),
        "n_points": ("n_points", "sum"),
    }
    if "n_weight" in per_seed.columns:
        aggregations["n_weight"] = ("n_weight", "sum")
    return (
        per_seed
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(**aggregations)
    )


def plot_family_sigma_grid(
    summary: pd.DataFrame,
    *,
    row_col: str,
    y_col: str,
    y_sem_col: str,
    outpath: Path,
    y_label: str,
    row_order: list | None = None,
) -> None:
    if summary.empty:
        return
    setup_plot_style()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    families = [f for f in ["vary_beta", "vary_opportunity", "manual"] if f in set(summary["family"].astype(str))]
    if not families:
        families = sorted(summary["family"].astype(str).unique())
    rows = row_order or sorted(summary[row_col].dropna().unique(), key=lambda x: str(x))
    rows = [row for row in rows if row in set(summary[row_col])]
    if not rows:
        return
    n_rows = len(rows)
    n_cols = len(families)
    fig_w, fig_h = figure_size(n_cols, n_rows, legend=True)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [max(0.8, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - RIGHT_MARGIN_IN / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
        wspace=0.42,
        hspace=0.55,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axes[row_i, col_i] = fig.add_subplot(
                gs[row_i, col_i],
                sharex=axes[0, 0] if row_i or col_i else None,
            )
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")
    styles = family_style_map(summary)
    legend_handles = []
    legend_labels = []
    seen_labels = set()
    for row_i, row_value in enumerate(rows):
        for col_i, family in enumerate(families):
            ax = axes[row_i, col_i]
            panel = summary[
                (summary[row_col].astype(str) == str(row_value))
                & (summary["family"].astype(str) == str(family))
            ]
            for (param_value, param_label), line in panel.groupby(["parameter_value", "parameter_label"], dropna=False):
                line = line.sort_values("sigma")
                color, marker = styles.get((str(family), float(param_value)), ("black", "o"))
                handle = ax.errorbar(
                    pd.to_numeric(line["sigma"], errors="coerce"),
                    pd.to_numeric(line[y_col], errors="coerce"),
                    yerr=pd.to_numeric(line[y_sem_col], errors="coerce"),
                    marker=marker,
                    linestyle="-",
                    linewidth=0.85,
                    markersize=2.4,
                    color=color,
                    markeredgecolor="black" if marker == "^" else color,
                    markeredgewidth=0.35 if marker == "^" else 0.0,
                )
                label = str(param_label)
                if label not in seen_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
                    seen_labels.add(label)
            if row_i == 0:
                ax.set_title(family_title(family), pad=1.5)
            if col_i == 0:
                ax.set_ylabel(safe_label(row_value), labelpad=2.0)
            else:
                ax.tick_params(labelleft=False)
            if row_i < n_rows - 1:
                ax.tick_params(labelbottom=False)
            ax.tick_params(pad=1.5)
            ax.axhline(0.0, color="black", linewidth=0.45, alpha=0.35)
    if legend_handles:
        legend_ax.legend(legend_handles, legend_labels, loc="center left", frameon=False, handlelength=1.0)
    fig.supxlabel("sigma", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(y_label, fontsize=PLOT_FONT_SIZE_PT, x=0.012)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Saved {outpath}", flush=True)


def plot_timestep_mae_sigma_rows(
    summary: pd.DataFrame,
    *,
    outpath: Path,
    y_label: str = "History decoding MAE",
) -> None:
    if summary.empty:
        return
    required = {
        "sigma",
        "family",
        "parameter_value",
        "parameter_label",
        "timestep_before_current_observation",
        "mean",
        "sem",
    }
    if not required.issubset(summary.columns):
        return
    setup_plot_style()
    work = summary.copy()
    work["sigma"] = pd.to_numeric(work["sigma"], errors="coerce")
    work["timestep_before_current_observation"] = pd.to_numeric(
        work["timestep_before_current_observation"],
        errors="coerce",
    )
    if "observations_in_trial" in work.columns:
        work["observations_in_trial"] = pd.to_numeric(
            work["observations_in_trial"],
            errors="coerce",
        )
    else:
        work["observations_in_trial"] = np.nan
    work = work[
        np.isfinite(work["sigma"])
        & np.isfinite(work["timestep_before_current_observation"])
        & np.isfinite(pd.to_numeric(work["mean"], errors="coerce"))
    ].copy()
    if work.empty:
        return
    sigmas = sorted(work["sigma"].unique())
    totals = sorted(work["observations_in_trial"][np.isfinite(work["observations_in_trial"])].unique())
    if not totals:
        totals = [np.nan]
    n_rows = len(sigmas)
    n_cols = len(totals)
    fig_w, fig_h = figure_size(n_cols, n_rows, legend=True)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [max(0.8, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - RIGHT_MARGIN_IN / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
        wspace=0.34,
        hspace=0.55,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axes[row_i, col_i] = fig.add_subplot(gs[row_i, col_i])
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")
    styles = family_style_map(work)
    legend_handles = []
    legend_labels = []
    seen_labels = set()
    for row_i, sigma in enumerate(sigmas):
        for col_i, total in enumerate(totals):
            ax = axes[row_i, col_i]
            panel = work[np.isclose(work["sigma"], sigma)]
            if np.isfinite(total):
                panel = panel[np.isclose(panel["observations_in_trial"], total)]
            for (family, param_value, param_label), line in panel.groupby(
                ["family", "parameter_value", "parameter_label"],
                dropna=False,
            ):
                line = line.sort_values("timestep_before_current_observation")
                color, marker = styles.get((str(family), float(param_value)), ("black", "o"))
                handle = ax.errorbar(
                    pd.to_numeric(line["timestep_before_current_observation"], errors="coerce"),
                    pd.to_numeric(line["mean"], errors="coerce"),
                    yerr=pd.to_numeric(line["sem"], errors="coerce"),
                    marker=marker,
                    linestyle="-",
                    linewidth=0.85,
                    markersize=2.4,
                    capsize=1.4,
                    color=color,
                    markeredgecolor="black" if marker == "^" else color,
                    markeredgewidth=0.35 if marker == "^" else 0.0,
                )
                label = str(param_label)
                if label not in seen_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
                    seen_labels.add(label)
            if row_i == 0:
                title = f"total obs = {int(total):d}" if np.isfinite(total) else "all totals"
                ax.set_title(title, pad=1.5)
            if col_i == 0:
                ax.set_ylabel(f"sigma = {sigma:g}", labelpad=2.0)
            else:
                ax.tick_params(labelleft=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(pad=1.5)
            if row_i < n_rows - 1:
                ax.tick_params(labelbottom=False)
    if legend_handles:
        legend_ax.legend(legend_handles, legend_labels, loc="center left", frameon=False, handlelength=1.0)
    fig.supxlabel("Current number\nof observations", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(y_label, fontsize=PLOT_FONT_SIZE_PT, x=0.012)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Saved {outpath}", flush=True)


def plot_abs_reward_diff_mae_sigma_rows(
    summary: pd.DataFrame,
    *,
    outpath: Path,
    y_label: str = "History decoding MAE",
) -> None:
    if summary.empty:
        return
    required = {
        "sigma",
        "family",
        "parameter_value",
        "parameter_label",
        "timestep_before_current_observation",
        "abs_actual_path_value_difference",
        "mean",
        "sem",
    }
    if not required.issubset(summary.columns):
        return
    setup_plot_style()
    work = summary.copy()
    work["sigma"] = pd.to_numeric(work["sigma"], errors="coerce")
    work["abs_actual_path_value_difference"] = pd.to_numeric(
        work["abs_actual_path_value_difference"],
        errors="coerce",
    )
    work["timestep_before_current_observation"] = pd.to_numeric(
        work["timestep_before_current_observation"],
        errors="coerce",
    )
    work = work[
        np.isfinite(work["sigma"])
        & np.isfinite(work["abs_actual_path_value_difference"])
        & np.isfinite(work["timestep_before_current_observation"])
        & np.isfinite(pd.to_numeric(work["mean"], errors="coerce"))
    ].copy()
    if work.empty:
        return
    sigmas = sorted(work["sigma"].unique())
    gaps = sorted(work["abs_actual_path_value_difference"].unique())
    if not gaps:
        return
    n_rows = len(sigmas)
    n_cols = len(gaps)
    fig_w, fig_h = figure_size(n_cols, n_rows, legend=True)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [max(0.8, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
        left=LEFT_MARGIN_IN / fig_w,
        right=1.0 - RIGHT_MARGIN_IN / fig_w,
        bottom=BOTTOM_MARGIN_IN / fig_h,
        top=1.0 - TOP_MARGIN_IN / fig_h,
        wspace=0.46,
        hspace=0.62,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axes[row_i, col_i] = fig.add_subplot(gs[row_i, col_i])
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")
    styles = family_style_map(work)
    legend_handles = []
    legend_labels = []
    seen_labels = set()
    x_values = pd.to_numeric(work["timestep_before_current_observation"], errors="coerce")
    x_min = float(np.nanmin(x_values)) if np.any(np.isfinite(x_values)) else 0.0
    x_max = float(np.nanmax(x_values)) if np.any(np.isfinite(x_values)) else 1.0
    for row_i, sigma in enumerate(sigmas):
        for col_i, gap in enumerate(gaps):
            ax = axes[row_i, col_i]
            panel = work[
                np.isclose(work["sigma"], sigma)
                & np.isclose(work["abs_actual_path_value_difference"], gap)
            ]
            for (family, param_value, param_label), line in panel.groupby(
                ["family", "parameter_value", "parameter_label"],
                dropna=False,
            ):
                line = line.sort_values("timestep_before_current_observation")
                color, marker = styles.get((str(family), float(param_value)), ("black", "o"))
                handle = ax.errorbar(
                    pd.to_numeric(line["timestep_before_current_observation"], errors="coerce"),
                    pd.to_numeric(line["mean"], errors="coerce"),
                    yerr=pd.to_numeric(line["sem"], errors="coerce"),
                    marker=marker,
                    linestyle="-",
                    linewidth=0.85,
                    markersize=2.4,
                    capsize=1.4,
                    color=color,
                    markeredgecolor="black" if marker == "^" else color,
                    markeredgewidth=0.35 if marker == "^" else 0.0,
                )
                label = str(param_label)
                if label not in seen_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
                    seen_labels.add(label)
            ax.set_xlim(max(0.0, x_min - 0.25), x_max + 0.25)
            ax.set_xticks([x for x in range(int(np.floor(x_min)), int(np.ceil(x_max)) + 1) if x >= 1])
            if row_i == 0:
                title = f"|dR| = {gap:g}"
                ax.set_title(title, pad=1.5)
            if col_i == 0:
                ax.set_ylabel(f"sigma = {sigma:g}", labelpad=2.0)
            else:
                ax.tick_params(labelleft=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(pad=1.5)
            if row_i < n_rows - 1:
                ax.tick_params(labelbottom=False)
    if legend_handles:
        legend_ax.legend(legend_handles, legend_labels, loc="center left", frameon=False, handlelength=1.0)
    fig.supxlabel("Current number\nof observations", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
    fig.supylabel(y_label, fontsize=PLOT_FONT_SIZE_PT, x=0.012)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Saved {outpath}", flush=True)


def float_filename_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def plot_reward_pair_mse_grids_by_sigma(
    summary: pd.DataFrame,
    *,
    outpath_prefix: Path,
    y_label: str = "History decoding MSE",
) -> None:
    if summary.empty:
        return
    required = {
        "sigma",
        "family",
        "parameter_value",
        "parameter_label",
        "timestep_before_current_observation",
        "actual_path_value_1",
        "actual_path_value_2",
        "mean",
        "sem",
    }
    if not required.issubset(summary.columns):
        return
    setup_plot_style()
    work = summary.copy()
    for col in [
        "sigma",
        "timestep_before_current_observation",
        "actual_path_value_1",
        "actual_path_value_2",
        "mean",
        "sem",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[
        np.isfinite(work["sigma"])
        & np.isfinite(work["timestep_before_current_observation"])
        & np.isfinite(work["actual_path_value_1"])
        & np.isfinite(work["actual_path_value_2"])
        & np.isfinite(work["mean"])
    ].copy()
    if work.empty:
        return
    styles = family_style_map(work)
    x_values = work["timestep_before_current_observation"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_values)) if np.any(np.isfinite(x_values)) else 1.0
    x_max = float(np.nanmax(x_values)) if np.any(np.isfinite(x_values)) else 1.0
    reward_1_values = sorted(work["actual_path_value_1"].unique(), reverse=True)
    reward_2_values = sorted(work["actual_path_value_2"].unique())
    for sigma in sorted(work["sigma"].unique()):
        sigma_work = work[np.isclose(work["sigma"], sigma)]
        if sigma_work.empty:
            continue
        n_rows = len(reward_1_values)
        n_cols = len(reward_2_values)
        fig_w, fig_h = figure_size(n_cols, n_rows, legend=True)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            n_rows,
            n_cols + 1,
            width_ratios=[1.0] * n_cols + [max(0.8, LEGEND_WIDTH_IN / PANEL_WIDTH_IN)],
            left=LEFT_MARGIN_IN / fig_w,
            right=1.0 - RIGHT_MARGIN_IN / fig_w,
            bottom=BOTTOM_MARGIN_IN / fig_h,
            top=1.0 - TOP_MARGIN_IN / fig_h,
            wspace=0.44,
            hspace=0.58,
        )
        axes = np.empty((n_rows, n_cols), dtype=object)
        for row_i in range(n_rows):
            for col_i in range(n_cols):
                axes[row_i, col_i] = fig.add_subplot(gs[row_i, col_i])
        legend_ax = fig.add_subplot(gs[:, -1])
        legend_ax.axis("off")
        legend_handles = []
        legend_labels = []
        seen_labels = set()
        for row_i, reward_1 in enumerate(reward_1_values):
            for col_i, reward_2 in enumerate(reward_2_values):
                ax = axes[row_i, col_i]
                panel = sigma_work[
                    np.isclose(sigma_work["actual_path_value_1"], reward_1)
                    & np.isclose(sigma_work["actual_path_value_2"], reward_2)
                ]
                for (family, param_value, param_label), line in panel.groupby(
                    ["family", "parameter_value", "parameter_label"],
                    dropna=False,
                ):
                    line = line.sort_values("timestep_before_current_observation")
                    color, marker = styles.get((str(family), float(param_value)), ("black", "o"))
                    handle = ax.errorbar(
                        pd.to_numeric(line["timestep_before_current_observation"], errors="coerce"),
                        pd.to_numeric(line["mean"], errors="coerce"),
                        yerr=pd.to_numeric(line["sem"], errors="coerce"),
                        marker=marker,
                        linestyle="-",
                        linewidth=0.85,
                        markersize=2.4,
                        capsize=1.4,
                        color=color,
                        markeredgecolor="black" if marker == "^" else color,
                        markeredgewidth=0.35 if marker == "^" else 0.0,
                    )
                    label = str(param_label)
                    if label not in seen_labels:
                        legend_handles.append(handle)
                        legend_labels.append(label)
                        seen_labels.add(label)
                ax.set_xlim(max(0.0, x_min - 0.25), x_max + 0.25)
                ax.set_xticks([x for x in range(int(np.floor(x_min)), int(np.ceil(x_max)) + 1) if x >= 1])
                if row_i == 0:
                    ax.set_title(f"R2 = {reward_2:g}", pad=1.5)
                if col_i == 0:
                    ax.set_ylabel(f"R1 = {reward_1:g}", labelpad=2.0)
                else:
                    ax.tick_params(labelleft=False)
                if row_i < n_rows - 1:
                    ax.tick_params(labelbottom=False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(pad=1.5)
        if legend_handles:
            legend_ax.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                frameon=False,
                handlelength=1.0,
            )
        fig.suptitle(f"sigma = {sigma:g}", fontsize=PLOT_FONT_SIZE_PT, y=0.995)
        fig.supxlabel("Current number\nof observations", fontsize=PLOT_FONT_SIZE_PT, y=0.02)
        fig.supylabel(y_label, fontsize=PLOT_FONT_SIZE_PT, x=0.012)
        outpath = outpath_prefix.with_name(
            f"{outpath_prefix.name}_sigma_{float_filename_tag(float(sigma))}.png"
        )
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        print(f"Saved {outpath}", flush=True)


def normalize_extra_tests(tests: list[str] | None) -> set[str]:
    requested = set(tests or ["terminal_history_value_error"])
    if "none" in requested:
        return set()
    all_tests = {
        "terminal_history_decoder",
        "terminal_history_value_error",
    }
    if "all" in requested:
        return all_tests
    return requested & all_tests


def read_probe_for_extra_tests(outroot: Path, max_rows_per_combo: int = 0) -> pd.DataFrame:
    if max_rows_per_combo and max_rows_per_combo > 0:
        paths = sorted(path for path in outroot.glob("*/probe_state_dataset.csv") if path.is_file())
        frames = []
        for path in paths:
            frame = pd.read_csv(path, nrows=int(max_rows_per_combo))
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return ensure_sampled_latent_columns(pd.concat(frames, ignore_index=True))
    return ensure_sampled_latent_columns(
        read_or_collect_csv(outroot, "probe_state_dataset_all.csv", "probe_state_dataset.csv")
    )


def plot_hypothesis2_outputs(
    outroot: Path,
    *,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.25,
    extra_tests: list[str] | None = None,
    max_analysis_states_per_combo: int = 0,
    terminal_target_mode: str = "all",
    terminal_representation_mode: str = "minimal",
    input_type: str = "uniform",
    full_h2_plots: bool = False,
    requested_filter: dict | None = None,
    min_decoder_test_samples: int = 10,
    probe_decoder_types: list[str] | None = None,
    deep_probe_config: dict | None = None,
) -> None:
    figures_dir = outroot / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if requested_filter_is_active(requested_filter):
        print(f"Filtering plot-only diagnostics to requested values: {describe_requested_filter(requested_filter)}", flush=True)
    forced = (
        filter_requested_rows(
            read_or_collect_csv(outroot, "forced_history_pairwise_all.csv", "forced_history_pairwise.csv"),
            requested_filter,
        )
        if full_h2_plots
        else pd.DataFrame()
    )
    decoder = (
        filter_requested_rows(
            read_or_collect_csv(outroot, "latent_decoder_r2_all.csv", "latent_decoder_r2.csv"),
            requested_filter,
        )
        if full_h2_plots
        else pd.DataFrame()
    )
    variance = (
        filter_requested_rows(
            read_or_collect_csv(outroot, "variance_decomposition_r2_all.csv", "variance_decomposition_r2.csv"),
            requested_filter,
        )
        if full_h2_plots
        else pd.DataFrame()
    )
    requested_extra_tests = normalize_extra_tests(extra_tests)
    decoder_types_requested = probe_decoder_types or parse_decoder_types(None, full_h2_plots=full_h2_plots)
    decoder_type_filter = set(decoder_types_requested)
    decoder_type_order = ["linear", "quadratic", "deep"]
    nondefault_decoder_types = set(decoder_types_requested) != set(
        parse_decoder_types(None, full_h2_plots=full_h2_plots)
    )
    probe = pd.DataFrame()

    def get_probe() -> pd.DataFrame:
        nonlocal probe
        if probe.empty:
            probe = filter_requested_rows(
                read_probe_for_extra_tests(outroot, int(max_analysis_states_per_combo)),
                requested_filter,
            )
        return probe

    # The broad conditional/partial and matched-posterior tests are intentionally
    # disabled here; they load very large probe-state tables and are too slow for
    # normal plotting. Keep the focused terminal-history decoder tests below.
    conditional = pd.DataFrame()
    conditional_decoder = pd.DataFrame()

    terminal_decoder_path = outroot / (
        f"terminal_history_decoder_{terminal_representation_mode}_{terminal_target_mode}_r2_all.csv"
    )
    if terminal_decoder_path.exists() and full_h2_plots:
        terminal_decoder = filter_requested_rows(pd.read_csv(terminal_decoder_path), requested_filter)
    elif full_h2_plots and "terminal_history_decoder" in requested_extra_tests:
        probe_data = get_probe()
        terminal_decoder = run_terminal_history_decoders_from_probe(
            probe_data,
            ridge_alpha=ridge_alpha,
            test_fraction=test_fraction,
            target_mode=terminal_target_mode,
            representation_mode=terminal_representation_mode,
        ) if not probe_data.empty else pd.DataFrame()
        terminal_decoder = filter_requested_rows(terminal_decoder, requested_filter)
        if not terminal_decoder.empty:
            terminal_decoder.to_csv(terminal_decoder_path, index=False)
    else:
        terminal_decoder = pd.DataFrame()

    terminal_conditional_decoder = pd.DataFrame()

    value_error_cache_tag = (
        "loo_same_node_prior_fallback_scope_total_obs_final_unpaid_all_v4"
        if full_h2_plots
        else "loo_same_node_prior_fallback_scope_total_obs_final_unpaid_quadratic_z_sample_v4"
    )
    if nondefault_decoder_types:
        decoder_tag = "decoders_" + "_".join(decoder_types_requested)
        if "deep" in decoder_type_filter:
            decoder_tag += "_" + deep_probe_cache_tag(deep_probe_config or {})
        value_error_cache_tag = f"loo_same_node_prior_fallback_scope_total_obs_final_unpaid_{decoder_tag}_v2"
    filter_suffix = requested_filter_cache_suffix(requested_filter)
    terminal_value_error_base_path = outroot / (
        f"terminal_history_value_error_{terminal_representation_mode}_{terminal_target_mode}_{value_error_cache_tag}.csv"
    )
    terminal_value_error_path = outroot / (
        f"terminal_history_value_error_{terminal_representation_mode}_{terminal_target_mode}_{value_error_cache_tag}{filter_suffix}.csv"
    )
    value_mae_timestep_cache_tag = value_error_cache_tag + "_total_obs_reward_grid_mse_columns_v3"
    terminal_value_mae_timestep_base_path = outroot / (
        f"terminal_history_value_mae_by_timestep_{terminal_representation_mode}_{terminal_target_mode}_{value_mae_timestep_cache_tag}.csv"
    )
    terminal_value_mae_timestep_path = outroot / (
        f"terminal_history_value_mae_by_timestep_{terminal_representation_mode}_{terminal_target_mode}_{value_mae_timestep_cache_tag}{filter_suffix}.csv"
    )
    if terminal_value_error_path.exists():
        terminal_value_error = pd.read_csv(terminal_value_error_path)
    elif terminal_value_error_base_path.exists():
        terminal_value_error = filter_requested_rows(pd.read_csv(terminal_value_error_base_path), requested_filter)
        if filter_suffix and not terminal_value_error.empty:
            terminal_value_error.to_csv(terminal_value_error_path, index=False)
    elif "terminal_history_value_error" in requested_extra_tests:
        probe_data = get_probe()
        terminal_value_error = run_terminal_history_value_error_baselines_from_probe(
            probe_data,
            ridge_alpha=ridge_alpha,
            test_fraction=test_fraction,
            target_mode=terminal_target_mode,
            representation_mode=terminal_representation_mode,
            input_type=input_type,
            representation_filter=None if full_h2_plots else {"z_sample"},
            decoder_type_filter=decoder_type_filter,
            deep_config=deep_probe_config,
        ) if not probe_data.empty else pd.DataFrame()
        terminal_value_error = filter_requested_rows(terminal_value_error, requested_filter)
        if not terminal_value_error.empty:
            terminal_value_error.to_csv(terminal_value_error_path, index=False)
    else:
        terminal_value_error = pd.DataFrame()

    if terminal_value_mae_timestep_path.exists():
        terminal_value_mae_timestep = pd.read_csv(terminal_value_mae_timestep_path)
    elif terminal_value_mae_timestep_base_path.exists():
        terminal_value_mae_timestep = filter_requested_rows(
            pd.read_csv(terminal_value_mae_timestep_base_path),
            requested_filter,
        )
        if filter_suffix and not terminal_value_mae_timestep.empty:
            terminal_value_mae_timestep.to_csv(terminal_value_mae_timestep_path, index=False)
    elif "terminal_history_value_error" in requested_extra_tests:
        probe_data = get_probe()
        terminal_value_mae_timestep = run_terminal_history_value_mae_by_timestep_from_probe(
            probe_data,
            ridge_alpha=ridge_alpha,
            test_fraction=test_fraction,
            target_mode=terminal_target_mode,
            representation_mode=terminal_representation_mode,
            representation_filter=None if full_h2_plots else {"z_sample"},
            decoder_type_filter=decoder_type_filter,
            deep_config=deep_probe_config,
        ) if not probe_data.empty else pd.DataFrame()
        terminal_value_mae_timestep = filter_requested_rows(
            terminal_value_mae_timestep,
            requested_filter,
        )
        if not terminal_value_mae_timestep.empty:
            terminal_value_mae_timestep.to_csv(terminal_value_mae_timestep_path, index=False)
    else:
        terminal_value_mae_timestep = pd.DataFrame()

    matched = pd.DataFrame()

    if not forced.empty:
        forced_metrics = [
            ("prior_norm_latent_mu_l2", "Prior-norm\nlatent L2"),
            ("latent_mu_l2", "Raw latent\nL2"),
            ("terminal_prob_l1", "Terminal prob\nL1"),
            ("next_action_prob_l1", "Action prob\nL1"),
        ]
        for metric, y_label in forced_metrics:
            if metric not in forced:
                continue
            summary = summarize_seed_metric(
                forced,
                group_cols=["family", "parameter_value", "parameter_label", "sigma", "case_group"],
                value_col=metric,
            )
            summary.to_csv(figures_dir / f"forced_history_{metric}_summary.csv", index=False)
            plot_family_sigma_grid(
                summary,
                row_col="case_group",
                y_col="mean",
                y_sem_col="sem",
                outpath=figures_dir / f"forced_history_{metric}_by_sigma.png",
                y_label=y_label,
            )

    if not decoder.empty:
        decoder = decoder.copy()
        decoder["r2_clipped"] = pd.to_numeric(decoder["r2"], errors="coerce").clip(lower=-0.25, upper=1.0)
        representations = [
            rep for rep in ["z_mu", "z_mu_logsigma", "prior_norm_z_mu", "prior_norm_z_mu_logsigma", "pga_scores"]
            if rep in set(decoder["representation"].astype(str))
        ]
        for representation in representations:
            piece = decoder[decoder["representation"].astype(str) == representation]
            summary = summarize_seed_metric(
                piece,
                group_cols=["family", "parameter_value", "parameter_label", "sigma", "target_family"],
                value_col="r2_clipped",
            )
            summary.to_csv(figures_dir / f"decoder_{representation}_target_family_summary.csv", index=False)
            plot_family_sigma_grid(
                summary,
                row_col="target_family",
                y_col="mean",
                y_sem_col="sem",
                outpath=figures_dir / f"decoder_{representation}_r2_by_sigma.png",
                y_label="Decoder\nR2",
                row_order=["posterior_stats", "individual_sample_values", "sample_order"],
            )

    if not variance.empty:
        variance = variance.copy()
        variance["r2_clipped"] = pd.to_numeric(variance["r2"], errors="coerce").clip(lower=-0.25, upper=1.0)
        coordinate_types = [
            ct for ct in ["latent_dim", "prior_norm_latent_dim", "pga"]
            if ct in set(variance["coordinate_type"].astype(str))
        ]
        row_order = ["posterior_stats", "sample_values", "sample_order_nodes", "counts", "sample_values_and_order", "all_features"]
        for coordinate_type in coordinate_types:
            piece = variance[variance["coordinate_type"].astype(str) == coordinate_type]
            summary = summarize_seed_metric(
                piece,
                group_cols=["family", "parameter_value", "parameter_label", "sigma", "feature_group"],
                value_col="r2_clipped",
            )
            summary.to_csv(figures_dir / f"variance_{coordinate_type}_feature_group_summary.csv", index=False)
            plot_family_sigma_grid(
                summary,
                row_col="feature_group",
                y_col="mean",
                y_sem_col="sem",
                outpath=figures_dir / f"variance_{coordinate_type}_r2_by_sigma.png",
                y_label="Variance\nexplained R2",
                row_order=row_order,
            )

    if not conditional.empty:
        conditional = conditional.copy()
        conditional["delta_r2_clipped"] = pd.to_numeric(
            conditional["delta_r2"],
            errors="coerce",
        ).clip(lower=-0.25, upper=1.0)
        row_order = [
            "posterior_given_history",
            "history_given_posterior",
            "sample_values_given_posterior",
            "sample_order_given_posterior",
        ]
        for coordinate_type in [
            ct for ct in ["latent_mu", "prior_norm_latent_mu", "pga"]
            if ct in set(conditional["coordinate_type"].astype(str))
        ]:
            piece = conditional[conditional["coordinate_type"].astype(str) == coordinate_type]
            summary = summarize_seed_metric(
                piece,
                group_cols=["family", "parameter_value", "parameter_label", "sigma", "test"],
                value_col="delta_r2_clipped",
            )
            summary.to_csv(figures_dir / f"conditional_{coordinate_type}_partial_r2_summary.csv", index=False)
            plot_family_sigma_grid(
                summary,
                row_col="test",
                y_col="mean",
                y_sem_col="sem",
                outpath=figures_dir / f"conditional_{coordinate_type}_partial_r2_by_sigma.png",
                y_label="Incremental\nR2",
                row_order=row_order,
            )

    if not conditional_decoder.empty:
        conditional_decoder = conditional_decoder.copy()
        conditional_decoder["delta_r2_clipped"] = pd.to_numeric(
            conditional_decoder["delta_r2"],
            errors="coerce",
        ).clip(lower=-0.25, upper=1.0)
        representations = [
            rep for rep in [
                "z_sample",
                "prior_norm_z_sample",
                "z_mu",
                "z_mu_logsigma",
                "prior_norm_z_mu",
                "prior_norm_z_mu_logsigma",
                "pga_scores",
            ]
            if rep in set(conditional_decoder["representation"].astype(str))
        ]
        for representation in representations:
            piece = conditional_decoder[conditional_decoder["representation"].astype(str) == representation]
            summary = summarize_seed_metric(
                piece,
                group_cols=["family", "parameter_value", "parameter_label", "sigma", "target_family"],
                value_col="delta_r2_clipped",
            )
            summary.to_csv(
                figures_dir / f"conditional_decoder_{representation}_target_family_summary.csv",
                index=False,
            )
            plot_family_sigma_grid(
                summary,
                row_col="target_family",
                y_col="mean",
                y_sem_col="sem",
                outpath=figures_dir / f"conditional_decoder_{representation}_partial_r2_by_sigma.png",
                y_label="Sample info\nincremental R2",
                row_order=["individual_sample_values", "sample_order"],
            )

    if not terminal_decoder.empty:
        terminal_decoder = terminal_decoder.copy()
        terminal_decoder["r2_clipped"] = pd.to_numeric(
            terminal_decoder["r2"],
            errors="coerce",
        ).clip(lower=-0.25, upper=1.0)
        for decoder_type in [
            dt for dt in ["linear", "quadratic"]
            if dt in set(terminal_decoder["decoder_type"].astype(str))
        ]:
            type_piece = terminal_decoder[terminal_decoder["decoder_type"].astype(str) == decoder_type]
            for representation in [
                rep for rep in [
                    "z_sample",
                    "prior_norm_z_sample",
                    "z_mu",
                    "z_mu_logsigma",
                    "prior_norm_z_mu",
                    "prior_norm_z_mu_logsigma",
                    "pga_scores",
                ]
                if rep in set(type_piece["representation"].astype(str))
            ]:
                piece = type_piece[type_piece["representation"].astype(str) == representation]
                summary = summarize_seed_metric(
                    piece,
                    group_cols=["family", "parameter_value", "parameter_label", "sigma", "target_family"],
                    value_col="r2_clipped",
                )
                summary.to_csv(
                    figures_dir / f"terminal_history_decoder_{decoder_type}_{representation}_target_family_summary.csv",
                    index=False,
                )
                plot_family_sigma_grid(
                    summary,
                    row_col="target_family",
                    y_col="mean",
                    y_sem_col="sem",
                    outpath=figures_dir / f"terminal_history_decoder_{decoder_type}_{representation}_r2_by_sigma.png",
                    y_label="History\nDecoder R2",
                    row_order=[
                        "history_length",
                        "node_value_history",
                        "node_visit_presence",
                        "history_values",
                        "history_order",
                        "history_presence",
                    ],
                )

    if not terminal_conditional_decoder.empty:
        terminal_conditional_decoder = terminal_conditional_decoder.copy()
        terminal_conditional_decoder["delta_r2_clipped"] = pd.to_numeric(
            terminal_conditional_decoder["delta_r2"],
            errors="coerce",
        ).clip(lower=-0.25, upper=1.0)
        for decoder_type in [
            dt for dt in ["linear", "quadratic"]
            if dt in set(terminal_conditional_decoder["decoder_type"].astype(str))
        ]:
            type_piece = terminal_conditional_decoder[
                terminal_conditional_decoder["decoder_type"].astype(str) == decoder_type
            ]
            for representation in [
                rep for rep in [
                    "z_sample",
                    "prior_norm_z_sample",
                    "z_mu",
                    "z_mu_logsigma",
                    "prior_norm_z_mu",
                    "prior_norm_z_mu_logsigma",
                    "pga_scores",
                ]
                if rep in set(type_piece["representation"].astype(str))
            ]:
                piece = type_piece[type_piece["representation"].astype(str) == representation]
                summary = summarize_seed_metric(
                    piece,
                    group_cols=["family", "parameter_value", "parameter_label", "sigma", "target_family"],
                    value_col="delta_r2_clipped",
                )
                summary.to_csv(
                    figures_dir
                    / f"terminal_history_conditional_decoder_{decoder_type}_{representation}_target_family_summary.csv",
                    index=False,
                )
                plot_family_sigma_grid(
                    summary,
                    row_col="target_family",
                    y_col="mean",
                    y_sem_col="sem",
                    outpath=figures_dir
                    / f"terminal_history_conditional_decoder_{decoder_type}_{representation}_partial_r2_by_sigma.png",
                    y_label="History info\nincremental R2",
                    row_order=["history_values", "history_order", "history_presence"],
                )

    if not terminal_value_mae_timestep.empty:
        terminal_value_mae_timestep = terminal_value_mae_timestep.copy()
        terminal_value_mae_timestep["decoder_mae"] = pd.to_numeric(
            terminal_value_mae_timestep["decoder_mae"],
            errors="coerce",
        )
        if "decoder_mse" in terminal_value_mae_timestep.columns:
            terminal_value_mae_timestep["decoder_mse"] = pd.to_numeric(
                terminal_value_mae_timestep["decoder_mse"],
                errors="coerce",
            )
        decoder_types = [
            dt for dt in decoder_type_order
            if dt in set(terminal_value_mae_timestep["decoder_type"].astype(str))
        ]
        if not full_h2_plots:
            decoder_types = [dt for dt in decoder_types if dt in decoder_type_filter]
        for decoder_type in decoder_types:
            type_piece = terminal_value_mae_timestep[
                terminal_value_mae_timestep["decoder_type"].astype(str) == decoder_type
            ]
            representations = [
                rep for rep in [
                    "z_sample",
                    "prior_norm_z_sample",
                    "z_mu",
                    "z_mu_logsigma",
                    "prior_norm_z_mu",
                    "prior_norm_z_mu_logsigma",
                    "pga_scores",
                ]
                if rep in set(type_piece["representation"].astype(str))
            ]
            if not full_h2_plots:
                representations = [rep for rep in representations if rep == "z_sample"]
            for representation in representations:
                piece = type_piece[type_piece["representation"].astype(str) == representation]
                if piece.empty:
                    continue
                summary = summarize_seed_metric(
                    piece,
                    group_cols=[
                        "family",
                        "parameter_value",
                        "parameter_label",
                        "sigma",
                        "observations_in_trial",
                        "timestep_before_current_observation",
                    ],
                    value_col="decoder_mae",
                    weight_col="n_test",
                    min_weight=int(min_decoder_test_samples),
                )
                if summary.empty:
                    continue
                summary.to_csv(
                    figures_dir
                    / f"terminal_history_value_mae_by_timestep_{decoder_type}_{representation}_summary.csv",
                    index=False,
                )
                plot_timestep_mae_sigma_rows(
                    summary,
                    outpath=figures_dir
                    / f"terminal_history_value_mae_by_timestep_{decoder_type}_{representation}_sigma_rows.png",
                    y_label="Node history\ndecoder MAE",
                )
                if "abs_actual_path_value_difference" in piece.columns:
                    gap_piece = piece[
                        np.isfinite(
                            pd.to_numeric(
                                piece["abs_actual_path_value_difference"],
                                errors="coerce",
                            )
                        )
                    ].copy()
                    if not gap_piece.empty:
                        gap_summary = summarize_seed_metric(
                            gap_piece,
                            group_cols=[
                                "family",
                                "parameter_value",
                                "parameter_label",
                                "sigma",
                                "timestep_before_current_observation",
                                "abs_actual_path_value_difference",
                            ],
                            value_col="decoder_mae",
                            weight_col="n_test",
                            min_weight=int(min_decoder_test_samples),
                        )
                        if not gap_summary.empty:
                            gap_summary.to_csv(
                                figures_dir
                                / f"terminal_history_value_mae_by_abs_actual_reward_difference_{decoder_type}_{representation}_summary.csv",
                                index=False,
                            )
                            plot_abs_reward_diff_mae_sigma_rows(
                                gap_summary,
                                outpath=figures_dir
                                / f"terminal_history_value_mae_by_abs_actual_reward_difference_{decoder_type}_{representation}_sigma_rows.png",
                                y_label="Node history\ndecoder MAE",
                            )
                reward_pair_cols = {
                    "actual_path_value_1",
                    "actual_path_value_2",
                    "decoder_mse",
                }
                if reward_pair_cols.issubset(piece.columns):
                    reward_pair_piece = piece[
                        np.isfinite(
                            pd.to_numeric(piece["actual_path_value_1"], errors="coerce")
                        )
                        & np.isfinite(
                            pd.to_numeric(piece["actual_path_value_2"], errors="coerce")
                        )
                        & np.isfinite(pd.to_numeric(piece["decoder_mse"], errors="coerce"))
                    ].copy()
                    if not reward_pair_piece.empty:
                        reward_pair_summary = summarize_seed_metric(
                            reward_pair_piece,
                            group_cols=[
                                "family",
                                "parameter_value",
                                "parameter_label",
                                "sigma",
                                "timestep_before_current_observation",
                                "actual_path_value_1",
                                "actual_path_value_2",
                            ],
                            value_col="decoder_mse",
                            weight_col="n_test",
                            min_weight=int(min_decoder_test_samples),
                        )
                        if not reward_pair_summary.empty:
                            reward_pair_summary.to_csv(
                                figures_dir
                                / f"terminal_history_value_mse_by_node1_node2_actual_reward_{decoder_type}_{representation}_summary.csv",
                                index=False,
                            )
                            plot_reward_pair_mse_grids_by_sigma(
                                reward_pair_summary,
                                outpath_prefix=figures_dir
                                / f"terminal_history_value_mse_by_node1_node2_actual_reward_{decoder_type}_{representation}",
                                y_label="Node history\ndecoder MSE",
                            )

    if not terminal_value_error.empty:
        terminal_value_error = terminal_value_error.copy()
        value_error_specs = [
            (
                "decoder_mse",
                "Decoder MSE",
                "decoder_mse",
                False,
            ),
            (
                "mse_ratio_to_history_mean",
                "Decoder MSE /\nhistory-mean MSE",
                "ratio_to_history_mean",
                True,
            ),
            (
                "mse_ratio_to_random_mean_var",
                "Decoder MSE /\nrandom mean-var MSE",
                "ratio_to_random_mean_var",
                True,
            ),
            (
                "mse_diff_to_history_mean",
                "Decoder MSE -\nhistory-mean MSE",
                "diff_to_history_mean",
                False,
            ),
            (
                "mse_diff_to_random_mean_var",
                "Decoder MSE -\nrandom mean-var MSE",
                "diff_to_random_mean_var",
                False,
            ),
        ]
        if not full_h2_plots:
            value_error_specs = [
                (
                    "mse_ratio_to_random_mean_var",
                    "Decoder MSE /\nrandom mean-var MSE",
                    "ratio_to_random_mean_var",
                    True,
                )
            ]
        for metric, y_label, suffix, is_ratio in value_error_specs:
            if metric not in terminal_value_error.columns:
                continue
            metric_values = pd.to_numeric(
                terminal_value_error[metric],
                errors="coerce",
            )
            terminal_value_error[metric] = (
                metric_values.clip(lower=0.0, upper=5.0)
                if is_ratio
                else metric_values
            )
            decoder_types = [
                dt for dt in decoder_type_order
                if dt in set(terminal_value_error["decoder_type"].astype(str))
            ]
            if not full_h2_plots:
                decoder_types = [dt for dt in decoder_types if dt in decoder_type_filter]
            for decoder_type in decoder_types:
                type_piece = terminal_value_error[
                    terminal_value_error["decoder_type"].astype(str) == decoder_type
                ]
                representations = [
                    rep for rep in [
                        "z_sample",
                        "prior_norm_z_sample",
                        "z_mu",
                        "z_mu_logsigma",
                        "prior_norm_z_mu",
                        "prior_norm_z_mu_logsigma",
                        "pga_scores",
                    ]
                    if rep in set(type_piece["representation"].astype(str))
                ]
                if not full_h2_plots:
                    representations = [rep for rep in representations if rep == "z_sample"]
                for representation in representations:
                    piece = type_piece[type_piece["representation"].astype(str) == representation]
                    scope_values = (
                        ["all_paid_observations", "final_paid_observation"]
                        if "value_error_latent_scope" in piece.columns
                        else ["all_paid_observations"]
                    )
                    for scope in scope_values:
                        if "value_error_latent_scope" in piece.columns:
                            scope_piece = piece[piece["value_error_latent_scope"].astype(str) == scope]
                        else:
                            scope_piece = piece
                        if scope_piece.empty:
                            continue
                        scope_suffix = "" if scope == "all_paid_observations" else "_final_paid"
                        if "value_error_stratum" in scope_piece.columns:
                            pooled_piece = scope_piece[scope_piece["value_error_stratum"].astype(str) == "all"]
                        else:
                            pooled_piece = scope_piece
                        if is_ratio and "sigma" in pooled_piece.columns:
                            pooled_piece = pooled_piece[pd.to_numeric(pooled_piece["sigma"], errors="coerce") > 1e-12]
                        summary = summarize_seed_metric(
                            pooled_piece,
                            group_cols=["family", "parameter_value", "parameter_label", "sigma", "target_family"],
                            value_col=metric,
                            weight_col="n_test",
                            min_weight=int(min_decoder_test_samples),
                        )
                        summary.to_csv(
                            figures_dir
                            / f"terminal_history_value_error_{decoder_type}_{representation}_{suffix}{scope_suffix}_summary.csv",
                            index=False,
                        )
                        plot_family_sigma_grid(
                            summary,
                            row_col="target_family",
                            y_col="mean",
                            y_sem_col="sem",
                            outpath=figures_dir
                            / f"terminal_history_value_error_{decoder_type}_{representation}_{suffix}{scope_suffix}_by_sigma.png",
                            y_label=y_label,
                            row_order=["node_value_history", "history_values"],
                        )

                        if "value_error_stratum" not in scope_piece.columns or "observations_in_trial" not in scope_piece.columns:
                            continue
                        total_piece = scope_piece[
                            scope_piece["value_error_stratum"].astype(str) == "by_total_observations"
                        ].copy()
                        total_piece["observations_in_trial"] = pd.to_numeric(
                            total_piece["observations_in_trial"],
                            errors="coerce",
                        )
                        total_piece = total_piece[np.isfinite(total_piece["observations_in_trial"])]
                        if is_ratio and "sigma" in total_piece.columns:
                            total_piece = total_piece[pd.to_numeric(total_piece["sigma"], errors="coerce") > 1e-12]
                        total_summary = summarize_seed_metric(
                            total_piece,
                            group_cols=[
                                "family",
                                "parameter_value",
                                "parameter_label",
                                "sigma",
                                "observations_in_trial",
                            ],
                            value_col=metric,
                            weight_col="n_test",
                            min_weight=int(min_decoder_test_samples),
                        )
                        if total_summary.empty:
                            continue
                        total_summary["total_observation_label"] = total_summary["observations_in_trial"].map(
                            lambda x: f"total obs\n= {int(float(x))}" if pd.notna(x) else "total obs\n= NA"
                        )
                        row_order = [
                            f"total obs\n= {int(float(x))}"
                            for x in sorted(pd.to_numeric(total_summary["observations_in_trial"], errors="coerce").dropna().unique())
                        ]
                        total_summary.to_csv(
                            figures_dir
                            / f"terminal_history_value_error_{decoder_type}_{representation}_{suffix}{scope_suffix}_by_total_observations_summary.csv",
                            index=False,
                        )
                        plot_family_sigma_grid(
                            total_summary,
                            row_col="total_observation_label",
                            y_col="mean",
                            y_sem_col="sem",
                            outpath=figures_dir
                            / f"terminal_history_value_error_{decoder_type}_{representation}_{suffix}{scope_suffix}_by_total_observations_sigma.png",
                            y_label=y_label,
                            row_order=row_order,
                        )

    if not matched.empty:
        for metric, y_label in [
            ("latent_within_posterior_ms", "Within posterior\nlatent MS"),
            ("between_history_centroid_ms", "History centroid\nMS"),
        ]:
            if metric not in matched.columns:
                continue
            for coordinate_type in [
                ct for ct in ["latent_mu", "prior_norm_latent_mu", "pga"]
                if ct in set(matched["coordinate_type"].astype(str))
            ]:
                piece = matched[matched["coordinate_type"].astype(str) == coordinate_type]
                summary = summarize_seed_metric(
                    piece,
                    group_cols=["family", "parameter_value", "parameter_label", "sigma", "coordinate_type"],
                    value_col=metric,
                )
                summary.to_csv(figures_dir / f"matched_posterior_{coordinate_type}_{metric}_summary.csv", index=False)
                plot_family_sigma_grid(
                    summary,
                    row_col="coordinate_type",
                    y_col="mean",
                    y_sem_col="sem",
                    outpath=figures_dir / f"matched_posterior_{coordinate_type}_{metric}_by_sigma.png",
                    y_label=y_label,
                    row_order=[coordinate_type],
                )


def maybe_add_pga_scores(df: pd.DataFrame, latent_dim: int, include_pga: bool) -> pd.DataFrame:
    if not include_pga or df.empty:
        return df
    try:
        from analysis.plot_revisit_latent_density_gaussian_pga_jax import ProductGaussianPGA
    except Exception as exc:
        print(f"Warning: could not import ProductGaussianPGA ({exc}); skipping PGA scores.", flush=True)
        return df
    mu_cols = [f"z_mu_{k}" for k in range(latent_dim)]
    sigma_cols = [f"z_sigma_{k}" for k in range(latent_dim)]
    if not set(mu_cols + sigma_cols).issubset(df.columns):
        return df
    mu = numeric_cols(df, mu_cols)
    sigma = numeric_cols(df, sigma_cols)
    ok = np.all(np.isfinite(mu), axis=1) & np.all(np.isfinite(sigma), axis=1) & np.all(sigma > 0, axis=1)
    if ok.sum() < 20:
        return df
    out = df.copy()
    pga = ProductGaussianPGA(n_components=2, max_iters=30)
    scores = np.full((len(out), 2), np.nan, dtype=float)
    scores[ok] = pga.fit_transform(mu[ok], sigma[ok])
    out["pga_score_0"] = scores[:, 0]
    out["pga_score_1"] = scores[:, 1]
    return out


def combo_dir_name(
    *,
    family: str,
    parameter_name: str,
    parameter_value: float,
    seed: int,
    beta: float,
    opportunity: float,
    lambda_value: float,
    sigma: float,
    rnn_dim: int,
    latent_dim: int,
    tree_type: str,
) -> str:
    raw = (
        f"seed_{seed}_{family}_{parameter_name}_{parameter_value:g}_"
        f"beta_{beta:g}_opp_{opportunity:g}_lambda_{lambda_value:g}_sigma_{sigma:g}_"
        f"rnn_{rnn_dim}_latent_{latent_dim}_{tree_type}"
    )
    return raw.replace(".", "p")


def run_combo(
    args: argparse.Namespace,
    outroot: Path,
    *,
    family: str,
    parameter_name: str,
    parameter_value: float,
    seed: int,
    beta: float,
    opportunity: float,
    lambda_value: float,
    alpha: float,
    sigma: float,
    rnn_dim: int,
    latent_dim: int,
) -> dict[str, Path | str]:
    config = make_config(
        args,
        seed=seed,
        beta=beta,
        lambda_value=lambda_value,
        alpha=alpha,
        opportunity=opportunity,
        sigma=sigma,
        rnn_dim=rnn_dim,
        latent_dim=latent_dim,
    )
    task = jp.build_task(config.tree_size, config.tree_type, config.input_type)
    model, params = jp.load_state_for_sim(config, task)
    combo_dir = outroot / combo_dir_name(
        family=family,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        seed=seed,
        beta=beta,
        opportunity=opportunity,
        lambda_value=lambda_value,
        sigma=sigma,
        rnn_dim=rnn_dim,
        latent_dim=latent_dim,
        tree_type=task.tree_type,
    )
    combo_dir.mkdir(parents=True, exist_ok=True)

    metadata_cols = {
        "family": family,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "parameter_label": f"beta = {parameter_value:g}" if parameter_name == "beta" else f"opp = {parameter_value:g}",
        "seed": seed,
        "beta": beta,
        "lambda": lambda_value,
        "alpha": alpha,
        "opportunity_cost": opportunity,
        "sigma": sigma,
        "rnn_dim": rnn_dim,
        "latent_dim": latent_dim,
        "tree_type": task.tree_type,
    }

    forced_cases = history_case_templates(task, sigma)
    forced_df = force_history_batch(
        model=model,
        params=params,
        config=config,
        task=task,
        cases=forced_cases,
        posterior_sigma_floor=float(args.posterior_sigma_floor),
    )
    for key, value in metadata_cols.items():
        forced_df[key] = value
    forced_df.to_csv(combo_dir / "forced_history_states.csv", index=False)
    pairwise_df = pairwise_forced_history_metrics(forced_df, latent_dim)
    for key, value in metadata_cols.items():
        pairwise_df[key] = value
    pairwise_df.to_csv(combo_dir / "forced_history_pairwise.csv", index=False)

    max_history_length = min(int(args.max_history_length), int(args.max_observations_before_stop))
    probe_parts = []
    if args.probe_source in {"policy", "both"}:
        policy_probe_df = rollout_policy_probe_states(
            model=model,
            params=params,
            config=config,
            task=task,
            n_trials=int(args.probe_n_states),
            posterior_sigma_floor=float(args.posterior_sigma_floor),
            max_history_length=max_history_length,
            batch_size=int(args.batch_size),
        )
        policy_probe_df.to_csv(combo_dir / "policy_probe_state_dataset.csv", index=False)
        probe_parts.append(policy_probe_df)
    if args.probe_source in {"forced", "both"}:
        rng = np.random.default_rng(seed + 710_000)
        action_nodes, observations, true_rewards, history_lengths = generate_probe_histories(
            task=task,
            rng=rng,
            n_states=int(args.probe_n_states),
            max_history_length=max_history_length,
            sigma=sigma,
        )
        forced_probe_df = rollout_probe_states(
            model=model,
            params=params,
            config=config,
            task=task,
            action_nodes=action_nodes,
            observations=observations,
            true_rewards=true_rewards,
            history_lengths=history_lengths,
            posterior_sigma_floor=float(args.posterior_sigma_floor),
            batch_size=int(args.batch_size),
        )
        forced_probe_df["source"] = "forced"
        forced_probe_df.to_csv(combo_dir / "forced_probe_state_dataset.csv", index=False)
        probe_parts.append(forced_probe_df)
    probe_df = pd.concat(probe_parts, ignore_index=True) if probe_parts else pd.DataFrame()
    probe_df = maybe_add_pga_scores(probe_df, latent_dim, bool(args.include_pga))
    for key, value in metadata_cols.items():
        probe_df[key] = value
    probe_df.to_csv(combo_dir / "probe_state_dataset.csv", index=False)

    decoder_df = run_latent_decoders(
        probe_df,
        task,
        latent_dim,
        max_history_length,
        seed,
        float(args.ridge_alpha),
        float(args.test_fraction),
    )
    for key, value in metadata_cols.items():
        decoder_df[key] = value
    decoder_df.to_csv(combo_dir / "latent_decoder_r2.csv", index=False)

    variance_df = run_variance_decomposition(
        probe_df,
        task,
        latent_dim,
        max_history_length,
        seed,
        float(args.ridge_alpha),
        float(args.test_fraction),
    )
    for key, value in metadata_cols.items():
        variance_df[key] = value
    variance_df.to_csv(combo_dir / "variance_decomposition_r2.csv", index=False)
    print(f"Saved hypothesis-2 diagnostics to {combo_dir}", flush=True)
    return {
        "combo_dir": combo_dir,
        "forced_history_pairwise": combo_dir / "forced_history_pairwise.csv",
        "probe_state_dataset": combo_dir / "probe_state_dataset.csv",
        "latent_decoder_r2": combo_dir / "latent_decoder_r2.csv",
        "variance_decomposition_r2": combo_dir / "variance_decomposition_r2.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hypothesis-2 revisit posterior-vs-sample representation tests.")
    parser.add_argument("tree", nargs="?", default=None)
    parser.add_argument("--preset-file", default=str(default_preset_file()))
    parser.add_argument("--output-root", "--results-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--alphas", nargs="+", default=["0.0"])
    parser.add_argument("--betas", "--vary-beta-values", dest="betas", default=None)
    parser.add_argument("--lambdas", dest="lambda_values", nargs="+", default=["100.0"])
    parser.add_argument("--opportunity-costs", "--vary-opportunity-values", dest="opportunity_costs", default=None)
    parser.add_argument("--sigmas", "--sigma-list", dest="sigmas", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--rnn-dims", default=None)
    parser.add_argument("--latent-dims", default=None)
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform")
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae")
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing hypothesis-2 CSVs from outdir and generate plots without rerunning model simulations.",
    )
    parser.add_argument(
        "--full-h2-plots",
        action="store_true",
        help=(
            "Generate the full legacy hypothesis-2 plot set. By default, plot-only "
            "generates only the four quadratic z_sample random-baseline ratio plots."
        ),
    )
    parser.add_argument("--allow-node-revisit", action="store_true")
    parser.add_argument("--max-observations-before-stop", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--probe-n-states", type=int, default=2000)
    parser.add_argument(
        "--probe-source",
        choices=["policy", "forced", "both"],
        default="policy",
        help=(
            "policy: collect latents only from histories reached by the model's own "
            "observe/stop policy. forced: use counterfactual forced random histories."
        ),
    )
    parser.add_argument("--max-history-length", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--posterior-sigma-floor", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--h2-extra-tests",
        nargs="+",
        choices=[
            "terminal_history_decoder",
            "terminal_history_value_error",
            "all",
            "none",
        ],
        default=["terminal_history_value_error"],
        help=(
            "Extra probe-state analyses to run during plotting. Default runs the "
            "terminal value-error baseline test. Slow mixed/partial tests are disabled."
        ),
    )
    parser.add_argument(
        "--max-analysis-states-per-combo",
        type=int,
        default=0,
        help=(
            "Use only the first N probe states per parameter combination for "
            "extra analyses. 0 means use all saved states."
        ),
    )
    parser.add_argument(
        "--min-decoder-test-samples",
        type=int,
        default=0,
        help=(
            "For terminal history value-error plots, drop decoder-target rows "
            "with fewer than this many held-out test examples. 0 disables this "
            "filter. Seed-level averages are still weighted by n_test."
        ),
    )
    parser.add_argument(
        "--probe-decoder-types",
        default=None,
        help=(
            "Comma/space-separated decoder types for terminal history value-error "
            "analyses: linear, quadratic, deep. Defaults preserve the old behavior."
        ),
    )
    parser.add_argument(
        "--deep-probe-hidden-dims",
        default="64,32",
        help="Hidden layer sizes for the deep probe decoder, e.g. 128,64.",
    )
    parser.add_argument("--deep-probe-epochs", type=int, default=DEFAULT_DEEP_PROBE_EPOCHS)
    parser.add_argument("--deep-probe-lr", type=float, default=DEFAULT_DEEP_PROBE_LR)
    parser.add_argument("--deep-probe-weight-decay", type=float, default=DEFAULT_DEEP_PROBE_WEIGHT_DECAY)
    parser.add_argument(
        "--terminal-decoder-target-mode",
        choices=["minimal", "values", "all"],
        default="all",
        help="Targets for terminal_history_decoder. all decodes values, order, presence, and history length.",
    )
    parser.add_argument(
        "--terminal-decoder-representation-mode",
        choices=["sample", "minimal", "mu_logsigma", "all"],
        default="sample",
        help="Latent representations for terminal_history_decoder.",
    )
    parser.add_argument("--include-pga", action="store_true")
    parser.add_argument("--return-target-mode", default="sampled_lambda")
    parser.add_argument("--sampled-lambda-critic", choices=["q", "value"], default="q")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--return-target-rollouts", type=int, default=8)
    parser.add_argument("--target-critic-update-interval", type=int, default=100)
    parser.add_argument("--target-critic-tau", type=float, default=1.0)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatches", type=int, default=1)
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    args = parser.parse_args()
    args.parameter_combos = None
    if args.tree is not None:
        return load_preset_rows(args)

    args.alphas = parse_csv_values(args.alphas, float)
    args.betas = parse_csv_values(args.betas or "", float)
    args.lambda_values = parse_csv_values(args.lambda_values, float)
    args.opportunity_costs = parse_csv_values(args.opportunity_costs or "0.0", float)
    args.sigmas = parse_csv_values(args.sigmas or "0.0", float)
    args.seeds = parse_csv_values(args.seeds, int)
    args.rnn_dims = parse_csv_values(args.rnn_dims or "16", int)
    args.latent_dims = parse_csv_values(args.latent_dims or "2", int)
    args.max_observations_before_stop = int(args.max_observations_before_stop or 10)
    args.outdir = args.outdir or "results/revisit_hypothesis2/manual"
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


def requested_filter_from_args(args: argparse.Namespace) -> dict:
    return {
        "parameter_combos": list(getattr(args, "parameter_combos", []) or []),
        "sigmas": list(getattr(args, "sigmas", []) or []),
        "seeds": list(getattr(args, "seeds", []) or []),
        "lambda_values": list(getattr(args, "lambda_values", []) or []),
        "alphas": list(getattr(args, "alphas", []) or []),
        "rnn_dims": list(getattr(args, "rnn_dims", []) or []),
        "latent_dims": list(getattr(args, "latent_dims", []) or []),
    }


def main() -> None:
    args = parse_args()
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    deep_probe_config = {
        "hidden_dims": parse_hidden_dims(args.deep_probe_hidden_dims),
        "epochs": int(args.deep_probe_epochs),
        "lr": float(args.deep_probe_lr),
        "weight_decay": float(args.deep_probe_weight_decay),
    }
    probe_decoder_types = parse_decoder_types(
        args.probe_decoder_types,
        full_h2_plots=bool(args.full_h2_plots),
    )
    if args.plot_only:
        plot_hypothesis2_outputs(
            outroot,
            ridge_alpha=float(args.ridge_alpha),
            test_fraction=float(args.test_fraction),
            extra_tests=list(args.h2_extra_tests),
            max_analysis_states_per_combo=int(args.max_analysis_states_per_combo),
            terminal_target_mode=str(args.terminal_decoder_target_mode),
            terminal_representation_mode=str(args.terminal_decoder_representation_mode),
            input_type=str(args.input_type),
            full_h2_plots=bool(args.full_h2_plots),
            requested_filter=requested_filter_from_args(args),
            min_decoder_test_samples=int(args.min_decoder_test_samples),
            probe_decoder_types=probe_decoder_types,
            deep_probe_config=deep_probe_config,
        )
        return
    aggregate_paths: dict[str, list[Path]] = {
        "forced_history_pairwise": [],
        "probe_state_dataset": [],
        "latent_decoder_r2": [],
        "variance_decomposition_r2": [],
    }
    for family, parameter_name, parameter_value, beta, opp in args.parameter_combos:
        for seed in args.seeds:
            for lambda_value in args.lambda_values:
                for alpha in args.alphas:
                    for sigma in args.sigmas:
                        for rnn_dim in args.rnn_dims:
                            for latent_dim in args.latent_dims:
                                outputs = run_combo(
                                    args,
                                    outroot,
                                    family=family,
                                    parameter_name=parameter_name,
                                    parameter_value=parameter_value,
                                    seed=int(seed),
                                    beta=float(beta),
                                    opportunity=float(opp),
                                    lambda_value=float(lambda_value),
                                    alpha=float(alpha),
                                    sigma=float(sigma),
                                    rnn_dim=int(rnn_dim),
                                    latent_dim=int(latent_dim),
                                )
                                for key in aggregate_paths:
                                    aggregate_paths[key].append(Path(outputs[key]))
    for key, paths in aggregate_paths.items():
        frames = [pd.read_csv(path) for path in paths if path.exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(outroot / f"{key}_all.csv", index=False)
    plot_hypothesis2_outputs(
        outroot,
        ridge_alpha=float(args.ridge_alpha),
        test_fraction=float(args.test_fraction),
        extra_tests=list(args.h2_extra_tests),
        max_analysis_states_per_combo=int(args.max_analysis_states_per_combo),
        terminal_target_mode=str(args.terminal_decoder_target_mode),
        terminal_representation_mode=str(args.terminal_decoder_representation_mode),
        input_type=str(args.input_type),
        full_h2_plots=bool(args.full_h2_plots),
        requested_filter=requested_filter_from_args(args),
        min_decoder_test_samples=int(args.min_decoder_test_samples),
        probe_decoder_types=probe_decoder_types,
        deep_probe_config=deep_probe_config,
    )


if __name__ == "__main__":
    main()
