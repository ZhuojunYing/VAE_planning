#!/usr/bin/env python3
"""Post-hoc effective-dimensionality and causal ablation diagnostics for JAX runs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp

from model_jax import planning as jp


def parse_list(value: str, cast):
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def make_config(
    *,
    beta: float,
    opportunity: float,
    seed: int,
    n_trials: int,
    rnn_units: int,
    latent_dim: int,
    max_observations: int,
    lambda_return: float,
    checkpoint_root: str,
) -> jp.RunConfig:
    return jp.RunConfig(
        lambda_=100.0,
        alpha=0.0,
        beta=float(beta),
        model_dir=checkpoint_root,
        epochs=120,
        input_type="uniform",
        seed=int(seed),
        tree_size=6,
        train_mode="sim",
        tree_type="disjoint3x2",
        opportunity_cost=float(opportunity),
        expansion_decision_version="lstm",
        model_variant="vae",
        rnn_units=int(rnn_units),
        latent_dim=int(latent_dim),
        sim_dir="outputs/jax_simulations",
        n_sim_trials=int(n_trials),
        num_envs=200,
        num_steps=int(max_observations) + 1,
        update_epochs=5,
        ppo_minibatches=1,
        steps_per_epoch=1,
        return_target_rollouts=8,
        return_target_mode="sampled_lambda",
        sampled_lambda_critic="q",
        lambda_return=float(lambda_return),
        target_critic_update_interval=100,
        target_critic_tau=1.0,
        backend=None,
        jit_training=True,
        profile_update_components=False,
        profile_update_components_every=10,
        enable_reconstruction=False,
        enable_probe=False,
        allow_node_revisit=True,
        max_observations_before_stop=int(max_observations),
        observation_sigma=0.0,
        kl_start_multiplier=1.0,
        kl_annealing_epochs=0,
        node_coverage_aux_coef=0.0,
        node_coverage_aux_epochs=0,
    )


def clone_model(
    model: jp.PlanningVAE,
    *,
    latent_ablate_to_prior: bool = False,
    latent_keep_dims: tuple[int, ...] = (),
    lstm_context_pca_mean: tuple[float, ...] = (),
    lstm_context_pca_components: tuple[tuple[float, ...], ...] = (),
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
        latent_perturb_mode=model.latent_perturb_mode,
        latent_perturb_timestep=model.latent_perturb_timestep,
        latent_perturb_scale=model.latent_perturb_scale,
        latent_ablate_to_prior=bool(latent_ablate_to_prior),
        latent_keep_dims=tuple(int(x) for x in latent_keep_dims),
        lstm_context_pca_mean=tuple(float(x) for x in lstm_context_pca_mean),
        lstm_context_pca_components=tuple(
            tuple(float(v) for v in row) for row in lstm_context_pca_components
        ),
    )


def rollout(model, params, config: jp.RunConfig, task: jp.TaskSpec, reset_rewards, rng_seed_offset: int):
    rng = jax.random.PRNGKey(config.seed + 100_000 + int(rng_seed_offset))
    reward_feature_dim = int(model.reward_feature_dim_override) or jp.reward_feature_dim_for_sigma(
        config.observation_sigma
    )
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
    return transitions


def stop_index_for_trial(transitions, trial: int) -> int | None:
    for t, trans in enumerate(transitions, start=1):
        if bool(np.asarray(trans.is_stop)[trial]):
            return t
    return None


def entropy(probs):
    p = np.asarray(probs, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p / total
    return float(-np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0)))


def summarize_rollout(transitions, reset_rewards, task: jp.TaskSpec, config: jp.RunConfig, label: str):
    rewards = np.asarray(reset_rewards)
    path_map = np.asarray(task.path_map, dtype=float)
    path_rewards = rewards @ path_map.T
    best_values = path_rewards.max(axis=1)
    denominator = float(best_values.mean())
    rng = np.random.default_rng(config.seed + 700_000 + abs(hash(label)) % 100_000)
    chosen = np.full(rewards.shape[0], -1, dtype=int)
    stop_t = np.full(rewards.shape[0], config.num_steps, dtype=int)
    term_entropy = np.full(rewards.shape[0], np.nan)
    terminal_probs_last = np.ones((rewards.shape[0], path_rewards.shape[1])) / path_rewards.shape[1]
    observed_nodes = [set() for _ in range(rewards.shape[0])]
    for t, trans in enumerate(transitions, start=1):
        is_stop = np.asarray(trans.is_stop) > 0.5
        node_index = np.asarray(trans.node_index)
        terminal_path_index = np.asarray(trans.terminal_path_index)
        terminal_probs = np.asarray(trans.action_output, dtype=float)
        terminal_probs_last = terminal_probs
        for trial in range(rewards.shape[0]):
            if chosen[trial] >= 0:
                continue
            node = node_index[trial]
            if np.isfinite(node) and int(node) >= 0:
                observed_nodes[trial].add(int(node) + 1)
            if is_stop[trial]:
                chosen[trial] = int(terminal_path_index[trial])
                stop_t[trial] = t
                term_entropy[trial] = entropy(terminal_probs[trial])
    missing = chosen < 0
    if np.any(missing):
        for trial in np.where(missing)[0]:
            probs = terminal_probs_last[trial]
            probs = np.where(np.isfinite(probs), probs, 0.0)
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones_like(probs) / len(probs)
            chosen[trial] = int(rng.choice(len(probs), p=probs))
            term_entropy[trial] = entropy(probs)
    chosen_values = path_rewards[np.arange(path_rewards.shape[0]), chosen]
    unique_nodes = np.asarray([len(x) for x in observed_nodes], dtype=float)
    path_groups = [(1, 2), (3, 4), (5, 6)]
    path_counts = np.zeros((len(observed_nodes), 3), dtype=int)
    for i, nodes in enumerate(observed_nodes):
        for p, group in enumerate(path_groups):
            path_counts[i, p] = sum(n in nodes for n in group)
    best_paths = path_rewards == best_values[:, None]
    chosen_best = best_paths[np.arange(path_rewards.shape[0]), chosen]
    return {
        "ablation": label,
        "n_trials": int(rewards.shape[0]),
        "norm_reward": float(chosen_values.mean() / denominator),
        "best_path_accuracy": float(chosen_best.mean()),
        "regret": float((best_values - chosen_values).mean()),
        "stop_timestep": float(stop_t.mean()),
        "observations_before_stop": float((stop_t - 1).mean()),
        "unique_nodes": float(unique_nodes.mean()),
        "all_nodes": float(np.mean(unique_nodes == 6)),
        "all_paths_touched": float(np.mean(np.all(path_counts > 0, axis=1))),
        "terminal_entropy": float(np.nanmean(term_entropy)),
    }


def valid_matrix(transitions, name: str, use_observe_mask: bool):
    chunks = []
    for trans in transitions:
        value = np.asarray(getattr(trans, name), dtype=float)
        valid = np.asarray(trans.valid) > 0.5
        if use_observe_mask:
            valid &= np.asarray(trans.is_observe) > 0.5
        chunks.append(value[valid])
    if not chunks:
        return np.zeros((0, 0), dtype=float)
    return np.concatenate(chunks, axis=0)


def prior_normalized_z(transitions):
    chunks = []
    for trans in transitions:
        valid = (np.asarray(trans.valid) > 0.5) & (np.asarray(trans.is_observe) > 0.5)
        z_mu = np.asarray(trans.z_mu, dtype=float)
        prior_mu = np.asarray(trans.prior_mu, dtype=float)
        prior_sigma = np.sqrt(np.exp(np.asarray(trans.prior_logvar, dtype=float)) + 1e-6)
        chunks.append(((z_mu - prior_mu) / prior_sigma)[valid])
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=float)


def per_dim_kl(transitions):
    chunks = []
    for trans in transitions:
        valid = (np.asarray(trans.valid) > 0.5) & (np.asarray(trans.is_observe) > 0.5)
        z_mu = np.asarray(trans.z_mu, dtype=float)
        z_logvar = np.asarray(trans.z_logvar, dtype=float)
        prior_mu = np.asarray(trans.prior_mu, dtype=float)
        prior_logvar = np.asarray(trans.prior_logvar, dtype=float)
        prior_var = np.exp(prior_logvar) + 1e-6
        post_var = np.exp(np.clip(z_logvar, -10.0, 10.0))
        kl = 0.5 * (
            np.log(prior_var + 1e-6)
            - np.log(post_var + 1e-6)
            + (post_var + (z_mu - prior_mu) ** 2) / (prior_var + 1e-6)
            - 1.0
        )
        chunks.append(kl[valid])
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=float)


def pca_summary(X, condition, seed, representation):
    X = np.asarray(X, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)] if X.size else X
    if X.shape[0] < 3 or X.shape[1] == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), np.zeros((0,))
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (s**2) / max(X.shape[0] - 1, 1)
    ratios = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    cum = np.cumsum(ratios)
    rows = []
    for i, (ev, rr, cc) in enumerate(zip(eig, ratios, cum), start=1):
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "representation": representation,
                "pc": i,
                "eigenvalue": float(ev),
                "variance_ratio": float(rr),
                "cumulative_variance": float(cc),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "dims_95": int(np.searchsorted(cum, 0.95) + 1),
                "dims_99": int(np.searchsorted(cum, 0.99) + 1),
            }
        )
    return rows, vt, mean, ratios


def run_one(condition_name, beta, opportunity, seed, args):
    task = jp.build_task(6, "disjoint3x2", "uniform")
    config = make_config(
        beta=beta,
        opportunity=opportunity,
        seed=seed,
        n_trials=args.n_trials,
        rnn_units=args.rnn_units,
        latent_dim=args.latent_dim,
        max_observations=args.max_observations,
        lambda_return=args.lambda_return,
        checkpoint_root=args.checkpoint_root,
    )
    model, params = jp.load_state_for_sim(config, task)
    rng = jax.random.PRNGKey(seed + 500_000)
    reset_rewards = jp.sample_reward_matrix(
        rng,
        int(args.n_trials),
        task.num_nodes,
        jnp.asarray(task.reward_values, dtype=jnp.float32),
    )
    normal_transitions = rollout(model, params, config, task, reset_rewards, 0)

    pca_rows = []
    matrices = {
        "lstm_context": valid_matrix(normal_transitions, "expansion_input", False),
        "z_mu": valid_matrix(normal_transitions, "z_mu", True),
        "z_sample": valid_matrix(normal_transitions, "z_sample", True),
        "prior_normalized_z_mu": prior_normalized_z(normal_transitions),
    }
    pca_cache = {}
    for name, X in matrices.items():
        rows, components, mean, ratios = pca_summary(X, condition_name, seed, name)
        pca_rows.extend(rows)
        pca_cache[name] = (components, mean, ratios)

    kl = per_dim_kl(normal_transitions)
    latent_order = np.argsort(-np.nanmean(kl, axis=0)) if kl.size else np.arange(args.latent_dim)
    kl_rows = [
        {
            "condition": condition_name,
            "seed": seed,
            "latent_dim_index": int(i),
            "rank_by_mean_kl": int(np.where(latent_order == i)[0][0] + 1),
            "mean_kl": float(np.nanmean(kl[:, i])) if kl.size else np.nan,
        }
        for i in range(args.latent_dim)
    ]

    ablation_rows = []
    baseline = summarize_rollout(normal_transitions, reset_rewards, task, config, "none")
    baseline.update({"condition": condition_name, "seed": seed, "ablation_type": "none", "keep": args.rnn_units})
    ablation_rows.append(baseline)

    for k in args.latent_keeps:
        keep = tuple(int(i) for i in latent_order[: min(int(k), args.latent_dim)])
        ablated_model = clone_model(
            model,
            latent_ablate_to_prior=True,
            latent_keep_dims=keep,
        )
        trans = rollout(ablated_model, params, config, task, reset_rewards, 10_000 + int(k))
        row = summarize_rollout(trans, reset_rewards, task, config, f"latent_keep_{k}")
        row.update({"condition": condition_name, "seed": seed, "ablation_type": "latent_dim", "keep": int(k)})
        ablation_rows.append(row)

    components, mean, _ratios = pca_cache["lstm_context"]
    for k in args.hidden_pc_keeps:
        keep_components = components[: min(int(k), components.shape[0])]
        ablated_model = clone_model(
            model,
            lstm_context_pca_mean=tuple(float(x) for x in mean),
            lstm_context_pca_components=tuple(tuple(float(v) for v in row) for row in keep_components),
        )
        trans = rollout(ablated_model, params, config, task, reset_rewards, 20_000 + int(k))
        row = summarize_rollout(trans, reset_rewards, task, config, f"lstm_pc_keep_{k}")
        row.update({"condition": condition_name, "seed": seed, "ablation_type": "lstm_pc", "keep": int(k)})
        ablation_rows.append(row)

    return pca_rows, kl_rows, ablation_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seeds", default="4,5,6")
    parser.add_argument("--rnn-units", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--max-observations", type=int, default=20)
    parser.add_argument("--lambda-return", type=float, default=0.8)
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--outdir", default="analysis_outputs/effective_dimensionality_ablation_jax")
    parser.add_argument("--latent-keeps", default="0,1,2,4,8,16")
    parser.add_argument("--hidden-pc-keeps", default="0,2,4,8,16,24,32")
    args = parser.parse_args()
    args.seeds = parse_list(args.seeds, int)
    args.latent_keeps = parse_list(args.latent_keeps, int)
    args.hidden_pc_keeps = parse_list(args.hidden_pc_keeps, int)

    conditions = [
        ("beta1000_opp0.02", 1000.0, 0.02),
        ("beta100_opp0", 100.0, 0.0),
    ]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_pca, all_kl, all_ablation = [], [], []
    for condition_name, beta, opp in conditions:
        for seed in args.seeds:
            print(f"Running {condition_name}, seed={seed}", flush=True)
            pca_rows, kl_rows, ablation_rows = run_one(condition_name, beta, opp, seed, args)
            all_pca.extend(pca_rows)
            all_kl.extend(kl_rows)
            all_ablation.extend(ablation_rows)

    pca_df = pd.DataFrame(all_pca)
    kl_df = pd.DataFrame(all_kl)
    ablation_df = pd.DataFrame(all_ablation)
    pca_df.to_csv(outdir / "pca_eigen_spectra.csv", index=False)
    kl_df.to_csv(outdir / "latent_dim_mean_kl.csv", index=False)
    ablation_df.to_csv(outdir / "simulation_ablation_metrics.csv", index=False)

    summary_cols = [
        "norm_reward",
        "best_path_accuracy",
        "observations_before_stop",
        "unique_nodes",
        "all_nodes",
        "all_paths_touched",
        "terminal_entropy",
    ]
    print("\nEffective dims:")
    if len(pca_df):
        print(
            pca_df.groupby(["condition", "representation", "seed"])[["dims_95", "dims_99"]]
            .first()
            .groupby(["condition", "representation"])
            .mean()
            .round(2)
            .to_string()
        )
    print("\nAblation means:")
    print(
        ablation_df.groupby(["condition", "ablation_type", "keep"])[summary_cols]
        .mean()
        .round(4)
        .to_string()
    )
    print(f"\nSaved CSVs to {outdir}")


if __name__ == "__main__":
    main()
