"""Utilities for latent posterior angle analyses of trained planning VRNNs."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_UNIFORM_REWARDS = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float32)
BINARY_REWARDS = np.array([0, 1], dtype=np.float32)


def normalize_tree_type(raw_tree_type: str, tree_size: int) -> str:
    key = str(raw_tree_type).strip().lower()
    aliases = {
        "": "legacy",
        "auto": "legacy",
        "default": "legacy",
        "legacy": "legacy",
        "3armed": "bandit3",
        "3_arm": "bandit3",
        "3_armed": "bandit3",
        "3-armed": "bandit3",
        "3armedbandit": "bandit3",
        "3_arm_bandit": "bandit3",
        "3-armed-bandit": "bandit3",
        "three_arm_bandit": "bandit3",
        "bandit3": "bandit3",
        "3bandit": "bandit3",
        "4armed": "bandit4",
        "4_arm": "bandit4",
        "4_armed": "bandit4",
        "4-armed": "bandit4",
        "4armedbandit": "bandit4",
        "4_arm_bandit": "bandit4",
        "4-armed-bandit": "bandit4",
        "four_arm_bandit": "bandit4",
        "bandit4": "bandit4",
        "4bandit": "bandit4",
        "2x2": "disjoint2x2",
        "2x2_disjoint": "disjoint2x2",
        "disjoint2x2": "disjoint2x2",
        "disjoint_2x2": "disjoint2x2",
        "3x2": "disjoint3x2",
        "3x2_disjoint": "disjoint3x2",
        "disjoint3x2": "disjoint3x2",
        "disjoint_3x2": "disjoint3x2",
    }
    if key == "bandit":
        if tree_size in (3, 4):
            return f"bandit{tree_size}"
        raise ValueError("tree_type='bandit' requires tree_size 3 or 4.")
    if key not in aliases:
        raise ValueError(f"Unknown tree_type={raw_tree_type!r}.")
    normalized = aliases[key]
    if normalized == "legacy" and tree_size == 3:
        return "bandit3"
    return normalized


def build_bandit_tree(num_arms: int) -> Dict[str, Dict[str, List[str]]]:
    tree = {"0": {}}
    for node in range(1, num_arms + 1):
        tree["0"][f"arm{node}"] = [-1, str(node)]
        tree[str(node)] = {}
    return tree


def build_disjoint_path_tree(num_paths: int, nodes_per_path: int) -> Dict[str, Dict[str, List[str]]]:
    tree = {"0": {}}
    node = 1
    for path_idx in range(1, num_paths + 1):
        first_node = str(node)
        tree["0"][f"path{path_idx}"] = [-1, first_node]
        for depth in range(nodes_per_path):
            current_node = str(node)
            if depth == nodes_per_path - 1:
                tree[current_node] = {}
            else:
                next_node = str(node + 1)
                tree[current_node] = {f"path{path_idx}_next{depth + 1}": [-1, next_node]}
            node += 1
    return tree


def decision_tree_for(tree_size: int, tree_type: str) -> Tuple[str, Dict]:
    tree_type = normalize_tree_type(tree_type, tree_size)
    if tree_type == "bandit3":
        if tree_size != 3:
            raise ValueError("tree_type='bandit3' requires tree_size=3.")
        return "bandit", build_bandit_tree(3)
    if tree_type == "bandit4":
        if tree_size != 4:
            raise ValueError("tree_type='bandit4' requires tree_size=4.")
        return "bandit", build_bandit_tree(4)
    if tree_type == "disjoint2x2":
        if tree_size != 4:
            raise ValueError("tree_type='disjoint2x2' requires tree_size=4.")
        return "disjoint", build_disjoint_path_tree(2, 2)
    if tree_type == "disjoint3x2":
        if tree_size != 6:
            raise ValueError("tree_type='disjoint3x2' requires tree_size=6.")
        return "disjoint", build_disjoint_path_tree(3, 2)
    if tree_type == "legacy" and tree_size == 2:
        return "legacy", {
            "0": {"right": [-1, "1"], "up": [-1, "2"]},
            "1": {},
            "2": {},
        }
    if tree_type == "legacy" and tree_size == 6:
        return "legacy", {
            "0": {"right": [-1, "1"], "up": [-1, "4"]},
            "1": {"right": [-1, "2"], "up": [-1, "3"]},
            "2": {},
            "3": {},
            "4": {"right": [-1, "5"], "up": [-1, "6"]},
            "5": {},
            "6": {},
        }
    raise ValueError(f"Unsupported tree_size={tree_size}, tree_type={tree_type!r}.")


def reward_norm_value(tree_size: int, input_type: str, index_path_map: Dict[int, Sequence[int]]) -> float:
    rewards = BINARY_REWARDS if input_type == "binary" else DEFAULT_UNIFORM_REWARDS
    paths = list(index_path_map.values())
    if not paths:
        return 1.0
    expected_max = 0.0
    for values in itertools.product(rewards, repeat=tree_size):
        node_rewards = np.asarray(values, dtype=float)
        path_values = [float(np.sum(node_rewards[np.asarray(path, dtype=int) - 1])) for path in paths]
        expected_max += max(path_values)
    return expected_max / float(len(rewards) ** tree_size)


def path_map_from_index_path_map(index_path_map: Dict[int, Sequence[int]], time_steps: int) -> np.ndarray:
    rows = []
    for node_indices in index_path_map.values():
        row = np.zeros(time_steps, dtype=np.float32)
        for node in node_indices:
            idx = int(node) - 1
            if 0 <= idx < time_steps:
                row[idx] = 1.0
        rows.append(row)
    return np.stack(rows, axis=0)


def path_covariance(path_map: np.ndarray) -> np.ndarray:
    num_paths, time_steps = path_map.shape
    cov = np.zeros((num_paths, num_paths, time_steps), dtype=np.float32)
    for i in range(num_paths):
        for j in range(num_paths):
            cov[i, j, :] = path_map[i, :] * path_map[j, :]
    return cov


def make_model_config(
    helper_module,
    *,
    tree_size: int,
    tree_type: str,
    input_type: str,
    seed: int,
    rnn_dim: int,
    latent_dim: int,
    expansion_decision_version: str,
    model_variant: str,
    checkpoint_root: str,
) -> SimpleNamespace:
    normalized_tree_type = normalize_tree_type(tree_type, tree_size)
    tree_label, decision_tree = decision_tree_for(tree_size, normalized_tree_type)
    (
        path_names,
        path_leaf_dict,
        sibling_map,
        node_path_map,
        node_path_name,
        path_indices,
        node_indices,
        est_best_path_map,
        path_node_map,
    ) = helper_module.analyze_tree_paths(decision_tree)
    index_path_map = {path_indices[i]: node_indices[i] for i in range(len(path_indices))}
    path_map_np = path_map_from_index_path_map(index_path_map, tree_size)
    return SimpleNamespace(
        tree_size=tree_size,
        time_steps=tree_size,
        tree_type=tree_label,
        tree_name_suffix="" if normalized_tree_type == "legacy" else f"_{normalized_tree_type}",
        input_type=input_type,
        seed=seed,
        rnn_units=int(rnn_dim),
        latent_dim=int(latent_dim),
        output_dim=int(rnn_dim),
        num_paths=len(path_names),
        index_path_map=index_path_map,
        path_map=path_map_np,
        path_map_np=path_map_np,
        path_cov_mat=path_covariance(path_map_np),
        reward_norm_value=reward_norm_value(tree_size, input_type, index_path_map),
        expansion_decision_version=expansion_decision_version,
        model_variant=model_variant,
        dir_name=checkpoint_root,
        sim_dir_name=checkpoint_root.replace("model", "simulation"),
    )


def float_token(value: float) -> str:
    value = float(value)
    return f"{value:.12g}"


def tree_file_label(tree_size: int, tree_type: str) -> str:
    normalized = normalize_tree_type(tree_type, tree_size)
    suffix = "" if normalized == "legacy" else f"_{normalized}"
    return f"{tree_size}n{suffix}"


def candidate_filename_tokens(
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
) -> List[str]:
    tree_label = tree_file_label(tree_size, tree_type)
    return [
        f"lambda_{float_token(lambda_value)}",
        f"alpha_{float_token(alpha)}",
        f"beta_{float_token(beta)}",
        f"seed_{int(seed)}",
        tree_label,
        f"rnn_{int(rnn_dim)}_latent_{int(latent_dim)}",
        f"opportunity_{float_token(opportunity_cost)}",
        f"expansion_{expansion_decision_version}",
        f"variant_{model_variant}",
    ]


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


@dataclass
class CheckpointMatch:
    path: Path
    score: int
    mtime: float
    reasons: List[str]


def score_checkpoint(
    path: Path,
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
) -> Optional[CheckpointMatch]:
    name = path.name
    if not name.endswith(".weights.h5"):
        return None
    lamb = _extract_float(r"lambda_([0-9eE.+-]+)", name)
    alp = _extract_float(r"alpha_([0-9eE.+-]+)", name)
    bet = _extract_float(r"beta_([0-9eE.+-]+)", name)
    found_seed = _extract_int(r"seed_([0-9]+)", name)
    if not (_close(lamb, lambda_value) and _close(alp, alpha) and _close(bet, beta) and found_seed == seed):
        return None

    score = 100
    reasons = ["lambda/alpha/beta/seed"]
    expected_tree = tree_file_label(tree_size, tree_type)
    if expected_tree in name:
        score += 20
        reasons.append("tree")
    else:
        return None

    found_rnn = _extract_int(r"rnn_([0-9]+)", name)
    found_latent = _extract_int(r"latent_([0-9]+)", name)
    if found_rnn is None and found_latent is None:
        if int(rnn_dim) == 64 and int(latent_dim) == 32:
            score += 2
            reasons.append("legacy-no-architecture")
        else:
            return None
    elif found_rnn == int(rnn_dim) and found_latent == int(latent_dim):
        score += 30
        reasons.append("architecture")
    else:
        return None

    opp = _extract_float(r"opportunity_([0-9eE.+-]+)", name)
    if opp is None:
        if abs(float(opportunity_cost)) < 1e-8:
            score += 1
            reasons.append("legacy-no-opportunity")
        else:
            return None
    elif _close(opp, opportunity_cost):
        score += 15
        reasons.append("opportunity")
    else:
        return None

    if f"expansion_{expansion_decision_version}" in name:
        score += 8
        reasons.append("expansion")
    if f"variant_{model_variant}" in name or (model_variant == "vae" and "variant_" not in name):
        score += 8
        reasons.append("variant")
    return CheckpointMatch(path=path, score=score, mtime=path.stat().st_mtime, reasons=reasons)


def find_checkpoint(
    checkpoint_root: str,
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
    root = Path(checkpoint_root)
    matches = []
    for path in root.rglob("*.weights.h5"):
        match = score_checkpoint(
            path,
            lambda_value=lambda_value,
            alpha=alpha,
            beta=beta,
            seed=seed,
            tree_size=tree_size,
            tree_type=tree_type,
            rnn_dim=rnn_dim,
            latent_dim=latent_dim,
            opportunity_cost=opportunity_cost,
            expansion_decision_version=expansion_decision_version,
            model_variant=model_variant,
        )
        if match is not None:
            matches.append(match)
    if not matches:
        return None, ["no matching .weights.h5 file found"]
    matches.sort(key=lambda item: (item.score, item.mtime), reverse=True)
    top_score = matches[0].score
    top = [item for item in matches if item.score == top_score]
    if len(top) > 1:
        top_sorted = sorted(top, key=lambda item: item.mtime, reverse=True)
        if len(top_sorted) > 1 and abs(top_sorted[0].mtime - top_sorted[1].mtime) < 1e-6:
            details = [str(item.path) for item in top_sorted]
            return None, ["ambiguous checkpoint matches with identical score/mtime", *details]
        return top_sorted[0].path, [
            "multiple matches; chose most recent top-scoring checkpoint",
            *[str(item.path) for item in top_sorted],
        ]
    return matches[0].path, [f"matched score={matches[0].score} reasons={','.join(matches[0].reasons)}"]


def sample_rewards(n_trials: int, time_steps: int, input_type: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = BINARY_REWARDS if input_type == "binary" else DEFAULT_UNIFORM_REWARDS
    return rng.choice(values, size=(n_trials, time_steps)).astype(np.float32)


def compute_path_values(rewards: np.ndarray, index_path_map: Dict[int, Sequence[int]]) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    path_values = []
    for node_indices in index_path_map.values():
        idx = np.asarray(node_indices, dtype=int) - 1
        path_values.append(np.sum(rewards[:, idx], axis=1))
    return np.stack(path_values, axis=1)


def compute_current_best_path_variables(
    rewards: np.ndarray,
    observed_masks: np.ndarray,
    index_path_map: Dict[int, Sequence[int]],
) -> Dict[str, np.ndarray]:
    rewards = np.asarray(rewards, dtype=float)
    observed_masks = np.asarray(observed_masks, dtype=bool)
    n_trials, n_steps, n_nodes = observed_masks.shape
    paths = [np.asarray(nodes, dtype=int) - 1 for nodes in index_path_map.values()]
    n_paths = len(paths)
    current_path_values = np.zeros((n_trials, n_steps, n_paths), dtype=float)
    for p_idx, node_idx in enumerate(paths):
        observed_rewards = rewards[:, None, node_idx] * observed_masks[:, :, node_idx]
        current_path_values[:, :, p_idx] = np.sum(observed_rewards, axis=2)

    sorted_values = np.sort(current_path_values, axis=2)
    best_values = sorted_values[:, :, -1]
    second_best = sorted_values[:, :, -2] if n_paths > 1 else np.full_like(best_values, np.nan)
    margins = best_values - second_best
    tie_flags = np.sum(np.isclose(current_path_values, best_values[:, :, None]), axis=2) > 1
    best_paths = np.argmax(current_path_values, axis=2).astype(float)
    best_paths[tie_flags] = np.nan
    switches = np.full((n_trials, n_steps), np.nan, dtype=float)
    for t in range(1, n_steps):
        valid = ~np.isnan(best_paths[:, t]) & ~np.isnan(best_paths[:, t - 1])
        switches[valid, t] = best_paths[valid, t] != best_paths[valid, t - 1]
    return {
        "current_path_values": current_path_values,
        "current_best_path": best_paths,
        "current_best_path_value": best_values,
        "current_best_path_margin": margins,
        "current_best_path_switch": switches,
        "tie_flag": tie_flags,
    }


def add_angle_features(df, latent_dim: int):
    z_mu_cols = [f"z_mu_{i}" for i in range(latent_dim)]
    z_logvar_cols = [f"z_logvar_{i}" for i in range(latent_dim)]
    z_sigma_cols = [f"z_sigma_{i}" for i in range(latent_dim)]
    for i in range(latent_dim):
        df[z_sigma_cols[i]] = np.exp(0.5 * df[z_logvar_cols[i]])
    if latent_dim != 2:
        return df

    mu0 = df["z_mu_0"].to_numpy(dtype=float)
    mu1 = df["z_mu_1"].to_numpy(dtype=float)
    df["angle_mu"] = np.arctan2(mu1, mu0)
    df["radius_mu"] = np.sqrt(mu0**2 + mu1**2)
    df["sin_angle_mu"] = np.sin(df["angle_mu"])
    df["cos_angle_mu"] = np.cos(df["angle_mu"])
    df["sigma_mean"] = df[["z_sigma_0", "z_sigma_1"]].mean(axis=1)
    df["sigma_product"] = df["z_sigma_0"] * df["z_sigma_1"]
    df["sigma_ratio"] = df["z_sigma_0"] / (df["z_sigma_1"] + 1e-8)
    df["logvar_mean"] = df[["z_logvar_0", "z_logvar_1"]].mean(axis=1)

    if "prior_mu_0" in df and "prior_mu_1" in df:
        p0 = df["prior_mu_0"].to_numpy(dtype=float)
        p1 = df["prior_mu_1"].to_numpy(dtype=float)
        df["prior_angle_mu"] = np.arctan2(p1, p0)
        df["prior_radius_mu"] = np.sqrt(p0**2 + p1**2)
        d0 = mu0 - p0
        d1 = mu1 - p1
        df["delta_mu_0"] = d0
        df["delta_mu_1"] = d1
        df["delta_angle_mu"] = np.arctan2(d1, d0)
        df["delta_radius_mu"] = np.sqrt(d0**2 + d1**2)
        df["sin_delta_angle_mu"] = np.sin(df["delta_angle_mu"])
        df["cos_delta_angle_mu"] = np.cos(df["delta_angle_mu"])

    for dim in range(2):
        x = df[f"z_mu_{dim}"].to_numpy(dtype=float) / math.sqrt(2.0)
        y = df[f"z_sigma_{dim}"].to_numpy(dtype=float)
        z = x + 1j * y
        # Exploratory only: a diagonal 2D posterior is a product of two
        # Gaussian manifolds, not one single 2D Poincare disk.
        w = (z - 1j) / (z + 1j)
        angle = np.angle(w)
        df[f"gm_disk_x_{dim}"] = np.real(w)
        df[f"gm_disk_y_{dim}"] = np.imag(w)
        df[f"gm_disk_radius_{dim}"] = np.abs(w)
        df[f"gm_disk_angle_{dim}"] = angle
        df[f"gm_disk_sin_angle_{dim}"] = np.sin(angle)
        df[f"gm_disk_cos_angle_{dim}"] = np.cos(angle)
    return df


def safe_json_dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"))

