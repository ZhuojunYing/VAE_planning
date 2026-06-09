#!/usr/bin/env python
"""Precompute small exact-policy diagnostic summaries for R plotting.

The R diagnostics can be very slow for tasks such as disjoint3x2 because they
enumerate 8^6 reward assignments for each exact time cost. This script performs
that exact-policy rollout once in Python and saves compact CSV summaries that
plot_time_cost_policy_diagnostics.R can read directly.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from itertools import product, permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_SPECS = {
    "default2": ((0,), (1,)),
    "bandit3": ((0,), (1,), (2,)),
    "bandit4": ((0,), (1,), (2,), (3,)),
    "disjoint2x2": ((0, 1), (2, 3)),
    "disjoint3x2": ((0, 1), (2, 3), (4, 5)),
}


def parse_float_list(value: str) -> List[float]:
    if not value or value.strip().lower() in {"auto", "all"}:
        return []
    return [float(piece) for piece in value.split(",") if piece.strip()]


def fmt_num(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:g}"


def canonical_path_state(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(sorted(float(value) for value in values))


def canonical_state(path_states: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    clean = tuple(canonical_path_state(path_state) for path_state in path_states)
    return tuple(sorted(clean, key=lambda item: (len(item), item)))


def state_label(path_states: Sequence[Sequence[float]]) -> str:
    parts = []
    for path_state in canonical_state(path_states):
        parts.append("[" + ",".join(fmt_num(value) for value in path_state) + "]")
    return ";".join(parts)


def parse_path_state(value) -> Tuple[float, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    text = str(value).strip()
    if not text:
        return tuple()
    clean = text.strip("[]")
    if not clean:
        return tuple()
    return tuple(float(piece) for piece in clean.split(",") if piece.strip())


def same_path_state(a: Sequence[float], b: Sequence[float]) -> bool:
    return canonical_path_state(a) == canonical_path_state(b)


def normalize_cost(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-8 else value


def cost_key(value: float) -> str:
    return f"{normalize_cost(value):.12g}"


def task_file(exact_dir: str, task: str, suffix: str) -> str:
    return os.path.join(
        exact_dir,
        ".breakpoint_task_runs",
        f"exact_time_cost_{task}_{suffix}.csv",
    )


def main_file(exact_dir: str, suffix: str) -> str:
    return os.path.join(exact_dir, f"exact_time_cost_{suffix}.csv")


def zero_file(zero_exact_dir: str, suffix: str) -> str:
    return os.path.join(zero_exact_dir, f"exact_time_cost_{suffix}.csv")


def read_filtered_csv(path: str, task: str, costs: Sequence[float], keep_cols: Sequence[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(keep_cols))
    requested = np.array([normalize_cost(cost) for cost in costs], dtype=float)
    chunks = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in keep_cols, chunksize=200_000):
        if "task" in chunk.columns:
            chunk = chunk[chunk["task"] == task]
        if len(requested) and "time_cost" in chunk.columns:
            times = pd.to_numeric(chunk["time_cost"], errors="coerce").to_numpy(dtype=float)
            keep = np.zeros(len(chunk), dtype=bool)
            for cost in requested:
                keep |= np.abs(times - cost) < 1e-8
            chunk = chunk[keep]
        if len(chunk):
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=list(keep_cols))
    return pd.concat(chunks, ignore_index=True).drop_duplicates()


def load_exact_tables(args, task: str, requested_costs: Sequence[float]):
    action_cols = [
        "task", "time_cost", "reward_values", "state_label", "action_kind",
        "observe_path_state", "q_value",
    ]
    state_cols = ["task", "time_cost", "state_label", "observed_count"]
    occupancy_cols = [
        "task", "time_cost", "reward_values", "decision_timestep",
        "state_label", "state_mass", "observed_count", "p_stop", "p_continue",
    ]

    nonzero_costs = [cost for cost in requested_costs if abs(cost) >= 1e-8]
    action_frames = []
    state_frames = []
    occupancy_frames = []
    if nonzero_costs:
        action_path = task_file(args.exact_dir, task, "actions")
        if not os.path.exists(action_path):
            action_path = main_file(args.exact_dir, "actions")
        state_path = task_file(args.exact_dir, task, "states")
        if not os.path.exists(state_path):
            state_path = main_file(args.exact_dir, "states")
        occupancy_path = task_file(args.exact_dir, task, "occupancy")
        if not os.path.exists(occupancy_path):
            occupancy_path = main_file(args.exact_dir, "occupancy")
        action_frames.append(read_filtered_csv(action_path, task, nonzero_costs, action_cols))
        state_frames.append(read_filtered_csv(state_path, task, nonzero_costs, state_cols))
        occupancy_frames.append(read_filtered_csv(occupancy_path, task, nonzero_costs, occupancy_cols))

    if any(abs(cost) < 1e-8 for cost in requested_costs):
        action_frames.append(read_filtered_csv(zero_file(args.zero_exact_dir, "actions"), task, [0.0], action_cols))
        state_frames.append(read_filtered_csv(zero_file(args.zero_exact_dir, "states"), task, [0.0], state_cols))
        occupancy_frames.append(read_filtered_csv(zero_file(args.zero_exact_dir, "occupancy"), task, [0.0], occupancy_cols))

    actions = pd.concat(action_frames, ignore_index=True).drop_duplicates() if action_frames else pd.DataFrame(columns=action_cols)
    states = pd.concat(state_frames, ignore_index=True).drop_duplicates() if state_frames else pd.DataFrame(columns=state_cols)
    occupancy = pd.concat(occupancy_frames, ignore_index=True).drop_duplicates() if occupancy_frames else pd.DataFrame(columns=occupancy_cols)
    actions["time_cost"] = pd.to_numeric(actions["time_cost"], errors="coerce").map(normalize_cost)
    states["time_cost"] = pd.to_numeric(states["time_cost"], errors="coerce").map(normalize_cost)
    occupancy["time_cost"] = pd.to_numeric(occupancy["time_cost"], errors="coerce").map(normalize_cost)
    actions["q_value"] = pd.to_numeric(actions["q_value"], errors="coerce")
    states["observed_count"] = pd.to_numeric(states["observed_count"], errors="coerce").astype("Int64")
    occupancy["decision_timestep"] = pd.to_numeric(occupancy["decision_timestep"], errors="coerce").astype("Int64")
    occupancy["observed_count"] = pd.to_numeric(occupancy["observed_count"], errors="coerce")
    occupancy["state_mass"] = pd.to_numeric(occupancy["state_mass"], errors="coerce")
    occupancy["p_stop"] = pd.to_numeric(occupancy["p_stop"], errors="coerce")
    occupancy["p_continue"] = pd.to_numeric(occupancy["p_continue"], errors="coerce")
    return actions, states, occupancy


def stop_prefer_action(rows: pd.DataFrame, tol: float = 1e-10) -> Optional[pd.Series]:
    if rows.empty:
        return None
    q_values = pd.to_numeric(rows["q_value"], errors="coerce")
    if q_values.isna().all():
        return None
    best_q = q_values.max()
    best = rows[np.abs(q_values - best_q) <= tol]
    stop_rows = best[best["action_kind"].astype(str) == "stop"]
    if len(stop_rows):
        return stop_rows.iloc[0]
    observe_rows = best[best["action_kind"].astype(str) == "observe"]
    if len(observe_rows):
        return observe_rows.iloc[0]
    return None


def p_continue_stop_prefer(rows: pd.DataFrame, tol: float = 1e-10) -> float:
    action = stop_prefer_action(rows, tol=tol)
    if action is None:
        return np.nan
    return 0.0 if str(action["action_kind"]) == "stop" else 1.0


def reward_values_from_actions(actions: pd.DataFrame) -> List[float]:
    if actions.empty or "reward_values" not in actions:
        return []
    text = str(actions["reward_values"].dropna().iloc[0])
    return [float(piece) for piece in text.split(",") if piece.strip()]


def expected_best_path_reward(task: str, reward_values: Sequence[float]) -> float:
    paths = TASK_SPECS[task]
    node_count = max(max(path) for path in paths) + 1
    total = 0.0
    count = 0
    for rewards in product(reward_values, repeat=node_count):
        path_rewards = [sum(rewards[node] for node in path) for path in paths]
        total += max(path_rewards)
        count += 1
    return total / count


def normalized_reward(chosen: float, reward_norm: float) -> float:
    if not np.isfinite(chosen) or not np.isfinite(reward_norm) or abs(reward_norm) < 1e-12:
        return np.nan
    return float(chosen) / float(reward_norm)


def build_actions_by_cost_state(actions: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    out = {}
    for (time_cost, label), piece in actions.groupby(["time_cost", "state_label"], dropna=False):
        out[(cost_key(time_cost), str(label))] = piece
    return out


def simulate_trials_for_cost(
    task: str,
    time_cost: float,
    actions_by_key: Dict[Tuple[str, str], pd.DataFrame],
    reward_values: Sequence[float],
    reward_norm: float,
) -> pd.DataFrame:
    paths = TASK_SPECS[task]
    path_count = len(paths)
    node_count = max(node for path in paths for node in path) + 1
    rows = []
    key_cost = cost_key(time_cost)

    for rewards_tuple in product(reward_values, repeat=node_count):
        rewards = np.asarray(rewards_tuple, dtype=float)
        observed_values = [[] for _ in range(path_count)]
        observed_nodes = [set() for _ in range(path_count)]
        observations = 0

        while True:
            label = state_label(observed_values)
            state_actions = actions_by_key.get((key_cost, label))
            if state_actions is None or state_actions.empty:
                break
            action = stop_prefer_action(state_actions)
            if action is None or str(action["action_kind"]) == "stop" or observations >= node_count:
                break
            target = parse_path_state(action.get("observe_path_state", ""))
            selected_path = None
            for path_i, path in enumerate(paths):
                if len(observed_nodes[path_i]) < len(path) and same_path_state(observed_values[path_i], target):
                    selected_path = path_i
                    break
            if selected_path is None:
                break
            candidate_nodes = [node for node in paths[selected_path] if node not in observed_nodes[selected_path]]
            if not candidate_nodes:
                break
            selected_node = candidate_nodes[0]
            observed_nodes[selected_path].add(selected_node)
            observed_values[selected_path].append(float(rewards[selected_node]))
            observations += 1

        path_rewards = np.asarray([
            np.sum(rewards[list(path)])
            for path in paths
        ], dtype=float)
        prior_mean = float(np.mean(reward_values))
        posterior_values = np.asarray([
            sum(observed_values[path_i]) + (len(paths[path_i]) - len(observed_nodes[path_i])) * prior_mean
            for path_i in range(path_count)
        ], dtype=float)
        chosen_path = int(np.argmax(posterior_values))
        chosen_reward = float(path_rewards[chosen_path])
        reward_bits = float(np.log2(len(reward_values)))
        reward_nats = float(np.log(len(reward_values)))
        cumulative_observation_count = observations * (observations + 1) / 2
        rows.append({
            "time_cost": normalize_cost(time_cost),
            "kl_paid_total": 0.0,
            "chosen_path_reward": chosen_reward,
            "normalized_chosen_path_reward": normalized_reward(chosen_reward, reward_norm),
            "observations_before_stop": observations,
            "final_raw_reward_information_bits": observations * reward_bits,
            "final_raw_reward_information_nats": observations * reward_nats,
            "cumulative_raw_reward_information_bits": cumulative_observation_count * reward_bits,
            "cumulative_raw_reward_information_nats": cumulative_observation_count * reward_nats,
        })
    return pd.DataFrame(rows)


def build_trial_summary(task: str, actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    reward_values = reward_values_from_actions(actions)
    if not reward_values:
        return pd.DataFrame()
    actions_by_key = build_actions_by_cost_state(actions)
    reward_norm = expected_best_path_reward(task, reward_values)
    trial_frames = []
    for time_cost in sorted(actions["time_cost"].dropna().unique()):
        trial_frames.append(simulate_trials_for_cost(task, time_cost, actions_by_key, reward_values, reward_norm))
    trials = pd.concat(trial_frames, ignore_index=True)
    summary = trials.groupby("time_cost", as_index=False).agg(
        kl_paid_total=("kl_paid_total", "mean"),
        chosen_path_reward=("chosen_path_reward", "mean"),
        normalized_chosen_path_reward=("normalized_chosen_path_reward", "mean"),
        observations_before_stop=("observations_before_stop", "mean"),
        var_observations_before_stop=("observations_before_stop", lambda x: float(np.var(x, ddof=0))),
        final_raw_reward_information_bits=("final_raw_reward_information_bits", "mean"),
        final_raw_reward_information_nats=("final_raw_reward_information_nats", "mean"),
        cumulative_raw_reward_information_bits=("cumulative_raw_reward_information_bits", "mean"),
        cumulative_raw_reward_information_nats=("cumulative_raw_reward_information_nats", "mean"),
        n=("kl_paid_total", "size"),
    )
    return summary


def build_information_by_timestep_summary(actions: pd.DataFrame, occupancy: pd.DataFrame) -> pd.DataFrame:
    if occupancy.empty:
        return pd.DataFrame()
    reward_values = reward_values_from_actions(actions)
    n_reward_values = len(reward_values)
    if n_reward_values <= 0:
        return pd.DataFrame()
    reward_bits = float(np.log2(n_reward_values))
    reward_nats = float(np.log(n_reward_values))
    rows = []
    for (time_cost, decision_timestep), piece in occupancy.groupby(["time_cost", "decision_timestep"], dropna=False):
        masses = pd.to_numeric(piece["state_mass"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(masses) & (masses > 0)
        if not np.any(valid):
            continue
        masses = masses[valid]
        piece_valid = piece.loc[valid].copy()
        active_mass = float(np.sum(masses))
        if active_mass <= 0:
            continue
        probs = masses / active_mass
        state_surprisal_bits = -np.log2(probs)
        state_entropy_bits = float(np.sum(probs * state_surprisal_bits))
        observed_count = pd.to_numeric(piece_valid["observed_count"], errors="coerce").to_numpy(dtype=float)
        expected_observed_count = float(np.sum(probs * observed_count))
        p_stop_values = pd.to_numeric(piece_valid["p_stop"], errors="coerce").to_numpy(dtype=float)
        p_continue_values = pd.to_numeric(piece_valid["p_continue"], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "time_cost": normalize_cost(time_cost),
            "decision_timestep": int(decision_timestep),
            "active_mass": active_mass,
            "n_active_states": int(len(masses)),
            "expected_observed_count": expected_observed_count,
            "state_entropy_bits": state_entropy_bits,
            "state_entropy_nats": float(state_entropy_bits * np.log(2.0)),
            "expected_state_surprisal_bits": state_entropy_bits,
            "expected_state_surprisal_nats": float(state_entropy_bits * np.log(2.0)),
            "raw_reward_information_bits": expected_observed_count * reward_bits,
            "raw_reward_information_nats": expected_observed_count * reward_nats,
            "bits_per_observed_reward": reward_bits,
            "nats_per_observed_reward": reward_nats,
            "n_reward_values": n_reward_values,
            "p_stop": float(np.sum(probs * p_stop_values)),
            "p_continue": float(np.sum(probs * p_continue_values)),
        })
    return pd.DataFrame(rows)


def parse_state_label(label: str) -> List[Tuple[float, ...]]:
    parts = str(label).split(";")
    return [parse_path_state(part) for part in parts]


def build_best_continue_summary(task: str, actions: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    paths = TASK_SPECS[task]
    node_count = max(node for path in paths for node in path) + 1
    actions_by_key = build_actions_by_cost_state(actions)
    rows = []
    for row in states.itertuples(index=False):
        observed_count = int(row.observed_count) if not pd.isna(row.observed_count) else -1
        if observed_count <= 0 or observed_count >= node_count:
            continue
        state = parse_state_label(row.state_label)
        path_values = np.asarray([sum(path_state) for path_state in state], dtype=float)
        path_counts = np.asarray([len(path_state) for path_state in state], dtype=int)
        observed_values = path_values[path_counts > 0]
        if observed_values.size == 0:
            continue
        best_path_value = float(np.max(observed_values))
        best_indices = np.where((path_counts > 0) & np.isclose(path_values, best_path_value))[0]
        best_path_complete = any(path_counts[idx] >= len(paths[idx]) for idx in best_indices)
        state_actions = actions_by_key.get((cost_key(row.time_cost), str(row.state_label)))
        if state_actions is None:
            continue
        rows.append({
            "time_cost": normalize_cost(row.time_cost),
            "decision_timestep": observed_count + 1,
            "best_path_value": best_path_value,
            "best_path_complete": "complete" if best_path_complete else "incomplete",
            "p_continue": p_continue_stop_prefer(state_actions),
            "mass": 1.0,
        })
    if not rows:
        return pd.DataFrame()
    dat = pd.DataFrame(rows)
    group_cols = ["time_cost", "decision_timestep", "best_path_value"]
    if task == "disjoint3x2":
        group_cols.append("best_path_complete")
    summary = dat.groupby(group_cols, as_index=False).agg(
        p_continue=("p_continue", "mean"),
        n=("mass", "sum"),
    )
    return summary


def ordered_node_sequences(nodes: Sequence[int], k: int):
    if k == 0:
        yield tuple()
        return
    for seq in permutations(nodes, k):
        yield seq


def build_difference_continue_summary(task: str, actions: pd.DataFrame) -> pd.DataFrame:
    paths = TASK_SPECS[task]
    if len(paths) != 2:
        return pd.DataFrame()
    reward_values = reward_values_from_actions(actions)
    actions_by_key = build_actions_by_cost_state(actions)
    node_to_path = {}
    for path_i, path in enumerate(paths):
        for node in path:
            node_to_path[node] = path_i
    node_count = max(node_to_path) + 1
    rows = []
    for time_cost in sorted(actions["time_cost"].dropna().unique()):
        key_cost = cost_key(time_cost)
        for decision_timestep in range(2, node_count + 1):
            observed_count = decision_timestep - 1
            for node_sequence in ordered_node_sequences(range(node_count), observed_count):
                for reward_tuple in product(reward_values, repeat=observed_count):
                    observed_values = [[] for _ in paths]
                    for node, reward in zip(node_sequence, reward_tuple):
                        observed_values[node_to_path[node]].append(float(reward))
                    label = state_label(observed_values)
                    state_actions = actions_by_key.get((key_cost, label))
                    if state_actions is None:
                        continue
                    current_path = node_to_path[node_sequence[-1]]
                    other_path = 1 - current_path
                    rows.append({
                        "time_cost": normalize_cost(time_cost),
                        "decision_timestep": decision_timestep,
                        "path_value_difference": sum(observed_values[current_path]) - sum(observed_values[other_path]),
                        "p_continue": p_continue_stop_prefer(state_actions),
                        "mass": 1.0,
                    })
    if not rows:
        return pd.DataFrame()
    dat = pd.DataFrame(rows)
    return dat.groupby(["time_cost", "decision_timestep", "path_value_difference"], as_index=False).agg(
        p_continue=("p_continue", "mean"),
        n=("mass", "sum"),
    )


def write_csv(path: str, dat: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dat.to_csv(path, index=False)
    print(f"Saved {path} ({len(dat)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--time-costs", default="all")
    parser.add_argument("--exact-dir", default="analyses/exp_binary/results/exact_time_cost")
    parser.add_argument("--zero-exact-dir", default="analyses/exp_binary/results/exact_time_cost_zero")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    task = args.task
    requested_costs = parse_float_list(args.time_costs)
    if requested_costs:
        requested_costs = sorted(set([0.0] + [normalize_cost(cost) for cost in requested_costs]))
    else:
        # Read all task-specific nonzero costs plus zero if available.
        summary_path = task_file(args.exact_dir, task, "summary")
        costs = []
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path, usecols=["time_cost"])
            costs.extend(pd.to_numeric(summary["time_cost"], errors="coerce").dropna().tolist())
        costs.append(0.0)
        requested_costs = sorted(set(normalize_cost(cost) for cost in costs))

    actions, states, occupancy = load_exact_tables(args, task, requested_costs)
    if actions.empty:
        raise SystemExit(f"No exact action rows found for task={task}, costs={requested_costs}")

    output_dir = args.output_dir or os.path.join(args.exact_dir, ".policy_diagnostics")
    prefix = args.output_prefix or f"exact_time_cost_{task}"
    trial_summary = build_trial_summary(task, actions)
    write_csv(os.path.join(output_dir, f"{prefix}_trial_summary.csv"), trial_summary)

    information_summary = build_information_by_timestep_summary(actions, occupancy)
    write_csv(os.path.join(output_dir, f"{prefix}_information_by_timestep_summary.csv"), information_summary)

    if task in {"bandit3", "bandit4", "disjoint3x2"}:
        continue_summary = build_best_continue_summary(task, actions, states)
        write_csv(os.path.join(output_dir, f"{prefix}_continue_best_summary.csv"), continue_summary)
    elif task in {"default2", "disjoint2x2"}:
        continue_summary = build_difference_continue_summary(task, actions)
        write_csv(os.path.join(output_dir, f"{prefix}_continue_difference_summary.csv"), continue_summary)


if __name__ == "__main__":
    main()
