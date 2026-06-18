#!/usr/bin/env python3
"""Exact dynamic-programming solution for the small expansion tasks.

Assumptions:
  * rewards are sampled independently and uniformly from the requested alphabet;
  * observed rewards are remembered perfectly;
  * every observation, including the first, costs a fixed time/opportunity cost;
  * stopping commits to the path with the largest posterior expected reward.

States are canonicalized by exchangeable paths and exchangeable nodes within a
path. Shared-node tasks use explicit node-level states because path exchange
would lose information about the shared structure.
"""

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from functools import lru_cache


DEFAULT_REWARDS = (-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)
BINARY_REWARDS = (0.0, 1.0)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    num_paths: int
    path_length: int
    tree_size: int
    tree_config: str
    paths_override: tuple = None
    node_count: int = None

    @property
    def n_nodes(self):
        if self.node_count is not None:
            return self.node_count
        return self.num_paths * self.path_length

    @property
    def paths(self):
        if self.paths_override is not None:
            return self.paths_override
        paths = []
        node = 0
        for _ in range(self.num_paths):
            path_nodes = tuple(range(node, node + self.path_length))
            paths.append(path_nodes)
            node += self.path_length
        return tuple(paths)

    @property
    def has_shared_nodes(self):
        path_nodes = [node for path in self.paths for node in path]
        return len(set(path_nodes)) < len(path_nodes)


TASKS = {
    "default2": TaskSpec("default2", 2, 1, 2, ""),
    "bandit3": TaskSpec("bandit3", 3, 1, 3, "bandit3"),
    "bandit4": TaskSpec("bandit4", 4, 1, 4, "bandit4"),
    "disjoint2x2": TaskSpec("disjoint2x2", 2, 2, 4, "disjoint2x2"),
    "disjoint3x2": TaskSpec("disjoint3x2", 3, 2, 6, "disjoint3x2"),
    "legacy6": TaskSpec(
        "legacy6",
        4,
        2,
        6,
        "",
        paths_override=((0, 1), (0, 2), (3, 4), (3, 5)),
        node_count=6,
    ),
}

TASK_ALIASES = {
    "2": "default2",
    "2node": "default2",
    "legacy2": "default2",
    "default": "default2",
    "default2": "default2",
    "bandit3": "bandit3",
    "3armed": "bandit3",
    "3_arm": "bandit3",
    "3-armed": "bandit3",
    "bandit4": "bandit4",
    "4armed": "bandit4",
    "4_arm": "bandit4",
    "4-armed": "bandit4",
    "2x2": "disjoint2x2",
    "disjoint2x2": "disjoint2x2",
    "disjoint_2x2": "disjoint2x2",
    "3x2": "disjoint3x2",
    "disjoint3x2": "disjoint3x2",
    "disjoint_3x2": "disjoint3x2",
    "6": "legacy6",
    "6node": "legacy6",
    "6_node": "legacy6",
    "legacy6": "legacy6",
    "sharedmiddle6": "legacy6",
    "shared_middle6": "legacy6",
    "shared_middle_6": "legacy6",
    "shared_middle": "legacy6",
}


def parse_float_list(text):
    return tuple(float(part.strip()) for part in str(text).split(",") if part.strip())


def normalize_task_names(raw_tasks):
    if len(raw_tasks) == 1 and raw_tasks[0].strip().lower() == "all":
        return tuple(TASKS)
    names = []
    for raw_task in raw_tasks:
        key = raw_task.strip().lower()
        if key not in TASK_ALIASES:
            valid = ", ".join(sorted(TASK_ALIASES))
            raise ValueError(f"Unknown task {raw_task!r}. Valid tasks/aliases: {valid}.")
        name = TASK_ALIASES[key]
        if name not in names:
            names.append(name)
    return tuple(names)


def fmt_num(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def canonical_path_state(values):
    return tuple(sorted(values))


def canonical_state(path_states):
    clean = tuple(canonical_path_state(path_state) for path_state in path_states)
    return tuple(sorted(clean, key=lambda x: (len(x), x)))


def empty_state(task):
    return canonical_state(tuple(() for _ in range(task.num_paths)))


def replace_one_path_state(state, old_path_state, new_path_state):
    path_states = list(state)
    for i, path_state in enumerate(path_states):
        if path_state == old_path_state:
            path_states[i] = canonical_path_state(new_path_state)
            return canonical_state(path_states)
    raise ValueError("Path state to replace was not present in state.")


def state_label(state):
    pieces = []
    for path_state in state:
        pieces.append("[" + ",".join(fmt_num(value) for value in path_state) + "]")
    return ";".join(pieces)


def json_path_state(path_state):
    return json.dumps(list(path_state), separators=(",", ":"))


def reward_norm(task, rewards):
    total = 0.0
    count = 0
    for assignment in itertools.product(rewards, repeat=task.n_nodes):
        best_path = max(sum(assignment[node] for node in path) for path in task.paths)
        total += best_path
        count += 1
    return total / count


def terminal_path_values_raw(state, task, prior_mean):
    values = []
    for path_state in state:
        missing = task.path_length - len(path_state)
        values.append(sum(path_state) + missing * prior_mean)
    return tuple(values)


def observed_count(state):
    return sum(len(path_state) for path_state in state)


class ExactTimeCostSolver:
    def __init__(
        self,
        task,
        rewards,
        time_cost,
        normalize_reward=True,
        min_observations_before_stop=0,
        tie_tol=1e-10,
        tie_mode="uniform_class",
    ):
        self.task = task
        self.rewards = tuple(rewards)
        self.reward_prob = 1.0 / len(self.rewards)
        self.prior_mean = sum(self.rewards) / len(self.rewards)
        self.time_cost = float(time_cost)
        self.normalize_reward = bool(normalize_reward)
        self.reward_norm = reward_norm(task, self.rewards) if normalize_reward else 1.0
        self.min_observations_before_stop = int(min_observations_before_stop)
        self.tie_tol = float(tie_tol)
        self.tie_mode = tie_mode
        self.seen_states = set()

    def scale_reward(self, value):
        return value / self.reward_norm

    def terminal_value_raw(self, state):
        return max(terminal_path_values_raw(state, self.task, self.prior_mean))

    def terminal_value(self, state):
        return self.scale_reward(self.terminal_value_raw(state))

    def stop_is_legal(self, state):
        return observed_count(state) >= self.min_observations_before_stop

    def observe_cost(self, state):
        return self.time_cost

    def unique_observe_actions(self, state):
        seen = set()
        for path_state in state:
            if len(path_state) >= self.task.path_length or path_state in seen:
                continue
            seen.add(path_state)
            matching_paths = sum(1 for candidate in state if candidate == path_state)
            unobserved_nodes_per_path = self.task.path_length - len(path_state)
            concrete_count = matching_paths * unobserved_nodes_per_path
            yield {
                "kind": "observe",
                "path_state": path_state,
                "concrete_count": concrete_count,
            }

    @lru_cache(maxsize=None)
    def solve(self, state):
        self.seen_states.add(state)
        actions = []
        terminal_raw = self.terminal_value_raw(state)
        terminal_value = self.scale_reward(terminal_raw)

        if self.stop_is_legal(state):
            actions.append({
                "kind": "stop",
                "path_state": None,
                "q_value": terminal_value,
                "terminal_value_raw": terminal_raw,
                "concrete_count": 1,
            })

        for action in self.unique_observe_actions(state):
            expected_future_value = 0.0
            for reward in self.rewards:
                next_path_state = canonical_path_state(action["path_state"] + (reward,))
                next_state = replace_one_path_state(state, action["path_state"], next_path_state)
                expected_future_value += self.reward_prob * self.solve(next_state)["value"]
            q_value = -self.observe_cost(state) + expected_future_value
            actions.append({
                "kind": "observe",
                "path_state": action["path_state"],
                "q_value": q_value,
                "terminal_value_raw": "",
                "concrete_count": action["concrete_count"],
            })

        if not actions:
            raise RuntimeError(f"No legal actions in state {state_label(state)}")

        best_q = max(action["q_value"] for action in actions)
        for action in actions:
            action["optimal"] = abs(action["q_value"] - best_q) <= self.tie_tol
        self.add_tie_policy_probs(actions)
        return {
            "value": best_q,
            "terminal_value": terminal_value,
            "terminal_value_raw": terminal_raw,
            "actions": tuple(actions),
        }

    def add_tie_policy_probs(self, actions):
        optimal_indices = [i for i, action in enumerate(actions) if action["optimal"]]
        for action in actions:
            action["tie_policy_prob"] = 0.0
        if self.tie_mode == "first":
            actions[optimal_indices[0]]["tie_policy_prob"] = 1.0
            return
        if self.tie_mode == "uniform_concrete":
            denom = sum(actions[i]["concrete_count"] for i in optimal_indices)
            for i in optimal_indices:
                actions[i]["tie_policy_prob"] = actions[i]["concrete_count"] / denom
            return
        prob = 1.0 / len(optimal_indices)
        for i in optimal_indices:
            actions[i]["tie_policy_prob"] = prob

    def state_rows(self):
        initial = empty_state(self.task)
        self.solve(initial)
        rows = []
        for state_id, state in enumerate(sorted(self.seen_states, key=state_label)):
            result = self.solve(state)
            actions = result["actions"]
            p_stop = sum(action["tie_policy_prob"] for action in actions if action["kind"] == "stop")
            p_continue = 1.0 - p_stop
            optimal_observe_path_states = [
                list(action["path_state"])
                for action in actions
                if action["kind"] == "observe" and action["optimal"]
            ]
            if p_stop > 0.0 and p_continue > 0.0:
                best_action_kind = "tie"
            elif p_stop > 0.0:
                best_action_kind = "stop"
            else:
                best_action_kind = "observe"
            rows.append({
                "state_id": state_id,
                "state": state,
                "state_label": state_label(state),
                "observed_count": observed_count(state),
                "unobserved_count": self.task.n_nodes - observed_count(state),
                "value": result["value"],
                "terminal_value": result["terminal_value"],
                "terminal_value_raw": result["terminal_value_raw"],
                "p_stop_optimal": p_stop,
                "p_continue_optimal": p_continue,
                "best_action_kind": best_action_kind,
                "optimal_observe_path_states": json.dumps(optimal_observe_path_states, separators=(",", ":")),
                "n_action_classes": len(actions),
                "n_optimal_action_classes": sum(1 for action in actions if action["optimal"]),
            })
        return rows

    def action_rows(self, state_id_by_state):
        rows = []
        for state in sorted(self.seen_states, key=state_label):
            result = self.solve(state)
            for action_index, action in enumerate(result["actions"]):
                rows.append({
                    "state_id": state_id_by_state[state],
                    "state_label": state_label(state),
                    "action_index": action_index,
                    "action_kind": action["kind"],
                    "observe_path_state": "" if action["path_state"] is None else json_path_state(action["path_state"]),
                    "observe_node": "",
                    "q_value": action["q_value"],
                    "optimal": action["optimal"],
                    "tie_policy_prob": action["tie_policy_prob"],
                    "concrete_action_count": action["concrete_count"],
                })
        return rows

    def occupancy_rows(self, state_id_by_state):
        rows = []
        summary = []
        active = {empty_state(self.task): 1.0}
        decision_timestep = 1
        while active:
            next_active = {}
            mass = sum(active.values())
            weighted_p_stop = 0.0
            weighted_value = 0.0
            weighted_terminal_value = 0.0
            for state, state_mass in sorted(active.items(), key=lambda item: state_label(item[0])):
                result = self.solve(state)
                actions = result["actions"]
                p_stop = sum(action["tie_policy_prob"] for action in actions if action["kind"] == "stop")
                weighted_p_stop += state_mass * p_stop
                weighted_value += state_mass * result["value"]
                weighted_terminal_value += state_mass * result["terminal_value"]
                rows.append({
                    "decision_timestep": decision_timestep,
                    "state_id": state_id_by_state[state],
                    "state_label": state_label(state),
                    "state_mass": state_mass,
                    "observed_count": observed_count(state),
                    "value": result["value"],
                    "terminal_value": result["terminal_value"],
                    "terminal_value_raw": result["terminal_value_raw"],
                    "p_stop": p_stop,
                    "p_continue": 1.0 - p_stop,
                })
                for action in actions:
                    action_prob = action["tie_policy_prob"]
                    if action_prob <= 0.0 or action["kind"] != "observe":
                        continue
                    transition_mass = state_mass * action_prob
                    for reward in self.rewards:
                        next_path_state = canonical_path_state(action["path_state"] + (reward,))
                        next_state = replace_one_path_state(state, action["path_state"], next_path_state)
                        next_active[next_state] = next_active.get(next_state, 0.0) + transition_mass * self.reward_prob
            summary.append({
                "decision_timestep": decision_timestep,
                "active_state_count": len(active),
                "active_mass": mass,
                "p_stop": weighted_p_stop / mass,
                "p_continue": 1.0 - weighted_p_stop / mass,
                "expected_value": weighted_value / mass,
                "expected_terminal_value": weighted_terminal_value / mass,
            })
            active = {
                state: state_mass
                for state, state_mass in next_active.items()
                if state_mass > 0.0 and observed_count(state) <= self.task.n_nodes
            }
            decision_timestep += 1
            if decision_timestep > self.task.n_nodes + 1:
                break
        return rows, summary


def empty_node_state(task):
    return tuple(None for _ in range(task.n_nodes))


def node_state_label(state):
    return "[" + ",".join("_" if value is None else fmt_num(value) for value in state) + "]"


def node_observed_count(state):
    return sum(value is not None for value in state)


def replace_node_value(state, node, reward):
    values = list(state)
    values[node] = reward
    return tuple(values)


def terminal_node_path_values_raw(state, task, prior_mean):
    values = []
    for path in task.paths:
        values.append(sum(prior_mean if state[node] is None else state[node] for node in path))
    return tuple(values)


class GeneralNodeExactTimeCostSolver(ExactTimeCostSolver):
    def terminal_value_raw(self, state):
        return max(terminal_node_path_values_raw(state, self.task, self.prior_mean))

    def stop_is_legal(self, state):
        return node_observed_count(state) >= self.min_observations_before_stop

    def observe_cost(self, state):
        return self.time_cost

    def unique_observe_actions(self, state):
        for node, value in enumerate(state):
            if value is None:
                yield {"kind": "observe", "node": node, "concrete_count": 1}

    @lru_cache(maxsize=None)
    def solve(self, state):
        self.seen_states.add(state)
        actions = []
        terminal_raw = self.terminal_value_raw(state)
        terminal_value = self.scale_reward(terminal_raw)
        if self.stop_is_legal(state):
            actions.append({
                "kind": "stop",
                "node": None,
                "q_value": terminal_value,
                "terminal_value_raw": terminal_raw,
                "concrete_count": 1,
            })
        for action in self.unique_observe_actions(state):
            expected_future_value = 0.0
            for reward in self.rewards:
                next_state = replace_node_value(state, action["node"], reward)
                expected_future_value += self.reward_prob * self.solve(next_state)["value"]
            actions.append({
                "kind": "observe",
                "node": action["node"],
                "q_value": -self.observe_cost(state) + expected_future_value,
                "terminal_value_raw": "",
                "concrete_count": action["concrete_count"],
            })
        if not actions:
            raise RuntimeError(f"No legal actions in state {node_state_label(state)}")
        best_q = max(action["q_value"] for action in actions)
        for action in actions:
            action["optimal"] = abs(action["q_value"] - best_q) <= self.tie_tol
        self.add_tie_policy_probs(actions)
        return {
            "value": best_q,
            "terminal_value": terminal_value,
            "terminal_value_raw": terminal_raw,
            "actions": tuple(actions),
        }

    def state_rows(self):
        initial = empty_node_state(self.task)
        self.solve(initial)
        rows = []
        for state_id, state in enumerate(sorted(self.seen_states, key=node_state_label)):
            result = self.solve(state)
            actions = result["actions"]
            p_stop = sum(action["tie_policy_prob"] for action in actions if action["kind"] == "stop")
            p_continue = 1.0 - p_stop
            optimal_observe_nodes = [
                action["node"] + 1
                for action in actions
                if action["kind"] == "observe" and action["optimal"]
            ]
            if p_stop > 0.0 and p_continue > 0.0:
                best_action_kind = "tie"
            elif p_stop > 0.0:
                best_action_kind = "stop"
            else:
                best_action_kind = "observe"
            rows.append({
                "state_id": state_id,
                "state": state,
                "state_label": node_state_label(state),
                "observed_count": node_observed_count(state),
                "unobserved_count": self.task.n_nodes - node_observed_count(state),
                "value": result["value"],
                "terminal_value": result["terminal_value"],
                "terminal_value_raw": result["terminal_value_raw"],
                "p_stop_optimal": p_stop,
                "p_continue_optimal": p_continue,
                "best_action_kind": best_action_kind,
                "optimal_observe_path_states": "",
                "optimal_observe_nodes": json.dumps(optimal_observe_nodes, separators=(",", ":")),
                "n_action_classes": len(actions),
                "n_optimal_action_classes": sum(1 for action in actions if action["optimal"]),
            })
        return rows

    def action_rows(self, state_id_by_state):
        rows = []
        for state in sorted(self.seen_states, key=node_state_label):
            result = self.solve(state)
            for action_index, action in enumerate(result["actions"]):
                rows.append({
                    "state_id": state_id_by_state[state],
                    "state_label": node_state_label(state),
                    "action_index": action_index,
                    "action_kind": action["kind"],
                    "observe_path_state": "",
                    "observe_node": "" if action["node"] is None else action["node"] + 1,
                    "q_value": action["q_value"],
                    "optimal": action["optimal"],
                    "tie_policy_prob": action["tie_policy_prob"],
                    "concrete_action_count": action["concrete_count"],
                })
        return rows

    def occupancy_rows(self, state_id_by_state):
        rows = []
        summary = []
        active = {empty_node_state(self.task): 1.0}
        decision_timestep = 1
        while active:
            next_active = {}
            mass = sum(active.values())
            weighted_p_stop = 0.0
            weighted_value = 0.0
            weighted_terminal_value = 0.0
            for state, state_mass in sorted(active.items(), key=lambda item: node_state_label(item[0])):
                result = self.solve(state)
                actions = result["actions"]
                p_stop = sum(action["tie_policy_prob"] for action in actions if action["kind"] == "stop")
                weighted_p_stop += state_mass * p_stop
                weighted_value += state_mass * result["value"]
                weighted_terminal_value += state_mass * result["terminal_value"]
                rows.append({
                    "decision_timestep": decision_timestep,
                    "state_id": state_id_by_state[state],
                    "state_label": node_state_label(state),
                    "state_mass": state_mass,
                    "observed_count": node_observed_count(state),
                    "value": result["value"],
                    "terminal_value": result["terminal_value"],
                    "terminal_value_raw": result["terminal_value_raw"],
                    "p_stop": p_stop,
                    "p_continue": 1.0 - p_stop,
                })
                for action in actions:
                    action_prob = action["tie_policy_prob"]
                    if action_prob <= 0.0 or action["kind"] != "observe":
                        continue
                    transition_mass = state_mass * action_prob
                    for reward in self.rewards:
                        next_state = replace_node_value(state, action["node"], reward)
                        next_active[next_state] = next_active.get(next_state, 0.0) + transition_mass * self.reward_prob
            summary.append({
                "decision_timestep": decision_timestep,
                "active_state_count": len(active),
                "active_mass": mass,
                "p_stop": weighted_p_stop / mass,
                "p_continue": 1.0 - weighted_p_stop / mass,
                "expected_value": weighted_value / mass,
                "expected_terminal_value": weighted_terminal_value / mass,
            })
            active = {
                state: state_mass
                for state, state_mass in next_active.items()
                if state_mass > 0.0 and node_observed_count(state) <= self.task.n_nodes
            }
            decision_timestep += 1
            if decision_timestep > self.task.n_nodes + 1:
                break
        return rows, summary


def attach_metadata(rows, task, solver):
    out = []
    for row in rows:
        row_out = {
            "task": task.name,
            "tree_size": task.tree_size,
            "tree_config": task.tree_config,
            "num_paths": task.num_paths,
            "path_length": task.path_length,
            "n_nodes": task.n_nodes,
            "time_cost": solver.time_cost,
            "reward_values": ",".join(fmt_num(value) for value in solver.rewards),
            "reward_prior_mean": solver.prior_mean,
            "normalize_reward": solver.normalize_reward,
            "reward_norm": solver.reward_norm,
            "min_observations_before_stop": solver.min_observations_before_stop,
            "tie_mode": solver.tie_mode,
        }
        row_out.update(row)
        row_out.pop("state", None)
        out.append(row_out)
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_solver(task, rewards, time_cost, args):
    solver_cls = GeneralNodeExactTimeCostSolver if task.has_shared_nodes else ExactTimeCostSolver
    return solver_cls(
        task=task,
        rewards=rewards,
        time_cost=time_cost,
        normalize_reward=not args.no_normalize_reward,
        min_observations_before_stop=args.min_observations_before_stop,
        tie_tol=args.tie_tol,
        tie_mode=args.tie_mode,
    )


def initial_state(task):
    return empty_node_state(task) if task.has_shared_nodes else empty_state(task)


def solve_all(args):
    task_names = normalize_task_names(args.task)
    rewards = BINARY_REWARDS if args.input_type == "binary" else parse_float_list(args.reward_values)
    time_costs = parse_float_list(args.time_costs)
    all_state_rows = []
    all_action_rows = []
    all_occupancy_rows = []
    all_summary_rows = []
    for task_name in task_names:
        task = TASKS[task_name]
        for time_cost in time_costs:
            solver = make_solver(task, rewards, time_cost, args)
            state_rows = solver.state_rows()
            state_id_by_state = {row["state"]: row["state_id"] for row in state_rows}
            action_rows = solver.action_rows(state_id_by_state)
            occupancy_rows, summary_rows = solver.occupancy_rows(state_id_by_state)
            all_state_rows.extend(attach_metadata(state_rows, task, solver))
            all_action_rows.extend(attach_metadata(action_rows, task, solver))
            all_occupancy_rows.extend(attach_metadata(occupancy_rows, task, solver))
            all_summary_rows.extend(attach_metadata(summary_rows, task, solver))
            print(
                f"{task.name} cost={fmt_num(time_cost)}: "
                f"{len(state_rows)} canonical states, "
                f"V0={solver.solve(initial_state(task))['value']:.8g}, "
                f"reward_norm={solver.reward_norm:.8g}"
            )
    prefix = args.output_prefix
    write_csv(os.path.join(args.output_dir, f"{prefix}_states.csv"), all_state_rows)
    write_csv(os.path.join(args.output_dir, f"{prefix}_actions.csv"), all_action_rows)
    write_csv(os.path.join(args.output_dir, f"{prefix}_occupancy.csv"), all_occupancy_rows)
    write_csv(os.path.join(args.output_dir, f"{prefix}_summary.csv"), all_summary_rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Calculate exact perfect-memory optimal policies with time cost only."
    )
    parser.add_argument("--task", nargs="+", default=["all"])
    parser.add_argument("--time-costs", default="0.0,0.01,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--input-type", choices=("uniform", "binary"), default="uniform")
    parser.add_argument(
        "--reward-values",
        default=",".join(fmt_num(value) for value in DEFAULT_REWARDS),
    )
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--min-observations-before-stop", type=int, default=0)
    parser.add_argument(
        "--tie-mode",
        choices=("uniform_class", "uniform_concrete", "first"),
        default="uniform_class",
    )
    parser.add_argument("--tie-tol", type=float, default=1e-10)
    parser.add_argument("--output-dir", default="analyses/exp_binary/results/exact_time_cost")
    parser.add_argument("--output-prefix", default="exact_time_cost")
    return parser


def main():
    args = build_arg_parser().parse_args()
    solve_all(args)


if __name__ == "__main__":
    main()
