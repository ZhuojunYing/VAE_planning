#!/usr/bin/env python3
"""Generate representative time costs where exact policies change.

The value of any policy is affine in the per-observation time cost:

    expected terminal reward - time_cost * expected_observations

This script computes the upper envelope of those affine values by dynamic
programming, then stores one non-boundary representative time cost from each
initial-state interval.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import exact_time_cost_solution as exact


INF = float("inf")
EPS = 1e-10


@dataclass(frozen=True)
class Segment:
    lo: float
    hi: float
    intercept: float
    slope: float
    action_kind: str
    action_detail: str


def segment_value(segment, cost):
    return segment.intercept + segment.slope * cost


def finite_breaks(segments):
    points = {0.0}
    for segment in segments:
        if math.isfinite(segment.lo):
            points.add(max(0.0, segment.lo))
        if math.isfinite(segment.hi):
            points.add(max(0.0, segment.hi))
    return points


def interval_midpoint(lo, hi):
    if math.isinf(hi):
        return lo + max(1e-3, 0.1 * max(1.0, abs(lo)))
    return (lo + hi) / 2.0


def find_segment(segments, cost):
    for segment in segments:
        if segment.lo - EPS <= cost and (cost < segment.hi - EPS or math.isinf(segment.hi)):
            return segment
    for segment in reversed(segments):
        if abs(cost - segment.hi) <= EPS:
            return segment
    raise RuntimeError(f"No segment covers cost={cost}.")


def merge_adjacent(segments):
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        same_line = (
            abs(previous.intercept - segment.intercept) <= 1e-9
            and abs(previous.slope - segment.slope) <= 1e-9
            and previous.action_kind == segment.action_kind
            and previous.action_detail == segment.action_detail
        )
        if same_line and abs(previous.hi - segment.lo) <= 1e-9:
            merged[-1] = Segment(
                previous.lo,
                segment.hi,
                previous.intercept,
                previous.slope,
                previous.action_kind,
                previous.action_detail,
            )
        else:
            merged.append(segment)
    return merged


def action_sort_key(segment):
    action_rank = 0 if segment.action_kind == "stop" else 1
    return (action_rank, segment.action_detail)


def upper_envelope(candidate_segments):
    points = {0.0}
    for action_segments in candidate_segments:
        points.update(finite_breaks(action_segments))
    base_points = sorted(points)
    if not base_points or base_points[0] > 0.0:
        base_points.insert(0, 0.0)
    base_intervals = []
    for lo, hi in zip(base_points, base_points[1:]):
        if hi - lo > EPS:
            base_intervals.append((lo, hi))
    base_intervals.append((base_points[-1], INF))

    out = []
    for lo, hi in base_intervals:
        active = []
        probe = interval_midpoint(lo, hi)
        for action_segments in candidate_segments:
            try:
                active.append(find_segment(action_segments, probe))
            except RuntimeError:
                continue
        if not active:
            continue

        cuts = {lo}
        if math.isfinite(hi):
            cuts.add(hi)
        for i, left in enumerate(active):
            for right in active[i + 1:]:
                denom = left.slope - right.slope
                if abs(denom) <= EPS:
                    continue
                crossing = (right.intercept - left.intercept) / denom
                if crossing > lo + EPS and (math.isinf(hi) or crossing < hi - EPS):
                    cuts.add(crossing)

        sorted_cuts = sorted(cuts)
        if math.isinf(hi):
            intervals = [(a, b) for a, b in zip(sorted_cuts, sorted_cuts[1:])]
            intervals.append((sorted_cuts[-1], INF))
        else:
            intervals = [(a, b) for a, b in zip(sorted_cuts, sorted_cuts[1:])]

        for sub_lo, sub_hi in intervals:
            if math.isfinite(sub_hi) and sub_hi - sub_lo <= EPS:
                continue
            sub_probe = interval_midpoint(sub_lo, sub_hi)
            best_value = max(segment_value(segment, sub_probe) for segment in active)
            winners = [
                segment
                for segment in active
                if abs(segment_value(segment, sub_probe) - best_value) <= 1e-9
            ]
            winner = sorted(winners, key=action_sort_key)[0]
            out.append(Segment(
                sub_lo,
                sub_hi,
                winner.intercept,
                winner.slope,
                winner.action_kind,
                winner.action_detail,
            ))

    return merge_adjacent(out)


class PiecewiseDynamics:
    def __init__(self, task, rewards, normalize_reward=True, min_observations_before_stop=0):
        self.task = task
        self.rewards = tuple(rewards)
        self.reward_prob = 1.0 / len(self.rewards)
        self.prior_mean = sum(self.rewards) / len(self.rewards)
        self.normalize_reward = bool(normalize_reward)
        self.reward_norm = exact.reward_norm(task, self.rewards) if normalize_reward else 1.0
        self.min_observations_before_stop = int(min_observations_before_stop)
        self.shared_nodes = task.has_shared_nodes

    def scale_reward(self, value):
        return value / self.reward_norm

    def empty_state(self):
        if self.shared_nodes:
            return exact.empty_node_state(self.task)
        return exact.empty_state(self.task)

    def observed_count(self, state):
        if self.shared_nodes:
            return exact.node_observed_count(state)
        return exact.observed_count(state)

    def state_label(self, state):
        if self.shared_nodes:
            return exact.node_state_label(state)
        return exact.state_label(state)

    def stop_value(self, state):
        if self.shared_nodes:
            raw = max(exact.terminal_node_path_values_raw(state, self.task, self.prior_mean))
        else:
            raw = max(exact.terminal_path_values_raw(state, self.task, self.prior_mean))
        return self.scale_reward(raw)

    def stop_is_legal(self, state):
        return self.observed_count(state) >= self.min_observations_before_stop

    def observe_actions(self, state):
        if self.shared_nodes:
            for node, value in enumerate(state):
                if value is None:
                    yield ("observe", str(node + 1), node)
            return

        seen = set()
        for path_state in state:
            if len(path_state) >= self.task.path_length or path_state in seen:
                continue
            seen.add(path_state)
            yield ("observe", exact.json_path_state(path_state), path_state)

    def transition(self, state, payload, reward):
        if self.shared_nodes:
            return exact.replace_node_value(state, payload, reward)
        next_path_state = exact.canonical_path_state(payload + (reward,))
        return exact.replace_one_path_state(state, payload, next_path_state)

    @lru_cache(maxsize=None)
    def value_segments(self, state):
        candidates = []
        if self.stop_is_legal(state):
            candidates.append([Segment(0.0, INF, self.stop_value(state), 0.0, "stop", "")])

        for action_kind, action_detail, payload in self.observe_actions(state):
            child_segments = [
                self.value_segments(self.transition(state, payload, reward))
                for reward in self.rewards
            ]
            candidates.append(self.combine_observe_children(state, child_segments, action_kind, action_detail))

        if not candidates:
            raise RuntimeError(f"No legal actions in state {self.state_label(state)}")
        return tuple(upper_envelope(candidates))

    def observe_cost_slope(self, state):
        return -1.0

    def combine_observe_children(self, state, child_segments, action_kind, action_detail):
        points = {0.0}
        for segments in child_segments:
            points.update(finite_breaks(segments))
        points = sorted(points)
        intervals = []
        for lo, hi in zip(points, points[1:]):
            if hi - lo > EPS:
                intervals.append((lo, hi))
        intervals.append((points[-1], INF))

        out = []
        for lo, hi in intervals:
            probe = interval_midpoint(lo, hi)
            pieces = [find_segment(segments, probe) for segments in child_segments]
            intercept = sum(piece.intercept for piece in pieces) * self.reward_prob
            slope = self.observe_cost_slope(state) + sum(piece.slope for piece in pieces) * self.reward_prob
            out.append(Segment(lo, hi, intercept, slope, action_kind, action_detail))
        return merge_adjacent(out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate representative exact time costs from policy breakpoints."
    )
    parser.add_argument("--task", nargs="+", default=["all"])
    parser.add_argument("--input-type", choices=("uniform", "binary"), default="uniform")
    parser.add_argument(
        "--reward-values",
        default=",".join(exact.fmt_num(value) for value in exact.DEFAULT_REWARDS),
    )
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--min-observations-before-stop", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="analyses/exp_binary/results/exact_time_cost_breakpoints",
    )
    parser.add_argument("--output-prefix", default="exact_time_cost")
    parser.add_argument(
        "--sample-cost-count",
        type=int,
        default=10,
        help=(
            "Maximum number of representative costs to keep per task. "
            "Costs are sampled evenly by index after all breakpoint intervals are computed. "
            "Use 0 to keep every representative cost."
        ),
    )
    return parser.parse_args()


def fmt_float(value):
    if value is None or math.isinf(value):
        return ""
    return f"{value:.12g}"


def representative_cost(segment):
    if math.isinf(segment.hi):
        return segment.lo + max(1e-3, 0.1 * max(1.0, abs(segment.lo)))
    return (segment.lo + segment.hi) / 2.0


def sampled_indices(n_values, sample_count):
    if sample_count <= 0 or n_values <= sample_count:
        return list(range(n_values))
    if sample_count == 1:
        return [0]
    indices = [
        int(round(i * (n_values - 1) / (sample_count - 1)))
        for i in range(sample_count)
    ]
    out = []
    for index in indices:
        if index not in out:
            out.append(index)
    candidate = 0
    while len(out) < sample_count and candidate < n_values:
        if candidate not in out:
            out.append(candidate)
        candidate += 1
    return sorted(out)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def main():
    args = parse_args()
    task_names = exact.normalize_task_names(args.task)
    rewards = exact.BINARY_REWARDS if args.input_type == "binary" else exact.parse_float_list(args.reward_values)

    interval_rows = []
    cost_rows = []
    for task_name in task_names:
        task = exact.TASKS[task_name]
        dynamics = PiecewiseDynamics(
            task=task,
            rewards=rewards,
            normalize_reward=not args.no_normalize_reward,
            min_observations_before_stop=args.min_observations_before_stop,
        )
        segments = list(dynamics.value_segments(dynamics.empty_state()))
        all_task_costs = []
        task_interval_rows = []
        for interval_index, segment in enumerate(segments, start=1):
            cost = representative_cost(segment)
            all_task_costs.append(cost)
            task_interval_rows.append({
                "task": task.name,
                "tree_size": task.tree_size,
                "tree_config": task.tree_config,
                "interval_index": interval_index,
                "lower_time_cost": fmt_float(segment.lo),
                "upper_time_cost": fmt_float(segment.hi),
                "representative_time_cost": fmt_float(cost),
                "initial_action_kind": segment.action_kind,
                "initial_action_detail": segment.action_detail,
                "value_intercept": fmt_float(segment.intercept),
                "value_slope": fmt_float(segment.slope),
                "expected_observations": fmt_float(-segment.slope),
                "reward_norm": fmt_float(dynamics.reward_norm),
            })

        keep_indices = sampled_indices(len(all_task_costs), args.sample_cost_count)
        keep_interval_indices = {index + 1 for index in keep_indices}
        for row in task_interval_rows:
            row["sampled_for_exact_run"] = str(
                int(int(row["interval_index"]) in keep_interval_indices)
            )
            interval_rows.append(row)

        task_costs = [all_task_costs[index] for index in keep_indices]
        for index in keep_indices:
            interval_index = index + 1
            cost_rows.append({
                "task": task.name,
                "interval_index": interval_index,
                "representative_time_cost": fmt_float(all_task_costs[index]),
                "sample_index": keep_indices.index(index) + 1,
                "n_full_intervals": len(all_task_costs),
            })

        cost_text = ",".join(fmt_float(cost) for cost in task_costs)
        all_cost_text = ",".join(fmt_float(cost) for cost in all_task_costs)
        write_text(
            os.path.join(args.output_dir, f"{args.output_prefix}_{task.name}_costs.txt"),
            cost_text + "\n",
        )
        write_text(
            os.path.join(args.output_dir, f"{args.output_prefix}_{task.name}_all_costs.txt"),
            all_cost_text + "\n",
        )
        print(
            f"{task.name}: sampled {len(task_costs)} of {len(all_task_costs)} "
            f"representative costs -> {cost_text}"
        )

    write_csv(
        os.path.join(args.output_dir, f"{args.output_prefix}_breakpoint_intervals.csv"),
        interval_rows,
    )
    write_csv(
        os.path.join(args.output_dir, f"{args.output_prefix}_representative_costs.csv"),
        cost_rows,
    )


if __name__ == "__main__":
    main()
