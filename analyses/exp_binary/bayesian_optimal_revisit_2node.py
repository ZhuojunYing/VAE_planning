#!/usr/bin/env python3
"""Bayes-optimal dynamic program for the noisy 2-node revisit task.

This script solves a two-armed, one-node-per-arm bandit where each latent reward
is drawn independently from a discrete prior, but observations can be noisy and
nodes may be revisited.  For sigma > 0, the belief state for each node is
continuous but two sufficient statistics are enough under Gaussian observation
noise:

    n_i = number of observations of node i
    s_i = sum of observed scalar rewards for node i

The finite-horizon value function is represented on interpolation grids over
(n1, s1, n2, s2).  Future observations are integrated with Gauss-Hermite
quadrature over the posterior predictive distribution.

The Bellman equation is:

    V(n1,s1,n2,s2) = max(
        stop_value,
        -c + E[y1 | n1,s1] V(n1+1,s1+y1,n2,s2),
        -c + E[y2 | n2,s2] V(n1,s1,n2+1,s2+y2)
    )

where stop_value is the larger posterior mean reward, optionally normalized by
the expected maximum reward of the two latent rewards under the prior.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_REWARDS = (-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)
EPS = 1e-12


def parse_float_list(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(text).split(",") if part.strip())


def fmt_num(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.12g}"


def safe_name(value: float) -> str:
    return fmt_num(value).replace("-", "m").replace(".", "p")


def expected_best_two_node_reward(reward_values: Sequence[float]) -> float:
    rewards = np.asarray(reward_values, dtype=float)
    vals = np.maximum(rewards[:, None], rewards[None, :])
    return float(np.mean(vals))


def logsumexp(values: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    out = max_values + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def clipped_searchsorted(grid: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower interpolation indices and fractions for a 1D grid."""
    if len(grid) == 1:
        idx = np.zeros_like(query, dtype=np.int64)
        frac = np.zeros_like(query, dtype=float)
        return idx, frac
    clipped = np.clip(query, grid[0], grid[-1])
    idx = np.searchsorted(grid, clipped, side="right") - 1
    idx = np.clip(idx, 0, len(grid) - 2)
    denom = grid[idx + 1] - grid[idx]
    frac = np.where(np.abs(denom) > EPS, (clipped - grid[idx]) / denom, 0.0)
    return idx.astype(np.int64), frac.astype(float)


def interp_axis0(table: np.ndarray, x_grid: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Interpolate a 2D table along axis 0; axis 1 is already aligned."""
    idx, frac = clipped_searchsorted(x_grid, np.asarray(x_query, dtype=float))
    if len(x_grid) == 1:
        return np.repeat(table[0:1, :], len(idx), axis=0)
    return (1.0 - frac)[:, None] * table[idx, :] + frac[:, None] * table[idx + 1, :]


def interp_axis1(table: np.ndarray, y_grid: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    """Interpolate a 2D table along axis 1; axis 0 is already aligned."""
    idx, frac = clipped_searchsorted(y_grid, np.asarray(y_query, dtype=float))
    if len(y_grid) == 1:
        return np.repeat(table[:, 0:1], len(idx), axis=1)
    return (1.0 - frac)[None, :] * table[:, idx] + frac[None, :] * table[:, idx + 1]


def bilinear(table: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation for one point."""
    ix, fx = clipped_searchsorted(x_grid, np.asarray([x], dtype=float))
    iy, fy = clipped_searchsorted(y_grid, np.asarray([y], dtype=float))
    ix0 = int(ix[0])
    iy0 = int(iy[0])
    fx0 = float(fx[0])
    fy0 = float(fy[0])
    if len(x_grid) == 1 and len(y_grid) == 1:
        return float(table[0, 0])
    if len(x_grid) == 1:
        return float((1.0 - fy0) * table[0, iy0] + fy0 * table[0, iy0 + 1])
    if len(y_grid) == 1:
        return float((1.0 - fx0) * table[ix0, 0] + fx0 * table[ix0 + 1, 0])
    return float(
        (1.0 - fx0) * (1.0 - fy0) * table[ix0, iy0]
        + fx0 * (1.0 - fy0) * table[ix0 + 1, iy0]
        + (1.0 - fx0) * fy0 * table[ix0, iy0 + 1]
        + fx0 * fy0 * table[ix0 + 1, iy0 + 1]
    )


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    value: float
    q_stop: float
    q_observe_1: float
    q_observe_2: float
    posterior_mean_1: float
    posterior_mean_2: float
    p_stop: float
    p_observe_1: float
    p_observe_2: float


class NoisyBayesRevisit2NodeDP:
    """Continuous-belief finite-horizon DP for sigma > 0."""

    def __init__(
        self,
        reward_values: Sequence[float] = DEFAULT_REWARDS,
        sigma: float = 1.0,
        time_cost: float = 0.1,
        max_observations: int = 10,
        grid_size: int = 161,
        grid_sigma_bound: float = 7.0,
        quadrature_order: int = 21,
        normalize_reward: bool = True,
        min_observations_before_stop: int = 0,
        tie_tol: float = 1e-9,
    ):
        if sigma <= 0.0:
            raise ValueError("NoisyBayesRevisit2NodeDP requires sigma > 0. Use NoiselessRevisit2NodeDP for sigma=0.")
        self.reward_values = np.asarray(reward_values, dtype=float)
        self.sigma = float(sigma)
        self.time_cost = float(time_cost)
        self.max_observations = int(max_observations)
        self.grid_size = int(grid_size)
        self.grid_sigma_bound = float(grid_sigma_bound)
        self.quadrature_order = int(quadrature_order)
        self.normalize_reward = bool(normalize_reward)
        self.reward_norm = expected_best_two_node_reward(self.reward_values) if normalize_reward else 1.0
        self.min_observations_before_stop = int(min_observations_before_stop)
        self.tie_tol = float(tie_tol)
        self.prior_prob = np.ones(len(self.reward_values), dtype=float) / len(self.reward_values)
        self.gh_x, self.gh_w = np.polynomial.hermite.hermgauss(self.quadrature_order)
        self.gh_w = self.gh_w / math.sqrt(math.pi)
        self.grids = self._build_sum_grids()
        self.values: Dict[Tuple[int, int], np.ndarray] = {}
        self.q_tables: Dict[Tuple[int, int], Mapping[str, np.ndarray]] = {}

    def _build_sum_grids(self) -> Dict[int, np.ndarray]:
        grids: Dict[int, np.ndarray] = {0: np.asarray([0.0], dtype=float)}
        r_min = float(np.min(self.reward_values))
        r_max = float(np.max(self.reward_values))
        for n in range(1, self.max_observations + 1):
            # This bound is intentionally based on sqrt(n), not n, because it
            # tracks the central mass of the Gaussian sum. Interpolation clamps
            # rare far-tail quadrature points.
            margin = self.grid_sigma_bound * self.sigma * math.sqrt(n)
            lo = n * r_min - margin
            hi = n * r_max + margin
            grids[n] = np.linspace(lo, hi, self.grid_size, dtype=float)
        return grids

    def posterior_probs(self, n: int, s: np.ndarray | float) -> np.ndarray:
        s_arr = np.asarray(s, dtype=float)
        if n <= 0:
            shape = s_arr.shape + (len(self.reward_values),)
            return np.broadcast_to(self.prior_prob, shape).copy()
        rewards = self.reward_values
        logp = -(n * rewards[None, :] ** 2 - 2.0 * s_arr.reshape(-1, 1) * rewards[None, :])
        logp = logp / (2.0 * self.sigma ** 2)
        logp = logp - logsumexp(logp, axis=1, keepdims=True)
        probs = np.exp(logp)
        return probs.reshape(s_arr.shape + (len(rewards),))

    def posterior_mean(self, n: int, s: np.ndarray | float) -> np.ndarray:
        probs = self.posterior_probs(n, s)
        return np.sum(probs * self.reward_values, axis=-1)

    def stop_value_grid(self, n1: int, n2: int) -> np.ndarray:
        s1_grid = self.grids[n1]
        s2_grid = self.grids[n2]
        m1 = self.posterior_mean(n1, s1_grid)
        m2 = self.posterior_mean(n2, s2_grid)
        stop = np.maximum(m1[:, None], m2[None, :]) / self.reward_norm
        if n1 + n2 < self.min_observations_before_stop:
            stop[:] = -np.inf
        return stop

    def _observe_node1_grid(self, n1: int, n2: int) -> np.ndarray:
        s1_grid = self.grids[n1]
        s2_grid = self.grids[n2]
        next_value = self.values[(n1 + 1, n2)]
        next_s1_grid = self.grids[n1 + 1]
        probs = self.posterior_probs(n1, s1_grid)
        out = np.zeros((len(s1_grid), len(s2_grid)), dtype=float)
        noise_scale = math.sqrt(2.0) * self.sigma
        for reward_index, reward in enumerate(self.reward_values):
            reward_weight = probs[:, reward_index]
            if np.max(reward_weight) <= 0.0:
                continue
            for x, w in zip(self.gh_x, self.gh_w):
                s_next = s1_grid + reward + noise_scale * x
                continuation = interp_axis0(next_value, next_s1_grid, s_next)
                out += w * reward_weight[:, None] * continuation
        return -self.time_cost + out

    def _observe_node2_grid(self, n1: int, n2: int) -> np.ndarray:
        s1_grid = self.grids[n1]
        s2_grid = self.grids[n2]
        next_value = self.values[(n1, n2 + 1)]
        next_s2_grid = self.grids[n2 + 1]
        probs = self.posterior_probs(n2, s2_grid)
        out = np.zeros((len(s1_grid), len(s2_grid)), dtype=float)
        noise_scale = math.sqrt(2.0) * self.sigma
        for reward_index, reward in enumerate(self.reward_values):
            reward_weight = probs[:, reward_index]
            if np.max(reward_weight) <= 0.0:
                continue
            for x, w in zip(self.gh_x, self.gh_w):
                s_next = s2_grid + reward + noise_scale * x
                continuation = interp_axis1(next_value, next_s2_grid, s_next)
                out += w * reward_weight[None, :] * continuation
        return -self.time_cost + out

    def solve(self) -> "NoisyBayesRevisit2NodeDP":
        for total_observed in range(self.max_observations, -1, -1):
            for n1 in range(total_observed + 1):
                n2 = total_observed - n1
                q_stop = self.stop_value_grid(n1, n2)
                candidates = [q_stop]
                q_observe_1 = np.full_like(q_stop, -np.inf)
                q_observe_2 = np.full_like(q_stop, -np.inf)
                if total_observed < self.max_observations:
                    q_observe_1 = self._observe_node1_grid(n1, n2)
                    q_observe_2 = self._observe_node2_grid(n1, n2)
                    candidates.extend([q_observe_1, q_observe_2])
                value = np.maximum.reduce(candidates)
                self.values[(n1, n2)] = value
                self.q_tables[(n1, n2)] = {
                    "stop": q_stop,
                    "observe_1": q_observe_1,
                    "observe_2": q_observe_2,
                }
        return self

    def _interp_table(self, name: str, n1: int, s1: float, n2: int, s2: float) -> float:
        table = self.q_tables[(n1, n2)][name] if name != "value" else self.values[(n1, n2)]
        return bilinear(table, self.grids[n1], self.grids[n2], s1, s2)

    def decision(self, n1: int, s1: float, n2: int, s2: float) -> PolicyDecision:
        if not self.values:
            self.solve()
        q_stop = self._interp_table("stop", n1, s1, n2, s2)
        q_obs1 = self._interp_table("observe_1", n1, s1, n2, s2)
        q_obs2 = self._interp_table("observe_2", n1, s1, n2, s2)
        q = np.asarray([q_stop, q_obs1, q_obs2], dtype=float)
        best = np.nanmax(q)
        is_best = np.isfinite(q) & (np.abs(q - best) <= self.tie_tol)
        probs = is_best.astype(float) / max(float(np.sum(is_best)), 1.0)
        action = ("stop", "observe_1", "observe_2")[int(np.argmax(q))]
        return PolicyDecision(
            action=action,
            value=best,
            q_stop=q_stop,
            q_observe_1=q_obs1,
            q_observe_2=q_obs2,
            posterior_mean_1=float(self.posterior_mean(n1, s1)),
            posterior_mean_2=float(self.posterior_mean(n2, s2)),
            p_stop=float(probs[0]),
            p_observe_1=float(probs[1]),
            p_observe_2=float(probs[2]),
        )


class NoiselessRevisit2NodeDP:
    """Exact finite DP for sigma=0.

    Re-observing a known node has no information value, so optimal revisit
    actions are excluded when time cost is nonnegative.
    """

    def __init__(
        self,
        reward_values: Sequence[float] = DEFAULT_REWARDS,
        time_cost: float = 0.1,
        max_observations: int = 10,
        normalize_reward: bool = True,
        min_observations_before_stop: int = 0,
        tie_tol: float = 1e-9,
    ):
        self.reward_values = tuple(float(x) for x in reward_values)
        self.time_cost = float(time_cost)
        self.max_observations = int(max_observations)
        self.normalize_reward = bool(normalize_reward)
        self.reward_norm = expected_best_two_node_reward(self.reward_values) if normalize_reward else 1.0
        self.prior_mean = float(np.mean(self.reward_values))
        self.min_observations_before_stop = int(min_observations_before_stop)
        self.tie_tol = float(tie_tol)

    def _mean(self, value: Optional[float]) -> float:
        return self.prior_mean if value is None else float(value)

    @lru_cache(maxsize=None)
    def _solve(self, value1: Optional[float], value2: Optional[float], observed_count: int) -> Tuple[float, float, float, float]:
        q_stop = max(self._mean(value1), self._mean(value2)) / self.reward_norm
        if observed_count < self.min_observations_before_stop:
            q_stop = -np.inf
        if observed_count >= self.max_observations:
            return q_stop, q_stop, -np.inf, -np.inf
        q_obs1 = -np.inf
        q_obs2 = -np.inf
        if value1 is None:
            q_obs1 = -self.time_cost + float(np.mean([
                self._solve(reward, value2, observed_count + 1)[0]
                for reward in self.reward_values
            ]))
        if value2 is None:
            q_obs2 = -self.time_cost + float(np.mean([
                self._solve(value1, reward, observed_count + 1)[0]
                for reward in self.reward_values
            ]))
        value = max(q_stop, q_obs1, q_obs2)
        return value, q_stop, q_obs1, q_obs2

    def solve(self) -> "NoiselessRevisit2NodeDP":
        self._solve(None, None, 0)
        return self

    def decision(self, n1: int, s1: float, n2: int, s2: float) -> PolicyDecision:
        value1 = None if n1 == 0 else float(s1 / n1)
        value2 = None if n2 == 0 else float(s2 / n2)
        total = int(n1 + n2)
        value, q_stop, q_obs1, q_obs2 = self._solve(value1, value2, total)
        q = np.asarray([q_stop, q_obs1, q_obs2], dtype=float)
        best = np.nanmax(q)
        is_best = np.isfinite(q) & (np.abs(q - best) <= self.tie_tol)
        probs = is_best.astype(float) / max(float(np.sum(is_best)), 1.0)
        action = ("stop", "observe_1", "observe_2")[int(np.argmax(q))]
        return PolicyDecision(
            action=action,
            value=value,
            q_stop=q_stop,
            q_observe_1=q_obs1,
            q_observe_2=q_obs2,
            posterior_mean_1=self._mean(value1),
            posterior_mean_2=self._mean(value2),
            p_stop=float(probs[0]),
            p_observe_1=float(probs[1]),
            p_observe_2=float(probs[2]),
        )


def build_solver(args: argparse.Namespace, time_cost: float):
    kwargs = {
        "reward_values": parse_float_list(args.reward_values),
        "time_cost": time_cost,
        "max_observations": args.max_observations,
        "normalize_reward": not args.no_normalize_reward,
        "min_observations_before_stop": args.min_observations_before_stop,
        "tie_tol": args.tie_tol,
    }
    if args.sigma <= 0.0:
        return NoiselessRevisit2NodeDP(**kwargs).solve()
    return NoisyBayesRevisit2NodeDP(
        sigma=args.sigma,
        grid_size=args.grid_size,
        grid_sigma_bound=args.grid_sigma_bound,
        quadrature_order=args.quadrature_order,
        **kwargs,
    ).solve()


def sample_action(decision: PolicyDecision, rng: np.random.Generator, tie_mode: str) -> str:
    if tie_mode == "first":
        return decision.action
    probs = np.asarray([decision.p_stop, decision.p_observe_1, decision.p_observe_2], dtype=float)
    if np.sum(probs) <= 0.0:
        return decision.action
    idx = int(rng.choice(3, p=probs / np.sum(probs)))
    return ("stop", "observe_1", "observe_2")[idx]


def choose_terminal_path(decision: PolicyDecision, rng: np.random.Generator, tie_mode: str) -> int:
    if abs(decision.posterior_mean_1 - decision.posterior_mean_2) <= 1e-12:
        if tie_mode == "first":
            return 1
        return int(rng.choice([1, 2]))
    return 1 if decision.posterior_mean_1 > decision.posterior_mean_2 else 2


def posterior_kl_to_prior(probs: np.ndarray, prior: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    prior = np.asarray(prior, dtype=float)
    valid = probs > 0
    if not np.any(valid):
        return 0.0
    return float(np.sum(probs[valid] * (np.log(probs[valid]) - np.log(prior[valid]))))


def node_belief_kl(solver, n: int, s: float) -> float:
    if n <= 0:
        return 0.0
    if hasattr(solver, "posterior_probs"):
        probs = np.asarray(solver.posterior_probs(n, s), dtype=float)
        prior = np.asarray(solver.prior_prob, dtype=float)
        return posterior_kl_to_prior(probs, prior)
    # With noiseless observations, one observation identifies the discrete
    # reward exactly, so the posterior is a point mass over the prior support.
    return math.log(len(solver.reward_values))


def total_belief_kl(solver, n1: int, s1: float, n2: int, s2: float) -> Tuple[float, float, float]:
    kl_1 = node_belief_kl(solver, n1, s1)
    kl_2 = node_belief_kl(solver, n2, s2)
    return kl_1 + kl_2, kl_1, kl_2


def simulate_policy(
    solver,
    reward_values: Sequence[float],
    sigma: float,
    time_cost: float,
    n_trials: int,
    seed: int,
    tie_mode: str = "uniform",
) -> List[Mapping[str, object]]:
    rng = np.random.default_rng(seed)
    rewards = np.asarray(reward_values, dtype=float)
    rows: List[Mapping[str, object]] = []
    for trial in range(int(n_trials)):
        true_rewards = rng.choice(rewards, size=2, replace=True)
        n1 = n2 = 0
        s1 = s2 = 0.0
        chosen_path = None
        stop_decision = None
        row: Dict[str, object] = {
            "graph": trial,
            "reward_node1": float(true_rewards[0]),
            "reward_node2": float(true_rewards[1]),
            "time_cost": float(time_cost),
            "observation_sigma": float(sigma),
        }
        for decision_timestep in range(1, solver.max_observations + 2):
            decision = solver.decision(n1, s1, n2, s2)
            belief_kl, belief_kl_1, belief_kl_2 = total_belief_kl(solver, n1, s1, n2, s2)
            action = sample_action(decision, rng, tie_mode)
            forced_stop = n1 + n2 >= solver.max_observations
            if forced_stop:
                action = "stop"
            prefix = f"t{decision_timestep}"
            row[f"belief_kl_{prefix}"] = belief_kl
            row[f"belief_kl_node1_{prefix}"] = belief_kl_1
            row[f"belief_kl_node2_{prefix}"] = belief_kl_2
            row[f"q_stop_{prefix}"] = decision.q_stop
            row[f"q_observe_1_{prefix}"] = decision.q_observe_1
            row[f"q_observe_2_{prefix}"] = decision.q_observe_2
            row[f"posterior_mean_1_{prefix}"] = decision.posterior_mean_1
            row[f"posterior_mean_2_{prefix}"] = decision.posterior_mean_2
            row[f"p_stop_{prefix}"] = decision.p_stop
            row[f"p_observe_1_{prefix}"] = decision.p_observe_1
            row[f"p_observe_2_{prefix}"] = decision.p_observe_2
            row[f"stop_{prefix}"] = action == "stop"
            row[f"expanded_node_{prefix}"] = ""
            row[f"expanded_reward_{prefix}"] = ""
            row[f"belief_kl_after_observation_{prefix}"] = ""
            row[f"kl_d_{prefix}"] = ""
            if action == "stop":
                chosen_path = choose_terminal_path(decision, rng, tie_mode)
                stop_decision = decision_timestep
                break
            node = 1 if action == "observe_1" else 2
            true_reward = true_rewards[node - 1]
            observed = float(true_reward) if sigma <= 0.0 else float(rng.normal(true_reward, sigma))
            if node == 1:
                n1 += 1
                s1 += observed
            else:
                n2 += 1
                s2 += observed
            after_kl, after_kl_1, after_kl_2 = total_belief_kl(solver, n1, s1, n2, s2)
            row[f"expanded_node_{prefix}"] = node
            row[f"expanded_reward_{prefix}"] = observed
            row[f"belief_kl_after_observation_{prefix}"] = after_kl
            row[f"belief_kl_node1_after_observation_{prefix}"] = after_kl_1
            row[f"belief_kl_node2_after_observation_{prefix}"] = after_kl_2
            row[f"kl_d_{prefix}"] = after_kl
        if chosen_path is None:
            decision = solver.decision(n1, s1, n2, s2)
            chosen_path = choose_terminal_path(decision, rng, tie_mode)
            stop_decision = solver.max_observations + 1
        chosen_value = float(true_rewards[chosen_path - 1])
        row["chosen_path"] = int(chosen_path)
        row["V"] = chosen_value
        row["normalized_chosen_path_reward"] = chosen_value / solver.reward_norm
        row["observations_before_stop"] = int(n1 + n2)
        row["stop_decision_timestep"] = int(stop_decision)
        row["visits_node1"] = int(n1)
        row["visits_node2"] = int(n2)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-costs", default="0.0", help="Comma-separated observation costs.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Observation noise SD. Use 0 for noiseless exact observations.")
    parser.add_argument("--reward-values", default=",".join(fmt_num(x) for x in DEFAULT_REWARDS))
    parser.add_argument("--max-observations", type=int, default=10)
    parser.add_argument("--min-observations-before-stop", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--grid-sigma-bound", type=float, default=7.0)
    parser.add_argument("--quadrature-order", type=int, default=21)
    parser.add_argument("--tie-tol", type=float, default=1e-9)
    parser.add_argument("--tie-mode", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--simulate-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="analyses/exp_binary/results/bayesian_revisit_2node")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    reward_values = parse_float_list(args.reward_values)
    summary_rows = []
    for time_cost in parse_float_list(args.time_costs):
        solver = build_solver(args, time_cost)
        initial = solver.decision(0, 0.0, 0, 0.0)
        summary_rows.append({
            "time_cost": float(time_cost),
            "sigma": float(args.sigma),
            "seed": int(args.seed),
            "max_observations": int(args.max_observations),
            "reward_values": ",".join(fmt_num(x) for x in reward_values),
            "reward_norm": solver.reward_norm,
            "initial_value": initial.value,
            "initial_q_stop": initial.q_stop,
            "initial_q_observe_1": initial.q_observe_1,
            "initial_q_observe_2": initial.q_observe_2,
            "initial_best_action": initial.action,
            "initial_p_stop": initial.p_stop,
            "initial_p_observe_1": initial.p_observe_1,
            "initial_p_observe_2": initial.p_observe_2,
            "grid_size": int(args.grid_size) if args.sigma > 0.0 else "",
            "grid_sigma_bound": float(args.grid_sigma_bound) if args.sigma > 0.0 else "",
            "quadrature_order": int(args.quadrature_order) if args.sigma > 0.0 else "",
        })
        if args.simulate_trials > 0:
            sim_rows = simulate_policy(
                solver,
                reward_values=reward_values,
                sigma=args.sigma,
                time_cost=time_cost,
                n_trials=args.simulate_trials,
                seed=args.seed,
                tie_mode=args.tie_mode,
            )
            sim_name = (
                f"bayesian_revisit_2node_sim_sigma_{safe_name(args.sigma)}"
                f"_cost_{safe_name(time_cost)}_maxobs_{args.max_observations}"
                f"_seed_{args.seed}.csv"
            )
            write_csv(outdir / sim_name, sim_rows)
    summary_name = (
        f"bayesian_revisit_2node_summary_sigma_{safe_name(args.sigma)}"
        f"_maxobs_{args.max_observations}_seed_{args.seed}.csv"
    )
    write_csv(outdir / summary_name, summary_rows)
    print(f"Saved {outdir / summary_name}")


if __name__ == "__main__":
    main()
