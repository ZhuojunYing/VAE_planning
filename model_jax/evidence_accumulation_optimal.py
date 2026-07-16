"""Bayes-optimal dynamic program for the continuous evidence task.

This script implements a normative finite-horizon observer for the
continuous evidence-accumulation task in ``model_jax/evidence_accumulation.py``.
It intentionally contains no neural network, VAE, PPO update, learned critic,
or representation cost.  The only optimized objective is expected terminal
choice reward minus opportunity costs for continuation actions.

Timing convention
-----------------
The convention matches the learned evidence model.  At trial reset, the first
evidence sample is already available.  On each decision step the observer first
incorporates the current sample, then chooses one of ``CONTINUE``,
``CHOOSE_A`` (left), or ``CHOOSE_B`` (right).  A ``CONTINUE`` action pays the
opportunity cost and advances to the next evidence sample.  A terminal choice
ends the trial immediately.  At ``max_observations_before_stop`` observations,
``CONTINUE`` is unavailable and the observer must choose A or B.

Belief state and Bellman equation
---------------------------------
For known coherence magnitude ``c`` and observation noise ``sigma``, the
belief state is the posterior log odds that B/right is correct:

    L_t = log P(y=+1 | o_1:t) / P(y=-1 | o_1:t).

With equal prior, ``L_0 = 0``.  The Gaussian likelihood ratio update is

    L_t = L_{t-1} + (2 c / sigma^2) o_t.

After ``t`` observations and before the next action, terminal values are

    Q_A(L) = 1 - sigmoid(L),      Q_B(L) = sigmoid(L).

For ``t < T`` the continuation value is

    Q_continue,t(L) = -c_opp + E[V_{t+1}(L + (2 c / sigma^2) o) | L],

where the predictive distribution is the posterior mixture

    p(o | L) = sigmoid(L) N(c, sigma^2) + (1 - sigmoid(L)) N(-c, sigma^2).

The expectation is evaluated with Gauss-Hermite quadrature and the value
function is represented on a one-dimensional symmetric LLR grid with linear
interpolation.  The initial dynamic-program value reported by this script is
the expected value after the first free observation, matching the learned
model's first-observation-before-action timing.

The observer assumes the run's coherence magnitude is known.  A hidden
coherence observer would need a joint posterior over choice and coherence and
is not implemented here.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


CONTINUE = 0
CHOOSE_A = 1
CHOOSE_B = 2
NUM_ACTIONS = 3


@dataclass(frozen=True)
class OptimalConfig:
    opportunity_cost: float
    coherence: float
    observation_noise_std: float
    max_observations_before_stop: int
    num_trials: int
    seed: int
    sim_dir: str
    policy_dir: str
    input_type: str
    correct_reward: float
    incorrect_reward: float
    llr_grid_size: int
    llr_grid_max: float | None
    quadrature_order: int
    tie_tolerance: float
    boundary_fraction: float
    enable_x64: bool
    coherence_known_to_agent: bool
    reuse_policy: bool
    save_policy_table: bool
    choice_at_end_only: bool = False


@dataclass
class PolicySolution:
    llr_grid: np.ndarray
    p_right_grid: np.ndarray
    q_choose_a: np.ndarray
    q_choose_b: np.ndarray
    q_continue: np.ndarray
    value: np.ndarray
    optimal_action: np.ndarray
    boundary_fraction_quadrature: float
    initial_value_after_first_observation: float


_POLICY_CACHE: dict[str, PolicySolution] = {}


def parse_float_values(raw_values: Iterable[str] | str | None) -> list[float]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    out: list[float] = []
    for raw in raw_values:
        for piece in str(raw).replace(",", " ").split():
            if piece.strip():
                out.append(float(piece))
    return out


def label_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def stable_sigmoid(x):
    return jax.nn.sigmoid(x)


def gaussian_logpdf(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)


def default_llr_grid_max(coherence: float, sigma: float, max_observations: int) -> float:
    if coherence <= 0.0:
        return 12.0
    mean_final = max_observations * 2.0 * coherence * coherence / (sigma * sigma)
    sd_final = math.sqrt(max_observations) * 2.0 * coherence / sigma
    return max(12.0, float(mean_final + 7.0 * sd_final))


def interp_on_grid(x, grid, values):
    return jnp.interp(x, grid, values, left=values[0], right=values[-1])


def terminal_policy_from_q(q_a: float, q_b: float, tol: float, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    probs = np.zeros(NUM_ACTIONS, dtype=float)
    if q_b > q_a + tol:
        probs[CHOOSE_B] = 1.0
        return CHOOSE_B, probs
    if q_a > q_b + tol:
        probs[CHOOSE_A] = 1.0
        return CHOOSE_A, probs
    probs[CHOOSE_A] = 0.5
    probs[CHOOSE_B] = 0.5
    return (CHOOSE_B if rng.random() < 0.5 else CHOOSE_A), probs


def action_entropy(probs: np.ndarray) -> float:
    keep = probs > 0
    if not np.any(keep):
        return 0.0
    return float(-np.sum(probs[keep] * np.log(probs[keep])))


def policy_stem(config: OptimalConfig) -> str:
    observer_label = "_observer_endchoice" if bool(config.choice_at_end_only) else ""
    return (
        f"evidence_optimal_opportunity_{config.opportunity_cost:g}_"
        f"coherence_{config.coherence:g}_obsstd_{config.observation_noise_std:g}_"
        f"maxobs_{config.max_observations_before_stop}_"
        f"reward_{config.correct_reward:g}_{config.incorrect_reward:g}_"
        f"grid_{config.llr_grid_size}_quad_{config.quadrature_order}"
        f"{observer_label}"
    )


def simulation_stem(config: OptimalConfig) -> str:
    observer_label = "_observer_endchoice" if bool(config.choice_at_end_only) else ""
    return (
        f"evidence_optimal_opportunity_{config.opportunity_cost:g}_"
        f"coherence_{config.coherence:g}_obsstd_{config.observation_noise_std:g}_"
        f"maxobs_{config.max_observations_before_stop}_seed_{config.seed}_"
        f"reward_{config.correct_reward:g}_{config.incorrect_reward:g}"
        f"{observer_label}"
    )


def solve_policy(config: OptimalConfig) -> PolicySolution:
    if not config.coherence_known_to_agent:
        raise NotImplementedError(
            "coherence_known_to_agent=False requires a joint posterior over choice and coherence; "
            "the current optimal DP only supports known coherence."
        )
    if config.observation_noise_std <= 0:
        raise ValueError("--observation-noise-std must be positive for the Gaussian evidence model.")
    if config.max_observations_before_stop < 1:
        raise ValueError("--max-observations-before-stop must be at least 1.")
    if config.llr_grid_size < 101:
        raise ValueError("--llr-grid-size should be at least 101.")
    if config.llr_grid_size % 2 == 0:
        raise ValueError("--llr-grid-size must be odd so L=0 lies on the grid.")

    if config.enable_x64:
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if config.enable_x64 else jnp.float32
    T = int(config.max_observations_before_stop)
    c = float(config.coherence)
    sigma = float(config.observation_noise_std)
    llr_max = float(config.llr_grid_max or default_llr_grid_max(c, sigma, T))
    grid = jnp.linspace(-llr_max, llr_max, int(config.llr_grid_size), dtype=dtype)
    p_right = stable_sigmoid(grid)
    reward_span = float(config.correct_reward) - float(config.incorrect_reward)
    q_a_terminal = float(config.incorrect_reward) + reward_span * (1.0 - p_right)
    q_b_terminal = float(config.incorrect_reward) + reward_span * p_right

    nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(config.quadrature_order))
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    weights = jnp.asarray(weights_np, dtype=dtype) / jnp.sqrt(jnp.asarray(jnp.pi, dtype=dtype))
    scale = jnp.asarray(0.0 if c == 0.0 else 2.0 * c / (sigma * sigma), dtype=dtype)
    obs_pos = jnp.asarray(c, dtype=dtype) + jnp.sqrt(jnp.asarray(2.0, dtype=dtype)) * jnp.asarray(sigma, dtype=dtype) * nodes
    obs_neg = -jnp.asarray(c, dtype=dtype) + jnp.sqrt(jnp.asarray(2.0, dtype=dtype)) * jnp.asarray(sigma, dtype=dtype) * nodes
    llr_inc_pos = scale * obs_pos
    llr_inc_neg = scale * obs_neg

    q_a_all = []
    q_b_all = []
    q_c_all = []
    value_all = []
    action_all = []
    boundary_hits = []

    next_value = jnp.maximum(q_a_terminal, q_b_terminal)
    for t in range(T, 0, -1):
        q_a = q_a_terminal
        q_b = q_b_terminal
        if t == T:
            q_c = jnp.full_like(grid, -jnp.inf)
        else:
            eval_pos = grid[:, None] + llr_inc_pos[None, :]
            eval_neg = grid[:, None] + llr_inc_neg[None, :]
            boundary_hits.append(
                jnp.mean(((eval_pos <= grid[0]) | (eval_pos >= grid[-1]) | (eval_neg <= grid[0]) | (eval_neg >= grid[-1])).astype(dtype))
            )
            v_pos = interp_on_grid(eval_pos, grid, next_value)
            v_neg = interp_on_grid(eval_neg, grid, next_value)
            expected_pos = jnp.sum(weights[None, :] * v_pos, axis=1)
            expected_neg = jnp.sum(weights[None, :] * v_neg, axis=1)
            q_c = -float(config.opportunity_cost) + p_right * expected_pos + (1.0 - p_right) * expected_neg
        best_stop = jnp.maximum(q_a, q_b)
        terminal_action = jnp.where(q_b > q_a + config.tie_tolerance, CHOOSE_B, CHOOSE_A)
        if bool(config.choice_at_end_only) and t < T:
            value = q_c
            action = jnp.full_like(terminal_action, CONTINUE)
        else:
            value = jnp.maximum(best_stop, q_c)
            action = jnp.where(q_c > best_stop + config.tie_tolerance, CONTINUE, terminal_action)
        q_a_all.append(q_a)
        q_b_all.append(q_b)
        q_c_all.append(q_c)
        value_all.append(value)
        action_all.append(action.astype(jnp.int32))
        next_value = value

    # Built backward; convert to timestep order 1..T.
    q_a_arr = jnp.stack(q_a_all[::-1], axis=0)
    q_b_arr = jnp.stack(q_b_all[::-1], axis=0)
    q_c_arr = jnp.stack(q_c_all[::-1], axis=0)
    value_arr = jnp.stack(value_all[::-1], axis=0)
    action_arr = jnp.stack(action_all[::-1], axis=0)

    init_value = initial_value_after_first_observation(
        np.asarray(grid),
        np.asarray(value_arr[0]),
        c,
        sigma,
        weights_np,
        nodes_np,
    )

    boundary_fraction = float(np.mean(np.asarray(jax.device_get(jnp.asarray(boundary_hits))))) if boundary_hits else 0.0
    return PolicySolution(
        llr_grid=np.asarray(jax.device_get(grid)),
        p_right_grid=np.asarray(jax.device_get(p_right)),
        q_choose_a=np.asarray(jax.device_get(q_a_arr)),
        q_choose_b=np.asarray(jax.device_get(q_b_arr)),
        q_continue=np.asarray(jax.device_get(q_c_arr)),
        value=np.asarray(jax.device_get(value_arr)),
        optimal_action=np.asarray(jax.device_get(action_arr)),
        boundary_fraction_quadrature=boundary_fraction,
        initial_value_after_first_observation=float(init_value),
    )


def initial_value_after_first_observation(
    llr_grid: np.ndarray,
    value_after_first: np.ndarray,
    coherence: float,
    sigma: float,
    weights: np.ndarray,
    nodes: np.ndarray,
) -> float:
    scale = 0.0 if coherence == 0.0 else 2.0 * coherence / (sigma * sigma)
    obs_pos = coherence + math.sqrt(2.0) * sigma * nodes
    obs_neg = -coherence + math.sqrt(2.0) * sigma * nodes
    l_pos = scale * obs_pos
    l_neg = scale * obs_neg
    v_pos = np.interp(l_pos, llr_grid, value_after_first, left=value_after_first[0], right=value_after_first[-1])
    v_neg = np.interp(l_neg, llr_grid, value_after_first, left=value_after_first[0], right=value_after_first[-1])
    return float(0.5 * np.sum((weights / math.sqrt(math.pi)) * v_pos) + 0.5 * np.sum((weights / math.sqrt(math.pi)) * v_neg))


def interpolate_policy(solution: PolicySolution, timestep: int, llr: float) -> tuple[float, float, float, float]:
    idx = int(timestep) - 1
    q_a = float(np.interp(llr, solution.llr_grid, solution.q_choose_a[idx], left=solution.q_choose_a[idx, 0], right=solution.q_choose_a[idx, -1]))
    q_b = float(np.interp(llr, solution.llr_grid, solution.q_choose_b[idx], left=solution.q_choose_b[idx, 0], right=solution.q_choose_b[idx, -1]))
    q_c = float(np.interp(llr, solution.llr_grid, solution.q_continue[idx], left=solution.q_continue[idx, 0], right=solution.q_continue[idx, -1]))
    value = float(np.interp(llr, solution.llr_grid, solution.value[idx], left=solution.value[idx, 0], right=solution.value[idx, -1]))
    return q_a, q_b, q_c, value


def simulate_optimal_policy(config: OptimalConfig, solution: PolicySolution) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(int(config.seed))
    T = int(config.max_observations_before_stop)
    c = float(config.coherence)
    sigma = float(config.observation_noise_std)
    llr_scale = 0.0 if c == 0.0 else 2.0 * c / (sigma * sigma)
    n = int(config.num_trials)
    correct_choice = rng.choice(np.asarray([-1, 1], dtype=int), size=n)
    correct_action = np.where(correct_choice < 0, CHOOSE_A, CHOOSE_B)
    means = correct_choice.astype(float)[:, None] * c
    observations = rng.normal(loc=means, scale=sigma, size=(n, T))

    rows: list[dict] = []
    boundary_threshold = config.boundary_fraction * float(np.max(np.abs(solution.llr_grid)))
    near_boundary_count = 0
    state_count = 0
    for trial in range(n):
        llr = 0.0
        cumulative = 0.0
        total_opp = 0.0
        terminal_action = CHOOSE_A
        terminal_reward = 0.0
        num_observations = T
        trajectory: dict[str, object] = {}
        done = False
        for t in range(1, T + 1):
            obs = float(observations[trial, t - 1])
            cumulative += obs
            llr += llr_scale * obs
            p_right = float(1.0 / (1.0 + math.exp(-max(min(llr, 700.0), -700.0))))
            q_a, q_b, q_c, value = interpolate_policy(solution, t, llr)
            state_count += 1
            if abs(llr) >= boundary_threshold:
                near_boundary_count += 1
            if bool(config.choice_at_end_only) and t < T:
                action = CONTINUE
                probs = np.zeros(NUM_ACTIONS, dtype=float)
                probs[CONTINUE] = 1.0
            elif t == T:
                action, probs = terminal_policy_from_q(q_a, q_b, config.tie_tolerance, rng)
            else:
                best_stop = max(q_a, q_b)
                if q_c > best_stop + config.tie_tolerance:
                    action = CONTINUE
                    probs = np.zeros(NUM_ACTIONS, dtype=float)
                    probs[CONTINUE] = 1.0
                else:
                    action, probs = terminal_policy_from_q(q_a, q_b, config.tie_tolerance, rng)
            trajectory[f"valid_t{t}"] = True
            trajectory[f"evidence_sample_t{t}"] = obs
            trajectory[f"cumulative_evidence_t{t}"] = cumulative
            trajectory[f"oracle_cumulative_llr_t{t}"] = llr
            trajectory[f"llr_t{t}"] = llr
            trajectory[f"p_right_t{t}"] = p_right
            trajectory[f"action_t{t}"] = int(action)
            trajectory[f"continue_t{t}"] = bool(action == CONTINUE)
            trajectory[f"stop_t{t}"] = bool(action != CONTINUE)
            trajectory[f"kl_d_t{t}"] = 0.0
            trajectory[f"kl_d_obs_t{t}"] = 0.0
            trajectory[f"memory_cost_t{t}"] = 0.0
            trajectory[f"opportunity_cost_t{t}"] = float(config.opportunity_cost if action == CONTINUE else 0.0)
            trajectory[f"policy_continue_t{t}"] = float(probs[CONTINUE])
            trajectory[f"policy_choose_a_t{t}"] = float(probs[CHOOSE_A])
            trajectory[f"policy_choose_b_t{t}"] = float(probs[CHOOSE_B])
            trajectory[f"q_choose_a_t{t}"] = q_a
            trajectory[f"q_choose_b_t{t}"] = q_b
            trajectory[f"q_continue_t{t}"] = q_c
            trajectory[f"value_pred_t{t}"] = value
            trajectory[f"action_policy_entropy_t{t}"] = action_entropy(probs)
            if action == CONTINUE:
                total_opp += float(config.opportunity_cost)
                continue
            terminal_action = int(action)
            terminal_reward = float(config.correct_reward if terminal_action == int(correct_action[trial]) else config.incorrect_reward)
            num_observations = t
            done = True
            break
        if not done:
            terminal_reward = float(config.correct_reward if terminal_action == int(correct_action[trial]) else config.incorrect_reward)
            num_observations = T

        for t in range(num_observations + 1, T + 1):
            trajectory[f"valid_t{t}"] = False
            trajectory[f"evidence_sample_t{t}"] = np.nan
            trajectory[f"cumulative_evidence_t{t}"] = np.nan
            trajectory[f"oracle_cumulative_llr_t{t}"] = np.nan
            trajectory[f"llr_t{t}"] = np.nan
            trajectory[f"p_right_t{t}"] = np.nan
            trajectory[f"action_t{t}"] = -1
            trajectory[f"continue_t{t}"] = False
            trajectory[f"stop_t{t}"] = False
            trajectory[f"kl_d_t{t}"] = 0.0
            trajectory[f"kl_d_obs_t{t}"] = 0.0
            trajectory[f"memory_cost_t{t}"] = 0.0
            trajectory[f"opportunity_cost_t{t}"] = 0.0
            trajectory[f"policy_continue_t{t}"] = np.nan
            trajectory[f"policy_choose_a_t{t}"] = np.nan
            trajectory[f"policy_choose_b_t{t}"] = np.nan
            trajectory[f"q_choose_a_t{t}"] = np.nan
            trajectory[f"q_choose_b_t{t}"] = np.nan
            trajectory[f"q_continue_t{t}"] = np.nan
            trajectory[f"value_pred_t{t}"] = np.nan
            trajectory[f"action_policy_entropy_t{t}"] = np.nan

        decision_cum = float(trajectory[f"cumulative_evidence_t{num_observations}"])
        decision_llr = float(trajectory[f"llr_t{num_observations}"])
        decision_p = float(trajectory[f"p_right_t{num_observations}"])
        row = {
            "graph": trial,
            "trial_id": trial,
            "seed": int(config.seed),
            "model_type": "optimal",
            "input_type": str(config.input_type),
            "opportunity_cost": float(config.opportunity_cost),
            "coherence": float(config.coherence),
            "signed_coherence": float(correct_choice[trial] * config.coherence),
            "observation_noise_std": float(config.observation_noise_std),
            "max_observations_before_stop": int(T),
            "choice_at_end_only": bool(config.choice_at_end_only),
            "correct_reward": float(config.correct_reward),
            "incorrect_reward": float(config.incorrect_reward),
            "correct_choice": int(correct_choice[trial]),
            "correct_action": int(correct_action[trial]),
            "terminal_action": int(terminal_action),
            "choose_right": int(terminal_action == CHOOSE_B),
            "correct": int(terminal_action == int(correct_action[trial])),
            "choose_correct": int(terminal_action == int(correct_action[trial])),
            "num_observations": int(num_observations),
            "num_continue_actions": int(max(num_observations - 1, 0)),
            "stopping_time": int(num_observations),
            "terminal_reward": terminal_reward,
            "total_opportunity_cost": total_opp,
            "total_memory_cost": 0.0,
            "total_kl_paid": 0.0,
            "total_reward": terminal_reward - total_opp,
            "cumulative_evidence_at_decision": decision_cum,
            "decision_cumulative_evidence": decision_cum,
            "decision_oracle_cumulative_llr": decision_llr,
            "final_llr": decision_llr,
            "final_p_right": decision_p,
        }
        row.update(trajectory)
        rows.append(row)

    diagnostics = pd.DataFrame(
        [
            {
                "near_or_outside_grid_fraction": near_boundary_count / max(state_count, 1),
                "quadrature_boundary_fraction": solution.boundary_fraction_quadrature,
                "state_count": state_count,
                "near_boundary_count": near_boundary_count,
            }
        ]
    )
    return rows, diagnostics


def boundary_rows(config: OptimalConfig, solution: PolicySolution) -> list[dict]:
    out = []
    tol = float(config.tie_tolerance)
    for idx in range(config.max_observations_before_stop):
        q_stop = np.maximum(solution.q_choose_a[idx], solution.q_choose_b[idx])
        cont = solution.q_continue[idx] > q_stop + tol
        if np.any(cont):
            lower = float(np.min(solution.llr_grid[cont]))
            upper = float(np.max(solution.llr_grid[cont]))
            lower_p = float(1.0 / (1.0 + np.exp(-lower)))
            upper_p = float(1.0 / (1.0 + np.exp(-upper)))
            has_cont = True
        else:
            lower = upper = lower_p = upper_p = np.nan
            has_cont = False
        out.append(
            {
                "timestep": idx + 1,
                "has_continuation_region": has_cont,
                "lower_llr_boundary": lower,
                "upper_llr_boundary": upper,
                "lower_p_right_boundary": lower_p,
                "upper_p_right_boundary": upper_p,
            }
        )
    return out


def save_policy_outputs(config: OptimalConfig, solution: PolicySolution) -> tuple[Path, Path, Path]:
    policy_dir = Path(config.policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    stem = policy_stem(config)
    npz_path = policy_dir / f"{stem}.npz"
    table_path = policy_dir / f"{stem}_policy_table.csv"
    boundary_path = policy_dir / f"{stem}_boundaries.csv"
    np.savez_compressed(
        npz_path,
        llr_grid=solution.llr_grid,
        p_right_grid=solution.p_right_grid,
        q_choose_a=solution.q_choose_a,
        q_choose_b=solution.q_choose_b,
        q_continue=solution.q_continue,
        value=solution.value,
        optimal_action=solution.optimal_action,
        boundary_fraction_quadrature=solution.boundary_fraction_quadrature,
        initial_value_after_first_observation=solution.initial_value_after_first_observation,
    )
    if config.save_policy_table:
        n_t = int(config.max_observations_before_stop)
        n_g = int(solution.llr_grid.shape[0])
        pd.DataFrame(
            {
                "timestep": np.repeat(np.arange(1, n_t + 1), n_g),
                "llr_grid": np.tile(solution.llr_grid, n_t),
                "p_right_grid": np.tile(solution.p_right_grid, n_t),
                "q_choose_a": solution.q_choose_a.reshape(-1),
                "q_choose_b": solution.q_choose_b.reshape(-1),
                "q_continue": solution.q_continue.reshape(-1),
                "value": solution.value.reshape(-1),
                "optimal_action": solution.optimal_action.reshape(-1),
            }
        ).to_csv(table_path, index=False)
    pd.DataFrame(boundary_rows(config, solution)).to_csv(boundary_path, index=False)
    return npz_path, table_path, boundary_path


def load_policy_npz(config: OptimalConfig) -> PolicySolution | None:
    path = Path(config.policy_dir) / f"{policy_stem(config)}.npz"
    if not config.reuse_policy or not path.exists():
        return None
    dat = np.load(path)
    return PolicySolution(
        llr_grid=dat["llr_grid"],
        p_right_grid=dat["p_right_grid"],
        q_choose_a=dat["q_choose_a"],
        q_choose_b=dat["q_choose_b"],
        q_continue=dat["q_continue"],
        value=dat["value"],
        optimal_action=dat["optimal_action"],
        boundary_fraction_quadrature=float(dat["boundary_fraction_quadrature"]),
        initial_value_after_first_observation=float(dat["initial_value_after_first_observation"]),
    )


def sem(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def save_condition_summary(rows: list[dict], config: OptimalConfig, solution: PolicySolution, diagnostics: pd.DataFrame, out_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    summary = (
        df.groupby(["model_type", "opportunity_cost", "coherence", "signed_coherence", "observation_noise_std", "choice_at_end_only"], dropna=False)
        .agg(
            n=("graph", "count"),
            p_choose_right=("choose_right", "mean"),
            p_choose_correct=("choose_correct", "mean"),
            se_choose_correct=("choose_correct", sem),
            mean_num_observations=("num_observations", "mean"),
            median_num_observations=("num_observations", "median"),
            se_num_observations=("num_observations", sem),
            mean_num_continue_actions=("num_continue_actions", "mean"),
            median_num_continue_actions=("num_continue_actions", "median"),
            mean_terminal_reward=("terminal_reward", "mean"),
            se_terminal_reward=("terminal_reward", sem),
            mean_opportunity_cost=("total_opportunity_cost", "mean"),
            se_opportunity_cost=("total_opportunity_cost", sem),
            mean_total_return=("total_reward", "mean"),
            se_total_return=("total_reward", sem),
            mean_decision_cumulative_evidence=("decision_cumulative_evidence", "mean"),
            mean_final_llr=("final_llr", "mean"),
            p_choose_a=("terminal_action", lambda x: np.mean(np.asarray(x) == CHOOSE_A)),
            p_choose_b=("terminal_action", lambda x: np.mean(np.asarray(x) == CHOOSE_B)),
        )
        .reset_index()
    )
    overall_return = float(df["total_reward"].mean())
    summary["estimated_initial_optimal_value"] = solution.initial_value_after_first_observation
    summary["empirical_mean_simulated_return"] = overall_return
    summary["dp_minus_empirical_return"] = solution.initial_value_after_first_observation - overall_return
    summary["near_or_outside_grid_fraction"] = float(diagnostics["near_or_outside_grid_fraction"].iloc[0])
    summary["quadrature_boundary_fraction"] = float(diagnostics["quadrature_boundary_fraction"].iloc[0])
    summary.to_csv(out_path, index=False)


def save_outputs(config: OptimalConfig, solution: PolicySolution, rows: list[dict], diagnostics: pd.DataFrame) -> tuple[Path, Path, Path]:
    sim_dir = Path(config.sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    stem = simulation_stem(config)
    trial_path = sim_dir / f"{stem}_{config.input_type}.csv"
    summary_path = sim_dir / f"{stem}_{config.input_type}_summary.csv"
    diagnostics_path = sim_dir / f"{stem}_{config.input_type}_diagnostics.csv"
    pd.DataFrame(rows).to_csv(trial_path, index=False)
    save_condition_summary(rows, config, solution, diagnostics, summary_path)
    diagnostics.to_csv(diagnostics_path, index=False)
    return trial_path, summary_path, diagnostics_path


def make_config_from_args(args, opportunity_cost: float, coherence: float, obsstd: float, seed: int) -> OptimalConfig:
    return OptimalConfig(
        opportunity_cost=float(opportunity_cost),
        coherence=float(coherence),
        observation_noise_std=float(obsstd),
        max_observations_before_stop=max(int(args.max_observations_before_stop), 1),
        num_trials=int(args.num_trials),
        seed=int(seed),
        sim_dir=str(args.sim_dir),
        policy_dir=str(args.policy_dir),
        input_type=str(args.input_type),
        correct_reward=float(args.correct_reward),
        incorrect_reward=float(args.incorrect_reward),
        llr_grid_size=int(args.llr_grid_size),
        llr_grid_max=None if args.llr_grid_max is None else float(args.llr_grid_max),
        quadrature_order=int(args.quadrature_order),
        tie_tolerance=float(args.tie_tolerance),
        boundary_fraction=float(args.boundary_fraction),
        enable_x64=bool(args.enable_x64),
        coherence_known_to_agent=bool(args.coherence_known_to_agent),
        reuse_policy=bool(args.reuse_policy),
        save_policy_table=not bool(args.no_policy_table),
        choice_at_end_only=bool(args.choice_at_end_only),
    )


def run_one(config: OptimalConfig) -> None:
    print(
        "Optimal evidence combo: "
        f"opp={config.opportunity_cost:g}, coherence={config.coherence:g}, "
        f"obsstd={config.observation_noise_std:g}, seed={config.seed}, "
        f"maxobs={config.max_observations_before_stop}, n={config.num_trials}, "
        f"choice_at_end_only={config.choice_at_end_only}",
        flush=True,
    )
    stem = policy_stem(config)
    solution = _POLICY_CACHE.get(stem)
    if solution is not None:
        print(f"Reusing in-memory optimal policy for {stem}", flush=True)
    else:
        solution = load_policy_npz(config)
    if solution is None:
        solution = solve_policy(config)
        npz_path, table_path, boundary_path = save_policy_outputs(config, solution)
        _POLICY_CACHE[stem] = solution
        print(f"Saved optimal policy cache to: {npz_path}", flush=True)
        if config.save_policy_table:
            print(f"Saved optimal policy table to: {table_path}", flush=True)
        print(f"Saved optimal boundaries to: {boundary_path}", flush=True)
    else:
        _POLICY_CACHE[stem] = solution
        print(f"Loaded/reused cached optimal policy: {Path(config.policy_dir) / (stem + '.npz')}", flush=True)
    if solution.boundary_fraction_quadrature > 1e-4:
        print(
            "Warning: some quadrature evaluation points reached the LLR grid boundary "
            f"(fraction={solution.boundary_fraction_quadrature:.6g}). Consider increasing --llr-grid-max.",
            flush=True,
        )
    rows, diagnostics = simulate_optimal_policy(config, solution)
    trial_path, summary_path, diagnostics_path = save_outputs(config, solution, rows, diagnostics)
    grid_frac = float(diagnostics["near_or_outside_grid_fraction"].iloc[0])
    if grid_frac > 1e-4:
        print(
            "Warning: simulation states approached the LLR grid boundary "
            f"(fraction={grid_frac:.6g}). Consider increasing --llr-grid-max.",
            flush=True,
        )
    mean_return = float(pd.DataFrame(rows)["total_reward"].mean())
    print(f"Saved optimal evidence trials to: {trial_path}", flush=True)
    print(f"Saved optimal evidence summary to: {summary_path}", flush=True)
    print(f"Saved optimal evidence diagnostics to: {diagnostics_path}", flush=True)
    print(
        f"DP initial value after first observation={solution.initial_value_after_first_observation:.6f}; "
        f"empirical return={mean_return:.6f}; difference={solution.initial_value_after_first_observation - mean_return:.6f}",
        flush=True,
    )


def run_validation_tests() -> None:
    print("Running evidence_accumulation_optimal self-tests...", flush=True)
    c = 0.2
    sigma = 0.5
    obs = 0.37
    l_prev = -0.4
    l_update = l_prev + (2.0 * c / (sigma * sigma)) * obs
    direct = l_prev + float(gaussian_logpdf(jnp.asarray(obs), c, sigma) - gaussian_logpdf(jnp.asarray(obs), -c, sigma))
    assert abs(l_update - direct) < 1e-6, (l_update, direct)

    p0 = float(stable_sigmoid(jnp.asarray(0.0)))
    assert abs((1.0 - p0) - 0.5) < 1e-12
    assert abs(p0 - 0.5) < 1e-12

    cfg = OptimalConfig(
        opportunity_cost=0.02,
        coherence=0.2,
        observation_noise_std=0.5,
        max_observations_before_stop=5,
        num_trials=2000,
        seed=0,
        sim_dir="/tmp",
        policy_dir="/tmp",
        input_type="evidence",
        correct_reward=1.0,
        incorrect_reward=0.0,
        llr_grid_size=1001,
        llr_grid_max=8.0,
        quadrature_order=21,
        tie_tolerance=1e-10,
        boundary_fraction=0.98,
        enable_x64=True,
        coherence_known_to_agent=True,
        reuse_policy=False,
        save_policy_table=False,
    )
    sol = solve_policy(cfg)
    assert np.max(np.abs(sol.value - sol.value[:, ::-1])) < 2e-4
    assert np.max(np.abs(sol.q_choose_a - sol.q_choose_b[:, ::-1])) < 2e-4
    bounds = pd.DataFrame(boundary_rows(cfg, sol))
    finite = bounds["has_continuation_region"].to_numpy(dtype=bool)
    if np.any(finite):
        assert np.nanmax(np.abs(bounds.loc[finite, "lower_llr_boundary"].to_numpy() + bounds.loc[finite, "upper_llr_boundary"].to_numpy())) < 0.05

    zero_cfg = OptimalConfig(**{**cfg.__dict__, "coherence": 0.0, "opportunity_cost": 0.01})
    zero_sol = solve_policy(zero_cfg)
    assert int(zero_sol.optimal_action[0, zero_cfg.llr_grid_size // 2]) != CONTINUE
    zero_rows, _ = simulate_optimal_policy(zero_cfg, zero_sol)
    zero_df = pd.DataFrame(zero_rows)
    assert abs(float(zero_df["choose_correct"].mean()) - 0.5) < 0.05
    assert float(zero_df["num_observations"].mean()) < 1.05

    free_cfg = OptimalConfig(**{**cfg.__dict__, "opportunity_cost": 0.0})
    free_sol = solve_policy(free_cfg)
    center = free_cfg.llr_grid_size // 2
    # With finite precision and stop-preferring ties, exact equality may stop,
    # but continuation cannot have lower value than stopping at the center.
    assert free_sol.q_continue[0, center] >= max(free_sol.q_choose_a[0, center], free_sol.q_choose_b[0, center]) - 1e-5

    high_cost = solve_policy(OptimalConfig(**{**cfg.__dict__, "opportunity_cost": 0.2}))
    low_cont = np.mean(sol.optimal_action == CONTINUE)
    high_cont = np.mean(high_cost.optimal_action == CONTINUE)
    assert high_cont <= low_cont + 1e-6

    # Bellman consistency at sampled grid points.
    for t in range(cfg.max_observations_before_stop):
        idx = np.linspace(0, cfg.llr_grid_size - 1, 25, dtype=int)
        lhs = sol.value[t, idx]
        rhs = np.maximum.reduce([sol.q_choose_a[t, idx], sol.q_choose_b[t, idx], sol.q_continue[t, idx]])
        assert np.max(np.abs(lhs - rhs)) < 2e-5

    small_cfg = OptimalConfig(**{**cfg.__dict__, "num_trials": 20})
    small_sol = solve_policy(small_cfg)
    rows, _ = simulate_optimal_policy(small_cfg, small_sol)
    assert len(rows) == 20
    jit_sigmoid = jax.jit(stable_sigmoid)
    assert np.all(np.isfinite(np.asarray(jit_sigmoid(jnp.asarray([-1.0, 0.0, 1.0])))))
    print("All evidence_accumulation_optimal self-tests passed.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Bayes-optimal dynamic-program baseline for the JAX evidence accumulation task.")
    parser.add_argument("--opportunity-costs", "--opportunity-cost", "--opportunity", nargs="*", default=["0.0"])
    parser.add_argument("--coherence-values", "--coherences", nargs="*", default=["0,0.05,0.1,0.2,0.4,0.8"])
    parser.add_argument("--observation-noise-std", "--observation-noise-std-list", "--obsstd", "--sigma", nargs="*", default=["1.0"])
    parser.add_argument("--max-observations-before-stop", "--maxobs", type=int, default=int(os.environ.get("MAX_OBSERVATIONS_BEFORE_STOP", "10")))
    parser.add_argument("--num-trials", "--n-sim-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--sim-dir", "--output-dir", default="outputs/jax_simulations_evi")
    parser.add_argument("--policy-dir", default="outputs/evidence_optimal_policies")
    parser.add_argument("--input-type", default="evidence")
    parser.add_argument("--correct-reward", type=float, default=1.0)
    parser.add_argument("--incorrect-reward", type=float, default=0.0)
    parser.add_argument("--llr-grid-size", type=int, default=12001)
    parser.add_argument("--llr-grid-max", type=float, default=None)
    parser.add_argument("--quadrature-order", type=int, default=41)
    parser.add_argument("--tie-tolerance", type=float, default=1e-10)
    parser.add_argument("--boundary-fraction", type=float, default=0.98)
    parser.add_argument("--enable-x64", action="store_true", default=True)
    parser.add_argument("--disable-x64", dest="enable_x64", action="store_false")
    parser.add_argument("--coherence-known-to-agent", action="store_true", default=True)
    parser.add_argument("--coherence-hidden", dest="coherence_known_to_agent", action="store_false")
    parser.add_argument("--reuse-policy", action="store_true")
    parser.add_argument("--no-policy-table", action="store_true", help="Still saves the compressed policy cache and boundary table.")
    parser.add_argument(
        "--choice-at-end-only",
        "--observer-only",
        "--observer-end-choice",
        action="store_true",
        default=os.environ.get("CHOICE_AT_END_ONLY", "").strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "Evaluate a fixed-duration observer baseline: continue is forced until "
            "--max-observations-before-stop observations have been incorporated, "
            "then the observer chooses A/B. Enabled runs add _observer_endchoice "
            "to policy and simulation filenames."
        ),
    )
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_validation_tests()
        return
    if not args.coherence_known_to_agent:
        raise NotImplementedError(
            "Hidden-coherence optimal evidence accumulation is not implemented. "
            "Use --coherence-known-to-agent, or implement a joint posterior over choice and coherence."
        )
    opportunities = parse_float_values(args.opportunity_costs)
    coherences = parse_float_values(args.coherence_values)
    obsstds = parse_float_values(args.observation_noise_std)
    seeds = parse_float_values(args.seeds) if args.seeds is not None else [float(args.seed)]
    if not opportunities:
        raise ValueError("--opportunity-costs must contain at least one value.")
    if not coherences:
        raise ValueError("--coherence-values must contain at least one value.")
    if any(c < 0 for c in coherences):
        raise ValueError("--coherence-values are nonnegative magnitudes.")
    if not obsstds:
        raise ValueError("--observation-noise-std must contain at least one value.")
    if any(s <= 0 for s in obsstds):
        raise ValueError("--observation-noise-std values must be positive.")
    combos = list(itertools.product(opportunities, coherences, obsstds, [int(s) for s in seeds]))
    print(f"Running {len(combos)} optimal evidence parameter combination(s).", flush=True)
    for opportunity, coherence, obsstd, seed in combos:
        config = make_config_from_args(args, opportunity, coherence, obsstd, seed)
        run_one(config)


if __name__ == "__main__":
    main()
