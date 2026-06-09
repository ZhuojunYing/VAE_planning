#!/usr/bin/env python
"""Analyze whether 2D latent posterior geometry tracks planning variables.

Interpretation notes:
radius_mu measures absolute posterior mean position relative to the latent
coordinate origin. delta_radius_mu measures how far the posterior mean moved
relative to the learned prior at the same timestep. posterior_prior_kl measures
the information/update magnitude relative to the learned prior. If
delta_angle_mu or delta_radius_mu improves prediction beyond posterior_prior_kl,
then the direction of the posterior update carries task-relevant information
beyond how much information was encoded. If raw posterior radius performs better
than prior-relative radius, then the absolute posterior position may be more
informative than update-from-prior direction.

The scatter plots by timestep test whether path identity appears as different
clusters, axes, or angular sectors in the 2D posterior mean space. These plots
are used to check whether different timesteps use different axes or angular
sectors in the 2D latent posterior mean space. If timestep 1 separates paths
along z_mu_0 but timestep 2 separates paths along z_mu_1, this suggests the path
code may rotate or reorganize over time.

The per-seed heatmaps test whether the apparent predictor identity is stable
across random seeds. If z_mu_0 predicts current best path at timestep 1 in one
seed but z_mu_1 predicts it in another seed, this suggests coordinate identity is
arbitrary and may reflect latent rotation or permutation. If angle-based
predictors remain predictive across seeds even when coordinate-specific
predictors swap, the more robust interpretation is that latent direction carries
path information. Because latent dimensions can rotate or permute across
independently trained models, claims about fixed dimensions should only be made
if the pattern is consistent across seeds or after alignment.

The previous delta variables measure posterior-prior displacement at a single
timestep. The temporal variables measure actual posterior movement from timestep
1 to timestep 2. Path switch is naturally a transition-level event, so temporal
direction from t1 to t2 is the more direct predictor.

The Gaussian/Fisher half-plane coordinates are x = mu / sqrt(2), y = sigma for
each univariate latent posterior dimension. A diagonal 2D latent posterior is a
product of two Gaussian half-planes, not one single 2D half-plane. Therefore,
visualizations are created separately for latent dimension 0 and latent
dimension 1. If temporal half-plane direction predicts path switch, this
suggests that decision-relevant switches correspond to specific
posterior-distribution movement directions. If only temporal radius/magnitude
predicts switch, the switch may be reflected mainly in update size rather than
update direction. Do not interpret these plots as proof that the latent space is
globally hyperbolic; they are visualizations of posterior distributions in their
Gaussian/Fisher geometry.

The reward-encoding analysis is designed to be robust to latent-axis flips,
rotations, and permutations across random seeds. Coordinate-specific slopes may
change sign across seeds, so the main question is whether the 2D posterior
vector as a whole contains reward information. The t1 analyses test whether the
first posterior update encodes the first observed reward. The t2 analyses test
whether the second posterior update encodes the current reward in a way that
depends on the first observed reward. A positive delta_R2_interaction means that
the R1 x R2 interaction explains additional variance in the posterior vector
beyond the main effects of R1 and R2. If the interaction improves prediction of
the t2 posterior vector across seeds, this supports the interpretation that the
second update is contextualized by the first observed reward rather than simply
encoding R2 independently. Because q(z_t) reconstructs h_{t+1}, which is passed
to the LSTM at the next timestep, t2 posterior analyses should only be
interpreted as task-relevant when the reconstructed h_{t+1} is used downstream.
Do not interpret this analysis as proving a fixed latent dimension encodes
reward; interpret it as evidence that the posterior update geometry contains
reward-context information. R2_minus_R1 should be tested as a separate
theoretically motivated predictor, not simply added to a model that already
includes R1 and R2, because it is linearly dependent on them. Reward difference
R2 - R1 tests relative value or contrast between current and previous evidence.
Absolute reward difference abs(R2 - R1) tests discrepancy or update magnitude
regardless of direction. Path-value margin tests a more direct choice-certainty
interpretation. The interaction model R1 + R2 + R1:R2 tests whether the
encoding of the current reward depends on the previous reward. The strongest
evidence for a relative-value interpretation would be reward_difference
performing as well as or better than R2 alone and approaching the performance of
the full interaction model. The strongest evidence for a choice-certainty
interpretation would be current_best_path_margin_t2 predicting the posterior
vector well, especially if it predicts sigma or posterior uncertainty.

The geometry-meaning analyses test whether geometric summaries of the posterior
have interpretable task meanings. Posterior angle is hypothesized to track path
identity or direction of relative evidence. Posterior radius is hypothesized to
track value magnitude, commitment, or distance from a neutral latent state.
Posterior sigma is hypothesized to track uncertainty, precision, or information
allocation. Posterior-prior delta angle/radius is hypothesized to track the
direction and magnitude of the update from the learned prior. Posterior-prior KL
is hypothesized to track the amount of information encoded. Within-path analyses
test whether value or certainty is represented beyond path identity. Because
latent axes can rotate, flip, or permute across seeds, do not make strong claims
about fixed dimensions unless they are consistent across seeds; prefer claims
about feature families such as angle, radius, sigma, and posterior-prior update.

The existing mu_radius and mu_angle features are Euclidean polar coordinates of
the 2D posterior mean vector [mu_0, mu_1]. The half-plane reward-geometry
analyses instead treat each univariate posterior dimension separately as a
Gaussian/Fisher half-plane point (mu_l / sqrt(2), sigma_l). Poincare disk
features map each univariate Gaussian posterior dimension from the half-plane to
the disk using w = (z - i) / (z + i). Angle variables are circular, so raw angle
correlations should be treated cautiously; the preferred angular regression uses
sin(angle) and cos(angle). The t1 analysis asks whether the first observed
reward is organized along radius or angle in the Gaussian/Fisher geometry. The
t2-by-R1 analysis asks whether the relationship between the second observed
reward and Gaussian/Fisher geometry depends on the first observed reward. A sign
flip in the reward_t2 relationship across reward_t1 groups supports the idea
that the second update is contextualized by the first observed reward. Do not
interpret half-plane radius as a true hyperbolic geodesic distance unless the
origin/reference is explicitly defined. The Poincare disk radius is closer to
distance from the canonical reference distribution but should still be
interpreted cautiously.

The latent 2D aggregate density plots visualize the product posterior
q(z_0, z_1) = q(z_0) q(z_1) in ordinary 2D latent sample space. They are
different from Gaussian/Fisher half-plane plots. The half-plane plots visualize
each univariate posterior component as a point (mu_l / sqrt(2), sigma_l), while
the 2D density plots visualize Normal([mu_0, mu_1], diag([sigma_0^2,
sigma_1^2])) and average those trial-level densities. For t1, these plots test
whether the first observed reward organizes the 2D posterior update
distribution. For t2, conditioning on reward_t1 and coloring by reward_t2 tests
whether the second observed reward produces distinct posterior update densities
given the first observation. Do not interpret these density plots as direct
hyperbolic half-plane plots.

Learned-prior-relative Gaussian/Fisher analyses use the learned prior at each
timestep as the reference distribution. For each univariate latent dimension,
posterior and prior Gaussians are represented in half-plane coordinates
x = mu / sqrt(2), y = sigma. The delta_halfplane_* variables are exploratory
chart-coordinate posterior-prior displacements; they are not exact Riemannian
log-map vectors. The prior_relative_fisher_distance variables use the
univariate Gaussian Fisher distance d_F = sqrt(2) * arcosh(1 + ||p-q||^2 /
(2 y_p y_q)) in this half-plane convention. posterior_prior_kl remains a
separate information/update magnitude feature and should not be treated as
identical to Fisher distance.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_DIR = REPO_ROOT / "model"
for path in (str(REPO_ROOT), str(MODEL_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from latent_angle_utils import (  # noqa: E402
    add_angle_features,
    compute_current_best_path_variables,
    compute_path_values,
    find_checkpoint,
    make_model_config,
    sample_rewards,
    safe_json_dumps,
)


BASE_OUTPUT_COUNT = 47
TERMINAL_PATH_OUTPUT_INDEX = 28
OBSERVATION_KL_OUTPUT_INDEX = 29
MM_PER_INCH = 25.4
PANEL_SIZE_IN = 60.0 / MM_PER_INCH
PANEL_FONT_SIZE = 7
COLORBAR_WIDTH_IN = 0.34
TITLE_EXTRA_HEIGHT_IN = 0.22


def panel_figsize(ncols: int = 1, nrows: int = 1, *, colorbar: bool = False, title: bool = False) -> Tuple[float, float]:
    width = PANEL_SIZE_IN * max(int(ncols), 1)
    height = PANEL_SIZE_IN * max(int(nrows), 1)
    if colorbar:
        width += COLORBAR_WIDTH_IN
    if title:
        height += TITLE_EXTRA_HEIGHT_IN
    return width, height


def apply_7pt_plot_style(plt):
    plt.rcParams.update({
        "font.size": PANEL_FONT_SIZE,
        "axes.titlesize": PANEL_FONT_SIZE,
        "axes.labelsize": PANEL_FONT_SIZE,
        "xtick.labelsize": PANEL_FONT_SIZE,
        "ytick.labelsize": PANEL_FONT_SIZE,
        "legend.fontsize": PANEL_FONT_SIZE,
        "legend.title_fontsize": PANEL_FONT_SIZE,
        "figure.titlesize": PANEL_FONT_SIZE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def add_first_row_colorbar(fig, axes, mappable, label: str, *, width: float = 0.025, pad: float = 0.025):
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 0:
        row_axes = [axes_array.item()]
    elif axes_array.ndim == 1:
        row_axes = list(axes_array)
    else:
        row_axes = list(axes_array[0, :])
    visible_axes = [ax for ax in row_axes if ax.get_visible()]
    if not visible_axes:
        return None
    fig.canvas.draw()
    bbox = visible_axes[-1].get_position()
    cax = fig.add_axes([bbox.x1 + pad, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=PANEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=PANEL_FONT_SIZE)
    return cbar


FEATURE_SETS = {
    "timestep_only": ["timestep"],
    "raw_angle_only": ["sin_angle_mu", "cos_angle_mu"],
    "raw_angle_plus_raw_radius": ["sin_angle_mu", "cos_angle_mu", "radius_mu"],
    "prior_delta_angle_only": ["sin_delta_angle_mu", "cos_delta_angle_mu"],
    "prior_delta_angle_plus_delta_radius": [
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "delta_radius_mu",
    ],
    "kl_only": ["posterior_prior_kl"],
    "kl_plus_prior_delta": [
        "posterior_prior_kl",
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "delta_radius_mu",
    ],
    "raw_plus_prior_delta": [
        "sin_angle_mu",
        "cos_angle_mu",
        "radius_mu",
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "delta_radius_mu",
    ],
    "full_latent_summary": [
        "z_mu_0",
        "z_mu_1",
        "z_logvar_0",
        "z_logvar_1",
        "radius_mu",
        "sin_angle_mu",
        "cos_angle_mu",
        "prior_mu_0",
        "prior_mu_1",
        "prior_logvar_0",
        "prior_logvar_1",
        "delta_radius_mu",
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "posterior_prior_kl",
    ],
}

PRIOR_DIAGNOSTIC_FEATURE_SETS = {
    "kl_only": ["posterior_prior_kl"],
    "prior_delta_angle_only": ["sin_delta_angle_mu", "cos_delta_angle_mu"],
    "prior_delta_angle_plus_delta_radius": [
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "delta_radius_mu",
    ],
    "kl_plus_prior_delta": [
        "posterior_prior_kl",
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "delta_radius_mu",
    ],
    "raw_angle_plus_raw_radius": ["sin_angle_mu", "cos_angle_mu", "radius_mu"],
    "raw_plus_prior_delta": [
        "sin_angle_mu",
        "cos_angle_mu",
        "radius_mu",
        "sin_delta_angle_mu",
        "cos_delta_angle_mu",
        "delta_radius_mu",
    ],
    "full_latent_summary": FEATURE_SETS["full_latent_summary"],
}

RADIUS_COMPARISON_FEATURE_SETS = [
    "raw_angle_plus_raw_radius",
    "prior_delta_angle_plus_delta_radius",
    "kl_plus_prior_delta",
    "raw_plus_prior_delta",
    "full_latent_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--betas", nargs="+", type=float, required=True)
    parser.add_argument("--lambdas", dest="lambda_values", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--rnn-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[32])
    parser.add_argument(
        "--opportunity-costs",
        "--opportunity-cost",
        dest="opportunity_costs",
        nargs="+",
        type=float,
        default=[0.0],
        help="Opportunity cost values to match in trained checkpoints. Default: 0.0.",
    )
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", default="analysis_outputs/latent_angle_planning")
    parser.add_argument("--checkpoint-root", default="outputs/models")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="legacy")
    parser.add_argument("--input-type", default="uniform", choices=["uniform", "binary"])
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae", choices=["vae", "rnn"])
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or TensorFlow device string.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-within-path-n", type=int, default=50)
    parser.add_argument("--min-reward-group-n", type=int, default=10)
    parser.add_argument(
        "--latent-density-grid-n",
        type=int,
        default=150,
        help="Grid resolution for 2D posterior aggregate density plots. Default: 150.",
    )
    parser.add_argument("--analysis-seed-offset", type=int, default=100_000)
    parser.add_argument(
        "--plot-t1-halfplane-by-observed-value",
        action="store_true",
        help=(
            "Also plot timestep-1 Gaussian half-plane panels for latent dims 0/1, "
            "colored by current observed path value and split by the value observed at timestep 1."
        ),
    )
    parser.add_argument(
        "--plot-t1-t2-halfplane-by-t1-observed-value",
        action="store_true",
        help=(
            "Also plot Gaussian half-plane panels for timesteps 1 and 2, colored by "
            "current observed path value and split by the value observed at timestep 1."
        ),
    )
    return parser.parse_args()


def configure_tensorflow(device: str):
    if device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    if device.lower() in ("cuda", "gpu"):
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    return tf


def import_model_modules():
    helper = importlib.import_module("helper")
    simulate = importlib.import_module("simulate")
    return helper, simulate


def tensor_to_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def build_and_load_model(tf, simulate_module, config, alpha, beta, lambda_value, opportunity_cost, checkpoint_path):
    model = simulate_module.build_model(config, alpha, beta, lambda_value, opportunity_cost)
    dummy = tf.zeros((1, config.time_steps, 1), dtype=tf.float32)
    try:
        _ = model(
            dummy,
            training=False,
            compute_losses=False,
            expansion_epsilon=0.0,
            return_analysis_tensors=True,
            analysis_use_posterior_mean=True,
        )
    except TypeError:
        _ = model(
            dummy,
            training=False,
            compute_losses=False,
            expansion_epsilon=0.0,
        )
    if hasattr(model, "build_target_critic"):
        model.build_target_critic()
    model.load_weights(str(checkpoint_path))
    return model


def posterior_prior_kl(z_mu, z_logvar, prior_mu, prior_logvar):
    z_logvar = np.clip(np.asarray(z_logvar, dtype=float), -10.0, 10.0)
    prior_logvar = np.clip(np.asarray(prior_logvar, dtype=float), -10.0, 10.0)
    z_var = np.exp(z_logvar) + 1e-6
    prior_var = np.exp(prior_logvar) + 1e-6
    return -0.5 * np.mean(
        1.0 + z_logvar - prior_logvar - ((z_mu - prior_mu) ** 2 + z_var) / prior_var,
        axis=-1,
    )


def gaussian_fisher_distance_from_halfplane(x1, y1, x2, y2, eps: float = 1e-8):
    """Univariate Gaussian Fisher distance via x=mu/sqrt(2), y=sigma half-plane."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y1 = np.maximum(np.asarray(y1, dtype=float), eps)
    y2 = np.maximum(np.asarray(y2, dtype=float), eps)
    arg = 1.0 + ((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2.0 * y1 * y2)
    arg = np.maximum(arg, 1.0)
    dist = math.sqrt(2.0) * np.arccosh(arg)
    dist = np.where(np.isfinite(dist) & (dist >= 0.0), dist, np.nan)
    return dist


def halfplane_to_canonical_disk(x, y, eps: float = 1e-8):
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), eps)
    z = x + 1j * y
    w = (z - 1j) / (z + 1j)
    r = np.abs(w)
    too_large = r >= 1.0
    if np.any(too_large):
        w = np.where(too_large, w / np.maximum(r, eps) * (1.0 - eps), w)
    return w


def disk_isometry_center_prior(post_w, prior_w, eps: float = 1e-8):
    post_w = np.asarray(post_w, dtype=np.complex128)
    prior_w = np.asarray(prior_w, dtype=np.complex128)
    denom = 1.0 - np.conj(prior_w) * post_w
    centered = np.where(
        np.abs(denom) > eps,
        (post_w - prior_w) / denom,
        np.nan + 1j * np.nan,
    )
    r = np.abs(centered)
    too_large = r >= 1.0
    if np.any(too_large):
        centered = np.where(too_large, centered / np.maximum(r, eps) * (1.0 - eps), centered)
    return centered


def run_model_trials(tf, model, rewards, batch_size: int):
    outputs_by_batch = []
    n_trials = rewards.shape[0]
    for start in range(0, n_trials, batch_size):
        batch_rewards = rewards[start : start + batch_size]
        tensor = tf.constant(batch_rewards[:, :, None], dtype=tf.float32)
        try:
            outputs = model(
                tensor,
                training=False,
                compute_losses=False,
                expansion_epsilon=0.0,
                return_analysis_tensors=True,
                analysis_use_posterior_mean=True,
            )
        except TypeError:
            outputs = model(
                tensor,
                training=False,
                compute_losses=False,
                expansion_epsilon=0.0,
            )
        outputs_by_batch.append(tuple(tensor_to_numpy(item) for item in outputs))

    merged = []
    n_outputs = len(outputs_by_batch[0])
    for output_i in range(n_outputs):
        parts = [batch[output_i] for batch in outputs_by_batch]
        arr0 = np.asarray(parts[0])
        if arr0.shape and arr0.shape[0] == parts[0].shape[0]:
            merged.append(np.concatenate(parts, axis=0))
        else:
            merged.append(parts[0])
    return tuple(merged)


def observed_node_sequence(node_selections: np.ndarray, stop_decisions: np.ndarray, time_steps: int) -> np.ndarray:
    node_selections = np.asarray(node_selections, dtype=int)
    stop_decisions = np.asarray(stop_decisions, dtype=bool)
    observed = np.full(node_selections.shape, np.nan, dtype=float)
    stopped = np.zeros(node_selections.shape[0], dtype=bool)
    for t in range(node_selections.shape[1]):
        selected = node_selections[:, t]
        is_observe = (~stopped) & (selected < time_steps) & (~stop_decisions[:, t])
        observed[is_observe, t] = selected[is_observe] + 1
        stopped = stopped | stop_decisions[:, t] | (selected >= time_steps)
    return observed


def trial_timestep_dataframe(
    *,
    metadata: Dict,
    config,
    rewards: np.ndarray,
    outputs: Tuple,
) -> pd.DataFrame:
    z_mu = np.asarray(outputs[11], dtype=float)
    node_selections = np.asarray(outputs[12], dtype=int)
    stop_decisions = np.asarray(outputs[13][:, :, 0], dtype=bool)
    observed_masks = np.asarray(outputs[14], dtype=bool)
    action_probs = np.asarray(outputs[15], dtype=float)
    paid_kl = np.asarray(outputs[16][:, :, 0], dtype=float)
    model_selected_path = np.asarray(outputs[TERMINAL_PATH_OUTPUT_INDEX], dtype=int)
    observation_kl = np.asarray(outputs[OBSERVATION_KL_OUTPUT_INDEX][:, :, 0], dtype=float)
    has_analysis_tensors = (
        len(outputs) >= BASE_OUTPUT_COUNT + 3
        and np.asarray(outputs[-3]).shape == z_mu.shape
        and np.asarray(outputs[-2]).shape == z_mu.shape
        and np.asarray(outputs[-1]).shape == z_mu.shape
    )
    if has_analysis_tensors:
        z_logvar = np.asarray(outputs[-3], dtype=float)
        prior_mu = np.asarray(outputs[-2], dtype=float)
        prior_logvar = np.asarray(outputs[-1], dtype=float)
    else:
        z_logvar = np.full_like(z_mu, np.nan, dtype=float)
        prior_mu = np.full_like(z_mu, np.nan, dtype=float)
        prior_logvar = np.full_like(z_mu, np.nan, dtype=float)
    z_sigma = np.exp(0.5 * np.clip(z_logvar, -10.0, 10.0))
    prior_sigma = np.exp(0.5 * np.clip(prior_logvar, -10.0, 10.0))
    pp_kl = posterior_prior_kl(z_mu, z_logvar, prior_mu, prior_logvar)

    path_values = compute_path_values(rewards, config.index_path_map)
    final_best_value = np.max(path_values, axis=1)
    final_tie = np.sum(np.isclose(path_values, final_best_value[:, None]), axis=1) > 1
    final_optimal_path = np.argmax(path_values, axis=1).astype(float)
    final_optimal_path[final_tie] = np.nan
    current = compute_current_best_path_variables(rewards, observed_masks, config.index_path_map)
    observed_nodes = observed_node_sequence(node_selections, stop_decisions, config.time_steps)
    observed_values = np.full(observed_nodes.shape, np.nan, dtype=float)
    for trial_i in range(observed_nodes.shape[0]):
        for t in range(observed_nodes.shape[1]):
            node = observed_nodes[trial_i, t]
            if np.isfinite(node):
                node_idx = int(node) - 1
                if 0 <= node_idx < rewards.shape[1]:
                    observed_values[trial_i, t] = rewards[trial_i, node_idx]

    rows = []
    n_trials, n_steps, latent_dim = z_mu.shape
    for trial_i in range(n_trials):
        final_path_values_json = safe_json_dumps(path_values[trial_i].astype(float).tolist())
        reward_values_json = safe_json_dumps(rewards[trial_i].astype(float).tolist())
        observed_order = [
            int(node)
            for node in observed_nodes[trial_i]
            if np.isfinite(node)
        ]
        observed_order_json = safe_json_dumps(observed_order)
        t1_observed_value = observed_values[trial_i, 0] if n_steps > 0 else np.nan
        t2_observed_value = observed_values[trial_i, 1] if n_steps > 1 else np.nan
        reward_t2_minus_reward_t1 = (
            t2_observed_value - t1_observed_value
            if np.isfinite(t1_observed_value) and np.isfinite(t2_observed_value)
            else np.nan
        )
        abs_reward_t2_minus_reward_t1 = (
            abs(reward_t2_minus_reward_t1)
            if np.isfinite(reward_t2_minus_reward_t1)
            else np.nan
        )
        t1_t2_interaction = (
            t1_observed_value * t2_observed_value
            if np.isfinite(t1_observed_value) and np.isfinite(t2_observed_value)
            else np.nan
        )
        current_best_path_margin_t2 = (
            current["current_best_path_margin"][trial_i, 1]
            if n_steps > 1
            else np.nan
        )
        current_best_path_t1 = current["current_best_path"][trial_i, 0] if n_steps > 0 else np.nan
        current_best_path_t2 = current["current_best_path"][trial_i, 1] if n_steps > 1 else np.nan
        current_best_path_value_t1 = current["current_best_path_value"][trial_i, 0] if n_steps > 0 else np.nan
        current_best_path_value_t2 = current["current_best_path_value"][trial_i, 1] if n_steps > 1 else np.nan
        current_best_path_margin_t1 = current["current_best_path_margin"][trial_i, 0] if n_steps > 0 else np.nan
        current_best_path_switch_t2 = current["current_best_path_switch"][trial_i, 1] if n_steps > 1 else np.nan
        for t in range(n_steps):
            observed_this_step = np.isfinite(observed_values[trial_i, t])
            stopped_this_step = bool(stop_decisions[trial_i, t])
            observed_next_step = bool(
                (t + 1 < n_steps)
                and np.isfinite(observed_values[trial_i, t + 1])
                and not bool(stop_decisions[trial_i, t + 1])
            )
            qz_used_downstream = bool(
                observed_this_step
                and not stopped_this_step
                and observed_next_step
            )
            row = {
                **metadata,
                "trial_id": int(trial_i),
                "trial_uid": f"{metadata['model_id']}::trial_{trial_i}",
                "timestep": int(t + 1),
                "observed_node": observed_nodes[trial_i, t],
                "observed_value": observed_values[trial_i, t],
                "t1_observed_value": t1_observed_value,
                "reward_t1": t1_observed_value,
                "reward_t2": t2_observed_value,
                "reward_t2_minus_reward_t1": reward_t2_minus_reward_t1,
                "abs_reward_t2_minus_reward_t1": abs_reward_t2_minus_reward_t1,
                "reward_t1_x_reward_t2": t1_t2_interaction,
                "reward_t1_reward_t2_interaction": t1_t2_interaction,
                "current_best_path_t1": current_best_path_t1,
                "current_best_path_t2": current_best_path_t2,
                "current_best_path_value_t1": current_best_path_value_t1,
                "current_best_path_value_t2": current_best_path_value_t2,
                "current_best_path_margin_t1": current_best_path_margin_t1,
                "current_best_path_margin_t2": current_best_path_margin_t2,
                "current_best_path_switch_t2": current_best_path_switch_t2,
                "stopped_at_timestep": stopped_this_step,
                "observed_at_timestep": bool(observed_this_step),
                "qz_used_downstream": qz_used_downstream,
                "observed_nodes_so_far": safe_json_dumps(
                    [
                        int(node)
                        for node in observed_nodes[trial_i, : t + 1]
                        if np.isfinite(node)
                    ]
                ),
                "observed_node_order": observed_order_json,
                "reward_values": reward_values_json,
                "current_best_path": current["current_best_path"][trial_i, t],
                "current_best_path_value": current["current_best_path_value"][trial_i, t],
                "current_best_path_margin": current["current_best_path_margin"][trial_i, t],
                "current_best_path_switch": current["current_best_path_switch"][trial_i, t],
                "tie_flag": bool(current["tie_flag"][trial_i, t]),
                "final_optimal_path": final_optimal_path[trial_i],
                "final_path_values": final_path_values_json,
                "model_action_probs": safe_json_dumps(action_probs[trial_i, t].astype(float).tolist()),
                "model_selected_path": int(model_selected_path[trial_i]),
                "paid_kl": paid_kl[trial_i, t],
                "observation_kl": observation_kl[trial_i, t],
                "posterior_prior_kl": pp_kl[trial_i, t],
            }
            for dim in range(latent_dim):
                row[f"z_mu_{dim}"] = z_mu[trial_i, t, dim]
                row[f"z_logvar_{dim}"] = z_logvar[trial_i, t, dim]
                row[f"z_sigma_{dim}"] = z_sigma[trial_i, t, dim]
                row[f"prior_mu_{dim}"] = prior_mu[trial_i, t, dim]
                row[f"prior_logvar_{dim}"] = prior_logvar[trial_i, t, dim]
                row[f"prior_sigma_{dim}"] = prior_sigma[trial_i, t, dim]
            rows.append(row)
    df = pd.DataFrame(rows)
    return add_angle_features(df, latent_dim)


def available_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [col for col in cols if col in df.columns]


def make_group_splits(groups, n_splits: int):
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    n_splits = min(n_splits, len(unique_groups))
    return GroupKFold(n_splits=n_splits)


def majority_baseline(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    if len(values) == 0:
        return np.nan
    return float(np.max(counts) / np.sum(counts))


def feature_set_metadata(feature_name: str, features: Sequence[str]) -> Dict:
    features = list(features)
    return {
        "feature_set": feature_name,
        "features": ",".join(features),
        "uses_raw_radius": "radius_mu" in features,
        "uses_prior_relative_radius": "delta_radius_mu" in features,
        "uses_kl": "posterior_prior_kl" in features,
        "uses_prior_parameters": any(
            col == "posterior_prior_kl" or
            col.startswith("prior_") or
            col.startswith("delta_") or
            col.startswith("sin_delta") or
            col.startswith("cos_delta")
            for col in features
        ),
    }


def skipped_feature_row(
    *,
    group_name: Dict,
    target_col: str,
    feature_name: str,
    features: Sequence[str],
    status: str,
    error_message: str,
) -> Dict:
    return {
        **group_name,
        "target": target_col,
        **feature_set_metadata(feature_name, features),
        "n": 0,
        "status": status,
        "error_message": error_message,
    }


def missing_feature_status(missing_cols: Sequence[str]) -> str:
    prior_markers = ("prior_", "delta_", "sin_delta", "cos_delta")
    if any(col.startswith(prior_markers) for col in missing_cols):
        return "skipped_missing_prior"
    return "skipped_missing_features"


def run_classification_cv(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_sets: Dict[str, List[str]],
    grouping_cols: Sequence[str],
    cv_folds: int,
    binary: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rows = []
    confusions = {}
    for group_name, group_df in grouped_views(df, grouping_cols):
        for feature_name, features in feature_sets.items():
            cols = available_columns(group_df, features)
            if len(cols) != len(features):
                missing = [col for col in features if col not in cols]
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status=missing_feature_status(missing),
                    error_message=f"missing columns: {','.join(missing)}",
                ))
                continue
            work = group_df.loc[
                group_df[target_col].notna() & np.isfinite(group_df[cols]).all(axis=1),
                cols + [target_col, "trial_uid"],
            ].copy()
            if work[target_col].nunique() < 2:
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status="skipped_insufficient_classes",
                    error_message="target has fewer than two classes after filtering",
                ))
                continue
            y = work[target_col].astype(int).to_numpy()
            x = work[cols].to_numpy(dtype=float)
            groups = work["trial_uid"].to_numpy()
            splitter = make_group_splits(groups, cv_folds)
            if not splitter:
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status="skipped_insufficient_groups",
                    error_message="fewer than two trial groups available",
                ))
                continue
            preds = np.full_like(y, fill_value=-1)
            prob_pos = np.full(y.shape, np.nan, dtype=float)
            for train_idx, test_idx in splitter.split(x, y, groups):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        multi_class="auto",
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                )
                model.fit(x[train_idx], y[train_idx])
                preds[test_idx] = model.predict(x[test_idx])
                if binary and hasattr(model[-1], "classes_"):
                    probs = model.predict_proba(x[test_idx])
                    if probs.shape[1] == 2:
                        prob_pos[test_idx] = probs[:, 1]
            valid = preds >= 0
            if not np.any(valid):
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status="failed_no_valid_predictions",
                    error_message="no cross-validated predictions were produced",
                ))
                continue
            yv = y[valid]
            pv = preds[valid]
            row = {
                **group_name,
                "target": target_col,
                **feature_set_metadata(feature_name, features),
                "n": int(len(yv)),
                "n_classes": int(len(np.unique(yv))),
                "status": "ok",
                "error_message": "",
                "accuracy": accuracy_score(yv, pv),
                "balanced_accuracy": balanced_accuracy_score(yv, pv),
                "macro_f1": f1_score(yv, pv, average="macro", zero_division=0),
                "majority_class_baseline": majority_baseline(yv),
            }
            row["balanced_accuracy_minus_baseline"] = row["balanced_accuracy"] - row["majority_class_baseline"]
            if binary:
                valid_prob = np.isfinite(prob_pos[valid])
                if np.any(valid_prob) and len(np.unique(yv[valid_prob])) == 2:
                    row["roc_auc"] = roc_auc_score(yv[valid_prob], prob_pos[valid][valid_prob])
                    row["average_precision"] = average_precision_score(yv[valid_prob], prob_pos[valid][valid_prob])
            rows.append(row)
            cm_key = f"{target_col}::{feature_name}::" + "::".join(f"{k}={v}" for k, v in group_name.items())
            confusions[cm_key] = confusion_matrix(yv, pv)
    return pd.DataFrame(rows), confusions


def run_regression_cv(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_sets: Dict[str, List[str]],
    grouping_cols: Sequence[str],
    cv_folds: int,
) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for group_name, group_df in grouped_views(df, grouping_cols):
        for feature_name, features in feature_sets.items():
            cols = available_columns(group_df, features)
            if len(cols) != len(features):
                missing = [col for col in features if col not in cols]
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status=missing_feature_status(missing),
                    error_message=f"missing columns: {','.join(missing)}",
                ))
                continue
            work = group_df.loc[
                group_df[target_col].notna() & np.isfinite(group_df[cols]).all(axis=1),
                cols + [target_col, "trial_uid"],
            ].copy()
            if work[target_col].nunique() < 2:
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status="skipped_insufficient_target_variance",
                    error_message="target has fewer than two unique values after filtering",
                ))
                continue
            y = work[target_col].to_numpy(dtype=float)
            x = work[cols].to_numpy(dtype=float)
            groups = work["trial_uid"].to_numpy()
            splitter = make_group_splits(groups, cv_folds)
            if not splitter:
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status="skipped_insufficient_groups",
                    error_message="fewer than two trial groups available",
                ))
                continue
            preds = np.full(y.shape, np.nan, dtype=float)
            for train_idx, test_idx in splitter.split(x, y, groups):
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 21)))
                model.fit(x[train_idx], y[train_idx])
                preds[test_idx] = model.predict(x[test_idx])
            valid = np.isfinite(preds)
            if not np.any(valid):
                rows.append(skipped_feature_row(
                    group_name=group_name,
                    target_col=target_col,
                    feature_name=feature_name,
                    features=features,
                    status="failed_no_valid_predictions",
                    error_message="no cross-validated predictions were produced",
                ))
                continue
            yv = y[valid]
            pv = preds[valid]
            pear = pearsonr(yv, pv).statistic if len(np.unique(yv)) > 1 else np.nan
            spear = spearmanr(yv, pv).statistic if len(np.unique(yv)) > 1 else np.nan
            rows.append({
                **group_name,
                "target": target_col,
                **feature_set_metadata(feature_name, features),
                "n": int(len(yv)),
                "status": "ok",
                "error_message": "",
                "r2": r2_score(yv, pv),
                "pearson_r": pear,
                "spearman_r": spear,
                "mae": mean_absolute_error(yv, pv),
                "rmse": math.sqrt(mean_squared_error(yv, pv)),
            })
    return pd.DataFrame(rows)


def grouped_views(df: pd.DataFrame, grouping_cols: Sequence[str]):
    yield {"analysis_scope": "pooled"}, df
    for col in grouping_cols:
        if col not in df.columns:
            continue
        for value, piece in df.groupby(col):
            yield {"analysis_scope": f"by_{col}", col: value}, piece


def run_predictions(df: pd.DataFrame, outdir: Path, cv_folds: int):
    if not {"sin_angle_mu", "cos_angle_mu"}.issubset(df.columns):
        print("Angle analyses require latent_dim = 2. Skipping angle prediction models for latent_dim != 2.")
        return {}, {}
    non_tied = df[~df["tie_flag"].astype(bool)].copy()
    results = {}
    confusions = {}
    cls_results, cls_confusions = run_classification_cv(
        non_tied,
        target_col="current_best_path",
        feature_sets=FEATURE_SETS,
        grouping_cols=["timestep", "lambda_value"],
        cv_folds=cv_folds,
    )
    cls_results.to_csv(outdir / "current_best_path_prediction_results.csv", index=False)
    results["current_best_path"] = cls_results
    confusions.update(cls_confusions)

    switch_df = non_tied[non_tied["current_best_path_switch"].notna()].copy()
    switch_results, switch_confusions = run_classification_cv(
        switch_df,
        target_col="current_best_path_switch",
        feature_sets=FEATURE_SETS,
        grouping_cols=["timestep", "lambda_value"],
        cv_folds=cv_folds,
        binary=True,
    )
    switch_results.to_csv(outdir / "path_switch_prediction_results.csv", index=False)
    results["path_switch"] = switch_results
    confusions.update(switch_confusions)

    value_results = run_regression_cv(
        df,
        target_col="current_best_path_value",
        feature_sets=FEATURE_SETS,
        grouping_cols=["timestep", "lambda_value"],
        cv_folds=cv_folds,
    )
    value_results.to_csv(outdir / "current_best_path_value_prediction_results.csv", index=False)
    results["current_best_path_value"] = value_results

    prior_rows = []
    for target, target_df, is_regression, is_binary in [
        ("current_best_path", non_tied, False, False),
        ("current_best_path_switch", switch_df, False, True),
        ("current_best_path_value", df, True, False),
    ]:
        if is_regression:
            res = run_regression_cv(
                target_df,
                target_col=target,
                feature_sets=PRIOR_DIAGNOSTIC_FEATURE_SETS,
                grouping_cols=["lambda_value"],
                cv_folds=cv_folds,
            )
            metric = "r2"
        else:
            res, diag_confusions = run_classification_cv(
                target_df,
                target_col=target,
                feature_sets=PRIOR_DIAGNOSTIC_FEATURE_SETS,
                grouping_cols=["lambda_value"],
                cv_folds=cv_folds,
                binary=is_binary,
            )
            confusions.update(diag_confusions)
            metric = "balanced_accuracy"
        if len(res) == 0:
            continue
        res["diagnostic_target"] = target
        res["primary_metric"] = metric
        prior_rows.append(res)
    prior_results = pd.concat(prior_rows, ignore_index=True) if prior_rows else pd.DataFrame()
    if len(prior_results) > 0:
        prior_results = add_prior_improvement_columns(prior_results)
    prior_results.to_csv(outdir / "prior_diagnostics_results.csv", index=False)
    results["prior_diagnostics"] = prior_results
    np.savez(outdir / "current_best_path_confusion_matrices.npz", **confusions)
    return results, confusions


def add_prior_improvement_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comparison_metrics = [
        "balanced_accuracy",
        "macro_f1",
        "r2",
        "pearson_r",
        "spearman_r",
        "rmse",
    ]
    for baseline_name in ("kl_only", "raw_angle_plus_raw_radius"):
        for metric in comparison_metrics:
            df[f"improvement_over_{baseline_name}_{metric}"] = np.nan
    if "status" not in df.columns:
        df["status"] = "ok"
    ok_df = df[df["status"] == "ok"]
    for keys, piece in ok_df.groupby(["diagnostic_target", "analysis_scope"], dropna=False):
        idx = piece.index
        for baseline_name in ("kl_only", "raw_angle_plus_raw_radius"):
            baseline_rows = piece[piece["feature_set"] == baseline_name]
            if len(baseline_rows) == 0:
                continue
            for metric in comparison_metrics:
                if metric not in piece.columns:
                    continue
                baseline_values = baseline_rows[metric].dropna()
                if len(baseline_values) == 0:
                    continue
                baseline_value = baseline_values.max() if metric != "rmse" else baseline_values.min()
                if metric == "rmse":
                    df.loc[idx, f"improvement_over_{baseline_name}_{metric}"] = (
                        baseline_value - df.loc[idx, metric]
                    )
                else:
                    df.loc[idx, f"improvement_over_{baseline_name}_{metric}"] = (
                        df.loc[idx, metric] - baseline_value
                    )
    if "improvement_over_kl_only_balanced_accuracy" in df.columns:
        df["improvement_over_kl_only"] = df["improvement_over_kl_only_balanced_accuracy"]
    return df


def compute_orthogonality(df: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not {"z_mu_0", "z_mu_1"}.issubset(df.columns):
        empty = pd.DataFrame()
        empty.to_csv(outdir / "latent_orthogonality_diagnostics.csv", index=False)
        empty.to_csv(outdir / "latent_angle_distribution.csv", index=False)
        return empty, empty

    rows = []
    angle_rows = []
    for group_name, piece in grouped_views(df, ["timestep", "lambda_value", "seed"]):
        record = dict(group_name)
        for a, b, name in [
            ("z_mu_0", "z_mu_1", "corr_z_mu"),
            ("z_logvar_0", "z_logvar_1", "corr_z_logvar"),
            ("z_sigma_0", "z_sigma_1", "corr_z_sigma"),
            ("prior_mu_0", "prior_mu_1", "corr_prior_mu"),
            ("delta_mu_0", "delta_mu_1", "corr_delta_mu"),
        ]:
            if a in piece and b in piece and piece[a].notna().sum() > 2:
                record[name] = piece[[a, b]].corr().iloc[0, 1]
        mu = piece[["z_mu_0", "z_mu_1"]].dropna().to_numpy(dtype=float)
        if mu.shape[0] > 2:
            cov = np.cov(mu, rowvar=False)
            eig = np.linalg.eigvalsh(cov)
            eig = np.sort(eig)[::-1]
            record["cov_mu_eigenvalue_0"] = eig[0]
            record["cov_mu_eigenvalue_1"] = eig[1]
            record["explained_variance_ratio_0"] = eig[0] / np.maximum(np.sum(eig), 1e-12)
            record["condition_number"] = eig[0] / np.maximum(eig[1], 1e-12)
        if "angle_mu" in piece:
            angles = piece["angle_mu"].dropna().to_numpy(dtype=float)
            if angles.size:
                resultant = np.abs(np.mean(np.exp(1j * angles)))
                hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi), density=False)
                probs = hist / np.maximum(hist.sum(), 1)
                entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
                record["angle_mean_resultant_length"] = resultant
                record["angle_hist_entropy"] = entropy
                for bin_i, count in enumerate(hist):
                    angle_rows.append({**group_name, "angle_bin": bin_i, "count": int(count)})
        rows.append(record)
    ortho = pd.DataFrame(rows)
    angle_dist = pd.DataFrame(angle_rows)
    ortho.to_csv(outdir / "latent_orthogonality_diagnostics.csv", index=False)
    angle_dist.to_csv(outdir / "latent_angle_distribution.csv", index=False)
    return ortho, angle_dist


def make_plots(
    df: pd.DataFrame,
    prediction_results: Dict[str, pd.DataFrame],
    ortho: pd.DataFrame,
    outdir: Path,
    transition_df: Optional[pd.DataFrame] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_7pt_plot_style(plt)
    apply_7pt_plot_style(plt)

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = add_halfplane_coordinate_columns(df)
    plot_df = rows_after_observed_reward_used_downstream(df)
    if len(plot_df) == 0:
        plot_df = df.iloc[0:0].copy()
    if {"z_mu_0", "z_mu_1"}.issubset(plot_df.columns) and len(plot_df):
        sample = plot_df.sample(min(len(plot_df), 20000), random_state=0)
        if "current_best_path" in sample:
            plt.figure(figsize=panel_figsize(1, 1, title=True))
            plt.scatter(sample["z_mu_0"], sample["z_mu_1"], c=sample["current_best_path"], s=4, alpha=0.45)
            plt.xlabel("z_mu_0")
            plt.ylabel("z_mu_1")
            plt.colorbar(label="current_best_path")
            plt.tight_layout()
            plt.savefig(figdir / "latent_mu_scatter_by_current_best_path.png", dpi=180)
            plt.close()
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        plt.scatter(sample["z_mu_0"], sample["z_mu_1"], c=sample["timestep"], s=4, alpha=0.45)
        plt.xlabel("z_mu_0")
        plt.ylabel("z_mu_1")
        plt.colorbar(label="timestep")
        plt.tight_layout()
        plt.savefig(figdir / "latent_mu_scatter_by_timestep.png", dpi=180)
        plt.close()

    if {"angle_mu", "current_best_path"}.issubset(plot_df.columns):
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        hist_df = plot_df[~plot_df["tie_flag"].astype(bool)] if "tie_flag" in plot_df.columns else plot_df
        for path_value, piece in hist_df.groupby("current_best_path"):
            plt.hist(piece["angle_mu"].dropna(), bins=40, alpha=0.35, label=f"path {int(path_value)}")
        plt.xlabel("angle_mu")
        plt.ylabel("count")
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.savefig(figdir / "latent_angle_hist_by_current_best_path.png", dpi=180, bbox_inches="tight")
        plt.close()

    plot_prediction_metric(
        prediction_results.get("current_best_path"),
        metric="balanced_accuracy",
        path=figdir / "angle_prediction_by_timestep.png",
    )
    plot_prediction_metric(
        prediction_results.get("path_switch"),
        metric="balanced_accuracy",
        path=figdir / "path_switch_prediction_by_timestep.png",
    )
    plot_prediction_metric(
        prediction_results.get("current_best_path_value"),
        metric="r2",
        path=figdir / "best_path_value_prediction_by_timestep.png",
    )
    plot_radius_feature_comparison(
        prediction_results.get("current_best_path"),
        metric="balanced_accuracy",
        path=figdir / "radius_feature_comparison_current_best_path.png",
    )
    plot_radius_feature_comparison(
        prediction_results.get("path_switch"),
        metric="balanced_accuracy",
        path=figdir / "radius_feature_comparison_path_switch.png",
    )
    plot_radius_feature_comparison(
        prediction_results.get("current_best_path_value"),
        metric="r2",
        path=figdir / "radius_feature_comparison_best_path_value.png",
    )
    if {"delta_mu_0", "delta_mu_1", "current_best_path"}.issubset(df.columns):
        sample = df.sample(min(len(df), 20000), random_state=1)
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        plt.scatter(sample["delta_mu_0"], sample["delta_mu_1"], c=sample["current_best_path"], s=4, alpha=0.45)
        plt.xlabel("posterior_mu_0 - prior_mu_0")
        plt.ylabel("posterior_mu_1 - prior_mu_1")
        plt.colorbar(label="current_best_path")
        plt.tight_layout()
        plt.savefig(figdir / "posterior_prior_displacement_scatter.png", dpi=180)
        plt.close()
    if len(ortho) > 0 and "lambda_value" in ortho and "corr_z_mu" in ortho:
        piece = ortho[ortho["analysis_scope"] == "by_lambda_value"]
        if len(piece) > 0:
            plt.figure(figsize=panel_figsize(1, 1, title=True))
            plt.plot(piece["lambda_value"], piece["corr_z_mu"], marker="o")
            plt.xlabel("lambda_value")
            plt.ylabel("corr(z_mu_0, z_mu_1)")
            plt.xscale("log")
            plt.tight_layout()
            plt.savefig(figdir / "latent_orthogonality_by_lambda.png", dpi=180)
            plt.close()


def plot_prediction_metric(results: Optional[pd.DataFrame], *, metric: str, path: Path):
    if results is None or len(results) == 0 or metric not in results:
        return
    import matplotlib.pyplot as plt

    piece = results[results["analysis_scope"] == "by_timestep"].copy()
    if "status" in piece:
        piece = piece[piece["status"] == "ok"]
    if len(piece) == 0:
        return
    plt.figure(figsize=panel_figsize(1, 1, title=True))
    for feature_set, fs_piece in piece.groupby("feature_set"):
        fs_piece = fs_piece.sort_values("timestep")
        plt.plot(fs_piece["timestep"], fs_piece[metric], marker="o", label=feature_set)
    plt.xlabel("timestep")
    plt.ylabel(metric)
    plt.legend(frameon=False, fontsize=PANEL_FONT_SIZE, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_radius_feature_comparison(results: Optional[pd.DataFrame], *, metric: str, path: Path):
    if results is None or len(results) == 0 or metric not in results:
        return
    import matplotlib.pyplot as plt

    piece = results[
        (results["analysis_scope"] == "by_timestep") &
        (results["feature_set"].isin(RADIUS_COMPARISON_FEATURE_SETS))
    ].copy()
    if "status" in piece:
        piece = piece[piece["status"] == "ok"]
    if len(piece) == 0:
        piece = results[
            (results["analysis_scope"] == "pooled") &
            (results["feature_set"].isin(RADIUS_COMPARISON_FEATURE_SETS))
        ].copy()
        if "status" in piece:
            piece = piece[piece["status"] == "ok"]
    if len(piece) == 0:
        return

    plt.figure(figsize=panel_figsize(1, 1, title=True))
    if "timestep" in piece and piece["timestep"].notna().any():
        for feature_set, fs_piece in piece.groupby("feature_set"):
            fs_piece = fs_piece.sort_values("timestep")
            plt.plot(fs_piece["timestep"], fs_piece[metric], marker="o", label=feature_set)
        plt.xlabel("timestep")
    else:
        piece = piece.sort_values(metric, ascending=False)
        plt.bar(piece["feature_set"], piece[metric])
        plt.xticks(rotation=35, ha="right")
        plt.xlabel("feature set")
    plt.ylabel(metric)
    plt.legend(frameon=False, fontsize=PANEL_FONT_SIZE, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.68, 1])
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def summarize_outputs(
    *,
    outdir: Path,
    df: pd.DataFrame,
    failures: List[Dict],
    prediction_results: Dict[str, pd.DataFrame],
    ortho: pd.DataFrame,
    n_trials: int,
):
    def best_feature(results: Optional[pd.DataFrame], metric: str):
        if results is None or len(results) == 0 or metric not in results:
            return None
        pooled = results[results["analysis_scope"] == "pooled"]
        if len(pooled) == 0:
            pooled = results
        row = pooled.sort_values(metric, ascending=False).iloc[0]
        return {"feature_set": row["feature_set"], metric: float(row[metric])}

    prior = prediction_results.get("prior_diagnostics", pd.DataFrame())
    prior_improves = None
    if len(prior) > 0 and "improvement_over_kl_only" in prior:
        prior_improves = bool((prior["improvement_over_kl_only"].fillna(0) > 0).any())
    ortho_note = None
    if len(ortho) > 0 and "corr_z_mu" in ortho:
        pooled = ortho[ortho["analysis_scope"] == "pooled"]
        if len(pooled):
            corr = float(pooled["corr_z_mu"].iloc[0])
            ortho_note = "approximately_orthogonal" if abs(corr) < 0.3 else "correlated"

    summary = {
        "models_analyzed": int(df["model_id"].nunique()) if "model_id" in df else 0,
        "failures": failures,
        "n_trials_per_model": n_trials,
        "rows": int(len(df)),
        "best_current_best_path_feature_set": best_feature(
            prediction_results.get("current_best_path"), "balanced_accuracy"
        ),
        "best_path_switch_feature_set": best_feature(
            prediction_results.get("path_switch"), "balanced_accuracy"
        ),
        "best_best_path_value_feature_set": best_feature(
            prediction_results.get("current_best_path_value"), "r2"
        ),
        "prior_displacement_improves_beyond_kl": prior_improves,
        "latent_dimension_orthogonality_note": ortho_note,
    }
    with open(outdir / "latent_angle_analysis_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "latent_angle_analysis_summary.txt", "w") as handle:
        handle.write("Latent angle planning analysis summary\n")
        handle.write("======================================\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")


# Active prediction layer: main outputs below use one predictor at a time within
# each model/checkpoint and timestep. The earlier feature-set helpers are kept as
# legacy utilities but are not called by main().
MODEL_GROUP_COLUMNS = [
    "model_id",
    "checkpoint_path",
    "alpha",
    "beta",
    "lambda_value",
    "seed",
    "rnn_dim",
    "latent_dim",
    "opportunity_cost",
    "tree_size",
    "tree_type",
    "input_type",
    "expansion_decision_version",
    "model_variant",
]

SINGLE_PREDICTOR_TYPES = {
    "z_mu_0": "posterior_mean",
    "z_mu_1": "posterior_mean",
    "z_logvar_0": "posterior_uncertainty",
    "z_logvar_1": "posterior_uncertainty",
    "z_sigma_0": "posterior_uncertainty",
    "z_sigma_1": "posterior_uncertainty",
    "angle_mu": "posterior_angle",
    "sin_angle_mu": "posterior_angle",
    "cos_angle_mu": "posterior_angle",
    "radius_mu": "posterior_radius",
    "prior_mu_0": "prior",
    "prior_mu_1": "prior",
    "prior_logvar_0": "prior",
    "prior_logvar_1": "prior",
    "posterior_prior_kl": "posterior_prior_kl",
    "delta_mu_0": "posterior_prior_delta",
    "delta_mu_1": "posterior_prior_delta",
    "delta_angle_mu": "posterior_prior_delta",
    "sin_delta_angle_mu": "posterior_prior_delta",
    "cos_delta_angle_mu": "posterior_prior_delta",
    "delta_radius_mu": "posterior_prior_delta",
    "gm_disk_radius_0": "gaussian_manifold_disk",
    "gm_disk_angle_0": "gaussian_manifold_disk",
    "gm_disk_sin_angle_0": "gaussian_manifold_disk",
    "gm_disk_cos_angle_0": "gaussian_manifold_disk",
    "gm_disk_radius_1": "gaussian_manifold_disk",
    "gm_disk_angle_1": "gaussian_manifold_disk",
    "gm_disk_sin_angle_1": "gaussian_manifold_disk",
    "gm_disk_cos_angle_1": "gaussian_manifold_disk",
}

SINGLE_PREDICTORS = list(SINGLE_PREDICTOR_TYPES.keys())

TEMPORAL_PREDICTOR_TYPES = {
    "temporal_delta_mu_0": "temporal_mu_direction",
    "temporal_delta_mu_1": "temporal_mu_direction",
    "temporal_sin_delta_angle_mu": "temporal_mu_direction",
    "temporal_cos_delta_angle_mu": "temporal_mu_direction",
    "temporal_delta_radius_mu": "temporal_mu_radius",
    "temporal_delta_sigma_0": "temporal_uncertainty_change",
    "temporal_delta_sigma_1": "temporal_uncertainty_change",
    "temporal_delta_logvar_0": "temporal_uncertainty_change",
    "temporal_delta_logvar_1": "temporal_uncertainty_change",
    "temporal_delta_halfplane_x_0": "temporal_halfplane_direction_dim0",
    "temporal_delta_halfplane_y_0": "temporal_halfplane_direction_dim0",
    "temporal_halfplane_sin_delta_angle_0": "temporal_halfplane_direction_dim0",
    "temporal_halfplane_cos_delta_angle_0": "temporal_halfplane_direction_dim0",
    "temporal_halfplane_delta_radius_0": "temporal_halfplane_radius_dim0",
    "temporal_delta_halfplane_x_1": "temporal_halfplane_direction_dim1",
    "temporal_delta_halfplane_y_1": "temporal_halfplane_direction_dim1",
    "temporal_halfplane_sin_delta_angle_1": "temporal_halfplane_direction_dim1",
    "temporal_halfplane_cos_delta_angle_1": "temporal_halfplane_direction_dim1",
    "temporal_halfplane_delta_radius_1": "temporal_halfplane_radius_dim1",
    "sin_delta_angle_mu_t2": "posterior_prior_delta",
    "cos_delta_angle_mu_t2": "posterior_prior_delta",
    "delta_radius_mu_t2": "posterior_prior_delta",
    "delta_mu_0_t2": "posterior_prior_delta",
    "delta_mu_1_t2": "posterior_prior_delta",
    "posterior_prior_kl_t2": "posterior_prior_kl",
}

TEMPORAL_PREDICTORS = list(TEMPORAL_PREDICTOR_TYPES.keys())


def predictor_type_for(predictor: str) -> str:
    return TEMPORAL_PREDICTOR_TYPES.get(
        predictor,
        SINGLE_PREDICTOR_TYPES.get(predictor, "unknown"),
    )


def integer_suffix_columns(df: pd.DataFrame, prefix: str, suffix: str = "") -> List[int]:
    dims = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        if suffix and not col.endswith(suffix):
            continue
        token_end = len(col) - len(suffix) if suffix else len(col)
        token = col[len(prefix):token_end]
        if token.isdigit():
            dims.append(int(token))
    return sorted(set(dims))


def available_latent_dims(df: pd.DataFrame, suffix: str = "") -> List[int]:
    dims = integer_suffix_columns(df, "z_mu_", suffix=suffix)
    out = []
    for dim in dims:
        if (
            f"z_mu_{dim}{suffix}" in df.columns
            and f"z_sigma_{dim}{suffix}" in df.columns
            and f"z_logvar_{dim}{suffix}" in df.columns
        ):
            out.append(dim)
    return out


def available_halfplane_dims(df: pd.DataFrame, suffix: str = "") -> List[int]:
    dims = integer_suffix_columns(df, "halfplane_x_", suffix=suffix)
    return [
        dim for dim in dims
        if f"halfplane_x_{dim}{suffix}" in df.columns and f"halfplane_y_{dim}{suffix}" in df.columns
    ]


def add_halfplane_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for dim in available_latent_dims(df):
        mu_col = f"z_mu_{dim}"
        sigma_col = f"z_sigma_{dim}"
        df[f"halfplane_x_{dim}"] = pd.to_numeric(df[mu_col], errors="coerce") / math.sqrt(2.0)
        df[f"halfplane_y_{dim}"] = pd.to_numeric(df[sigma_col], errors="coerce")
    return df


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) != 0
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})


def strict_qz_used_downstream_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Rows whose latent state was carried forward to a later observation.

    The model first chooses whether to observe, then encodes the observed reward.
    If the next expansion decision stops, the just-created latent is not used for
    another observation and should be excluded from latent-geometry analyses.
    """
    required = {"timestep", "observed_at_timestep", "stopped_at_timestep"}
    if df is None or len(df) == 0 or not required.issubset(df.columns):
        return None
    key_cols = [col for col in ["model_id", "trial_uid"] if col in df.columns]
    if "trial_uid" not in key_cols and "trial_id" in df.columns:
        metadata_cols = [
            "model_id",
            "checkpoint_path",
            "seed",
            "lambda_value",
            "beta",
            "alpha",
            "opportunity_cost",
            "rnn_dim",
            "latent_dim",
            "tree_size",
            "tree_type",
            "input_type",
            "expansion_decision_version",
            "model_variant",
        ]
        key_cols = [col for col in metadata_cols if col in df.columns] + ["trial_id"]
    if not key_cols:
        return None

    work = df[key_cols + ["timestep", "observed_at_timestep", "stopped_at_timestep"]].copy()
    work["_row_index"] = np.arange(len(work))
    work["_timestep"] = pd.to_numeric(work["timestep"], errors="coerce")
    work["_observed"] = bool_series(work["observed_at_timestep"])
    work["_stopped"] = bool_series(work["stopped_at_timestep"])
    next_work = work[key_cols + ["_timestep", "_observed", "_stopped"]].copy()
    next_work["_timestep"] = next_work["_timestep"] - 1
    next_work = next_work.rename(
        columns={
            "_observed": "_next_observed",
            "_stopped": "_next_stopped",
        }
    )
    next_work = next_work.drop_duplicates(key_cols + ["_timestep"], keep="first")
    merged = work.merge(
        next_work,
        on=key_cols + ["_timestep"],
        how="left",
        sort=False,
    )
    mask = (
        merged["_observed"].fillna(False)
        & ~merged["_stopped"].fillna(False)
        & merged["_next_observed"].fillna(False)
        & ~merged["_next_stopped"].fillna(False)
    )
    out = pd.Series(False, index=df.index)
    out.iloc[merged["_row_index"].to_numpy(dtype=int)] = mask.to_numpy(dtype=bool)
    return out


def rows_after_observed_reward_used_downstream(df: pd.DataFrame, timestep: Optional[int] = None) -> pd.DataFrame:
    """Keep rows whose latent state was carried forward to another observation."""
    if df is None or len(df) == 0:
        return df.copy() if hasattr(df, "copy") else pd.DataFrame()
    source = df.copy()
    strict_mask = strict_qz_used_downstream_mask(source)
    if strict_mask is not None:
        mask = strict_mask
    elif "qz_used_downstream" in source.columns:
        mask = bool_series(source["qz_used_downstream"])
    else:
        mask = pd.Series(True, index=source.index)
        if "observed_at_timestep" in source.columns:
            mask &= bool_series(source["observed_at_timestep"])
        if "stopped_at_timestep" in source.columns:
            mask &= ~bool_series(source["stopped_at_timestep"])
    if timestep is not None and "timestep" in source.columns:
        mask &= pd.to_numeric(source["timestep"], errors="coerce") == timestep
    return source[mask].copy()


def transitions_after_observed_rewards_used_downstream(transition_df: pd.DataFrame) -> pd.DataFrame:
    """Keep transitions whose endpoint latent states were both used after the observed rewards."""
    if transition_df is None or len(transition_df) == 0:
        return transition_df.copy() if hasattr(transition_df, "copy") else pd.DataFrame()
    out = transition_df.copy()
    for suffix in ("t1", "t2"):
        qz_col = f"qz_used_downstream_{suffix}"
        obs_col = f"observed_at_timestep_{suffix}"
        stop_col = f"stopped_at_timestep_{suffix}"
        if qz_col in out.columns:
            out = out[bool_series(out[qz_col])].copy()
        else:
            if obs_col in out.columns:
                out = out[bool_series(out[obs_col])].copy()
            if stop_col in out.columns:
                out = out[~bool_series(out[stop_col])].copy()
    return out


def build_temporal_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per model x trial x adjacent timestep transition."""
    required = {
        "trial_uid",
        "trial_id",
        "timestep",
        "current_best_path",
        "current_best_path_value",
        "current_best_path_switch",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    latent_dims = available_latent_dims(df)
    if not latent_dims:
        return pd.DataFrame()

    key_cols = [col for col in MODEL_GROUP_COLUMNS + ["trial_id", "trial_uid"] if col in df.columns]
    timesteps = sorted(int(value) for value in df["timestep"].dropna().unique())
    if len(timesteps) < 2:
        return pd.DataFrame()

    def suffix_non_keys(piece: pd.DataFrame, suffix: str) -> pd.DataFrame:
        rename = {col: f"{col}_{suffix}" for col in piece.columns if col not in key_cols}
        return piece.rename(columns=rename)

    transition_frames = []
    for start_timestep, end_timestep in zip(timesteps[:-1], timesteps[1:]):
        start_df = df[df["timestep"] == start_timestep].copy()
        end_df = df[df["timestep"] == end_timestep].copy()
        if len(start_df) == 0 or len(end_df) == 0:
            continue
        merged = suffix_non_keys(start_df, "t1").merge(
            suffix_non_keys(end_df, "t2"),
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) == 0:
            continue

        out = merged[key_cols].copy()
        out["transition"] = f"t{start_timestep}_to_t{end_timestep}"
        out["transition_start_timestep"] = int(start_timestep)
        out["transition_end_timestep"] = int(end_timestep)
        for label in (
            "current_best_path",
            "current_best_path_value",
            "current_best_path_switch",
            "tie_flag",
            "observed_at_timestep",
            "stopped_at_timestep",
            "qz_used_downstream",
            "observed_value",
        ):
            for suffix in ("t1", "t2"):
                col = f"{label}_{suffix}"
                if col in merged.columns:
                    out[col] = merged[col]
        out["best_path_value_change_t1_to_t2"] = (
            pd.to_numeric(out["current_best_path_value_t2"], errors="coerce")
            - pd.to_numeric(out["current_best_path_value_t1"], errors="coerce")
        )

        for dim in latent_dims:
            for base in ("z_mu", "z_sigma", "z_logvar"):
                c1 = f"{base}_{dim}_t1"
                c2 = f"{base}_{dim}_t2"
                if c1 in merged.columns and c2 in merged.columns:
                    out[c1] = merged[c1]
                    out[c2] = merged[c2]
                    out[f"temporal_delta_{base.replace('z_', '')}_{dim}"] = (
                        pd.to_numeric(merged[c2], errors="coerce")
                        - pd.to_numeric(merged[c1], errors="coerce")
                    )

            x1 = pd.to_numeric(merged[f"z_mu_{dim}_t1"], errors="coerce") / math.sqrt(2.0)
            x2 = pd.to_numeric(merged[f"z_mu_{dim}_t2"], errors="coerce") / math.sqrt(2.0)
            y1 = pd.to_numeric(merged[f"z_sigma_{dim}_t1"], errors="coerce")
            y2 = pd.to_numeric(merged[f"z_sigma_{dim}_t2"], errors="coerce")
            out[f"halfplane_x_{dim}_t1"] = x1
            out[f"halfplane_x_{dim}_t2"] = x2
            out[f"halfplane_y_{dim}_t1"] = y1
            out[f"halfplane_y_{dim}_t2"] = y2
            out[f"temporal_delta_halfplane_x_{dim}"] = x2 - x1
            out[f"temporal_delta_halfplane_y_{dim}"] = y2 - y1
            out[f"temporal_halfplane_delta_angle_{dim}"] = np.arctan2(y2 - y1, x2 - x1)
            out[f"temporal_halfplane_delta_radius_{dim}"] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            out[f"temporal_halfplane_sin_delta_angle_{dim}"] = np.sin(
                out[f"temporal_halfplane_delta_angle_{dim}"]
            )
            out[f"temporal_halfplane_cos_delta_angle_{dim}"] = np.cos(
                out[f"temporal_halfplane_delta_angle_{dim}"]
            )

        if {"temporal_delta_mu_0", "temporal_delta_mu_1"}.issubset(out.columns):
            d0 = out["temporal_delta_mu_0"]
            d1 = out["temporal_delta_mu_1"]
            out["temporal_delta_angle_mu"] = np.arctan2(d1, d0)
            out["temporal_delta_radius_mu"] = np.sqrt(d0**2 + d1**2)
            out["temporal_sin_delta_angle_mu"] = np.sin(out["temporal_delta_angle_mu"])
            out["temporal_cos_delta_angle_mu"] = np.cos(out["temporal_delta_angle_mu"])

        prior_comparison_cols = [
            "posterior_prior_kl",
            "delta_angle_mu",
            "sin_delta_angle_mu",
            "cos_delta_angle_mu",
            "delta_radius_mu",
        ]
        prior_comparison_cols.extend(f"delta_mu_{dim}" for dim in latent_dims)
        for col in prior_comparison_cols:
            t2_col = f"{col}_t2"
            if t2_col in merged.columns:
                out[t2_col] = merged[t2_col]
        transition_frames.append(out)
    return pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()


def model_timestep_views(df: pd.DataFrame):
    group_cols = [col for col in MODEL_GROUP_COLUMNS + ["timestep"] if col in df.columns]
    for values, piece in df.groupby(group_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        yield dict(zip(group_cols, values)), piece


def model_transition_views(df: pd.DataFrame):
    group_cols = [col for col in MODEL_GROUP_COLUMNS + ["transition"] if col in df.columns]
    for values, piece in df.groupby(group_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        yield dict(zip(group_cols, values)), piece


def predictor_result_base(
    group_name: Dict,
    *,
    target_col: str,
    predictor: str,
    n_rows: int = 0,
    n_trials: int = 0,
    n_classes_or_target_variance=np.nan,
    cv_folds: int = 0,
    status: str,
    error_message: str = "",
) -> Dict:
    return {
        **group_name,
        "target": target_col,
        "predictor": predictor,
        "predictor_type": predictor_type_for(predictor),
        "n_rows": int(n_rows),
        "n_trials": int(n_trials),
        "n_classes_or_target_variance": n_classes_or_target_variance,
        "cv_folds": int(cv_folds),
        "status": status,
        "error_message": error_message,
    }


def single_predictor_work(piece: pd.DataFrame, target_col: str, predictor: str) -> pd.DataFrame:
    return piece.loc[
        piece[target_col].notna() & np.isfinite(piece[predictor]),
        [target_col, predictor, "trial_uid"],
    ].copy()


def predictor_missing_or_empty(piece: pd.DataFrame, predictor: str) -> Optional[str]:
    if predictor not in piece.columns:
        return f"missing predictor: {predictor}"
    values = pd.to_numeric(piece[predictor], errors="coerce")
    if not np.isfinite(values).any():
        return f"predictor has no finite values: {predictor}"
    return None


def valid_group_kfold(groups: np.ndarray, requested_folds: int):
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return None, 0
    folds = min(int(requested_folds), len(unique_groups))
    if folds < 2:
        return None, 0
    return GroupKFold(n_splits=folds), folds


def run_single_predictor_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    predictors: Sequence[str],
    cv_folds: int,
    binary: bool = False,
    view_fn=model_timestep_views,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for group_name, piece in view_fn(df):
        for predictor in predictors:
            missing_message = predictor_missing_or_empty(piece, predictor)
            if missing_message:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    status="skipped_missing_predictor",
                    error_message=missing_message,
                ))
                continue
            work = single_predictor_work(piece, target_col, predictor)
            n_trials = work["trial_uid"].nunique() if len(work) else 0
            if len(work) == 0:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    status="skipped_missing_predictor",
                    error_message="no rows remain after finite predictor/target filtering",
                ))
                continue
            if work[predictor].nunique(dropna=True) < 2:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    status="skipped_zero_variance",
                    error_message="predictor has no variance at this timestep",
                ))
                continue
            n_classes = int(work[target_col].nunique(dropna=True))
            if n_classes < 2:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    n_classes_or_target_variance=n_classes,
                    status="skipped_insufficient_classes",
                    error_message="target has fewer than two classes at this timestep",
                ))
                continue
            y = work[target_col].astype(int).to_numpy()
            x = work[[predictor]].to_numpy(dtype=float)
            groups = work["trial_uid"].to_numpy()
            splitter, actual_folds = valid_group_kfold(groups, cv_folds)
            if splitter is None:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    n_classes_or_target_variance=n_classes,
                    status="skipped_insufficient_groups",
                    error_message="fewer than two trial groups available",
                ))
                continue
            preds = np.full(y.shape, -1, dtype=int)
            prob_pos = np.full(y.shape, np.nan, dtype=float)
            for train_idx, test_idx in splitter.split(x, y, groups):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        multi_class="auto",
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                )
                model.fit(x[train_idx], y[train_idx])
                preds[test_idx] = model.predict(x[test_idx])
                if binary:
                    probs = model.predict_proba(x[test_idx])
                    classes = model[-1].classes_
                    if probs.shape[1] == 2 and 1 in classes:
                        prob_pos[test_idx] = probs[:, int(np.where(classes == 1)[0][0])]
            valid = preds >= 0
            if not np.any(valid):
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    n_classes_or_target_variance=n_classes,
                    cv_folds=actual_folds,
                    status="failed_no_valid_predictions",
                    error_message="no cross-validated predictions were produced",
                ))
                continue
            yv = y[valid]
            pv = preds[valid]
            row = predictor_result_base(
                group_name,
                target_col=target_col,
                predictor=predictor,
                n_rows=len(yv),
                n_trials=len(np.unique(groups[valid])),
                n_classes_or_target_variance=int(len(np.unique(yv))),
                cv_folds=actual_folds,
                status="ok",
            )
            row.update({
                "accuracy": accuracy_score(yv, pv),
                "balanced_accuracy": balanced_accuracy_score(yv, pv),
                "macro_f1": f1_score(yv, pv, average="macro", zero_division=0),
                "majority_class_baseline": majority_baseline(yv),
            })
            row["balanced_accuracy_minus_baseline"] = (
                row["balanced_accuracy"] - row["majority_class_baseline"]
            )
            if binary:
                valid_prob = np.isfinite(prob_pos[valid])
                if np.any(valid_prob) and len(np.unique(yv[valid_prob])) == 2:
                    row["roc_auc"] = roc_auc_score(yv[valid_prob], prob_pos[valid][valid_prob])
                    row["average_precision"] = average_precision_score(
                        yv[valid_prob], prob_pos[valid][valid_prob]
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def run_single_predictor_regression(
    df: pd.DataFrame,
    *,
    target_col: str,
    predictors: Sequence[str],
    cv_folds: int,
    view_fn=model_timestep_views,
) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for group_name, piece in view_fn(df):
        for predictor in predictors:
            missing_message = predictor_missing_or_empty(piece, predictor)
            if missing_message:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    status="skipped_missing_predictor",
                    error_message=missing_message,
                ))
                continue
            work = single_predictor_work(piece, target_col, predictor)
            n_trials = work["trial_uid"].nunique() if len(work) else 0
            if len(work) == 0:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    status="skipped_missing_predictor",
                    error_message="no rows remain after finite predictor/target filtering",
                ))
                continue
            if work[predictor].nunique(dropna=True) < 2:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    status="skipped_zero_variance",
                    error_message="predictor has no variance at this timestep",
                ))
                continue
            target_variance = float(np.var(work[target_col].to_numpy(dtype=float)))
            if not np.isfinite(target_variance) or target_variance <= 1e-12:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    n_classes_or_target_variance=target_variance,
                    status="skipped_insufficient_target_variance",
                    error_message="target has no variance at this timestep",
                ))
                continue
            y = work[target_col].to_numpy(dtype=float)
            x = work[[predictor]].to_numpy(dtype=float)
            groups = work["trial_uid"].to_numpy()
            splitter, actual_folds = valid_group_kfold(groups, cv_folds)
            if splitter is None:
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    n_classes_or_target_variance=target_variance,
                    status="skipped_insufficient_groups",
                    error_message="fewer than two trial groups available",
                ))
                continue
            preds = np.full(y.shape, np.nan, dtype=float)
            for train_idx, test_idx in splitter.split(x, y, groups):
                if np.var(y[train_idx]) <= 1e-12:
                    continue
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 21)))
                model.fit(x[train_idx], y[train_idx])
                preds[test_idx] = model.predict(x[test_idx])
            valid = np.isfinite(preds)
            if not np.any(valid):
                rows.append(predictor_result_base(
                    group_name,
                    target_col=target_col,
                    predictor=predictor,
                    n_rows=len(work),
                    n_trials=n_trials,
                    n_classes_or_target_variance=target_variance,
                    cv_folds=actual_folds,
                    status="failed_no_valid_predictions",
                    error_message="no cross-validated predictions were produced",
                ))
                continue
            yv = y[valid]
            pv = preds[valid]
            pear = pearsonr(yv, pv).statistic if len(np.unique(yv)) > 1 and len(np.unique(pv)) > 1 else np.nan
            spear = spearmanr(yv, pv).statistic if len(np.unique(yv)) > 1 and len(np.unique(pv)) > 1 else np.nan
            row = predictor_result_base(
                group_name,
                target_col=target_col,
                predictor=predictor,
                n_rows=len(yv),
                n_trials=len(np.unique(groups[valid])),
                n_classes_or_target_variance=float(np.var(yv)),
                cv_folds=actual_folds,
                status="ok",
            )
            row.update({
                "r2": r2_score(yv, pv),
                "pearson_r": pear,
                "spearman_r": spear,
                "mae": mean_absolute_error(yv, pv),
                "rmse": math.sqrt(mean_squared_error(yv, pv)),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_single_predictor_groups(prediction_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    target_metrics = {
        "current_best_path": "balanced_accuracy",
        "path_switch": "balanced_accuracy",
        "current_best_path_value": "r2",
    }
    for target_key, metric in target_metrics.items():
        res = prediction_results.get(target_key)
        if res is None or len(res) == 0 or metric not in res:
            continue
        ok = res[res["status"] == "ok"].copy()
        if len(ok) == 0:
            continue
        for (predictor_type, timestep), piece in ok.groupby(["predictor_type", "timestep"], dropna=False):
            rows.append({
                "target": piece["target"].iloc[0],
                "target_key": target_key,
                "timestep": timestep,
                "predictor_type": predictor_type,
                "metric": metric,
                "mean_metric": piece[metric].mean(),
                "max_metric": piece[metric].max(),
                "n_ok_models": int(len(piece)),
                "status": "ok",
                "error_message": "",
            })
    return pd.DataFrame(rows)


def run_predictions(df: pd.DataFrame, outdir: Path, cv_folds: int):
    if "z_mu_0" not in df.columns:
        print("Single-predictor latent analyses require at least one latent dimension. Skipping prediction models.")
        empty = pd.DataFrame()
        for name in [
            "current_best_path_prediction_results.csv",
            "path_switch_prediction_results.csv",
            "current_best_path_value_prediction_results.csv",
            "prior_diagnostics_results.csv",
        ]:
            empty.to_csv(outdir / name, index=False)
        return {}, {}

    predictors = SINGLE_PREDICTORS
    non_tied = df[~df["tie_flag"].astype(bool)].copy()
    results = {}

    cls_results = run_single_predictor_classification(
        non_tied,
        target_col="current_best_path",
        predictors=predictors,
        cv_folds=cv_folds,
        binary=False,
    )
    cls_results.to_csv(outdir / "current_best_path_prediction_results.csv", index=False)
    results["current_best_path"] = cls_results

    switch_df = non_tied[non_tied["current_best_path_switch"].notna()].copy()
    switch_results = run_single_predictor_classification(
        switch_df,
        target_col="current_best_path_switch",
        predictors=predictors,
        cv_folds=cv_folds,
        binary=True,
    )
    switch_results.to_csv(outdir / "path_switch_prediction_results.csv", index=False)
    results["path_switch"] = switch_results

    value_results = run_single_predictor_regression(
        df,
        target_col="current_best_path_value",
        predictors=predictors,
        cv_folds=cv_folds,
    )
    value_results.to_csv(outdir / "current_best_path_value_prediction_results.csv", index=False)
    results["current_best_path_value"] = value_results

    prior_diagnostics = summarize_single_predictor_groups(results)
    prior_diagnostics.to_csv(outdir / "prior_diagnostics_results.csv", index=False)
    results["prior_diagnostics"] = prior_diagnostics
    return results, {}


def run_temporal_direction_predictions(
    transition_df: pd.DataFrame,
    outdir: Path,
    cv_folds: int,
) -> Dict[str, pd.DataFrame]:
    result_files = {
        "temporal_current_best_path": "temporal_direction_current_best_path_prediction_results.csv",
        "temporal_path_switch": "temporal_direction_path_switch_prediction_results.csv",
        "temporal_best_path_value": "temporal_direction_best_path_value_prediction_results.csv",
        "temporal_best_path_value_change": "temporal_direction_best_path_value_change_prediction_results.csv",
    }
    if transition_df is None or len(transition_df) == 0:
        empty = pd.DataFrame()
        for filename in result_files.values():
            empty.to_csv(outdir / filename, index=False)
        empty.to_csv(outdir / "temporal_direction_analysis_summary.csv", index=False)
        return {key: empty for key in result_files}

    predictors = [predictor for predictor in TEMPORAL_PREDICTORS if predictor in transition_df.columns]
    if not predictors:
        empty = pd.DataFrame()
        for filename in result_files.values():
            empty.to_csv(outdir / filename, index=False)
        empty.to_csv(outdir / "temporal_direction_analysis_summary.csv", index=False)
        return {key: empty for key in result_files}

    results = {}
    non_tied_t2 = transition_df.copy()
    if "tie_flag_t2" in non_tied_t2.columns:
        non_tied_t2 = non_tied_t2[~non_tied_t2["tie_flag_t2"].astype(bool)].copy()

    cls_results = run_single_predictor_classification(
        non_tied_t2,
        target_col="current_best_path_t2",
        predictors=predictors,
        cv_folds=cv_folds,
        binary=False,
        view_fn=model_transition_views,
    )
    cls_results.to_csv(outdir / result_files["temporal_current_best_path"], index=False)
    results["temporal_current_best_path"] = cls_results

    switch_df = non_tied_t2[non_tied_t2["current_best_path_switch_t2"].notna()].copy()
    switch_results = run_single_predictor_classification(
        switch_df,
        target_col="current_best_path_switch_t2",
        predictors=predictors,
        cv_folds=cv_folds,
        binary=True,
        view_fn=model_transition_views,
    )
    switch_results.to_csv(outdir / result_files["temporal_path_switch"], index=False)
    results["temporal_path_switch"] = switch_results

    value_results = run_single_predictor_regression(
        transition_df,
        target_col="current_best_path_value_t2",
        predictors=predictors,
        cv_folds=cv_folds,
        view_fn=model_transition_views,
    )
    value_results.to_csv(outdir / result_files["temporal_best_path_value"], index=False)
    results["temporal_best_path_value"] = value_results

    value_change_results = run_single_predictor_regression(
        transition_df,
        target_col="best_path_value_change_t1_to_t2",
        predictors=predictors,
        cv_folds=cv_folds,
        view_fn=model_transition_views,
    )
    value_change_results.to_csv(outdir / result_files["temporal_best_path_value_change"], index=False)
    results["temporal_best_path_value_change"] = value_change_results

    temporal_summary = summarize_temporal_direction_results(results)
    temporal_summary.to_csv(outdir / "temporal_direction_analysis_summary.csv", index=False)
    return results


def summarize_temporal_direction_results(prediction_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    specs = {
        "temporal_current_best_path": ("current_best_path_t2", "balanced_accuracy"),
        "temporal_path_switch": ("current_best_path_switch_t2", "balanced_accuracy"),
        "temporal_best_path_value": ("current_best_path_value_t2", "r2"),
        "temporal_best_path_value_change": ("best_path_value_change_t1_to_t2", "r2"),
    }
    rows = []
    temporal_mu_types = {"temporal_mu_direction", "temporal_mu_radius"}
    temporal_halfplane_types = {
        "temporal_halfplane_direction_dim0",
        "temporal_halfplane_direction_dim1",
        "temporal_halfplane_radius_dim0",
        "temporal_halfplane_radius_dim1",
    }
    posterior_prior_types = {"posterior_prior_delta", "posterior_prior_kl"}
    for result_key, (target, metric) in specs.items():
        res = prediction_results.get(result_key)
        if res is None or len(res) == 0 or metric not in res:
            continue
        ok = res[res["status"] == "ok"].copy()
        if len(ok) == 0:
            continue
        agg = ok.groupby(["predictor", "predictor_type"], as_index=False, dropna=False)[metric].mean()

        def best_row(mask):
            piece = agg[mask].copy()
            if len(piece) == 0:
                return None
            return piece.sort_values(metric, ascending=False).iloc[0]

        best = best_row(np.ones(len(agg), dtype=bool))
        best_mu = best_row(agg["predictor_type"].isin(temporal_mu_types))
        best_halfplane = best_row(agg["predictor_type"].isin(temporal_halfplane_types))
        best_prior = best_row(agg["predictor_type"].isin(posterior_prior_types))
        if best is None:
            continue
        best_halfplane_value = np.nan if best_halfplane is None else float(best_halfplane[metric])
        best_mu_value = np.nan if best_mu is None else float(best_mu[metric])
        best_prior_value = np.nan if best_prior is None else float(best_prior[metric])
        temporal_values = np.asarray([best_mu_value, best_halfplane_value], dtype=float)
        best_temporal_value = (
            float(np.nanmax(temporal_values))
            if np.isfinite(temporal_values).any()
            else np.nan
        )
        rows.append({
            "target": target,
            "best_predictor": best["predictor"],
            "best_predictor_type": best["predictor_type"],
            "best_metric": metric,
            "best_metric_value": float(best[metric]),
            "best_temporal_mu_predictor": "" if best_mu is None else best_mu["predictor"],
            "best_temporal_halfplane_predictor": "" if best_halfplane is None else best_halfplane["predictor"],
            "best_posterior_prior_predictor": "" if best_prior is None else best_prior["predictor"],
            "temporal_halfplane_beats_temporal_mu": bool(
                np.isfinite(best_halfplane_value)
                and (not np.isfinite(best_mu_value) or best_halfplane_value > best_mu_value)
            ),
            "temporal_direction_beats_posterior_prior": bool(
                np.isfinite(best_temporal_value)
                and (
                    not np.isfinite(best_prior_value)
                    or best_temporal_value > best_prior_value
                )
            ),
        })
    return pd.DataFrame(rows)


def aggregate_single_predictor_metric(results: Optional[pd.DataFrame], metric: str) -> pd.DataFrame:
    if results is None or len(results) == 0 or metric not in results:
        return pd.DataFrame()
    piece = results[results["status"] == "ok"].copy() if "status" in results else results.copy()
    if len(piece) == 0:
        return pd.DataFrame()
    return (
        piece
        .groupby(["predictor", "predictor_type", "timestep"], as_index=False, dropna=False)[metric]
        .mean()
    )


def aggregate_single_predictor_metric_by_seed(results: Optional[pd.DataFrame], metric: str) -> pd.DataFrame:
    if results is None or len(results) == 0 or metric not in results or "seed" not in results:
        return pd.DataFrame()
    piece = results[results["status"] == "ok"].copy() if "status" in results else results.copy()
    if len(piece) == 0:
        return pd.DataFrame()
    return (
        piece
        .groupby(["seed", "predictor", "predictor_type", "timestep"], as_index=False, dropna=False)[metric]
        .mean()
    )


def metric_table(
    agg: pd.DataFrame,
    metric: str,
    *,
    predictor_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if len(agg) == 0:
        return pd.DataFrame()
    table = agg.pivot_table(index="predictor", columns="timestep", values=metric, aggfunc="mean")
    if predictor_order is None:
        predictor_order = table.mean(axis=1).sort_values(ascending=False).index.tolist()
    ordered = [predictor for predictor in predictor_order if predictor in table.index]
    extras = [predictor for predictor in table.index if predictor not in ordered]
    return table.loc[ordered + extras]


def file_token(value) -> str:
    text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def plot_single_predictor_metric(results: Optional[pd.DataFrame], *, metric: str, path: Path):
    agg = aggregate_single_predictor_metric(results, metric)
    if len(agg) == 0:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=panel_figsize(1, 1, title=True))
    for predictor, piece in agg.groupby("predictor"):
        piece = piece.sort_values("timestep")
        plt.plot(piece["timestep"], piece[metric], marker="o", linewidth=1.4, label=predictor)
    plt.xlabel("timestep")
    plt.ylabel(metric)
    plt.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_single_predictor_metric_by_type(
    results: Optional[pd.DataFrame],
    *,
    metric: str,
    figdir: Path,
    filename_prefix: str,
):
    agg = aggregate_single_predictor_metric(results, metric)
    if len(agg) == 0:
        return
    import matplotlib.pyplot as plt

    short_type_names = {
        "posterior_mean": "posterior_mean",
        "posterior_uncertainty": "posterior_uncertainty",
        "posterior_angle": "posterior_angle",
        "posterior_radius": "posterior_radius",
        "prior": "prior",
        "posterior_prior_delta": "prior_delta",
        "posterior_prior_kl": "posterior_prior_kl",
        "gaussian_manifold_disk": "gaussian_manifold_disk",
    }
    for predictor_type, type_piece in agg.groupby("predictor_type"):
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        for predictor, piece in type_piece.groupby("predictor"):
            piece = piece.sort_values("timestep")
            plt.plot(piece["timestep"], piece[metric], marker="o", label=predictor)
        plt.xlabel("timestep")
        plt.ylabel(metric)
        plt.legend(frameon=False, fontsize=PANEL_FONT_SIZE, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.72, 1])
        suffix = short_type_names.get(str(predictor_type), str(predictor_type))
        plt.savefig(figdir / f"{filename_prefix}_{suffix}.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_single_predictor_heatmap(
    results: Optional[pd.DataFrame],
    *,
    metric: str,
    path: Path,
    predictor_order: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    agg = aggregate_single_predictor_metric(results, metric)
    if len(agg) == 0:
        return
    import matplotlib.pyplot as plt

    table = metric_table(agg, metric, predictor_order=predictor_order)
    plt.figure(figsize=panel_figsize(1, 1, title=True))
    image = plt.imshow(
        table.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.yticks(np.arange(len(table.index)), table.index, fontsize=7)
    plt.xticks(np.arange(len(table.columns)), table.columns)
    plt.xlabel("timestep")
    plt.ylabel("predictor")
    plt.colorbar(image, label=metric)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def target_plot_specs() -> Dict[str, Dict[str, object]]:
    return {
        "current_best_path": {
            "metric": "balanced_accuracy",
            "filename": "single_predictor_current_best_path_heatmap",
            "label": "current_best_path",
        },
        "path_switch": {
            "metric": "balanced_accuracy",
            "filename": "single_predictor_path_switch_heatmap",
            "label": "current_best_path_switch",
        },
        "current_best_path_value": {
            "metric": "r2",
            "filename": "single_predictor_best_path_value_heatmap",
            "label": "current_best_path_value",
        },
    }


def plot_single_predictor_heatmaps_by_seed(prediction_results: Dict[str, pd.DataFrame], figdir: Path):
    for target_key, spec in target_plot_specs().items():
        metric = str(spec["metric"])
        filename = str(spec["filename"])
        agg_all = aggregate_single_predictor_metric(prediction_results.get(target_key), metric)
        agg_seed = aggregate_single_predictor_metric_by_seed(prediction_results.get(target_key), metric)
        if len(agg_all) == 0 or len(agg_seed) == 0:
            continue
        order = metric_table(agg_all, metric).index.tolist()
        finite_values = agg_seed[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_values) == 0:
            continue
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        for seed, seed_piece in agg_seed.groupby("seed", dropna=False):
            seed_table = metric_table(seed_piece, metric, predictor_order=order)
            if len(seed_table) == 0:
                continue
            import matplotlib.pyplot as plt

            plt.figure(
                figsize=panel_figsize(1, 1, title=True)
            )
            image = plt.imshow(
                seed_table.to_numpy(dtype=float),
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            plt.yticks(np.arange(len(seed_table.index)), seed_table.index, fontsize=7)
            plt.xticks(np.arange(len(seed_table.columns)), seed_table.columns)
            plt.xlabel("timestep")
            plt.ylabel("predictor")
            plt.title(f"seed {seed}")
            plt.colorbar(image, label=metric)
            plt.tight_layout()
            plt.savefig(figdir / f"{filename}_seed_{file_token(seed)}.png", dpi=180)
            plt.close()


def best_path_metric_lookup(results: Optional[pd.DataFrame]) -> Dict:
    lookup = {}
    if results is None or len(results) == 0 or "balanced_accuracy" not in results:
        return lookup
    ok = results[results["status"] == "ok"].copy()
    if len(ok) == 0:
        return lookup
    agg = (
        ok.groupby(["model_id", "timestep"], as_index=False, dropna=False)["balanced_accuracy"]
        .max()
    )
    for _, row in agg.iterrows():
        lookup[(row["model_id"], int(row["timestep"]))] = float(row["balanced_accuracy"])
    return lookup


def plot_latent_mu_scatter_by_timestep(
    df: pd.DataFrame,
    prediction_results: Dict[str, pd.DataFrame],
    figdir: Path,
):
    required = {"z_mu_0", "z_mu_1", "current_best_path", "timestep", "model_id"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    best_metric = best_path_metric_lookup(prediction_results.get("current_best_path"))
    filtered_df = rows_after_observed_reward_used_downstream(df)
    plot_df = filtered_df[
        filtered_df["current_best_path"].notna()
        & np.isfinite(filtered_df["z_mu_0"])
        & np.isfinite(filtered_df["z_mu_1"])
        & ~bool_series(filtered_df["tie_flag"])
    ].copy()
    if len(plot_df) == 0:
        return

    path_values = sorted(int(v) for v in plot_df["current_best_path"].dropna().unique())
    if len(path_values) == 0:
        return
    cmap = plt.get_cmap("tab10", len(path_values))
    boundaries = np.arange(len(path_values) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    path_to_color_index = {value: idx for idx, value in enumerate(path_values)}

    group_cols = [col for col in MODEL_GROUP_COLUMNS if col in plot_df.columns]
    if "model_id" not in group_cols:
        group_cols.append("model_id")
    for values, piece in plot_df.groupby(group_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(group_cols, values))
        timesteps = sorted(int(t) for t in piece["timestep"].dropna().unique())
        if not timesteps:
            continue
        xmin, xmax = piece["z_mu_0"].min(), piece["z_mu_0"].max()
        ymin, ymax = piece["z_mu_1"].min(), piece["z_mu_1"].max()
        xpad = max((xmax - xmin) * 0.05, 1e-3)
        ypad = max((ymax - ymin) * 0.05, 1e-3)
        fig, axes = plt.subplots(
            1,
            len(timesteps),
            figsize=panel_figsize(len(timesteps), 1, title=True),
            squeeze=False,
        )
        for ax, timestep in zip(axes[0], timesteps):
            t_piece = piece[piece["timestep"] == timestep]
            if len(t_piece) > 5000:
                t_piece = t_piece.sample(5000, random_state=17 + timestep)
            color_index = t_piece["current_best_path"].astype(int).map(path_to_color_index).to_numpy()
            scatter = ax.scatter(
                t_piece["z_mu_0"],
                t_piece["z_mu_1"],
                c=color_index,
                cmap=cmap,
                norm=norm,
                s=5,
                alpha=0.55,
                linewidths=0,
            )
            ba = best_metric.get((group_name.get("model_id"), timestep))
            ba_text = "" if ba is None else f"\nbest BA={ba:.3f}"
            ax.set_title(f"t={timestep}\nn={len(t_piece)}{ba_text}", fontsize=PANEL_FONT_SIZE)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_xlabel("z_mu_0")
            ax.set_ylabel("z_mu_1")
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), ticks=np.arange(len(path_values)))
        cbar.ax.set_yticklabels([str(v) for v in path_values])
        cbar.set_label("current_best_path")
        lambda_token = file_token(group_name.get("lambda_value", "na"))
        seed_token = file_token(group_name.get("seed", "na"))
        beta_token = file_token(group_name.get("beta", "na"))
        model_token = file_token(group_name.get("model_id", "model"))
        fig.suptitle(f"lambda={group_name.get('lambda_value')} seed={group_name.get('seed')}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.96, 0.92])
        fig.savefig(
            figdir / (
                "latent_mu_scatter_by_current_best_path_by_timestep_"
                f"lambda_{lambda_token}_seed_{seed_token}_beta_{beta_token}_{model_token}.png"
            ),
            dpi=180,
        )
        plt.close(fig)


def plot_temporal_direction_metric(results: Optional[pd.DataFrame], *, metric: str, path: Path):
    if results is None or len(results) == 0 or metric not in results:
        return
    ok = results[results["status"] == "ok"].copy() if "status" in results else results.copy()
    if len(ok) == 0:
        return
    agg = ok.groupby(["predictor", "predictor_type"], as_index=False, dropna=False)[metric].mean()
    if len(agg) == 0:
        return
    agg = agg.sort_values(metric, ascending=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    type_values = agg["predictor_type"].astype(str).unique().tolist()
    cmap = plt.get_cmap("tab20", max(len(type_values), 1))
    color_map = {ptype: cmap(i) for i, ptype in enumerate(type_values)}
    colors = [color_map[str(ptype)] for ptype in agg["predictor_type"]]
    fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
    ax.barh(agg["predictor"], agg[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_ylabel("predictor")
    ax.tick_params(axis="y", labelsize=7)
    handles = [Patch(color=color_map[ptype], label=ptype) for ptype in type_values]
    fig.legend(handles=handles, frameon=False, fontsize=PANEL_FONT_SIZE, bbox_to_anchor=(0.99, 0.98), loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.76, 1])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def halfplane_axis_limits(df: pd.DataFrame) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    limits = {}
    for dim in available_halfplane_dims(df):
        x = pd.to_numeric(df.get(f"halfplane_x_{dim}", pd.Series(dtype=float)), errors="coerce")
        y = pd.to_numeric(df.get(f"halfplane_y_{dim}", pd.Series(dtype=float)), errors="coerce")
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        xvals = x[finite].to_numpy(dtype=float)
        yvals = y[finite].to_numpy(dtype=float)
        xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        xpad = max((xmax - xmin) * 0.08, 1e-3)
        ypad = max((ymax - ymin) * 0.08, 1e-3)
        limits[dim] = ((xmin - xpad, xmax + xpad), (max(0.0, ymin - ypad), ymax + ypad))
    return limits


def temporal_halfplane_axis_limits(
    transition_df: pd.DataFrame,
) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    limits = {}
    if transition_df is None or len(transition_df) == 0:
        return limits
    for dim in available_halfplane_dims(transition_df, suffix="_t1"):
        x_cols = [f"halfplane_x_{dim}_t1", f"halfplane_x_{dim}_t2"]
        y_cols = [f"halfplane_y_{dim}_t1", f"halfplane_y_{dim}_t2"]
        if not set(x_cols + y_cols).issubset(transition_df.columns):
            continue
        x = pd.concat([pd.to_numeric(transition_df[col], errors="coerce") for col in x_cols], ignore_index=True)
        y = pd.concat([pd.to_numeric(transition_df[col], errors="coerce") for col in y_cols], ignore_index=True)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        xvals = x[finite].to_numpy(dtype=float)
        yvals = y[finite].to_numpy(dtype=float)
        xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        xpad = max((xmax - xmin) * 0.10, 1e-3)
        ypad = max((ymax - ymin) * 0.10, 1e-3)
        limits[dim] = ((xmin - xpad, xmax + xpad), (max(0.0, ymin - ypad), ymax + ypad))
    return limits


def draw_halfplane_panel(ax, piece: pd.DataFrame, dim: int, limits, *, color=None, c=None, cmap=None, label=None):
    if len(piece) == 0:
        ax.set_axis_off()
        return None
    if len(piece) > 6000:
        piece = piece.sample(6000, random_state=53 + dim)
    xcol = f"halfplane_x_{dim}"
    ycol = f"halfplane_y_{dim}"
    if c is None:
        try:
            artist = ax.hexbin(piece[xcol], piece[ycol], gridsize=35, mincnt=1, cmap="viridis")
        except Exception:
            artist = ax.scatter(piece[xcol], piece[ycol], s=4, alpha=0.25, linewidths=0, color=color, label=label)
    else:
        artist = ax.scatter(piece[xcol], piece[ycol], c=c, cmap=cmap, s=5, alpha=0.45, linewidths=0)
    ax.set_xlabel(f"mu_{dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
    ax.set_ylabel(f"sigma_{dim}", fontsize=PANEL_FONT_SIZE)
    if dim in limits:
        ax.set_xlim(*limits[dim][0])
        ax.set_ylim(*limits[dim][1])
    return artist


def plot_halfplane_current_best_path(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    required = {"current_best_path", "timestep", "tie_flag"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.patches import Patch

    plot_df = rows_after_observed_reward_used_downstream(df)
    plot_df = plot_df[plot_df["current_best_path"].notna() & ~plot_df["tie_flag"].astype(bool)].copy()
    if len(plot_df) == 0:
        return
    dims = available_halfplane_dims(plot_df)
    if not dims:
        return
    timesteps = sorted(int(v) for v in plot_df["timestep"].dropna().unique())
    path_values = sorted(plot_df["current_best_path"].dropna().unique())
    limits = limits or halfplane_axis_limits(plot_df)
    cmap = plt.get_cmap("tab10", max(len(path_values), 1))
    path_to_idx = {path_value: idx for idx, path_value in enumerate(path_values)}
    norm = BoundaryNorm(np.arange(len(path_values) + 1) - 0.5, cmap.N)

    fig, axes = plt.subplots(
        len(timesteps),
        len(dims),
        figsize=panel_figsize(len(dims), len(timesteps), title=True),
        squeeze=False,
    )
    scatter = None
    for row_i, timestep in enumerate(timesteps):
        t_piece = plot_df[plot_df["timestep"] == timestep]
        if len(t_piece) > 7000:
            t_piece = t_piece.sample(7000, random_state=41 + timestep)
        color_index = t_piece["current_best_path"].map(path_to_idx).to_numpy(dtype=float)
        for col_i, dim in enumerate(dims):
            ax = axes[row_i, col_i]
            scatter = draw_halfplane_panel(ax, t_piece, dim, limits, c=color_index, cmap=cmap)
            ax.set_title(f"t={timestep}, dim={dim}", fontsize=PANEL_FONT_SIZE)
            if scatter is not None:
                scatter.set_norm(norm)
    handles = [Patch(color=cmap(idx), label=f"path {path}") for path, idx in path_to_idx.items()]
    fig.legend(handles=handles, frameon=False, fontsize=PANEL_FONT_SIZE, bbox_to_anchor=(0.99, 0.98), loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(
        figdir / f"gaussian_halfplane_current_best_path_facets{filename_suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    for path_value in path_values:
        path_piece = plot_df[plot_df["current_best_path"] == path_value]
        if len(path_piece) == 0:
            continue
        fig, axes = plt.subplots(
            len(timesteps),
            len(dims),
            figsize=panel_figsize(len(dims), len(timesteps), title=True),
            squeeze=False,
        )
        last_artist = None
        for row_i, timestep in enumerate(timesteps):
            t_piece = path_piece[path_piece["timestep"] == timestep]
            for col_i, dim in enumerate(dims):
                ax = axes[row_i, col_i]
                last_artist = draw_halfplane_panel(ax, t_piece, dim, limits)
                ax.set_title(f"path={path_value}, t={timestep}, dim={dim}", fontsize=PANEL_FONT_SIZE)
        if last_artist is not None and hasattr(last_artist, "get_array"):
            fig.tight_layout(rect=[0, 0, 0.84, 1])
            cax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
            cbar = fig.colorbar(last_artist, cax=cax)
            cbar.set_label("count")
        else:
            fig.tight_layout(rect=[0, 0, 0.94, 1])
        fig.savefig(
            figdir / f"gaussian_halfplane_current_best_path_path_{file_token(path_value)}{filename_suffix}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_halfplane_best_path_value(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    required = {"current_best_path_value", "timestep"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plot_df = rows_after_observed_reward_used_downstream(df)
    plot_df = plot_df[plot_df["current_best_path_value"].notna()].copy()
    if len(plot_df) == 0:
        return
    dims = available_halfplane_dims(plot_df)
    if not dims:
        return
    timesteps = sorted(int(v) for v in plot_df["timestep"].dropna().unique())
    limits = limits or halfplane_axis_limits(plot_df)
    fig, axes = plt.subplots(
        len(timesteps),
        len(dims),
        figsize=panel_figsize(len(dims), len(timesteps), title=True),
        squeeze=False,
    )
    scatter = None
    for row_i, timestep in enumerate(timesteps):
        t_piece = plot_df[plot_df["timestep"] == timestep]
        if len(t_piece) > 8000:
            t_piece = t_piece.sample(8000, random_state=61 + timestep)
        c = pd.to_numeric(t_piece["current_best_path_value"], errors="coerce")
        for col_i, dim in enumerate(dims):
            ax = axes[row_i, col_i]
            scatter = draw_halfplane_panel(ax, t_piece, dim, limits, c=c, cmap="viridis")
            ax.set_title(f"t={timestep}, dim={dim}", fontsize=PANEL_FONT_SIZE)
    if scatter is not None:
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        cax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("current_best_path_value")
    else:
        fig.tight_layout(rect=[0, 0, 0.93, 1])
    fig.savefig(
        figdir / f"gaussian_halfplane_best_path_value_by_timestep_dim{filename_suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    value = pd.to_numeric(plot_df["current_best_path_value"], errors="coerce")
    if value.nunique(dropna=True) < 3:
        return
    try:
        plot_df["current_best_path_value_bin"] = pd.qcut(value, q=3, labels=["low", "medium", "high"], duplicates="drop")
    except ValueError:
        return
    bins = [str(v) for v in plot_df["current_best_path_value_bin"].dropna().unique()]
    if not bins:
        return
    bin_to_color = {name: plt.get_cmap("Set2", len(bins))(i) for i, name in enumerate(bins)}
    fig, axes = plt.subplots(
        len(timesteps),
        len(dims),
        figsize=panel_figsize(len(dims), len(timesteps), title=True),
        squeeze=False,
    )
    for row_i, timestep in enumerate(timesteps):
        t_piece = plot_df[plot_df["timestep"] == timestep]
        if len(t_piece) > 8000:
            t_piece = t_piece.sample(8000, random_state=71 + timestep)
        colors = t_piece["current_best_path_value_bin"].astype(str).map(bin_to_color)
        for col_i, dim in enumerate(dims):
            ax = axes[row_i, col_i]
            ax.scatter(
                t_piece[f"halfplane_x_{dim}"],
                t_piece[f"halfplane_y_{dim}"],
                color=list(colors),
                s=5,
                alpha=0.35,
                linewidths=0,
            )
            ax.set_xlabel(f"mu_{dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
            ax.set_ylabel(f"sigma_{dim}", fontsize=PANEL_FONT_SIZE)
            ax.set_title(f"t={timestep}, dim={dim}", fontsize=PANEL_FONT_SIZE)
            if dim in limits:
                ax.set_xlim(*limits[dim][0])
                ax.set_ylim(*limits[dim][1])
    handles = [Patch(color=color, label=name) for name, color in bin_to_color.items()]
    fig.legend(handles=handles, frameon=False, bbox_to_anchor=(0.99, 0.98), loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(
        figdir / f"gaussian_halfplane_best_path_value_quantile_facets{filename_suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_t1_halfplane_by_observed_value(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    required = {"timestep", "current_best_path_value", "t1_observed_value"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plot_df = rows_after_observed_reward_used_downstream(df, timestep=1)
    plot_df = plot_df[
        plot_df["current_best_path_value"].notna()
        & plot_df["t1_observed_value"].notna()
    ].copy()
    if len(plot_df) == 0:
        return
    available_dims = available_halfplane_dims(plot_df)
    dims = [dim for dim in (0, 1) if dim in available_dims]
    if not dims:
        dims = available_dims[:2]
    if not dims:
        return
    limits = limits or halfplane_axis_limits(plot_df)
    color_values = pd.to_numeric(plot_df["current_best_path_value"], errors="coerce")
    finite_color = np.isfinite(color_values)
    if not finite_color.any():
        return
    vmin = float(color_values[finite_color].min())
    vmax = float(color_values[finite_color].max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"

    def plot_piece(piece: pd.DataFrame, filename: str, title_suffix: str):
        finite_cols = ["current_best_path_value"]
        for dim in dims:
            finite_cols.extend([f"halfplane_x_{dim}", f"halfplane_y_{dim}"])
        piece = piece[
            piece["current_best_path_value"].notna()
            & np.isfinite(piece[finite_cols].apply(pd.to_numeric, errors="coerce")).all(axis=1)
        ].copy()
        if len(piece) == 0:
            return
        if len(piece) > 8000:
            piece = piece.sample(8000, random_state=131)
        fig, axes = plt.subplots(
            1,
            len(dims),
            figsize=panel_figsize(len(dims), 1, title=True),
            squeeze=False,
        )
        scatter = None
        c = pd.to_numeric(piece["current_best_path_value"], errors="coerce")
        for col_i, dim in enumerate(dims):
            ax = axes[0, col_i]
            scatter = ax.scatter(
                piece[f"halfplane_x_{dim}"],
                piece[f"halfplane_y_{dim}"],
                c=c,
                cmap=cmap,
                norm=norm,
                s=6,
                alpha=0.45,
                linewidths=0,
            )
            ax.set_xlabel(f"mu_{dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
            ax.set_ylabel(f"sigma_{dim}", fontsize=PANEL_FONT_SIZE)
            ax.set_title(f"t=1, dim={dim}{title_suffix}", fontsize=PANEL_FONT_SIZE)
            if dim in limits:
                ax.set_xlim(*limits[dim][0])
                ax.set_ylim(*limits[dim][1])
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        if scatter is not None:
            cax = fig.add_axes([0.88, 0.18, 0.025, 0.64])
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label("current_best_path_value")
        fig.savefig(figdir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    plot_piece(
        plot_df,
        f"gaussian_halfplane_t1_current_path_value_by_dim{filename_suffix}.png",
        "",
    )
    for observed_value in sorted(plot_df["t1_observed_value"].dropna().unique()):
        value_piece = plot_df[np.isclose(plot_df["t1_observed_value"], observed_value)]
        plot_piece(
            value_piece,
            (
                "gaussian_halfplane_t1_current_path_value_by_dim_"
                f"t1_observed_{file_token(observed_value)}{filename_suffix}.png"
            ),
            f", observed={observed_value:g}",
        )


def plot_t1_halfplane_observed_reward_by_dim(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    required = {"timestep", "t1_observed_value"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plot_df = rows_after_observed_reward_used_downstream(df, timestep=1)
    plot_df = plot_df[plot_df["t1_observed_value"].notna()].copy()
    if len(plot_df) == 0:
        return
    available_dims = available_halfplane_dims(plot_df)
    dims = [dim for dim in (0, 1) if dim in available_dims]
    if not dims:
        dims = available_dims[:2]
    if not dims:
        return
    finite_cols = ["t1_observed_value"]
    for dim in dims:
        finite_cols.extend([f"halfplane_x_{dim}", f"halfplane_y_{dim}"])
    plot_df = plot_df[
        np.isfinite(plot_df[finite_cols].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    ].copy()
    if len(plot_df) == 0:
        return
    if len(plot_df) > 8000:
        plot_df = plot_df.sample(8000, random_state=191)
    limits = limits or halfplane_axis_limits(plot_df)
    color_values = pd.to_numeric(plot_df["t1_observed_value"], errors="coerce")
    vmin = float(color_values.min())
    vmax = float(color_values.max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1,
        len(dims),
        figsize=panel_figsize(len(dims), 1, title=True),
        squeeze=False,
    )
    scatter = None
    for col_i, dim in enumerate(dims):
        ax = axes[0, col_i]
        scatter = ax.scatter(
            plot_df[f"halfplane_x_{dim}"],
            plot_df[f"halfplane_y_{dim}"],
            c=color_values,
            cmap="viridis",
            norm=norm,
            s=6,
            alpha=0.45,
            linewidths=0,
        )
        display_dim = dim + 1
        ax.set_xlabel(f"mu_{display_dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
        ax.set_ylabel(f"sigma_{display_dim}", fontsize=PANEL_FONT_SIZE)
        ax.set_title(f"t=1, latent dim {display_dim}", fontsize=PANEL_FONT_SIZE)
        if dim in limits:
            ax.set_xlim(*limits[dim][0])
            ax.set_ylim(*limits[dim][1])
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    if scatter is not None:
        cax = fig.add_axes([0.88, 0.18, 0.025, 0.64])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("observed reward")
    fig.savefig(
        figdir / f"gaussian_halfplane_t1_observed_reward_by_dim{filename_suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_poincare_disk_t1_t2_rewards_by_dim(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
):
    required = {"timestep", "reward_t1", "reward_t2"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plot_df = add_geometry_meaning_columns(df)
    dims = []
    for dim in (0, 1):
        needed = {
            f"poincare_disk_x_{dim}_t1",
            f"poincare_disk_y_{dim}_t1",
            f"poincare_disk_x_{dim}_t2",
            f"poincare_disk_y_{dim}_t2",
        }
        if needed.intersection(plot_df.columns):
            dims.append(dim)
    if not dims:
        return

    def reward_norm(piece: pd.DataFrame, reward_col: str):
        reward_values = pd.to_numeric(piece[reward_col], errors="coerce")
        reward_values = reward_values[np.isfinite(reward_values)]
        if len(reward_values) == 0:
            return None
        vmin = float(reward_values.min())
        vmax = float(reward_values.max())
        if math.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5
        return Normalize(vmin=vmin, vmax=vmax)

    def draw_disk_panel(ax, piece: pd.DataFrame, dim: int, timestep: int, reward_col: str, norm):
        x_col = f"poincare_disk_x_{dim}_t{timestep}"
        y_col = f"poincare_disk_y_{dim}_t{timestep}"
        if x_col not in piece.columns or y_col not in piece.columns:
            ax.set_axis_off()
            return None
        numeric = piece[[x_col, y_col, reward_col]].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric).all(axis=1)
        if not finite.any():
            ax.set_axis_off()
            return None
        scatter = ax.scatter(
            numeric.loc[finite, x_col],
            numeric.loc[finite, y_col],
            c=numeric.loc[finite, reward_col],
            cmap="viridis",
            norm=norm,
            s=6,
            alpha=0.45,
            linewidths=0,
        )
        circle = plt.Circle((0, 0), 1.0, fill=False, color="0.35", linewidth=0.8)
        ax.add_patch(circle)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        display_dim = dim + 1
        ax.set_xlabel(f"disk x dim {display_dim}", fontsize=PANEL_FONT_SIZE)
        ax.set_ylabel(f"disk y dim {display_dim}", fontsize=PANEL_FONT_SIZE)
        return scatter

    t1_piece = rows_after_observed_reward_used_downstream(plot_df, timestep=1)
    if len(t1_piece) > 8000:
        t1_piece = t1_piece.sample(8000, random_state=242)
    t1_norm = reward_norm(t1_piece, "reward_t1")
    if t1_norm is not None and len(t1_piece) > 0:
        fig, axes = plt.subplots(
            1,
            len(dims),
            figsize=panel_figsize(len(dims), 1, colorbar=True, title=True),
            squeeze=False,
        )
        scatter = None
        for col_i, dim in enumerate(dims):
            ax = axes[0, col_i]
            scatter = draw_disk_panel(ax, t1_piece, dim, 1, "reward_t1", t1_norm)
            ax.set_title(f"t=1, color=reward_t1", fontsize=PANEL_FONT_SIZE)
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        if scatter is not None:
            cax = fig.add_axes([0.88, 0.18, 0.025, 0.64])
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label("observed reward at t1")
        fig.savefig(
            figdir / f"poincare_disk_t1_observed_reward_by_dim{filename_suffix}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    t2_piece = rows_after_observed_reward_used_downstream(plot_df, timestep=2)
    t2_piece = t2_piece[pd.to_numeric(t2_piece["reward_t2"], errors="coerce").notna()].copy()
    if len(t2_piece) > 0:
        reward_t1_values = []
        reward_t1_numeric = pd.to_numeric(t2_piece["reward_t1"], errors="coerce")
        for reward_t1_value in sorted(reward_t1_numeric.dropna().unique()):
            group = t2_piece[np.isclose(reward_t1_numeric, reward_t1_value)]
            has_points = False
            for dim in dims:
                x_col = f"poincare_disk_x_{dim}_t2"
                y_col = f"poincare_disk_y_{dim}_t2"
                if x_col in group.columns and y_col in group.columns:
                    numeric = group[[x_col, y_col, "reward_t2"]].apply(pd.to_numeric, errors="coerce")
                    has_points = has_points or np.isfinite(numeric).all(axis=1).any()
            if has_points:
                reward_t1_values.append(reward_t1_value)
    else:
        reward_t1_values = []
    t2_norm = reward_norm(t2_piece, "reward_t2") if len(t2_piece) > 0 else None
    if t2_norm is not None and reward_t1_values:
        nrows = len(reward_t1_values)
        fig, axes = plt.subplots(
            nrows,
            len(dims),
            figsize=panel_figsize(len(dims), nrows, colorbar=True, title=True),
            squeeze=False,
        )
        scatter = None
        for row_i, reward_t1_value in enumerate(reward_t1_values):
            row_piece = t2_piece[
                np.isclose(pd.to_numeric(t2_piece["reward_t1"], errors="coerce"), reward_t1_value)
            ].copy()
            if len(row_piece) > 2000:
                row_piece = row_piece.sample(2000, random_state=251 + row_i)
            for col_i, dim in enumerate(dims):
                ax = axes[row_i, col_i]
                scatter = draw_disk_panel(ax, row_piece, dim, 2, "reward_t2", t2_norm)
                ax.set_title(f"t=2, R1={reward_t1_value:g}", fontsize=PANEL_FONT_SIZE)
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        if scatter is not None:
            cax = fig.add_axes([0.88, 0.18, 0.025, 0.64])
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label("observed reward at t2")
        fig.savefig(
            figdir / f"poincare_disk_t2_observed_reward_by_t1_observed_dim{filename_suffix}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_t1_t2_halfplane_by_t1_observed_value(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    plot_overview: bool = True,
    plot_sliced_pdf: bool = True,
):
    required = {"timestep", "current_best_path_value", "t1_observed_value"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    pieces = [
        rows_after_observed_reward_used_downstream(df, timestep=timestep)
        for timestep in (1, 2)
    ]
    plot_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    plot_df = plot_df[
        plot_df["current_best_path_value"].notna()
        & plot_df["t1_observed_value"].notna()
    ].copy()
    if len(plot_df) == 0:
        return
    available_dims = available_halfplane_dims(plot_df)
    dims = [dim for dim in (0, 1) if dim in available_dims]
    if not dims:
        dims = available_dims[:2]
    if not dims:
        return
    timesteps = [t for t in (1, 2) if (pd.to_numeric(plot_df["timestep"], errors="coerce") == t).any()]
    if not timesteps:
        return
    limits = limits or halfplane_axis_limits(plot_df)
    color_values = pd.to_numeric(plot_df["current_best_path_value"], errors="coerce")
    finite_color = np.isfinite(color_values)
    if not finite_color.any():
        return
    vmin = float(color_values[finite_color].min())
    vmax = float(color_values[finite_color].max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"

    def plot_piece(piece: pd.DataFrame, filename: str, title_suffix: str):
        finite_cols = ["current_best_path_value"]
        for dim in dims:
            finite_cols.extend([f"halfplane_x_{dim}", f"halfplane_y_{dim}"])
        numeric_finite = piece[finite_cols].apply(pd.to_numeric, errors="coerce")
        piece = piece[
            piece["current_best_path_value"].notna()
            & np.isfinite(numeric_finite).all(axis=1)
        ].copy()
        if len(piece) == 0:
            return
        fig, axes = plt.subplots(
            len(timesteps),
            len(dims),
            figsize=panel_figsize(len(dims), len(timesteps), title=True),
            squeeze=False,
        )
        scatter = None
        for row_i, timestep in enumerate(timesteps):
            t_piece = piece[pd.to_numeric(piece["timestep"], errors="coerce") == timestep].copy()
            if len(t_piece) > 8000:
                t_piece = t_piece.sample(8000, random_state=151 + timestep)
            c = pd.to_numeric(t_piece["current_best_path_value"], errors="coerce")
            for col_i, dim in enumerate(dims):
                ax = axes[row_i, col_i]
                if len(t_piece) == 0:
                    ax.set_axis_off()
                    continue
                scatter = ax.scatter(
                    t_piece[f"halfplane_x_{dim}"],
                    t_piece[f"halfplane_y_{dim}"],
                    c=c,
                    cmap=cmap,
                    norm=norm,
                    s=6,
                    alpha=0.45,
                    linewidths=0,
                )
                ax.set_xlabel(f"mu_{dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
                ax.set_ylabel(f"sigma_{dim}", fontsize=PANEL_FONT_SIZE)
                ax.set_title(f"t={timestep}, dim={dim}{title_suffix}", fontsize=PANEL_FONT_SIZE)
                if dim in limits:
                    ax.set_xlim(*limits[dim][0])
                    ax.set_ylim(*limits[dim][1])
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        if scatter is not None:
            cax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label("current_best_path_value")
        fig.savefig(figdir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    if plot_overview:
        plot_piece(
            plot_df,
            f"gaussian_halfplane_t1_t2_current_path_value_by_dim{filename_suffix}.png",
            "",
        )

    if not plot_sliced_pdf:
        return

    t2_df = rows_after_observed_reward_used_downstream(plot_df, timestep=2)
    if len(t2_df) == 0:
        return
    observed_values = sorted(t2_df["t1_observed_value"].dropna().unique())
    if not observed_values:
        return
    color_values_t2 = pd.to_numeric(t2_df["current_best_path_value"], errors="coerce")
    finite_color_t2 = np.isfinite(color_values_t2)
    if not finite_color_t2.any():
        return
    vmin_t2 = float(color_values_t2[finite_color_t2].min())
    vmax_t2 = float(color_values_t2[finite_color_t2].max())
    if math.isclose(vmin_t2, vmax_t2):
        vmin_t2 -= 0.5
        vmax_t2 += 0.5
    norm_t2 = Normalize(vmin=vmin_t2, vmax=vmax_t2)
    shared_ylim = None
    y_limits = [limits[dim][1] for dim in dims if dim in limits]
    if y_limits:
        shared_ylim = (
            min(limit[0] for limit in y_limits),
            max(limit[1] for limit in y_limits),
        )
    fig, axes = plt.subplots(
        len(observed_values),
        len(dims),
        figsize=panel_figsize(len(dims), len(observed_values), title=True),
        squeeze=False,
    )
    scatter = None
    for row_i, observed_value in enumerate(observed_values):
        value_piece = t2_df[np.isclose(t2_df["t1_observed_value"], observed_value)].copy()
        if len(value_piece) > 8000:
            value_piece = value_piece.sample(8000, random_state=181 + row_i)
        c = pd.to_numeric(value_piece["current_best_path_value"], errors="coerce")
        for col_i, dim in enumerate(dims):
            ax = axes[row_i, col_i]
            if len(value_piece) == 0:
                ax.set_axis_off()
                continue
            scatter = ax.scatter(
                value_piece[f"halfplane_x_{dim}"],
                value_piece[f"halfplane_y_{dim}"],
                c=c,
                cmap=cmap,
                norm=norm_t2,
                s=6,
                alpha=0.45,
                linewidths=0,
            )
            ax.set_xlabel(f"mu_{dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
            ax.set_ylabel(f"sigma_{dim}", fontsize=PANEL_FONT_SIZE)
            ax.set_title(f"t=2, dim={dim}, t1 observed={observed_value:g}", fontsize=PANEL_FONT_SIZE)
            if dim in limits:
                ax.set_xlim(*limits[dim][0])
                ax.set_ylim(*(shared_ylim or limits[dim][1]))
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    if scatter is not None:
        cax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("current_best_path_value at t=2")
    fig.savefig(
        figdir / f"gaussian_halfplane_t2_current_path_value_by_t1_observed_dim{filename_suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_halfplane_path_switch(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    required = {"current_best_path_switch", "timestep"}
    if not required.issubset(df.columns):
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plot_df = rows_after_observed_reward_used_downstream(df)
    plot_df = plot_df[plot_df["current_best_path_switch"].notna()].copy()
    if len(plot_df) == 0:
        return
    dims = available_halfplane_dims(plot_df)
    if not dims:
        return
    timesteps = sorted(int(v) for v in plot_df["timestep"].dropna().unique())
    switch_values = sorted(plot_df["current_best_path_switch"].dropna().unique())
    limits = limits or halfplane_axis_limits(plot_df)
    cmap = plt.get_cmap("tab10", max(len(switch_values), 1))
    switch_to_color = {value: cmap(idx) for idx, value in enumerate(switch_values)}
    fig, axes = plt.subplots(
        len(timesteps),
        len(dims),
        figsize=panel_figsize(len(dims), len(timesteps), title=True),
        squeeze=False,
    )
    for row_i, timestep in enumerate(timesteps):
        t_piece = plot_df[plot_df["timestep"] == timestep]
        if len(t_piece) > 8000:
            t_piece = t_piece.sample(8000, random_state=81 + timestep)
        colors = t_piece["current_best_path_switch"].map(switch_to_color)
        for col_i, dim in enumerate(dims):
            ax = axes[row_i, col_i]
            ax.scatter(
                t_piece[f"halfplane_x_{dim}"],
                t_piece[f"halfplane_y_{dim}"],
                color=list(colors),
                s=5,
                alpha=0.38,
                linewidths=0,
            )
            ax.set_xlabel(f"mu_{dim} / sqrt(2)", fontsize=PANEL_FONT_SIZE)
            ax.set_ylabel(f"sigma_{dim}", fontsize=PANEL_FONT_SIZE)
            ax.set_title(f"t={timestep}, dim={dim}", fontsize=PANEL_FONT_SIZE)
            if dim in limits:
                ax.set_xlim(*limits[dim][0])
                ax.set_ylim(*limits[dim][1])
    handles = [
        Patch(color=color, label=f"switch {value}")
        for value, color in switch_to_color.items()
    ]
    fig.legend(handles=handles, frameon=False, bbox_to_anchor=(0.99, 0.98), loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(
        figdir / f"gaussian_halfplane_path_switch_by_timestep_dim{filename_suffix}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temporal_halfplane_arrows(
    transition_df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    if transition_df is None or len(transition_df) == 0:
        return
    transition_df = transitions_after_observed_rewards_used_downstream(transition_df)
    if len(transition_df) == 0:
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    targets = [
        ("current_best_path_switch_t2", "switch", False),
        ("current_best_path_t2", "current_best_path", False),
        ("current_best_path_value_t2", "best_path_value", True),
        ("best_path_value_change_t1_to_t2", "value_change", True),
    ]
    transition_groups = (
        transition_df.groupby("transition", dropna=False)
        if "transition" in transition_df.columns
        else [("all_transitions", transition_df)]
    )
    for transition_label, transition_piece in transition_groups:
        transition_token = file_token(transition_label)
        for dim in available_halfplane_dims(transition_piece, suffix="_t1"):
            x1 = f"halfplane_x_{dim}_t1"
            y1 = f"halfplane_y_{dim}_t1"
            dx = f"temporal_delta_halfplane_x_{dim}"
            dy = f"temporal_delta_halfplane_y_{dim}"
            if not {x1, y1, dx, dy}.issubset(transition_piece.columns):
                continue
            for target, label, continuous in targets:
                if target not in transition_piece.columns:
                    continue
                piece = transition_piece[
                    transition_piece[target].notna()
                    & np.isfinite(transition_piece[[x1, y1, dx, dy]]).all(axis=1)
                ].copy()
                if len(piece) == 0:
                    continue
                if len(piece) > 1500:
                    piece = piece.sample(1500, random_state=101 + dim)
                fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
                continuous_mappable = None
                if continuous:
                    values = pd.to_numeric(piece[target], errors="coerce")
                    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
                    colors = plt.get_cmap("coolwarm")(norm(values))
                    ax.quiver(
                        piece[x1],
                        piece[y1],
                        piece[dx],
                        piece[dy],
                        color=colors,
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        alpha=0.45,
                        width=0.002,
                    )
                    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
                    sm.set_array([])
                    continuous_mappable = sm
                else:
                    categories = sorted(piece[target].dropna().unique())
                    cmap = plt.get_cmap("tab10", max(len(categories), 1))
                    handles = []
                    for idx, category in enumerate(categories):
                        cat_piece = piece[piece[target] == category]
                        ax.quiver(
                            cat_piece[x1],
                            cat_piece[y1],
                            cat_piece[dx],
                            cat_piece[dy],
                            color=cmap(idx),
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            alpha=0.42,
                            width=0.002,
                        )
                        handles.append(Patch(color=cmap(idx), label=str(category)))
                    fig.legend(handles=handles, frameon=False, bbox_to_anchor=(0.99, 0.98), loc="upper right")
                ax.set_xlabel(f"mu_{dim} at start / sqrt(2)")
                ax.set_ylabel(f"sigma_{dim} at start")
                ax.set_title(f"{transition_label} half-plane movement, dim {dim}")
                ax.axhline(0, color="0.85", linewidth=0.8)
                ax.axvline(0, color="0.85", linewidth=0.8)
                if limits and dim in limits:
                    ax.set_xlim(*limits[dim][0])
                    ax.set_ylim(*limits[dim][1])
                if continuous_mappable is not None:
                    fig.tight_layout(rect=[0, 0, 0.82, 1])
                    cax = fig.add_axes([0.86, 0.18, 0.035, 0.64])
                    cbar = fig.colorbar(continuous_mappable, cax=cax)
                    cbar.set_label(target)
                else:
                    fig.tight_layout(rect=[0, 0, 0.82, 1])
                fig.savefig(
                    figdir / (
                        f"gaussian_halfplane_temporal_arrows_by_{label}_"
                        f"{transition_token}_dim{dim}{filename_suffix}.png"
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(fig)


def plot_gaussian_halfplanes_by_seed(
    df: pd.DataFrame,
    transition_df: Optional[pd.DataFrame],
    figdir: Path,
    halfplane_limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    transition_limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    plot_t1_value_slices: bool = False,
    plot_t1_t2_value_slices: bool = False,
):
    if "seed" not in df.columns:
        return
    for seed, seed_df in df.groupby("seed", dropna=False):
        if len(seed_df) == 0:
            continue
        suffix = f"_seed_{file_token(seed)}"
        plot_halfplane_current_best_path(
            seed_df,
            figdir,
            filename_suffix=suffix,
            limits=halfplane_limits,
        )
        plot_halfplane_best_path_value(
            seed_df,
            figdir,
            filename_suffix=suffix,
            limits=halfplane_limits,
        )
        plot_halfplane_path_switch(
            seed_df,
            figdir,
            filename_suffix=suffix,
            limits=halfplane_limits,
        )
        if plot_t1_value_slices:
            plot_t1_halfplane_by_observed_value(
                seed_df,
                figdir,
                filename_suffix=suffix,
                limits=halfplane_limits,
            )
        if plot_t1_t2_value_slices:
            plot_t1_t2_halfplane_by_t1_observed_value(
                seed_df,
                figdir,
                filename_suffix=suffix,
                limits=halfplane_limits,
            )
        if transition_df is not None and len(transition_df) > 0 and "seed" in transition_df.columns:
            seed_transition = transition_df[transition_df["seed"] == seed].copy()
            plot_temporal_halfplane_arrows(
                seed_transition,
                figdir,
                filename_suffix=suffix,
                limits=transition_limits,
            )


def make_plots(
    df: pd.DataFrame,
    prediction_results: Dict[str, pd.DataFrame],
    ortho: pd.DataFrame,
    outdir: Path,
    transition_df: Optional[pd.DataFrame] = None,
    plot_t1_value_slices: bool = False,
    plot_t1_t2_value_slices: bool = False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = add_halfplane_coordinate_columns(df)
    plot_df = rows_after_observed_reward_used_downstream(df)
    if len(plot_df) == 0:
        plot_df = df.iloc[0:0].copy()
    if {"z_mu_0", "z_mu_1"}.issubset(plot_df.columns) and len(plot_df):
        sample = plot_df.sample(min(len(plot_df), 20000), random_state=0)
        if "current_best_path" in sample:
            plt.figure(figsize=panel_figsize(1, 1, title=True))
            plt.scatter(sample["z_mu_0"], sample["z_mu_1"], c=sample["current_best_path"], s=4, alpha=0.45)
            plt.xlabel("z_mu_0")
            plt.ylabel("z_mu_1")
            plt.colorbar(label="current_best_path")
            plt.tight_layout()
            plt.savefig(figdir / "latent_mu_scatter_by_current_best_path.png", dpi=180)
            plt.close()
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        plt.scatter(sample["z_mu_0"], sample["z_mu_1"], c=sample["timestep"], s=4, alpha=0.45)
        plt.xlabel("z_mu_0")
        plt.ylabel("z_mu_1")
        plt.colorbar(label="timestep")
        plt.tight_layout()
        plt.savefig(figdir / "latent_mu_scatter_by_timestep.png", dpi=180)
        plt.close()

    if {"angle_mu", "current_best_path"}.issubset(plot_df.columns) and len(plot_df):
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        hist_df = plot_df[~plot_df["tie_flag"].astype(bool)] if "tie_flag" in plot_df.columns else plot_df
        for path_value, piece in hist_df.groupby("current_best_path"):
            plt.hist(piece["angle_mu"].dropna(), bins=40, alpha=0.35, label=f"path {int(path_value)}")
        plt.xlabel("angle_mu")
        plt.ylabel("count")
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.savefig(figdir / "latent_angle_hist_by_current_best_path.png", dpi=180, bbox_inches="tight")
        plt.close()

    plot_latent_mu_scatter_by_timestep(plot_df, prediction_results, figdir)

    plot_single_predictor_metric(
        prediction_results.get("current_best_path"),
        metric="balanced_accuracy",
        path=figdir / "single_predictor_current_best_path_by_timestep.png",
    )
    plot_single_predictor_metric(
        prediction_results.get("path_switch"),
        metric="balanced_accuracy",
        path=figdir / "single_predictor_path_switch_by_timestep.png",
    )
    plot_single_predictor_metric(
        prediction_results.get("current_best_path_value"),
        metric="r2",
        path=figdir / "single_predictor_best_path_value_by_timestep.png",
    )
    plot_single_predictor_metric_by_type(
        prediction_results.get("current_best_path"),
        metric="balanced_accuracy",
        figdir=figdir,
        filename_prefix="single_predictor_current_best_path_by_timestep",
    )
    plot_single_predictor_metric_by_type(
        prediction_results.get("path_switch"),
        metric="balanced_accuracy",
        figdir=figdir,
        filename_prefix="single_predictor_path_switch_by_timestep",
    )
    plot_single_predictor_metric_by_type(
        prediction_results.get("current_best_path_value"),
        metric="r2",
        figdir=figdir,
        filename_prefix="single_predictor_best_path_value_by_timestep",
    )
    plot_single_predictor_heatmap(
        prediction_results.get("current_best_path"),
        metric="balanced_accuracy",
        path=figdir / "single_predictor_current_best_path_heatmap.png",
    )
    plot_single_predictor_heatmap(
        prediction_results.get("path_switch"),
        metric="balanced_accuracy",
        path=figdir / "single_predictor_path_switch_heatmap.png",
    )
    plot_single_predictor_heatmap(
        prediction_results.get("current_best_path_value"),
        metric="r2",
        path=figdir / "single_predictor_best_path_value_heatmap.png",
    )
    plot_single_predictor_heatmaps_by_seed(prediction_results, figdir)

    plot_temporal_direction_metric(
        prediction_results.get("temporal_current_best_path"),
        metric="balanced_accuracy",
        path=figdir / "temporal_direction_current_best_path_prediction.png",
    )
    plot_temporal_direction_metric(
        prediction_results.get("temporal_path_switch"),
        metric="balanced_accuracy",
        path=figdir / "temporal_direction_path_switch_prediction.png",
    )
    plot_temporal_direction_metric(
        prediction_results.get("temporal_best_path_value"),
        metric="r2",
        path=figdir / "temporal_direction_best_path_value_prediction.png",
    )
    plot_temporal_direction_metric(
        prediction_results.get("temporal_best_path_value_change"),
        metric="r2",
        path=figdir / "temporal_direction_best_path_value_change_prediction.png",
    )

    filtered_transition_df = transitions_after_observed_rewards_used_downstream(transition_df)
    halfplane_limits = halfplane_axis_limits(plot_df)
    transition_limits = temporal_halfplane_axis_limits(filtered_transition_df)
    plot_halfplane_current_best_path(plot_df, figdir, limits=halfplane_limits)
    plot_halfplane_best_path_value(plot_df, figdir, limits=halfplane_limits)
    plot_halfplane_path_switch(plot_df, figdir, limits=halfplane_limits)
    if plot_t1_value_slices:
        plot_t1_halfplane_by_observed_value(plot_df, figdir, limits=halfplane_limits)
    if plot_t1_t2_value_slices:
        plot_t1_t2_halfplane_by_t1_observed_value(plot_df, figdir, limits=halfplane_limits)
    plot_temporal_halfplane_arrows(filtered_transition_df, figdir, limits=transition_limits)
    plot_gaussian_halfplanes_by_seed(
        plot_df,
        filtered_transition_df,
        figdir,
        halfplane_limits=halfplane_limits,
        transition_limits=transition_limits,
        plot_t1_value_slices=plot_t1_value_slices,
        plot_t1_t2_value_slices=plot_t1_t2_value_slices,
    )

    if {"delta_mu_0", "delta_mu_1", "current_best_path"}.issubset(plot_df.columns) and len(plot_df):
        sample = plot_df.sample(min(len(plot_df), 20000), random_state=1)
        plt.figure(figsize=panel_figsize(1, 1, title=True))
        plt.scatter(sample["delta_mu_0"], sample["delta_mu_1"], c=sample["current_best_path"], s=4, alpha=0.45)
        plt.xlabel("posterior_mu_0 - prior_mu_0")
        plt.ylabel("posterior_mu_1 - prior_mu_1")
        plt.colorbar(label="current_best_path")
        plt.tight_layout()
        plt.savefig(figdir / "posterior_prior_displacement_scatter.png", dpi=180)
        plt.close()

    if len(ortho) > 0 and "lambda_value" in ortho and "corr_z_mu" in ortho:
        piece = ortho[ortho["analysis_scope"] == "by_lambda_value"]
        if len(piece) > 0:
            plt.figure(figsize=panel_figsize(1, 1, title=True))
            plt.plot(piece["lambda_value"], piece["corr_z_mu"], marker="o")
            plt.xlabel("lambda_value")
            plt.ylabel("corr(z_mu_0, z_mu_1)")
            plt.xscale("log")
            plt.tight_layout()
            plt.savefig(figdir / "latent_orthogonality_by_lambda.png", dpi=180)
            plt.close()


def best_single_predictor_by_timestep(results: Optional[pd.DataFrame], metric: str) -> List[Dict]:
    if results is None or len(results) == 0 or metric not in results:
        return []
    ok = results[results["status"] == "ok"].copy()
    if len(ok) == 0:
        return []
    agg = (
        ok.groupby(["timestep", "predictor", "predictor_type"], as_index=False, dropna=False)[metric]
        .mean()
    )
    rows = []
    for timestep, piece in agg.groupby("timestep", dropna=False):
        row = piece.sort_values(metric, ascending=False).iloc[0]
        rows.append({
            "timestep": int(timestep),
            "predictor": row["predictor"],
            "predictor_type": row["predictor_type"],
            metric: float(row[metric]),
        })
    return rows


def predictor_group_usefulness(prediction_results: Dict[str, pd.DataFrame]) -> Dict:
    target_metrics = {
        "current_best_path": ("balanced_accuracy", 0.5),
        "path_switch": ("balanced_accuracy", 0.5),
        "current_best_path_value": ("r2", 0.0),
    }
    groups = {
        "posterior_angle_predictors": lambda d: d["predictor_type"] == "posterior_angle",
        "raw_posterior_radius": lambda d: d["predictor"] == "radius_mu",
        "prior_relative_delta_predictors": lambda d: d["predictor_type"] == "posterior_prior_delta",
        "posterior_prior_kl": lambda d: d["predictor"] == "posterior_prior_kl",
    }
    out = {}
    for target_key, (metric, threshold) in target_metrics.items():
        res = prediction_results.get(target_key)
        target_out = {}
        if res is None or len(res) == 0 or metric not in res:
            out[target_key] = target_out
            continue
        ok = res[res["status"] == "ok"].copy()
        for group_name, selector in groups.items():
            piece = ok[selector(ok)] if len(ok) else ok
            if len(piece) == 0:
                target_out[group_name] = {
                    "available": False,
                    "useful": False,
                    "best_metric": None,
                    "metric": metric,
                }
                continue
            best_metric = float(piece[metric].max())
            target_out[group_name] = {
                "available": True,
                "useful": bool(best_metric > threshold),
                "best_metric": best_metric,
                "metric": metric,
            }
        out[target_key] = target_out
    return out


def target_result_specs() -> Dict[str, Dict[str, str]]:
    return {
        "current_best_path": {
            "target": "current_best_path",
            "metric": "balanced_accuracy",
        },
        "path_switch": {
            "target": "current_best_path_switch",
            "metric": "balanced_accuracy",
        },
        "current_best_path_value": {
            "target": "current_best_path_value",
            "metric": "r2",
        },
    }


def write_single_predictor_seed_consistency(
    prediction_results: Dict[str, pd.DataFrame],
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    for target_key, spec in target_result_specs().items():
        metric = spec["metric"]
        target_name = spec["target"]
        res = prediction_results.get(target_key)
        if res is None or len(res) == 0 or metric not in res or "seed" not in res:
            continue
        ok = res[res["status"] == "ok"].copy()
        if len(ok) == 0:
            continue
        agg = (
            ok.groupby(["seed", "timestep", "predictor", "predictor_type"], as_index=False, dropna=False)[metric]
            .mean()
        )
        best_by_seed = []
        for (seed, timestep), piece in agg.groupby(["seed", "timestep"], dropna=False):
            ranked = piece.sort_values(metric, ascending=False)
            best = ranked.iloc[0]
            top = ranked.head(5)
            detail_rows.append({
                "target": target_name,
                "timestep": int(timestep),
                "seed": seed,
                "best_predictor": best["predictor"],
                "best_predictor_type": best["predictor_type"],
                "best_metric_value": float(best[metric]),
                "metric": metric,
                "top_5_predictors": ",".join(top["predictor"].astype(str).tolist()),
                "top_5_metric_values": ",".join(f"{float(value):.8g}" for value in top[metric]),
            })
            best_by_seed.append({
                "target": target_name,
                "timestep": int(timestep),
                "seed": seed,
                "best_predictor": best["predictor"],
                "best_predictor_type": best["predictor_type"],
                "best_metric_value": float(best[metric]),
            })
        best_df = pd.DataFrame(best_by_seed)
        if len(best_df) == 0:
            continue
        for (target, timestep), piece in best_df.groupby(["target", "timestep"], dropna=False):
            counts = piece["best_predictor"].value_counts()
            most_common = counts.index[0]
            n_best = int(counts.iloc[0])
            type_piece = piece[piece["best_predictor"] == most_common]
            type_counts = type_piece["best_predictor_type"].value_counts()
            summary_rows.append({
                "target": target,
                "timestep": int(timestep),
                "most_common_best_predictor": most_common,
                "n_seeds_where_best": n_best,
                "most_common_best_predictor_type": type_counts.index[0] if len(type_counts) else "",
                "n_seeds_total": int(piece["seed"].nunique()),
                "best_predictor_consistency_fraction": n_best / max(int(piece["seed"].nunique()), 1),
            })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(outdir / "single_predictor_seed_consistency.csv", index=False)
    summary.to_csv(outdir / "single_predictor_seed_consistency_summary.csv", index=False)
    return detail, summary


COORDINATE_PREDICTORS = ["z_mu_0", "z_mu_1", "delta_mu_0", "delta_mu_1"]
ANGLE_STABILITY_PREDICTORS = [
    "sin_angle_mu",
    "cos_angle_mu",
    "sin_delta_angle_mu",
    "cos_delta_angle_mu",
]


def coordinate_axis_label(predictor: Optional[str]) -> str:
    if predictor == "z_mu_0":
        return "mu0"
    if predictor == "z_mu_1":
        return "mu1"
    if predictor == "delta_mu_0":
        return "delta_mu0"
    if predictor == "delta_mu_1":
        return "delta_mu1"
    if predictor:
        return "mixed"
    return "none"


def write_latent_axis_seed_stability(
    prediction_results: Dict[str, pd.DataFrame],
    outdir: Path,
) -> pd.DataFrame:
    rows = []
    summary_lines = [
        "Latent axis seed stability summary",
        "==================================",
        "",
        "If timestep 1 uses z_mu_0 in one seed but z_mu_1 in another, the identity of latent dimension 0 versus 1 is probably arbitrary due to latent rotation/permutation. In that case, we should make claims about latent direction/angle rather than fixed dimensions.",
        "If angle-based predictors are more stable across seeds than coordinate-specific predictors, the stronger claim is that path information is represented in latent direction rather than in a specific latent coordinate.",
        "If the same coordinate dominates across most seeds, then it may be meaningful to discuss that specific latent axis, but still interpret cautiously.",
        "",
    ]

    for target_key, spec in target_result_specs().items():
        metric = spec["metric"]
        target_name = spec["target"]
        res = prediction_results.get(target_key)
        if res is None or len(res) == 0 or metric not in res or "seed" not in res:
            continue
        ok = res[res["status"] == "ok"].copy()
        if len(ok) == 0:
            continue
        agg = (
            ok.groupby(["seed", "timestep", "predictor"], as_index=False, dropna=False)[metric]
            .mean()
        )
        for (seed, timestep), piece in agg.groupby(["seed", "timestep"], dropna=False):
            coord_piece = piece[piece["predictor"].isin(COORDINATE_PREDICTORS)].sort_values(
                metric, ascending=False
            )
            angle_piece = piece[piece["predictor"].isin(ANGLE_STABILITY_PREDICTORS)].sort_values(
                metric, ascending=False
            )
            best_coord = coord_piece.iloc[0] if len(coord_piece) else None
            best_angle = angle_piece.iloc[0] if len(angle_piece) else None
            coord_predictor = None if best_coord is None else str(best_coord["predictor"])
            angle_predictor = None if best_angle is None else str(best_angle["predictor"])
            coord_metric = np.nan if best_coord is None else float(best_coord[metric])
            angle_metric = np.nan if best_angle is None else float(best_angle[metric])
            rows.append({
                "target": target_name,
                "timestep": int(timestep),
                "seed": seed,
                "best_coordinate_predictor": coord_predictor or "",
                "best_coordinate_metric": coord_metric,
                "best_angle_predictor": angle_predictor or "",
                "best_angle_metric": angle_metric,
                "metric": metric,
                "coordinate_axis_dominant": coordinate_axis_label(coord_predictor),
                "angle_beats_coordinate": bool(np.isfinite(angle_metric) and (
                    not np.isfinite(coord_metric) or angle_metric > coord_metric
                )),
            })

    stability = pd.DataFrame(rows)
    stability.to_csv(outdir / "latent_axis_seed_stability.csv", index=False)

    if len(stability) == 0:
        summary_lines.append("No seed-level axis stability rows were available.")
    else:
        for (target, timestep), piece in stability.groupby(["target", "timestep"], dropna=False):
            axis_counts = piece["coordinate_axis_dominant"].value_counts()
            top_axis = axis_counts.index[0] if len(axis_counts) else "none"
            top_axis_n = int(axis_counts.iloc[0]) if len(axis_counts) else 0
            n_seeds = int(piece["seed"].nunique())
            angle_rate = float(piece["angle_beats_coordinate"].mean()) if len(piece) else float("nan")
            swaps = piece["coordinate_axis_dominant"].nunique() > 1
            summary_lines.append(
                f"{target}, timestep {int(timestep)}: most common coordinate={top_axis} "
                f"({top_axis_n}/{n_seeds} seeds); coordinate_swaps_across_seeds={swaps}; "
                f"angle_beats_coordinate_fraction={angle_rate:.3f}"
            )
    with open(outdir / "latent_axis_seed_stability_summary.txt", "w") as handle:
        handle.write("\n".join(summary_lines) + "\n")
    return stability


def write_seed_diagnostics(prediction_results: Dict[str, pd.DataFrame], outdir: Path) -> Dict[str, pd.DataFrame]:
    consistency, consistency_summary = write_single_predictor_seed_consistency(prediction_results, outdir)
    axis_stability = write_latent_axis_seed_stability(prediction_results, outdir)
    return {
        "single_predictor_seed_consistency": consistency,
        "single_predictor_seed_consistency_summary": consistency_summary,
        "latent_axis_seed_stability": axis_stability,
    }


def best_single_predictor_by_seed_timestep(results: Optional[pd.DataFrame], metric: str) -> List[Dict]:
    if results is None or len(results) == 0 or metric not in results or "seed" not in results:
        return []
    ok = results[results["status"] == "ok"].copy()
    if len(ok) == 0:
        return []
    agg = (
        ok.groupby(["seed", "timestep", "predictor", "predictor_type"], as_index=False, dropna=False)[metric]
        .mean()
    )
    rows = []
    for (seed, timestep), piece in agg.groupby(["seed", "timestep"], dropna=False):
        row = piece.sort_values(metric, ascending=False).iloc[0]
        rows.append({
            "seed": int(seed) if pd.notna(seed) else None,
            "timestep": int(timestep),
            "predictor": row["predictor"],
            "predictor_type": row["predictor_type"],
            metric: float(row[metric]),
        })
    return rows


def axis_stability_summary_dict(axis_stability: pd.DataFrame) -> Dict:
    if axis_stability is None or len(axis_stability) == 0:
        return {
            "coordinate_identity_stable_across_seeds": None,
            "angle_direction_more_stable_than_coordinates": None,
        }
    rows = []
    stable_votes = []
    angle_votes = []
    for (target, timestep), piece in axis_stability.groupby(["target", "timestep"], dropna=False):
        n_seeds = int(piece["seed"].nunique())
        axis_counts = piece["coordinate_axis_dominant"].value_counts()
        top_fraction = float(axis_counts.iloc[0] / max(n_seeds, 1)) if len(axis_counts) else 0.0
        coordinate_stable = top_fraction >= 0.75 and n_seeds > 1
        angle_fraction = float(piece["angle_beats_coordinate"].mean()) if len(piece) else 0.0
        angle_more_stable = angle_fraction >= 0.5
        stable_votes.append(coordinate_stable)
        angle_votes.append(angle_more_stable)
        rows.append({
            "target": target,
            "timestep": int(timestep),
            "most_common_coordinate_axis": axis_counts.index[0] if len(axis_counts) else "none",
            "coordinate_consistency_fraction": top_fraction,
            "coordinate_identity_stable": coordinate_stable,
            "angle_beats_coordinate_fraction": angle_fraction,
            "angle_direction_more_stable_than_coordinates": angle_more_stable,
        })
    return {
        "coordinate_identity_stable_across_seeds": bool(stable_votes and all(stable_votes)),
        "angle_direction_more_stable_than_coordinates": bool(angle_votes and any(angle_votes)),
        "by_target_timestep": rows,
    }


def summarize_outputs(
    *,
    outdir: Path,
    df: pd.DataFrame,
    failures: List[Dict],
    prediction_results: Dict[str, pd.DataFrame],
    ortho: pd.DataFrame,
    n_trials: int,
    seed_diagnostics: Optional[Dict[str, pd.DataFrame]] = None,
):
    ortho_note = None
    if len(ortho) > 0 and "corr_z_mu" in ortho:
        pooled = ortho[ortho["analysis_scope"] == "pooled"]
        if len(pooled):
            corr = float(pooled["corr_z_mu"].iloc[0])
            ortho_note = "approximately_orthogonal" if abs(corr) < 0.3 else "correlated"

    seed_diagnostics = seed_diagnostics or {}
    axis_summary = axis_stability_summary_dict(
        seed_diagnostics.get("latent_axis_seed_stability", pd.DataFrame())
    )
    temporal_summary_df = summarize_temporal_direction_results(prediction_results)
    temporal_summary_records = temporal_summary_df.to_dict(orient="records") if len(temporal_summary_df) else []

    def temporal_record(target: str) -> Optional[Dict]:
        for record in temporal_summary_records:
            if record.get("target") == target:
                return record
        return None

    summary = {
        "models_analyzed": int(df["model_id"].nunique()) if "model_id" in df else 0,
        "failures": failures,
        "n_trials_per_model": n_trials,
        "rows": int(len(df)),
        "best_single_predictor_current_best_path_by_timestep": best_single_predictor_by_timestep(
            prediction_results.get("current_best_path"), "balanced_accuracy"
        ),
        "best_single_predictor_path_switch_by_timestep": best_single_predictor_by_timestep(
            prediction_results.get("path_switch"), "balanced_accuracy"
        ),
        "best_single_predictor_current_best_path_value_by_timestep": best_single_predictor_by_timestep(
            prediction_results.get("current_best_path_value"), "r2"
        ),
        "best_single_predictor_current_best_path_by_seed_timestep": best_single_predictor_by_seed_timestep(
            prediction_results.get("current_best_path"), "balanced_accuracy"
        ),
        "best_single_predictor_path_switch_by_seed_timestep": best_single_predictor_by_seed_timestep(
            prediction_results.get("path_switch"), "balanced_accuracy"
        ),
        "best_single_predictor_current_best_path_value_by_seed_timestep": best_single_predictor_by_seed_timestep(
            prediction_results.get("current_best_path_value"), "r2"
        ),
        "coordinate_identity_stability": axis_summary,
        "predictor_group_usefulness": predictor_group_usefulness(prediction_results),
        "best_temporal_direction_current_best_path_t2": temporal_record("current_best_path_t2"),
        "best_temporal_direction_current_best_path_switch_t2": temporal_record("current_best_path_switch_t2"),
        "best_temporal_direction_current_best_path_value_t2": temporal_record("current_best_path_value_t2"),
        "best_temporal_direction_best_path_value_change_t1_to_t2": temporal_record(
            "best_path_value_change_t1_to_t2"
        ),
        "temporal_direction_analysis": temporal_summary_records,
        "latent_dimension_orthogonality_note": ortho_note,
    }
    with open(outdir / "latent_angle_analysis_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "latent_angle_analysis_summary.txt", "w") as handle:
        handle.write("Latent angle planning analysis summary\n")
        handle.write("======================================\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")


def reward_encoding_spec(
    analysis_name: str,
    timestep: int,
    target_vector: str,
    y_cols: Sequence[str],
    x_cols: Sequence[str],
    predictor_formula: str,
    predictor_family: str,
    primary: bool = False,
    requires_margin: bool = False,
) -> Dict:
    return {
        "analysis_name": analysis_name,
        "timestep": timestep,
        "target_vector": target_vector,
        "y_cols": list(y_cols),
        "x_cols": list(x_cols),
        "predictor_formula": predictor_formula,
        "predictor_family": predictor_family,
        "primary": primary,
        "requires_margin": requires_margin,
    }


def make_reward_encoding_analyses() -> List[Dict]:
    analyses = [
        reward_encoding_spec(
            "t1_mu_vector_from_R1",
            1,
            "t1_mu_vector",
            ["z_mu_0", "z_mu_1"],
            ["reward_t1"],
            "R1",
            "t1_R1",
            primary=True,
        ),
        reward_encoding_spec(
            "t1_sigma_vector_from_R1",
            1,
            "t1_sigma_vector",
            ["z_sigma_0", "z_sigma_1"],
            ["reward_t1"],
            "R1",
            "t1_R1",
            primary=True,
        ),
    ]
    t2_specs = [
        ("R1_only", "R1_only", ["reward_t1"], "R1"),
        ("R2_only", "R2_only", ["reward_t2"], "R2"),
        (
            "R2_minus_R1",
            "reward_difference",
            ["reward_t2_minus_reward_t1"],
            "R2 - R1",
        ),
        (
            "abs_R2_minus_R1",
            "absolute_reward_difference",
            ["abs_reward_t2_minus_reward_t1"],
            "abs(R2 - R1)",
        ),
        (
            "R1_R2_no_interaction",
            "R1_R2_main_effects",
            ["reward_t1", "reward_t2"],
            "R1 + R2",
        ),
        (
            "R1_R2_interaction",
            "R1_R2_interaction",
            ["reward_t1", "reward_t2", "reward_t1_x_reward_t2"],
            "R1 + R2 + R1:R2",
        ),
        (
            "current_best_path_margin",
            "choice_margin",
            ["current_best_path_margin_t2"],
            "current_best_path_margin_t2",
        ),
        (
            "R1_R2_margin",
            "R1_R2_margin",
            ["reward_t1", "reward_t2", "current_best_path_margin_t2"],
            "R1 + R2 + current_best_path_margin_t2",
        ),
    ]
    for suffix, family, x_cols, formula in t2_specs:
        requires_margin = family in {"choice_margin", "R1_R2_margin"}
        analyses.append(
            reward_encoding_spec(
                f"t2_mu_vector_from_{suffix}",
                2,
                "t2_mu_vector",
                ["z_mu_0", "z_mu_1"],
                x_cols,
                formula,
                family,
                primary=family == "R1_R2_interaction",
                requires_margin=requires_margin,
            )
        )
        analyses.append(
            reward_encoding_spec(
                f"t2_sigma_vector_from_{suffix}",
                2,
                "t2_sigma_vector",
                ["z_sigma_0", "z_sigma_1"],
                x_cols,
                formula,
                family,
                primary=family == "R1_R2_interaction",
                requires_margin=requires_margin,
            )
        )
    return analyses


REWARD_ENCODING_ANALYSES = make_reward_encoding_analyses()


def reward_encoding_group_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in MODEL_GROUP_COLUMNS if col in df.columns]


def add_reward_encoding_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "reward_t1" not in df.columns and "t1_observed_value" in df.columns:
        df["reward_t1"] = df["t1_observed_value"]
    if "reward_t2" not in df.columns and {"trial_uid", "timestep", "observed_value"}.issubset(df.columns):
        t2_values = (
            df[pd.to_numeric(df["timestep"], errors="coerce") == 2]
            .drop_duplicates("trial_uid")
            .set_index("trial_uid")["observed_value"]
        )
        df["reward_t2"] = df["trial_uid"].map(t2_values)
    if {"reward_t1", "reward_t2"}.issubset(df.columns):
        r1 = pd.to_numeric(df["reward_t1"], errors="coerce")
        r2 = pd.to_numeric(df["reward_t2"], errors="coerce")
        df["reward_t2_minus_reward_t1"] = r2 - r1
        df["abs_reward_t2_minus_reward_t1"] = (r2 - r1).abs()
        df["reward_t1_x_reward_t2"] = r1 * r2
        df["reward_t1_reward_t2_interaction"] = df["reward_t1_x_reward_t2"]
    needs_margin_t2 = "current_best_path_margin_t2" not in df.columns
    if not needs_margin_t2:
        needs_margin_t2 = not np.isfinite(
            pd.to_numeric(df["current_best_path_margin_t2"], errors="coerce")
        ).any()
    if needs_margin_t2 and {
        "trial_uid",
        "timestep",
        "current_best_path_margin",
    }.issubset(df.columns):
        margin_t2 = (
            df[pd.to_numeric(df["timestep"], errors="coerce") == 2]
            .drop_duplicates("trial_uid")
            .set_index("trial_uid")["current_best_path_margin"]
        )
        df["current_best_path_margin_t2"] = df["trial_uid"].map(margin_t2)
    return df


def add_geometry_meaning_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = add_reward_encoding_derived_columns(df)
    df = df.copy()
    if {"z_mu_0", "z_mu_1"}.issubset(df.columns):
        mu0 = pd.to_numeric(df["z_mu_0"], errors="coerce")
        mu1 = pd.to_numeric(df["z_mu_1"], errors="coerce")
        df["mu_radius"] = np.sqrt(mu0 ** 2 + mu1 ** 2)
        df["mu_angle"] = np.arctan2(mu1, mu0)
        df["mu_sin_angle"] = np.sin(df["mu_angle"])
        df["mu_cos_angle"] = np.cos(df["mu_angle"])
    if {"z_sigma_0", "z_sigma_1"}.issubset(df.columns):
        sig0 = pd.to_numeric(df["z_sigma_0"], errors="coerce")
        sig1 = pd.to_numeric(df["z_sigma_1"], errors="coerce")
        df["sigma_radius"] = np.sqrt(sig0 ** 2 + sig1 ** 2)
        df["sigma_angle"] = np.arctan2(sig1, sig0)
        df["sigma_sin_angle"] = np.sin(df["sigma_angle"])
        df["sigma_cos_angle"] = np.cos(df["sigma_angle"])
    if {"z_mu_0", "z_mu_1", "prior_mu_0", "prior_mu_1"}.issubset(df.columns):
        df["delta_mu_0"] = pd.to_numeric(df["z_mu_0"], errors="coerce") - pd.to_numeric(df["prior_mu_0"], errors="coerce")
        df["delta_mu_1"] = pd.to_numeric(df["z_mu_1"], errors="coerce") - pd.to_numeric(df["prior_mu_1"], errors="coerce")
        df["delta_mu_radius"] = np.sqrt(df["delta_mu_0"] ** 2 + df["delta_mu_1"] ** 2)
        df["delta_mu_angle"] = np.arctan2(df["delta_mu_1"], df["delta_mu_0"])
        df["delta_mu_sin_angle"] = np.sin(df["delta_mu_angle"])
        df["delta_mu_cos_angle"] = np.cos(df["delta_mu_angle"])
    for dim in available_latent_dims(df):
        mu_col = f"z_mu_{dim}"
        sigma_col = f"z_sigma_{dim}"
        if mu_col not in df.columns or sigma_col not in df.columns:
            continue
        x = pd.to_numeric(df[mu_col], errors="coerce") / math.sqrt(2.0)
        y = pd.to_numeric(df[sigma_col], errors="coerce")
        eu_x = pd.to_numeric(df[mu_col], errors="coerce")
        eu_y = y
        df[f"euclidean_x_{dim}"] = eu_x
        df[f"euclidean_y_{dim}"] = eu_y
        df[f"euclidean_radius_{dim}"] = np.sqrt(eu_x ** 2 + eu_y ** 2)
        df[f"euclidean_angle_{dim}"] = np.arctan2(eu_y, eu_x)
        df[f"euclidean_sin_angle_{dim}"] = np.sin(df[f"euclidean_angle_{dim}"])
        df[f"euclidean_cos_angle_{dim}"] = np.cos(df[f"euclidean_angle_{dim}"])
        df[f"halfplane_x_{dim}"] = x
        df[f"halfplane_y_{dim}"] = y
        df[f"post_halfplane_x_{dim}"] = x
        df[f"post_halfplane_y_{dim}"] = y
        df[f"halfplane_radius_{dim}"] = np.sqrt(x ** 2 + y ** 2)
        df[f"halfplane_angle_{dim}"] = np.arctan2(y, x)
        df[f"halfplane_sin_angle_{dim}"] = np.sin(df[f"halfplane_angle_{dim}"])
        df[f"halfplane_cos_angle_{dim}"] = np.cos(df[f"halfplane_angle_{dim}"])
        post_disk = halfplane_to_canonical_disk(x, y)
        df[f"canonical_disk_x_{dim}"] = np.real(post_disk)
        df[f"canonical_disk_y_{dim}"] = np.imag(post_disk)
        df[f"canonical_disk_radius_{dim}"] = np.abs(post_disk)
        df[f"canonical_disk_angle_{dim}"] = np.angle(post_disk)
        df[f"canonical_disk_sin_angle_{dim}"] = np.sin(df[f"canonical_disk_angle_{dim}"])
        df[f"canonical_disk_cos_angle_{dim}"] = np.cos(df[f"canonical_disk_angle_{dim}"])
        prior_mu_col = f"prior_mu_{dim}"
        prior_sigma_col = f"prior_sigma_{dim}"
        prior_logvar_col = f"prior_logvar_{dim}"
        if prior_sigma_col not in df.columns and prior_logvar_col in df.columns:
            df[prior_sigma_col] = np.exp(
                0.5 * np.clip(pd.to_numeric(df[prior_logvar_col], errors="coerce"), -10.0, 10.0)
            )
        if prior_mu_col in df.columns and prior_sigma_col in df.columns:
            prior_x = pd.to_numeric(df[prior_mu_col], errors="coerce") / math.sqrt(2.0)
            prior_y = pd.to_numeric(df[prior_sigma_col], errors="coerce")
            df[f"prior_halfplane_x_{dim}"] = prior_x
            df[f"prior_halfplane_y_{dim}"] = prior_y
            df[f"delta_halfplane_x_{dim}"] = x - prior_x
            df[f"delta_halfplane_y_{dim}"] = y - prior_y
            df[f"delta_halfplane_radius_{dim}"] = np.sqrt(
                df[f"delta_halfplane_x_{dim}"] ** 2 + df[f"delta_halfplane_y_{dim}"] ** 2
            )
            df[f"delta_halfplane_angle_{dim}"] = np.arctan2(
                df[f"delta_halfplane_y_{dim}"],
                df[f"delta_halfplane_x_{dim}"],
            )
            df[f"delta_halfplane_sin_angle_{dim}"] = np.sin(df[f"delta_halfplane_angle_{dim}"])
            df[f"delta_halfplane_cos_angle_{dim}"] = np.cos(df[f"delta_halfplane_angle_{dim}"])
            df[f"prior_relative_fisher_distance_{dim}"] = gaussian_fisher_distance_from_halfplane(
                x,
                y,
                prior_x,
                prior_y,
            )
            df[f"prior_centered_halfplane_x_{dim}"] = df[f"delta_halfplane_x_{dim}"]
            df[f"prior_centered_halfplane_y_{dim}"] = df[f"delta_halfplane_y_{dim}"]
            df[f"prior_centered_halfplane_radius_{dim}"] = df[f"delta_halfplane_radius_{dim}"]
            df[f"prior_centered_halfplane_angle_{dim}"] = df[f"delta_halfplane_angle_{dim}"]
            df[f"prior_centered_halfplane_sin_angle_{dim}"] = df[f"delta_halfplane_sin_angle_{dim}"]
            df[f"prior_centered_halfplane_cos_angle_{dim}"] = df[f"delta_halfplane_cos_angle_{dim}"]
            prior_disk = halfplane_to_canonical_disk(prior_x, prior_y)
            centered_disk = disk_isometry_center_prior(post_disk, prior_disk)
            df[f"prior_centered_disk_x_{dim}"] = np.real(centered_disk)
            df[f"prior_centered_disk_y_{dim}"] = np.imag(centered_disk)
            df[f"prior_centered_disk_radius_{dim}"] = np.abs(centered_disk)
            df[f"prior_centered_disk_angle_{dim}"] = np.angle(centered_disk)
            df[f"prior_centered_disk_sin_angle_{dim}"] = np.sin(df[f"prior_centered_disk_angle_{dim}"])
            df[f"prior_centered_disk_cos_angle_{dim}"] = np.cos(df[f"prior_centered_disk_angle_{dim}"])
            df[f"prior_centered_disk_hyperbolic_distance_{dim}"] = 2.0 * np.arctanh(
                np.minimum(df[f"prior_centered_disk_radius_{dim}"], 1.0 - 1e-8)
            )
            df[f"prior_centered_fisher_distance_{dim}"] = (
                math.sqrt(2.0) * df[f"prior_centered_disk_hyperbolic_distance_{dim}"]
            )
        denom = x ** 2 + (y + 1.0) ** 2
        df[f"poincare_disk_x_{dim}"] = (x ** 2 + y ** 2 - 1.0) / denom
        df[f"poincare_disk_y_{dim}"] = -2.0 * x / denom
        df[f"poincare_disk_radius_{dim}"] = np.sqrt(
            df[f"poincare_disk_x_{dim}"] ** 2 + df[f"poincare_disk_y_{dim}"] ** 2
        )
        df[f"poincare_disk_angle_{dim}"] = np.arctan2(
            df[f"poincare_disk_y_{dim}"],
            df[f"poincare_disk_x_{dim}"],
        )
        df[f"poincare_disk_sin_angle_{dim}"] = np.sin(df[f"poincare_disk_angle_{dim}"])
        df[f"poincare_disk_cos_angle_{dim}"] = np.cos(df[f"poincare_disk_angle_{dim}"])

    fisher_cols = [
        f"prior_relative_fisher_distance_{dim}"
        for dim in available_latent_dims(df)
        if f"prior_relative_fisher_distance_{dim}" in df.columns
    ]
    if fisher_cols:
        fisher_sq = df[fisher_cols].apply(pd.to_numeric, errors="coerce") ** 2
        df["prior_relative_fisher_distance_total"] = np.sqrt(fisher_sq.sum(axis=1, skipna=False))
        df["prior_relative_fisher_distance_mean"] = np.sqrt(fisher_sq.mean(axis=1, skipna=False))
    prior_centered_fisher_cols = [
        f"prior_centered_fisher_distance_{dim}"
        for dim in available_latent_dims(df)
        if f"prior_centered_fisher_distance_{dim}" in df.columns
    ]
    if prior_centered_fisher_cols:
        fisher_sq = df[prior_centered_fisher_cols].apply(pd.to_numeric, errors="coerce") ** 2
        df["prior_centered_fisher_distance_total"] = np.sqrt(fisher_sq.sum(axis=1, skipna=False))

    task_bases = [
        "current_best_path",
        "current_best_path_value",
        "current_best_path_margin",
        "current_best_path_switch",
    ]
    if {"trial_uid", "timestep"}.issubset(df.columns):
        timestep_numeric = pd.to_numeric(df["timestep"], errors="coerce")
        for t in (1, 2):
            t_df = df[timestep_numeric == t].drop_duplicates("trial_uid")
            if len(t_df) == 0:
                continue
            t_lookup = t_df.set_index("trial_uid")
            for base in task_bases:
                out_col = f"{base}_t{t}"
                if base in t_lookup.columns and (
                    out_col not in df.columns
                    or not np.isfinite(pd.to_numeric(df[out_col], errors="coerce")).any()
                ):
                    df[out_col] = df["trial_uid"].map(t_lookup[base])
            geometry_bases = [
                "mu_radius",
                "mu_angle",
                "mu_sin_angle",
                "mu_cos_angle",
                "sigma_radius",
                "sigma_angle",
                "sigma_sin_angle",
                "sigma_cos_angle",
                "delta_mu_0",
                "delta_mu_1",
                "delta_mu_radius",
                "delta_mu_angle",
                "delta_mu_sin_angle",
                "delta_mu_cos_angle",
                "posterior_prior_kl",
            ]
            geometry_bases.extend(
                col
                for col in df.columns
                if re.match(
                    r"^(z_mu|z_logvar|z_sigma|prior_mu|prior_logvar|prior_sigma|euclidean_x|euclidean_y|euclidean_radius|euclidean_angle|euclidean_sin_angle|euclidean_cos_angle|halfplane_x|halfplane_y|post_halfplane_x|post_halfplane_y|prior_halfplane_x|prior_halfplane_y|delta_halfplane_x|delta_halfplane_y|delta_halfplane_radius|delta_halfplane_angle|delta_halfplane_sin_angle|delta_halfplane_cos_angle|prior_relative_fisher_distance|prior_centered_halfplane_x|prior_centered_halfplane_y|prior_centered_halfplane_radius|prior_centered_halfplane_angle|prior_centered_halfplane_sin_angle|prior_centered_halfplane_cos_angle|canonical_disk_x|canonical_disk_y|canonical_disk_radius|canonical_disk_angle|canonical_disk_sin_angle|canonical_disk_cos_angle|prior_centered_disk_x|prior_centered_disk_y|prior_centered_disk_radius|prior_centered_disk_angle|prior_centered_disk_sin_angle|prior_centered_disk_cos_angle|prior_centered_disk_hyperbolic_distance|prior_centered_fisher_distance|halfplane_radius|halfplane_angle|halfplane_sin_angle|halfplane_cos_angle|poincare_disk_x|poincare_disk_y|poincare_disk_radius|poincare_disk_angle|poincare_disk_sin_angle|poincare_disk_cos_angle)_\d+$",
                    col,
                )
            )
            geometry_bases.extend([
                col for col in [
                    "prior_relative_fisher_distance_total",
                    "prior_relative_fisher_distance_mean",
                    "prior_centered_fisher_distance_total",
                ]
                if col in df.columns
            ])
            for base in geometry_bases:
                if base in t_lookup.columns:
                    df[f"{base}_t{t}"] = df["trial_uid"].map(t_lookup[base])
    return df


def safe_regression_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    xv = x[finite]
    yv = y[finite]
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return float("nan")
    try:
        if kind == "spearman":
            from scipy.stats import spearmanr

            return float(spearmanr(xv, yv).correlation)
        from scipy.stats import pearsonr

        return float(pearsonr(xv, yv).statistic)
    except Exception:
        return float("nan")


def json_safe_value(value):
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def skipped_reward_encoding_row(group_name: Dict, spec: Dict, status: str, error_message: str) -> Dict:
    return {
        **group_name,
        "analysis_name": spec["analysis_name"],
        "model_direction": "encoding",
        "timestep": spec["timestep"],
        "target_vector": spec["target_vector"],
        "predictor_formula": spec["predictor_formula"],
        "predictor_family": spec.get("predictor_family", ""),
        "n_trials": 0,
        "n_folds": 0,
        "overall_R2": np.nan,
        "mean_R2_across_dims": np.nan,
        "variance_weighted_R2": np.nan,
        "R2_dim0": np.nan,
        "R2_dim1": np.nan,
        "Pearson_r_dim0": np.nan,
        "Pearson_r_dim1": np.nan,
        "Spearman_r_dim0": np.nan,
        "Spearman_r_dim1": np.nan,
        "RMSE_dim0": np.nan,
        "RMSE_dim1": np.nan,
        "MAE_dim0": np.nan,
        "MAE_dim1": np.nan,
        "status": status,
        "error_message": error_message,
    }


def fit_reward_encoding_spec(
    group_name: Dict,
    group_df: pd.DataFrame,
    spec: Dict,
    cv_folds: int,
) -> Tuple[Dict, List[Dict]]:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    required = set(spec["x_cols"] + spec["y_cols"] + ["trial_uid", "timestep"])
    result_status = "ok"
    result_error_message = ""
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        if spec.get("requires_margin") and "current_best_path_margin_t2" in missing:
            return skipped_reward_encoding_row(
                group_name,
                spec,
                "skipped_missing_margin",
                "current_best_path_margin_t2 is unavailable",
            ), []
        return skipped_reward_encoding_row(
            group_name,
            spec,
            "skipped_missing_columns",
            f"missing columns: {','.join(missing)}",
        ), []

    work = rows_after_observed_reward_used_downstream(group_df, timestep=spec["timestep"])
    if "qz_used_downstream" not in group_df.columns:
        result_status = "warning_qz2_downstream_usage_unknown"
        result_error_message = "qz_used_downstream column is unavailable"
    if spec.get("requires_margin") and "current_best_path_margin_t2" in work.columns:
        margin_values = pd.to_numeric(work["current_best_path_margin_t2"], errors="coerce")
        if not np.isfinite(margin_values).any():
            return skipped_reward_encoding_row(
                group_name,
                spec,
                "skipped_missing_margin",
                "current_best_path_margin_t2 has no finite values after filtering",
            ), []
    cols = spec["x_cols"] + spec["y_cols"]
    numeric = work[cols].apply(pd.to_numeric, errors="coerce")
    work = work[np.isfinite(numeric).all(axis=1)].copy()
    if len(work) < 4:
        return skipped_reward_encoding_row(
            group_name,
            spec,
            "skipped_insufficient_trials",
            "fewer than four valid trials after filtering",
        ), []

    x = work[spec["x_cols"]].to_numpy(dtype=float)
    y = work[spec["y_cols"]].to_numpy(dtype=float)
    groups = work["trial_uid"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return skipped_reward_encoding_row(
            group_name,
            spec,
            "skipped_insufficient_groups",
            "fewer than two trial groups available",
        ), []
    n_splits = min(int(cv_folds), len(unique_groups))
    if n_splits < 2:
        return skipped_reward_encoding_row(
            group_name,
            spec,
            "skipped_insufficient_folds",
            "fewer than two cross-validation folds available",
        ), []

    from sklearn.model_selection import GroupKFold

    preds = np.full_like(y, np.nan, dtype=float)
    fold_ids = np.full(y.shape[0], -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        ])
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])
        fold_ids[test_idx] = fold_i

    valid = np.isfinite(preds).all(axis=1)
    if not valid.any():
        return skipped_reward_encoding_row(
            group_name,
            spec,
            "failed_no_valid_predictions",
            "no cross-validated predictions were produced",
        ), []
    yv = y[valid]
    pv = preds[valid]
    r2_dim0 = float(r2_score(yv[:, 0], pv[:, 0]))
    r2_dim1 = float(r2_score(yv[:, 1], pv[:, 1]))
    row = {
        **group_name,
        "analysis_name": spec["analysis_name"],
        "model_direction": "encoding",
        "timestep": spec["timestep"],
        "target_vector": spec["target_vector"],
        "predictor_formula": spec["predictor_formula"],
        "predictor_family": spec.get("predictor_family", ""),
        "n_trials": int(len(yv)),
        "n_folds": int(n_splits),
        "overall_R2": float(r2_score(yv, pv, multioutput="uniform_average")),
        "mean_R2_across_dims": float(np.nanmean([r2_dim0, r2_dim1])),
        "variance_weighted_R2": float(r2_score(yv, pv, multioutput="variance_weighted")),
        "R2_dim0": r2_dim0,
        "R2_dim1": r2_dim1,
        "Pearson_r_dim0": safe_regression_corr(yv[:, 0], pv[:, 0], "pearson"),
        "Pearson_r_dim1": safe_regression_corr(yv[:, 1], pv[:, 1], "pearson"),
        "Spearman_r_dim0": safe_regression_corr(yv[:, 0], pv[:, 0], "spearman"),
        "Spearman_r_dim1": safe_regression_corr(yv[:, 1], pv[:, 1], "spearman"),
        "RMSE_dim0": float(np.sqrt(mean_squared_error(yv[:, 0], pv[:, 0]))),
        "RMSE_dim1": float(np.sqrt(mean_squared_error(yv[:, 1], pv[:, 1]))),
        "MAE_dim0": float(mean_absolute_error(yv[:, 0], pv[:, 0])),
        "MAE_dim1": float(mean_absolute_error(yv[:, 1], pv[:, 1])),
        "status": result_status,
        "error_message": result_error_message,
    }

    predictions = []
    valid_indices = np.flatnonzero(valid)
    for out_i, work_i in enumerate(valid_indices):
        predictions.append({
            **group_name,
            "analysis_name": spec["analysis_name"],
            "model_direction": "encoding",
            "trial_id": work.iloc[work_i]["trial_id"] if "trial_id" in work.columns else work_i,
            "trial_uid": work.iloc[work_i]["trial_uid"],
            "fold": int(fold_ids[work_i]),
            "target_vector": spec["target_vector"],
            "predictor_formula": spec["predictor_formula"],
            "predictor_family": spec.get("predictor_family", ""),
            "observed_dim0": float(yv[out_i, 0]),
            "observed_dim1": float(yv[out_i, 1]),
            "predicted_dim0": float(pv[out_i, 0]),
            "predicted_dim1": float(pv[out_i, 1]),
        })
    return row, predictions


def reward_encoding_competing_model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = reward_encoding_group_columns(results)
    rows = []
    valid_statuses = {"ok", "warning_qz2_downstream_usage_unknown"}
    ok = results[results["status"].isin(valid_statuses)].copy()
    family_to_col = {
        "R1_only": "R2_R1_only",
        "R2_only": "R2_R2_only",
        "reward_difference": "R2_reward_difference",
        "absolute_reward_difference": "R2_absolute_reward_difference",
        "R1_R2_main_effects": "R2_R1_R2_main_effects",
        "R1_R2_interaction": "R2_R1_R2_interaction",
        "choice_margin": "R2_choice_margin",
        "R1_R2_margin": "R2_R1_R2_margin",
    }
    for target_vector in ("t2_mu_vector", "t2_sigma_vector"):
        piece = ok[ok["target_vector"] == target_vector]
        if len(piece) == 0:
            continue
        for values, group in piece.groupby(group_cols, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            group_name = dict(zip(group_cols, values))
            by_family = group.set_index("predictor_family")["variance_weighted_R2"].to_dict()
            values_by_col = {
                col: by_family.get(family, np.nan)
                for family, col in family_to_col.items()
            }
            finite_scores = {
                family: score
                for family, score in by_family.items()
                if family in family_to_col and np.isfinite(score)
            }
            if finite_scores:
                best_model, best_model_r2 = max(finite_scores.items(), key=lambda item: item[1])
            else:
                best_model, best_model_r2 = "", np.nan
            main_effects = values_by_col["R2_R1_R2_main_effects"]
            interaction = values_by_col["R2_R1_R2_interaction"]
            row = {
                **group_name,
                "target_vector": target_vector,
                **values_by_col,
                "R2_R1_R2_no_interaction": main_effects,
                "best_model": best_model,
                "best_model_R2": best_model_r2,
                "delta_R2_difference_vs_R2": (
                    values_by_col["R2_reward_difference"] - values_by_col["R2_R2_only"]
                ),
                "delta_R2_abs_difference_vs_R2": (
                    values_by_col["R2_absolute_reward_difference"] - values_by_col["R2_R2_only"]
                ),
                "delta_R2_interaction_vs_main_effects": interaction - main_effects,
                "delta_R2_interaction": interaction - main_effects,
                "delta_R2_margin_vs_R1_R2": values_by_col["R2_choice_margin"] - main_effects,
                "status": "ok" if finite_scores else "skipped_missing_model",
                "error_message": "" if finite_scores else "no successful competing reward-encoding models",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def reward_encoding_interaction_comparison(results: pd.DataFrame) -> pd.DataFrame:
    return reward_encoding_competing_model_comparison(results)


def plot_reward_encoding_outputs(results: pd.DataFrame, predictions: pd.DataFrame, interaction: pd.DataFrame, figdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_7pt_plot_style(plt)

    figdir.mkdir(parents=True, exist_ok=True)
    valid_statuses = {"ok", "warning_qz2_downstream_usage_unknown"}
    ok = results[results["status"].isin(valid_statuses)].copy()
    if len(ok) > 0:
        metric = "variance_weighted_R2" if "variance_weighted_R2" in ok.columns else "overall_R2"
        family_order = [
            "R1_only",
            "R2_only",
            "reward_difference",
            "absolute_reward_difference",
            "R1_R2_main_effects",
            "R1_R2_interaction",
            "choice_margin",
            "R1_R2_margin",
        ]
        t2_ok = ok[ok["target_vector"].isin(["t2_mu_vector", "t2_sigma_vector"])].copy()
        if len(t2_ok) > 0 and "predictor_family" in t2_ok.columns:
            targets = ["t2_mu_vector", "t2_sigma_vector"]
            fig, axes = plt.subplots(1, 2, figsize=panel_figsize(2, 1, title=True), squeeze=False, sharey=True)
            seeds = sorted(t2_ok["seed"].dropna().unique()) if "seed" in t2_ok else ["all"]
            x = np.arange(len(family_order), dtype=float)
            for ax, target in zip(axes[0], targets):
                target_piece = t2_ok[t2_ok["target_vector"] == target]
                for seed in seeds:
                    seed_piece = target_piece[target_piece["seed"] == seed] if "seed" in target_piece else target_piece
                    values = [
                        pd.to_numeric(
                            seed_piece[seed_piece["predictor_family"] == family][metric],
                            errors="coerce",
                        ).mean()
                        for family in family_order
                    ]
                    ax.plot(x, values, marker="o", linewidth=1.0, markersize=4, label=f"seed {seed}")
                ax.axhline(0, color="0.75", linewidth=0.8)
                ax.set_title(target, fontsize=PANEL_FONT_SIZE)
                ax.set_xticks(x)
                ax.set_xticklabels(family_order, rotation=35, ha="right", fontsize=PANEL_FONT_SIZE)
                ax.set_ylabel(metric, fontsize=PANEL_FONT_SIZE)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, frameon=False, bbox_to_anchor=(0.995, 0.98), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.86, 1])
            fig.savefig(figdir / "reward_encoding_competing_models_by_seed.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            diff_family_order = [
                "R2_only",
                "reward_difference",
                "absolute_reward_difference",
                "R1_R2_main_effects",
                "R1_R2_interaction",
                "choice_margin",
            ]
            fig, axes = plt.subplots(1, 2, figsize=panel_figsize(2, 1, title=True), squeeze=False, sharey=True)
            for ax, target in zip(axes[0], targets):
                target_piece = t2_ok[t2_ok["target_vector"] == target]
                for seed in seeds:
                    seed_piece = target_piece[target_piece["seed"] == seed] if "seed" in target_piece else target_piece
                    values = [
                        pd.to_numeric(
                            seed_piece[seed_piece["predictor_family"] == family][metric],
                            errors="coerce",
                        ).mean()
                        for family in diff_family_order
                    ]
                    ax.plot(np.arange(len(diff_family_order)), values, marker="o", linewidth=1.0, markersize=4, label=f"seed {seed}")
                ax.axhline(0, color="0.75", linewidth=0.8)
                ax.set_title(target, fontsize=PANEL_FONT_SIZE)
                ax.set_xticks(np.arange(len(diff_family_order)))
                ax.set_xticklabels(diff_family_order, rotation=35, ha="right", fontsize=PANEL_FONT_SIZE)
                ax.set_ylabel(metric, fontsize=PANEL_FONT_SIZE)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, frameon=False, bbox_to_anchor=(0.995, 0.98), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.86, 1])
            fig.savefig(figdir / "reward_encoding_difference_vs_interaction.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        plotted_names = [
            spec["analysis_name"]
            for spec in REWARD_ENCODING_ANALYSES
            if spec.get("primary")
        ] + [
            "t2_mu_vector_from_R1_only",
            "t2_mu_vector_from_R2_only",
        ]
        plotted_names = list(dict.fromkeys(plotted_names))
        plot_df = ok[ok["analysis_name"].isin(plotted_names)].copy()
        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
            seeds = sorted(plot_df["seed"].dropna().unique()) if "seed" in plot_df else ["all"]
            x_labels = [name for name in plotted_names if name in set(plot_df["analysis_name"])]
            x = np.arange(len(x_labels), dtype=float)
            width = 0.8 / max(len(seeds), 1)
            for seed_i, seed in enumerate(seeds):
                seed_piece = plot_df[plot_df["seed"] == seed] if "seed" in plot_df else plot_df
                values = [
                    pd.to_numeric(seed_piece[seed_piece["analysis_name"] == name][metric], errors="coerce").mean()
                    for name in x_labels
                ]
                ax.bar(x + (seed_i - (len(seeds) - 1) / 2) * width, values, width=width, label=f"seed {seed}")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=35, ha="right")
            ax.set_ylabel(metric)
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            fig.savefig(figdir / "reward_encoding_vector_r2_by_seed.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    if len(interaction) > 0 and "delta_R2_interaction" in interaction.columns:
        piece = interaction[interaction["status"].isin(valid_statuses)].copy()
        if len(piece) > 0:
            fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
            targets = sorted(piece["target_vector"].dropna().unique())
            x = np.arange(len(targets), dtype=float)
            seeds = sorted(piece["seed"].dropna().unique()) if "seed" in piece else ["all"]
            width = 0.8 / max(len(seeds), 1)
            for seed_i, seed in enumerate(seeds):
                seed_piece = piece[piece["seed"] == seed] if "seed" in piece else piece
                values = [
                    pd.to_numeric(seed_piece[seed_piece["target_vector"] == target]["delta_R2_interaction"], errors="coerce").mean()
                    for target in targets
                ]
                ax.scatter(x + (seed_i - (len(seeds) - 1) / 2) * width, values, label=f"seed {seed}")
            ax.axhline(0, color="0.65", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(targets)
            ax.set_ylabel("delta_R2_interaction")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            fig.savefig(figdir / "reward_encoding_interaction_delta_R2.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    if len(interaction) > 0 and "delta_R2_difference_vs_R2" in interaction.columns:
        piece = interaction[interaction["status"].isin(valid_statuses)].copy()
        if len(piece) > 0:
            fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
            targets = sorted(piece["target_vector"].dropna().unique())
            x = np.arange(len(targets), dtype=float)
            seeds = sorted(piece["seed"].dropna().unique()) if "seed" in piece else ["all"]
            width = 0.8 / max(len(seeds), 1)
            for seed_i, seed in enumerate(seeds):
                seed_piece = piece[piece["seed"] == seed] if "seed" in piece else piece
                values = [
                    pd.to_numeric(
                        seed_piece[seed_piece["target_vector"] == target]["delta_R2_difference_vs_R2"],
                        errors="coerce",
                    ).mean()
                    for target in targets
                ]
                ax.scatter(x + (seed_i - (len(seeds) - 1) / 2) * width, values, label=f"seed {seed}")
            ax.axhline(0, color="0.65", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(targets)
            ax.set_ylabel("delta_R2_difference_vs_R2")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            fig.savefig(figdir / "reward_encoding_reward_difference_delta_R2.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    key_names = [
        "t1_mu_vector_from_R1",
        "t1_sigma_vector_from_R1",
        "t2_mu_vector_from_R1_only",
        "t2_mu_vector_from_R2_only",
        "t2_sigma_vector_from_R1_only",
        "t2_sigma_vector_from_R2_only",
        "t2_mu_vector_from_R1_R2_interaction",
        "t2_sigma_vector_from_R1_R2_interaction",
        "t2_mu_vector_from_R2_minus_R1",
        "t2_sigma_vector_from_R2_minus_R1",
        "t2_mu_vector_from_abs_R2_minus_R1",
        "t2_sigma_vector_from_abs_R2_minus_R1",
        "t2_mu_vector_from_current_best_path_margin",
        "t2_sigma_vector_from_current_best_path_margin",
    ]
    predicted_plot_filename = {
        "t2_mu_vector_from_R1_only": "predicted_vs_observed_t2_mu_vector_from_R1.png",
        "t2_mu_vector_from_R2_only": "predicted_vs_observed_t2_mu_vector_from_R2.png",
        "t2_sigma_vector_from_R1_only": "predicted_vs_observed_t2_sigma_vector_from_R1.png",
        "t2_sigma_vector_from_R2_only": "predicted_vs_observed_t2_sigma_vector_from_R2.png",
        "t2_mu_vector_from_R2_minus_R1": "predicted_vs_observed_t2_mu_vector_from_R2_minus_R1.png",
        "t2_sigma_vector_from_R2_minus_R1": "predicted_vs_observed_t2_sigma_vector_from_R2_minus_R1.png",
        "t2_mu_vector_from_abs_R2_minus_R1": "predicted_vs_observed_t2_mu_vector_from_abs_R2_minus_R1.png",
        "t2_sigma_vector_from_abs_R2_minus_R1": "predicted_vs_observed_t2_sigma_vector_from_abs_R2_minus_R1.png",
        "t2_mu_vector_from_current_best_path_margin": "predicted_vs_observed_t2_mu_vector_from_current_best_path_margin.png",
        "t2_sigma_vector_from_current_best_path_margin": "predicted_vs_observed_t2_sigma_vector_from_current_best_path_margin.png",
    }
    for analysis_name in key_names:
        piece = predictions[predictions["analysis_name"] == analysis_name].copy() if len(predictions) else pd.DataFrame()
        if len(piece) == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=panel_figsize(2, 1, title=True), squeeze=False)
        for dim, ax in enumerate(axes[0]):
            obs = pd.to_numeric(piece[f"observed_dim{dim}"], errors="coerce")
            pred = pd.to_numeric(piece[f"predicted_dim{dim}"], errors="coerce")
            finite = np.isfinite(obs) & np.isfinite(pred)
            ax.scatter(obs[finite], pred[finite], s=8, alpha=0.35, linewidths=0)
            if finite.any():
                lo = float(min(obs[finite].min(), pred[finite].min()))
                hi = float(max(obs[finite].max(), pred[finite].max()))
                pad = max((hi - lo) * 0.05, 1e-3)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.55", linewidth=0.8)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"dim {dim}", fontsize=PANEL_FONT_SIZE)
            ax.set_xlabel("observed")
            ax.set_ylabel("predicted")
        fig.suptitle(analysis_name, fontsize=11)
        fig.tight_layout()
        filename = predicted_plot_filename.get(
            analysis_name,
            f"predicted_vs_observed_{analysis_name}.png",
        )
        fig.savefig(figdir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_reward_encoding_summary(
    outdir: Path,
    results: pd.DataFrame,
    interaction: pd.DataFrame,
    failures: List[Dict],
    n_trials: int,
):
    valid_statuses = {"ok", "warning_qz2_downstream_usage_unknown"}
    ok = results[results["status"].isin(valid_statuses)].copy() if len(results) else pd.DataFrame()
    best = {}
    target_summaries = {}
    if len(ok):
        for analysis_name, piece in ok.groupby("analysis_name", dropna=False):
            best_row = piece.sort_values("variance_weighted_R2", ascending=False).iloc[0]
            best[analysis_name] = {
                "best_variance_weighted_R2": float(best_row["variance_weighted_R2"]),
                "seed": best_row.get("seed"),
                "model_id": best_row.get("model_id", ""),
            }
        for target_vector, piece in ok[ok["timestep"] == 2].groupby("target_vector", dropna=False):
            family_scores = (
                piece.groupby("predictor_family", dropna=False)["variance_weighted_R2"]
                .mean()
                .sort_values(ascending=False)
            )
            target_summaries[target_vector] = {
                "best_model": str(family_scores.index[0]) if len(family_scores) else "",
                "best_model_mean_variance_weighted_R2": float(family_scores.iloc[0]) if len(family_scores) else None,
                "reward_difference_beats_R2": bool(
                    family_scores.get("reward_difference", -np.inf) >= family_scores.get("R2_only", np.inf)
                ) if "reward_difference" in family_scores and "R2_only" in family_scores else None,
                "absolute_reward_difference_beats_R2": bool(
                    family_scores.get("absolute_reward_difference", -np.inf) >= family_scores.get("R2_only", np.inf)
                ) if "absolute_reward_difference" in family_scores and "R2_only" in family_scores else None,
                "interaction_beats_main_effects": bool(
                    family_scores.get("R1_R2_interaction", -np.inf) >= family_scores.get("R1_R2_main_effects", np.inf)
                ) if "R1_R2_interaction" in family_scores and "R1_R2_main_effects" in family_scores else None,
                "choice_margin_mean_variance_weighted_R2": (
                    float(family_scores["choice_margin"]) if "choice_margin" in family_scores else None
                ),
            }
    summary = {
        "analysis": "reward_encoding_vector_regressions",
        "models_analyzed": int(results["model_id"].nunique()) if len(results) and "model_id" in results else 0,
        "n_trials_per_model": n_trials,
        "failures": failures,
        "best_by_analysis": best,
        "target_summaries": target_summaries,
        "competing_model_comparison_rows": interaction.to_dict(orient="records") if len(interaction) else [],
    }
    summary = json_safe_value(summary)
    with open(outdir / "reward_encoding_vector_r2_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "reward_encoding_vector_r2_summary.txt", "w") as handle:
        handle.write("Reward encoding vector R2 summary\n")
        handle.write("=================================\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
    with open(outdir / "reward_encoding_competing_model_summary.txt", "w") as handle:
        handle.write("Reward encoding competing model summary\n")
        handle.write("======================================\n")
        for target_vector, target_summary in summary.get("target_summaries", {}).items():
            handle.write(f"{target_vector}: {target_summary}\n")
        handle.write("\nInterpretation notes\n")
        handle.write(
            "If reward_difference predicts the posterior vector as well as or better than R2, "
            "the latent update may encode relative value rather than current reward alone.\n"
        )
        handle.write(
            "If absolute_reward_difference predicts sigma better than R2, posterior uncertainty "
            "may track update magnitude or evidence discrepancy.\n"
        )
        handle.write(
            "If R1_R2_interaction beats R1_R2_main_effects, the second posterior update is "
            "contextualized by the first observed reward in a nonlinear way.\n"
        )
        handle.write(
            "If choice_margin predicts the posterior vector well, the latent posterior may "
            "reflect choice certainty or decision margin.\n"
        )


def run_reward_encoding_analyses(
    df: pd.DataFrame,
    outdir: Path,
    cv_folds: int,
    failures: List[Dict],
    n_trials: int,
    make_plots: bool = True,
):
    df = add_reward_encoding_derived_columns(df)
    group_cols = reward_encoding_group_columns(df)
    rows = []
    prediction_rows = []
    if not group_cols:
        group_iter = [((), df)]
    else:
        group_iter = df.groupby(group_cols, dropna=False)
    for values, group_df in group_iter:
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(group_cols, values))
        for spec in REWARD_ENCODING_ANALYSES:
            row, preds = fit_reward_encoding_spec(group_name, group_df, spec, cv_folds)
            rows.append(row)
            prediction_rows.extend(preds)
    results = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    interaction = reward_encoding_competing_model_comparison(results)
    results.to_csv(outdir / "reward_encoding_vector_r2_results.csv", index=False)
    predictions.to_csv(outdir / "reward_encoding_cv_predictions.csv", index=False)
    interaction.to_csv(outdir / "reward_encoding_competing_model_comparison.csv", index=False)
    interaction.to_csv(outdir / "reward_encoding_interaction_comparison.csv", index=False)
    if make_plots:
        figdir = outdir / "figures"
        plot_reward_encoding_outputs(results, predictions, interaction, figdir)
    write_reward_encoding_summary(outdir, results, interaction, failures, n_trials)
    return results, predictions, interaction


def geometry_feature_group(predictor_family: str) -> str:
    name = str(predictor_family)
    if name.startswith("prior_relative_fisher_distance"):
        return "prior_relative_fisher_distance"
    if name.startswith("prior_relative_halfplane_angle"):
        return "prior_relative_halfplane_angle"
    if name.startswith("prior_relative_halfplane_radius"):
        return "prior_relative_halfplane_radius"
    if name.startswith("prior_relative_halfplane_dim"):
        return "prior_relative_halfplane_vector"
    if name.startswith("mu_angle"):
        return "posterior_angle"
    if name.startswith("mu_radius_plus_angle"):
        return "posterior_radius_plus_angle"
    if name.startswith("mu_radius"):
        return "posterior_radius"
    if name.startswith("sigma_vector"):
        return "posterior_sigma"
    if name.startswith("sigma_radius"):
        return "posterior_sigma_radius"
    if "delta" in name and "angle" in name:
        return "posterior_prior_delta_angle"
    if "delta" in name and "radius" in name:
        return "posterior_prior_delta_radius"
    if "kl" in name:
        return "posterior_prior_kl"
    if "halfplane" in name and "dim0" in name:
        return "halfplane_dim0"
    if "halfplane" in name and "dim1" in name:
        return "halfplane_dim1"
    return name


def suffixed(cols: Sequence[str], timestep: int) -> List[str]:
    return [f"{col}_t{timestep}" for col in cols]


GEOMETRY_DECODING_FEATURES = {
    1: [
        ("mu_radius_t1", ["mu_radius_t1"]),
        ("mu_angle_t1", ["mu_sin_angle_t1", "mu_cos_angle_t1"]),
        ("mu_radius_plus_angle_t1", ["mu_radius_t1", "mu_sin_angle_t1", "mu_cos_angle_t1"]),
        ("sigma_vector_t1", ["z_sigma_0_t1", "z_sigma_1_t1"]),
        ("sigma_radius_t1", ["sigma_radius_t1"]),
        ("posterior_prior_delta_radius_t1", ["delta_mu_radius_t1"]),
        ("posterior_prior_delta_angle_t1", ["delta_mu_sin_angle_t1", "delta_mu_cos_angle_t1"]),
        ("posterior_prior_kl_t1", ["posterior_prior_kl_t1"]),
        ("halfplane_dim0_t1", ["halfplane_x_0_t1", "halfplane_y_0_t1"]),
        ("halfplane_dim1_t1", ["halfplane_x_1_t1", "halfplane_y_1_t1"]),
        ("halfplane_angle_dim0_t1", ["halfplane_sin_angle_0_t1", "halfplane_cos_angle_0_t1"]),
        ("halfplane_angle_dim1_t1", ["halfplane_sin_angle_1_t1", "halfplane_cos_angle_1_t1"]),
        ("prior_relative_fisher_distance_total_t1", ["prior_relative_fisher_distance_total_t1"]),
        ("prior_relative_fisher_distance_dim0_t1", ["prior_relative_fisher_distance_0_t1"]),
        ("prior_relative_fisher_distance_dim1_t1", ["prior_relative_fisher_distance_1_t1"]),
        ("prior_relative_halfplane_angle_dim0_t1", ["delta_halfplane_sin_angle_0_t1", "delta_halfplane_cos_angle_0_t1"]),
        ("prior_relative_halfplane_angle_dim1_t1", ["delta_halfplane_sin_angle_1_t1", "delta_halfplane_cos_angle_1_t1"]),
        ("prior_relative_halfplane_radius_dim0_t1", ["delta_halfplane_radius_0_t1"]),
        ("prior_relative_halfplane_radius_dim1_t1", ["delta_halfplane_radius_1_t1"]),
        ("prior_relative_halfplane_dim0_t1", ["delta_halfplane_x_0_t1", "delta_halfplane_y_0_t1"]),
        ("prior_relative_halfplane_dim1_t1", ["delta_halfplane_x_1_t1", "delta_halfplane_y_1_t1"]),
    ],
    2: [
        ("mu_radius_t2", ["mu_radius_t2"]),
        ("mu_angle_t2", ["mu_sin_angle_t2", "mu_cos_angle_t2"]),
        ("mu_radius_plus_angle_t2", ["mu_radius_t2", "mu_sin_angle_t2", "mu_cos_angle_t2"]),
        ("sigma_vector_t2", ["z_sigma_0_t2", "z_sigma_1_t2"]),
        ("sigma_radius_t2", ["sigma_radius_t2"]),
        ("delta_mu_radius_t2", ["delta_mu_radius_t2"]),
        ("delta_mu_angle_t2", ["delta_mu_sin_angle_t2", "delta_mu_cos_angle_t2"]),
        ("delta_mu_radius_plus_angle_t2", ["delta_mu_radius_t2", "delta_mu_sin_angle_t2", "delta_mu_cos_angle_t2"]),
        ("posterior_prior_kl_t2", ["posterior_prior_kl_t2"]),
        ("halfplane_dim0_t2", ["halfplane_x_0_t2", "halfplane_y_0_t2"]),
        ("halfplane_dim1_t2", ["halfplane_x_1_t2", "halfplane_y_1_t2"]),
        ("halfplane_angle_dim0_t2", ["halfplane_sin_angle_0_t2", "halfplane_cos_angle_0_t2"]),
        ("halfplane_angle_dim1_t2", ["halfplane_sin_angle_1_t2", "halfplane_cos_angle_1_t2"]),
        ("prior_relative_fisher_distance_total_t2", ["prior_relative_fisher_distance_total_t2"]),
        ("prior_relative_fisher_distance_dim0_t2", ["prior_relative_fisher_distance_0_t2"]),
        ("prior_relative_fisher_distance_dim1_t2", ["prior_relative_fisher_distance_1_t2"]),
        ("prior_relative_halfplane_angle_dim0_t2", ["delta_halfplane_sin_angle_0_t2", "delta_halfplane_cos_angle_0_t2"]),
        ("prior_relative_halfplane_angle_dim1_t2", ["delta_halfplane_sin_angle_1_t2", "delta_halfplane_cos_angle_1_t2"]),
        ("prior_relative_halfplane_radius_dim0_t2", ["delta_halfplane_radius_0_t2"]),
        ("prior_relative_halfplane_radius_dim1_t2", ["delta_halfplane_radius_1_t2"]),
        ("prior_relative_halfplane_dim0_t2", ["delta_halfplane_x_0_t2", "delta_halfplane_y_0_t2"]),
        ("prior_relative_halfplane_dim1_t2", ["delta_halfplane_x_1_t2", "delta_halfplane_y_1_t2"]),
    ],
}


def make_geometry_to_task_specs() -> List[Dict]:
    specs = []
    t1_value_targets = [
        ("reward_t1", "regression"),
        ("current_best_path_value_t1", "regression"),
        ("current_best_path_margin_t1", "regression"),
    ]
    for target, target_type in t1_value_targets:
        for family, cols in GEOMETRY_DECODING_FEATURES[1]:
            specs.append({
                "analysis_direction": "geometry_to_task",
                "target_variable": target,
                "timestep": 1,
                "predictor_family": family,
                "predictor_columns": cols,
                "target_type": target_type,
            })
    for family, cols in GEOMETRY_DECODING_FEATURES[1]:
        specs.append({
            "analysis_direction": "geometry_to_task",
            "target_variable": "current_best_path_t1",
            "timestep": 1,
            "predictor_family": family,
            "predictor_columns": cols,
            "target_type": "classification",
        })
    for target, target_type in [
        ("reward_t2", "regression"),
        ("reward_t2_minus_reward_t1", "regression"),
        ("abs_reward_t2_minus_reward_t1", "regression"),
        ("current_best_path_value_t2", "regression"),
        ("current_best_path_margin_t2", "regression"),
        ("current_best_path_switch_t2", "binary"),
        ("current_best_path_t2", "classification"),
    ]:
        for family, cols in GEOMETRY_DECODING_FEATURES[2]:
            specs.append({
                "analysis_direction": "geometry_to_task",
                "target_variable": target,
                "timestep": 2,
                "predictor_family": family,
                "predictor_columns": cols,
                "target_type": target_type,
            })
    return specs


def make_task_to_geometry_specs() -> List[Dict]:
    t1_targets = [
        ("mu_radius_t1", ["mu_radius_t1"]),
        ("mu_angle_t1", ["mu_sin_angle_t1", "mu_cos_angle_t1"]),
        ("sigma_vector_t1", ["z_sigma_0_t1", "z_sigma_1_t1"]),
        ("sigma_radius_t1", ["sigma_radius_t1"]),
        ("delta_mu_radius_t1", ["delta_mu_radius_t1"]),
        ("delta_mu_angle_t1", ["delta_mu_sin_angle_t1", "delta_mu_cos_angle_t1"]),
        ("posterior_prior_kl_t1", ["posterior_prior_kl_t1"]),
        ("prior_relative_fisher_distance_total_t1", ["prior_relative_fisher_distance_total_t1"]),
        ("prior_relative_fisher_distance_0_t1", ["prior_relative_fisher_distance_0_t1"]),
        ("prior_relative_fisher_distance_1_t1", ["prior_relative_fisher_distance_1_t1"]),
        ("prior_relative_halfplane_angle_dim0_t1", ["delta_halfplane_sin_angle_0_t1", "delta_halfplane_cos_angle_0_t1"]),
        ("prior_relative_halfplane_angle_dim1_t1", ["delta_halfplane_sin_angle_1_t1", "delta_halfplane_cos_angle_1_t1"]),
        ("delta_halfplane_radius_0_t1", ["delta_halfplane_radius_0_t1"]),
        ("delta_halfplane_radius_1_t1", ["delta_halfplane_radius_1_t1"]),
    ]
    t2_targets = [
        ("mu_radius_t2", ["mu_radius_t2"]),
        ("mu_angle_t2", ["mu_sin_angle_t2", "mu_cos_angle_t2"]),
        ("sigma_vector_t2", ["z_sigma_0_t2", "z_sigma_1_t2"]),
        ("sigma_radius_t2", ["sigma_radius_t2"]),
        ("delta_mu_radius_t2", ["delta_mu_radius_t2"]),
        ("delta_mu_angle_t2", ["delta_mu_sin_angle_t2", "delta_mu_cos_angle_t2"]),
        ("posterior_prior_kl_t2", ["posterior_prior_kl_t2"]),
        ("prior_relative_fisher_distance_total_t2", ["prior_relative_fisher_distance_total_t2"]),
        ("prior_relative_fisher_distance_0_t2", ["prior_relative_fisher_distance_0_t2"]),
        ("prior_relative_fisher_distance_1_t2", ["prior_relative_fisher_distance_1_t2"]),
        ("prior_relative_halfplane_angle_dim0_t2", ["delta_halfplane_sin_angle_0_t2", "delta_halfplane_cos_angle_0_t2"]),
        ("prior_relative_halfplane_angle_dim1_t2", ["delta_halfplane_sin_angle_1_t2", "delta_halfplane_cos_angle_1_t2"]),
        ("delta_halfplane_radius_0_t2", ["delta_halfplane_radius_0_t2"]),
        ("delta_halfplane_radius_1_t2", ["delta_halfplane_radius_1_t2"]),
    ]
    t1_predictors = [
        ("R1_only", ["reward_t1"]),
        ("current_best_path_value_t1", ["current_best_path_value_t1"]),
        ("current_best_path_margin_t1", ["current_best_path_margin_t1"]),
    ]
    t2_predictors = [
        ("R2_only", ["reward_t2"]),
        ("reward_difference", ["reward_t2_minus_reward_t1"]),
        ("absolute_reward_difference", ["abs_reward_t2_minus_reward_t1"]),
        ("R1_R2_main_effects", ["reward_t1", "reward_t2"]),
        ("R1_R2_interaction", ["reward_t1", "reward_t2", "reward_t1_x_reward_t2"]),
        ("choice_margin", ["current_best_path_margin_t2"]),
        ("R1_R2_margin", ["reward_t1", "reward_t2", "current_best_path_margin_t2"]),
    ]
    specs = []
    for target, y_cols in t1_targets:
        for family, x_cols in t1_predictors:
            specs.append({
                "analysis_direction": "task_to_geometry",
                "target_geometry": target,
                "timestep": 1,
                "predictor_family": family,
                "predictor_columns": x_cols,
                "y_cols": y_cols,
            })
    for target, y_cols in t2_targets:
        for family, x_cols in t2_predictors:
            specs.append({
                "analysis_direction": "task_to_geometry",
                "target_geometry": target,
                "timestep": 2,
                "predictor_family": family,
                "predictor_columns": x_cols,
                "y_cols": y_cols,
            })
    return specs


def base_geometry_decoding_row(group_name: Dict, spec: Dict, status: str, error_message: str, n_trials: int = 0, n_folds: int = 0) -> Dict:
    return {
        **group_name,
        "analysis_direction": "geometry_to_task",
        "target_variable": spec["target_variable"],
        "timestep": spec["timestep"],
        "predictor_family": spec["predictor_family"],
        "feature_family": geometry_feature_group(spec["predictor_family"]),
        "predictor_columns": ",".join(spec["predictor_columns"]),
        "n_trials": int(n_trials),
        "n_folds": int(n_folds),
        "metric_primary": np.nan,
        "R2": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "Pearson_r": np.nan,
        "Spearman_r": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "status": status,
        "error_message": error_message,
    }


def base_task_to_geometry_row(group_name: Dict, spec: Dict, status: str, error_message: str, n_trials: int = 0, n_folds: int = 0) -> Dict:
    return {
        **group_name,
        "analysis_direction": "task_to_geometry",
        "target_geometry": spec["target_geometry"],
        "timestep": spec["timestep"],
        "predictor_family": spec["predictor_family"],
        "predictor_columns": ",".join(spec["predictor_columns"]),
        "n_trials": int(n_trials),
        "n_folds": int(n_folds),
        "overall_R2": np.nan,
        "variance_weighted_R2": np.nan,
        "mean_R2_across_dims": np.nan,
        "R2_dim0": np.nan,
        "R2_dim1": np.nan,
        "Pearson_r_dim0": np.nan,
        "Pearson_r_dim1": np.nan,
        "Spearman_r_dim0": np.nan,
        "Spearman_r_dim1": np.nan,
        "RMSE_dim0": np.nan,
        "RMSE_dim1": np.nan,
        "MAE_dim0": np.nan,
        "MAE_dim1": np.nan,
        "status": status,
        "error_message": error_message,
    }


def spec_uses_prior_relative_geometry(spec: Dict) -> bool:
    text = " ".join(
        [
            str(spec.get("predictor_family", "")),
            str(spec.get("target_geometry", "")),
            ",".join(spec.get("predictor_columns", [])),
            ",".join(spec.get("y_cols", [])),
        ]
    )
    return any(
        token in text
        for token in (
            "prior_relative",
            "delta_halfplane",
            "prior_halfplane",
        )
    )


def cv_splits_for_work(work: pd.DataFrame, cv_folds: int):
    from sklearn.model_selection import GroupKFold

    groups = work["trial_uid"].astype(str).to_numpy() if "trial_uid" in work.columns else work.index.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(cv_folds), len(unique_groups))
    if n_splits < 2:
        return [], groups, 0
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(work)), groups=groups)), groups, n_splits


def finite_model_frame(group_df: pd.DataFrame, cols: Sequence[str], timestep: int, qz_filter: bool) -> Tuple[pd.DataFrame, str, str]:
    status = "ok"
    error_message = ""
    if qz_filter:
        work = rows_after_observed_reward_used_downstream(group_df, timestep=timestep)
        if "qz_used_downstream" not in group_df.columns:
            status = "warning_qz2_downstream_usage_unknown"
            error_message = "qz_used_downstream column is unavailable"
    else:
        work = group_df[pd.to_numeric(group_df["timestep"], errors="coerce") == timestep].copy()
    numeric = work[list(cols)].apply(pd.to_numeric, errors="coerce")
    work = work[np.isfinite(numeric).all(axis=1)].copy()
    return work, status, error_message


def fit_geometry_to_task_spec(group_name: Dict, group_df: pd.DataFrame, spec: Dict, cv_folds: int) -> Dict:
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    required = set(spec["predictor_columns"] + [spec["target_variable"], "timestep"])
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        if spec_uses_prior_relative_geometry(spec):
            return base_geometry_decoding_row(group_name, spec, "skipped_missing_prior", f"missing prior-relative columns: {','.join(missing)}")
        if "margin" in spec["target_variable"] or any("margin" in col for col in spec["predictor_columns"]):
            return base_geometry_decoding_row(group_name, spec, "skipped_missing_margin", f"missing columns: {','.join(missing)}")
        return base_geometry_decoding_row(group_name, spec, "skipped_missing_columns", f"missing columns: {','.join(missing)}")
    work, status, error_message = finite_model_frame(
        group_df,
        spec["predictor_columns"] + [spec["target_variable"]],
        spec["timestep"],
        qz_filter=True,
    )
    if spec_uses_prior_relative_geometry(spec) and len(work) == 0:
        return base_geometry_decoding_row(group_name, spec, "skipped_missing_prior", "no finite prior-relative rows")
    if len(work) < 4:
        return base_geometry_decoding_row(group_name, spec, "skipped_too_few_rows", "fewer than four valid rows")
    x = work[spec["predictor_columns"]].to_numpy(dtype=float)
    if np.any(np.nanstd(x, axis=0) <= 1e-12):
        return base_geometry_decoding_row(group_name, spec, "skipped_no_predictor_variance", "one or more predictors have no variance", len(work))
    y = pd.to_numeric(work[spec["target_variable"]], errors="coerce").to_numpy()
    splits, _, n_splits = cv_splits_for_work(work, cv_folds)
    if n_splits < 2:
        return base_geometry_decoding_row(group_name, spec, "skipped_insufficient_folds", "fewer than two CV folds", len(work), n_splits)
    target_type = spec["target_type"]
    if target_type == "regression":
        if np.nanstd(y) <= 1e-12:
            return base_geometry_decoding_row(group_name, spec, "skipped_no_target_variance", "target has no variance", len(work), n_splits)
        pred = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in splits:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
            ])
            model.fit(x[train_idx], y[train_idx])
            pred[test_idx] = model.predict(x[test_idx])
        finite = np.isfinite(pred) & np.isfinite(y)
        row = base_geometry_decoding_row(group_name, spec, status, error_message, int(finite.sum()), n_splits)
        if finite.sum() >= 3:
            row.update({
                "R2": float(r2_score(y[finite], pred[finite])),
                "Pearson_r": safe_regression_corr(y[finite], pred[finite], "pearson"),
                "Spearman_r": safe_regression_corr(y[finite], pred[finite], "spearman"),
                "RMSE": float(np.sqrt(mean_squared_error(y[finite], pred[finite]))),
                "MAE": float(mean_absolute_error(y[finite], pred[finite])),
            })
            row["metric_primary"] = row["R2"]
        return row

    y_class = y.astype(int)
    classes = np.unique(y_class)
    if len(classes) < 2:
        return base_geometry_decoding_row(group_name, spec, "skipped_no_target_variance", "classification target has fewer than two classes", len(work), n_splits)
    pred = np.full(len(y_class), -999999, dtype=int)
    score = np.full(len(y_class), np.nan, dtype=float)
    for train_idx, test_idx in splits:
        if len(np.unique(y_class[train_idx])) < 2:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        model.fit(x[train_idx], y_class[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
        if target_type == "binary" and hasattr(model.named_steps["model"], "classes_"):
            proba = model.predict_proba(x[test_idx])
            if proba.shape[1] == 2:
                score[test_idx] = proba[:, 1]
    finite = pred != -999999
    row = base_geometry_decoding_row(group_name, spec, status, error_message, int(finite.sum()), n_splits)
    if finite.sum() > 0:
        row["balanced_accuracy"] = float(balanced_accuracy_score(y_class[finite], pred[finite]))
        row["macro_f1"] = float(f1_score(y_class[finite], pred[finite], average="macro"))
        row["metric_primary"] = row["balanced_accuracy"]
        if target_type == "binary" and np.isfinite(score[finite]).any() and len(np.unique(y_class[finite])) == 2:
            score_f = score[finite]
            finite_score = np.isfinite(score_f)
            row["roc_auc"] = float(roc_auc_score(y_class[finite][finite_score], score_f[finite_score]))
            row["average_precision"] = float(average_precision_score(y_class[finite][finite_score], score_f[finite_score]))
    return row


def fit_task_to_geometry_spec(group_name: Dict, group_df: pd.DataFrame, spec: Dict, cv_folds: int) -> Dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    required = set(spec["predictor_columns"] + spec["y_cols"] + ["timestep"])
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        if spec_uses_prior_relative_geometry(spec):
            return base_task_to_geometry_row(group_name, spec, "skipped_missing_prior", f"missing prior-relative columns: {','.join(missing)}")
        if any("margin" in col for col in spec["predictor_columns"]):
            return base_task_to_geometry_row(group_name, spec, "skipped_missing_margin", f"missing columns: {','.join(missing)}")
        return base_task_to_geometry_row(group_name, spec, "skipped_missing_columns", f"missing columns: {','.join(missing)}")
    work, status, error_message = finite_model_frame(
        group_df,
        spec["predictor_columns"] + spec["y_cols"],
        spec["timestep"],
        qz_filter=True,
    )
    if spec_uses_prior_relative_geometry(spec) and len(work) == 0:
        return base_task_to_geometry_row(group_name, spec, "skipped_missing_prior", "no finite prior-relative rows")
    if len(work) < 4:
        return base_task_to_geometry_row(group_name, spec, "skipped_too_few_rows", "fewer than four valid rows")
    x = work[spec["predictor_columns"]].to_numpy(dtype=float)
    y = work[spec["y_cols"]].to_numpy(dtype=float)
    if np.any(np.nanstd(x, axis=0) <= 1e-12):
        return base_task_to_geometry_row(group_name, spec, "skipped_no_predictor_variance", "one or more predictors have no variance", len(work))
    if np.any(np.nanstd(y, axis=0) <= 1e-12):
        return base_task_to_geometry_row(group_name, spec, "skipped_no_target_variance", "one or more target dimensions have no variance", len(work))
    splits, _, n_splits = cv_splits_for_work(work, cv_folds)
    if n_splits < 2:
        return base_task_to_geometry_row(group_name, spec, "skipped_insufficient_folds", "fewer than two CV folds", len(work), n_splits)
    pred = np.full_like(y, np.nan, dtype=float)
    for train_idx, test_idx in splits:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        ])
        model.fit(x[train_idx], y[train_idx])
        fold_pred = np.asarray(model.predict(x[test_idx]), dtype=float)
        if fold_pred.ndim == 1:
            fold_pred = fold_pred[:, None]
        pred[test_idx, :] = fold_pred
    valid = np.isfinite(pred).all(axis=1)
    row = base_task_to_geometry_row(group_name, spec, status, error_message, int(valid.sum()), n_splits)
    if valid.sum() >= 3:
        yv = y[valid]
        pv = pred[valid]
        dim_scores = [float(r2_score(yv[:, dim], pv[:, dim])) for dim in range(yv.shape[1])]
        row["overall_R2"] = float(r2_score(yv, pv, multioutput="uniform_average"))
        row["variance_weighted_R2"] = float(r2_score(yv, pv, multioutput="variance_weighted"))
        row["mean_R2_across_dims"] = float(np.nanmean(dim_scores))
        row["R2_dim0"] = dim_scores[0]
        row["R2_dim1"] = dim_scores[1] if len(dim_scores) > 1 else np.nan
        for dim in range(min(2, yv.shape[1])):
            row[f"Pearson_r_dim{dim}"] = safe_regression_corr(yv[:, dim], pv[:, dim], "pearson")
            row[f"Spearman_r_dim{dim}"] = safe_regression_corr(yv[:, dim], pv[:, dim], "spearman")
            row[f"RMSE_dim{dim}"] = float(np.sqrt(mean_squared_error(yv[:, dim], pv[:, dim])))
            row[f"MAE_dim{dim}"] = float(mean_absolute_error(yv[:, dim], pv[:, dim]))
    return row


def fit_within_path_geometry_decoding(group_name: Dict, group_df: pd.DataFrame, cv_folds: int, min_n: int) -> List[Dict]:
    rows = []
    for timestep, targets in {
        1: ["reward_t1", "current_best_path_value_t1", "current_best_path_margin_t1"],
        2: ["reward_t2", "reward_t2_minus_reward_t1", "abs_reward_t2_minus_reward_t1", "current_best_path_value_t2", "current_best_path_margin_t2"],
    }.items():
        path_col = f"current_best_path_t{timestep}"
        if path_col not in group_df.columns:
            continue
        for path_value, path_df in group_df.groupby(path_col, dropna=True):
            path_df = rows_after_observed_reward_used_downstream(path_df, timestep=timestep)
            for target in targets:
                for family, cols in [
                    item for item in GEOMETRY_DECODING_FEATURES[timestep]
                    if not item[0].startswith("halfplane_angle") and not item[0].startswith("halfplane_dim")
                ]:
                    spec = {
                        "target_variable": target,
                        "timestep": timestep,
                        "predictor_family": family,
                        "predictor_columns": cols,
                        "target_type": "regression",
                    }
                    row = {
                        **group_name,
                        "current_best_path": path_value,
                        "target_variable": target,
                        "timestep": timestep,
                        "predictor_family": family,
                        "feature_family": geometry_feature_group(family),
                        "n_trials": 0,
                        "metric_primary": np.nan,
                        "R2": np.nan,
                        "balanced_accuracy": np.nan,
                        "Pearson_r": np.nan,
                        "Spearman_r": np.nan,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "status": "ok",
                        "error_message": "",
                    }
                    if len(path_df) < min_n:
                        row.update({"n_trials": int(len(path_df)), "status": "skipped_too_few_rows", "error_message": f"fewer than min-within-path-n={min_n} rows"})
                        rows.append(row)
                        continue
                    fit_row = fit_geometry_to_task_spec(group_name, path_df, spec, cv_folds)
                    for key in ["n_trials", "metric_primary", "R2", "balanced_accuracy", "Pearson_r", "Spearman_r", "RMSE", "MAE", "status", "error_message"]:
                        row[key] = fit_row.get(key, row.get(key))
                    rows.append(row)
    return rows


def summarize_geometry_meaning(decoding: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = decoding[decoding["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])].copy() if len(decoding) else pd.DataFrame()
    if len(valid) == 0:
        return pd.DataFrame(), pd.DataFrame()
    valid["metric_primary"] = pd.to_numeric(valid["metric_primary"], errors="coerce")

    def finite_max(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")

    rows = []
    for (target, timestep), group in valid.groupby(["target_variable", "timestep"], dropna=False):
        group = group[np.isfinite(group["metric_primary"])]
        if len(group) == 0:
            continue
        mean_by_family = group.groupby("predictor_family")["metric_primary"].mean()
        best_family = mean_by_family.idxmax()
        feature_best = group.groupby("feature_family")["metric_primary"].mean()
        rows.append({
            "target_variable": target,
            "timestep": timestep,
            "best_predictor_family": best_family,
            "best_metric": "metric_primary",
            "best_metric_value": float(mean_by_family.max()),
            "best_angle_predictor_value": float(feature_best.get("posterior_angle", np.nan)),
            "best_radius_predictor_value": float(feature_best.get("posterior_radius", np.nan)),
            "best_sigma_predictor_value": finite_max([
                feature_best.get("posterior_sigma", np.nan),
                feature_best.get("posterior_sigma_radius", np.nan),
            ]),
            "best_delta_predictor_value": finite_max([
                feature_best.get("posterior_prior_delta_angle", np.nan),
                feature_best.get("posterior_prior_delta_radius", np.nan),
            ]),
            "best_kl_predictor_value": float(feature_best.get("posterior_prior_kl", np.nan)),
        })
    seed_means = (
        valid.groupby(["feature_family", "target_variable", "timestep", "seed"], dropna=False)["metric_primary"]
        .mean()
        .reset_index()
    )
    feature_summary = (
        seed_means.groupby(["feature_family", "target_variable", "timestep"], dropna=False)["metric_primary"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    if len(feature_summary):
        feature_summary["sem"] = feature_summary["std"] / np.sqrt(feature_summary["count"].clip(lower=1))
    return pd.DataFrame(rows), feature_summary


def plot_geometry_meaning_outputs(decoding: pd.DataFrame, encoding: pd.DataFrame, within_path: pd.DataFrame, summary: pd.DataFrame, figdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)

    def heatmap(data: pd.DataFrame, index: str, columns: str, values: str, path: str, title: str):
        if len(data) == 0:
            return
        pivot = data.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(title, fontsize=PANEL_FONT_SIZE)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figdir / path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    valid_dec = decoding[decoding["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])].copy() if len(decoding) else pd.DataFrame()
    if len(valid_dec):
        valid_dec["metric_primary"] = pd.to_numeric(valid_dec["metric_primary"], errors="coerce")
        heatmap(valid_dec[valid_dec["timestep"] == 1], "predictor_family", "target_variable", "metric_primary", "geometry_to_task_decoding_heatmap_t1.png", "Geometry to task decoding t1")
        heatmap(valid_dec[valid_dec["timestep"] == 2], "predictor_family", "target_variable", "metric_primary", "geometry_to_task_decoding_heatmap_t2.png", "Geometry to task decoding t2")

    valid_enc = encoding[encoding["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])].copy() if len(encoding) else pd.DataFrame()
    if len(valid_enc):
        valid_enc["variance_weighted_R2"] = pd.to_numeric(valid_enc["variance_weighted_R2"], errors="coerce")
        heatmap(valid_enc[valid_enc["timestep"] == 1], "predictor_family", "target_geometry", "variance_weighted_R2", "task_to_geometry_encoding_heatmap_t1.png", "Task to geometry encoding t1")
        heatmap(valid_enc[valid_enc["timestep"] == 2], "predictor_family", "target_geometry", "variance_weighted_R2", "task_to_geometry_encoding_heatmap_t2.png", "Task to geometry encoding t2")

    if len(valid_dec):
        seed_summary = valid_dec.groupby(["feature_family", "seed"], dropna=False)["metric_primary"].mean().reset_index()
        if len(seed_summary):
            fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
            families = sorted(seed_summary["feature_family"].dropna().unique())
            seeds = sorted(seed_summary["seed"].dropna().unique()) if "seed" in seed_summary else ["all"]
            x = np.arange(len(families), dtype=float)
            width = 0.8 / max(len(seeds), 1)
            for seed_i, seed in enumerate(seeds):
                piece = seed_summary[seed_summary["seed"] == seed]
                vals = [pd.to_numeric(piece[piece["feature_family"] == fam]["metric_primary"], errors="coerce").mean() for fam in families]
                ax.bar(x + (seed_i - (len(seeds) - 1) / 2) * width, vals, width=width, label=f"seed {seed}")
            ax.set_xticks(x)
            ax.set_xticklabels(families, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("mean primary metric")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            fig.savefig(figdir / "feature_family_summary_by_seed.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    valid_wp = within_path[within_path["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])].copy() if len(within_path) else pd.DataFrame()
    if len(valid_wp):
        valid_wp["metric_primary"] = pd.to_numeric(valid_wp["metric_primary"], errors="coerce")
        heatmap(valid_wp[valid_wp["timestep"] == 1], "predictor_family", "target_variable", "metric_primary", "within_path_value_coding_t1.png", "Within-path value coding t1")
        heatmap(valid_wp[valid_wp["timestep"] == 2], "predictor_family", "target_variable", "metric_primary", "within_path_value_coding_t2.png", "Within-path value coding t2")

    if len(summary):
        plot_cols = [
            "best_angle_predictor_value",
            "best_radius_predictor_value",
            "best_sigma_predictor_value",
            "best_delta_predictor_value",
            "best_kl_predictor_value",
        ]
        piece = summary.copy()
        labels = [f"{row.target_variable}\nt{row.timestep}" for row in piece.itertuples()]
        x = np.arange(len(labels), dtype=float)
        width = 0.8 / len(plot_cols)
        fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
        for i, col in enumerate(plot_cols):
            ax.bar(x + (i - (len(plot_cols) - 1) / 2) * width, pd.to_numeric(piece[col], errors="coerce"), width=width, label=col.replace("best_", "").replace("_predictor_value", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("primary metric")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        fig.savefig(figdir / "angle_radius_sigma_interpretation_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def prior_relative_status_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0 or "status" not in df.columns:
        return pd.DataFrame()
    return df[df["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])].copy()


def is_prior_relative_family(name: str) -> bool:
    name = str(name)
    return name.startswith("prior_relative_") or name.startswith("delta_halfplane")


def build_prior_relative_distance_vs_kl_results(decoding: pd.DataFrame) -> pd.DataFrame:
    valid = prior_relative_status_filter(decoding)
    if len(valid) == 0:
        return pd.DataFrame()
    valid["metric_primary"] = pd.to_numeric(valid["metric_primary"], errors="coerce")
    rows = []
    comparison_families = {
        "posterior_prior_kl": lambda s: s.startswith("posterior_prior_kl"),
        "prior_relative_fisher_distance_total": lambda s: s.startswith("prior_relative_fisher_distance_total"),
        "prior_relative_fisher_distance_dim0": lambda s: s.startswith("prior_relative_fisher_distance_dim0"),
        "prior_relative_fisher_distance_dim1": lambda s: s.startswith("prior_relative_fisher_distance_dim1"),
        "prior_relative_halfplane_angle_dim0": lambda s: s.startswith("prior_relative_halfplane_angle_dim0"),
        "prior_relative_halfplane_angle_dim1": lambda s: s.startswith("prior_relative_halfplane_angle_dim1"),
        "prior_relative_halfplane_radius_dim0": lambda s: s.startswith("prior_relative_halfplane_radius_dim0"),
        "prior_relative_halfplane_radius_dim1": lambda s: s.startswith("prior_relative_halfplane_radius_dim1"),
    }
    group_cols = [col for col in MODEL_GROUP_COLUMNS if col in valid.columns]
    for values, group in valid.groupby(group_cols + ["target_variable", "timestep"], dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(group_cols + ["target_variable", "timestep"], values))
        family_scores = group.groupby("predictor_family", dropna=False)["metric_primary"].mean()
        named_scores = {}
        for label, selector in comparison_families.items():
            matches = [
                score for family, score in family_scores.items()
                if selector(str(family)) and np.isfinite(score)
            ]
            named_scores[label] = float(np.nanmax(matches)) if matches else np.nan
        prior_scores = {
            family: score for family, score in family_scores.items()
            if is_prior_relative_family(str(family)) and np.isfinite(score)
        }
        if prior_scores:
            best_family = max(prior_scores, key=prior_scores.get)
            best_metric = float(prior_scores[best_family])
        else:
            best_family = ""
            best_metric = np.nan
        kl_metric = named_scores["posterior_prior_kl"]
        row.update(named_scores)
        row.update({
            "best_prior_relative_predictor": best_family,
            "best_prior_relative_metric": best_metric,
            "kl_metric": kl_metric,
            "delta_best_prior_relative_minus_kl": (
                best_metric - kl_metric
                if np.isfinite(best_metric) and np.isfinite(kl_metric)
                else np.nan
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_prior_relative_heatmap(
    data: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    path: Path,
    title: str,
):
    if data is None or len(data) == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = data.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(title, fontsize=PANEL_FONT_SIZE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finite_plot_limits(*arrays, pad: float = 0.08) -> Optional[Tuple[float, float]]:
    values = []
    for array in arrays:
        arr = pd.to_numeric(pd.Series(np.asarray(array).ravel()), errors="coerce").to_numpy(dtype=float)
        values.append(arr[np.isfinite(arr)])
    values = np.concatenate(values) if values else np.asarray([], dtype=float)
    if len(values) == 0:
        return None
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-12:
        span = max(abs(lo), 1.0)
        lo -= 0.5 * span
        hi += 0.5 * span
    else:
        lo -= pad * span
        hi += pad * span
    return lo, hi


def prior_posterior_halfplane_axis_limits(df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    limits = {}
    for timestep in (1, 2):
        t_df = rows_after_observed_reward_used_downstream(df, timestep=timestep)
        if len(t_df) == 0:
            continue
        for dim in (0, 1):
            required = [
                f"prior_halfplane_x_{dim}",
                f"prior_halfplane_y_{dim}",
                f"delta_halfplane_x_{dim}",
                f"delta_halfplane_y_{dim}",
            ]
            if not all(col in t_df.columns for col in required):
                continue
            prior_x = pd.to_numeric(t_df[f"prior_halfplane_x_{dim}"], errors="coerce")
            prior_y = pd.to_numeric(t_df[f"prior_halfplane_y_{dim}"], errors="coerce")
            post_x = prior_x + pd.to_numeric(t_df[f"delta_halfplane_x_{dim}"], errors="coerce")
            post_y = prior_y + pd.to_numeric(t_df[f"delta_halfplane_y_{dim}"], errors="coerce")
            xlim = finite_plot_limits(prior_x, post_x)
            ylim = finite_plot_limits(prior_y, post_y)
            if xlim is not None and ylim is not None:
                limits[(timestep, dim)] = (xlim, ylim)
    return limits


def prior_relative_scatter_axis_limits(df: pd.DataFrame, specs: Sequence[Tuple[int, str, str]]) -> Dict[Tuple[int, str, str], Tuple[Tuple[float, float], Tuple[float, float]]]:
    limits = {}
    for timestep, x_col, y_col in specs:
        t_df = rows_after_observed_reward_used_downstream(df, timestep=timestep)
        if len(t_df) == 0 or x_col not in t_df.columns or y_col not in t_df.columns:
            continue
        xlim = finite_plot_limits(t_df[x_col])
        ylim = finite_plot_limits(t_df[y_col])
        if xlim is not None and ylim is not None:
            limits[(timestep, x_col, y_col)] = (xlim, ylim)
    return limits


def prior_posterior_halfplane_color_limits(df: pd.DataFrame) -> Dict[Tuple[int, int, str], Tuple[float, float]]:
    limits = {}
    for timestep in (1, 2):
        t_df = rows_after_observed_reward_used_downstream(df, timestep=timestep)
        if len(t_df) == 0:
            continue
        color_specs = (
            [("reward", "reward_t1")]
            if timestep == 1
            else [
                ("reward2", "reward_t2"),
                ("reward_difference", "reward_t2_minus_reward_t1"),
                ("path_switch", "current_best_path_switch_t2"),
            ]
        )
        for dim in (0, 1):
            for label, color_col in color_specs:
                if color_col not in t_df.columns:
                    continue
                values = pd.to_numeric(t_df[color_col], errors="coerce")
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                limits[(timestep, dim, label)] = (float(values.min()), float(values.max()))
    return limits


def prior_relative_scatter_color_limits(
    df: pd.DataFrame,
    specs: Sequence[Tuple[int, str, str, Optional[str]]],
) -> Dict[Tuple[int, str, str], Tuple[float, float]]:
    limits = {}
    for timestep, x_col, y_col, color_col in specs:
        if not color_col:
            continue
        t_df = rows_after_observed_reward_used_downstream(df, timestep=timestep)
        if len(t_df) == 0 or color_col not in t_df.columns:
            continue
        values = pd.to_numeric(t_df[color_col], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        limits[(timestep, x_col, y_col)] = (float(values.min()), float(values.max()))
    return limits


def plot_prior_posterior_halfplane_arrows(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    axis_limits: Optional[Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    color_limits: Optional[Dict[Tuple[int, int, str], Tuple[float, float]]] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    apply_7pt_plot_style(plt)

    for timestep in (1, 2):
        t_df = rows_after_observed_reward_used_downstream(df, timestep=timestep)
        if len(t_df) == 0:
            continue
        color_specs = (
            [("reward", "reward_t1")]
            if timestep == 1
            else [
                ("reward2", "reward_t2"),
                ("reward_difference", "reward_t2_minus_reward_t1"),
                ("path_switch", "current_best_path_switch_t2"),
            ]
        )
        for dim in (0, 1):
            required = [
                f"prior_halfplane_x_{dim}",
                f"prior_halfplane_y_{dim}",
                f"delta_halfplane_x_{dim}",
                f"delta_halfplane_y_{dim}",
            ]
            if not all(col in t_df.columns for col in required):
                continue
            numeric = t_df[required].apply(pd.to_numeric, errors="coerce")
            for label, color_col in color_specs:
                if color_col not in t_df.columns:
                    continue
                piece = t_df.copy()
                color_values = pd.to_numeric(piece[color_col], errors="coerce")
                finite = np.isfinite(numeric).all(axis=1) & np.isfinite(color_values)
                piece = piece[finite].copy()
                if len(piece) == 0:
                    continue
                if len(piece) > 1200:
                    piece = piece.sample(1200, random_state=101 + timestep * 10 + dim)
                    color_values = pd.to_numeric(piece[color_col], errors="coerce")
                if color_limits and (timestep, dim, label) in color_limits:
                    vmin, vmax = color_limits[(timestep, dim, label)]
                else:
                    vmin = float(color_values.min())
                    vmax = float(color_values.max())
                if math.isclose(vmin, vmax):
                    vmin -= 0.5
                    vmax += 0.5
                norm = Normalize(vmin=vmin, vmax=vmax)
                colors = plt.get_cmap("coolwarm")(norm(color_values))
                fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
                ax.quiver(
                    pd.to_numeric(piece[f"prior_halfplane_x_{dim}"], errors="coerce"),
                    pd.to_numeric(piece[f"prior_halfplane_y_{dim}"], errors="coerce"),
                    pd.to_numeric(piece[f"delta_halfplane_x_{dim}"], errors="coerce"),
                    pd.to_numeric(piece[f"delta_halfplane_y_{dim}"], errors="coerce"),
                    color=colors,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    alpha=0.45,
                    width=0.002,
                )
                ax.set_xlabel("prior mu / sqrt(2)")
                ax.set_ylabel("prior sigma")
                ax.set_title(f"prior to posterior, t{timestep}, dim {dim}", fontsize=PANEL_FONT_SIZE)
                if axis_limits and (timestep, dim) in axis_limits:
                    ax.set_xlim(*axis_limits[(timestep, dim)][0])
                    ax.set_ylim(*axis_limits[(timestep, dim)][1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
                sm.set_array([])
                fig.tight_layout(rect=[0, 0, 0.84, 1])
                cax = fig.add_axes([0.87, 0.18, 0.03, 0.64])
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label(color_col)
                fig.savefig(
                    figdir / f"prior_posterior_halfplane_t{timestep}_dim{dim}_{label}{filename_suffix}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(fig)


def plot_prior_relative_distance_and_angle(
    df: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
    axis_limits: Optional[Dict[Tuple[int, str, str], Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    color_limits: Optional[Dict[Tuple[int, str, str], Tuple[float, float]]] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scatter_plot(timestep: int, x_col: str, y_col: str, filename: str, color_col: Optional[str] = None):
        t_df = rows_after_observed_reward_used_downstream(df, timestep=timestep)
        required = [x_col, y_col] + ([color_col] if color_col else [])
        if len(t_df) == 0 or not all(col in t_df.columns for col in required):
            return
        numeric = t_df[required].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric).all(axis=1)
        piece = numeric[finite]
        if len(piece) == 0:
            return
        fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
        if color_col:
            vmin = vmax = None
            if color_limits and (timestep, x_col, y_col) in color_limits:
                vmin, vmax = color_limits[(timestep, x_col, y_col)]
            sc = ax.scatter(piece[x_col], piece[y_col], c=piece[color_col], cmap="coolwarm", vmin=vmin, vmax=vmax, s=9, alpha=0.45, linewidths=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_col)
        else:
            ax.scatter(piece[x_col], piece[y_col], s=9, alpha=0.45, linewidths=0)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        if axis_limits and (timestep, x_col, y_col) in axis_limits:
            ax.set_xlim(*axis_limits[(timestep, x_col, y_col)][0])
            ax.set_ylim(*axis_limits[(timestep, x_col, y_col)][1])
        fig.tight_layout()
        fig.savefig(figdir / filename.replace(".png", f"{filename_suffix}.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    scatter_plot(1, "reward_t1", "prior_relative_fisher_distance_total_t1", "prior_relative_distance_vs_reward_t1.png")
    scatter_plot(2, "reward_t2", "prior_relative_fisher_distance_total_t2", "prior_relative_distance_vs_reward_t2.png")
    scatter_plot(2, "reward_t2_minus_reward_t1", "prior_relative_fisher_distance_total_t2", "prior_relative_distance_vs_reward_difference_t2.png")
    scatter_plot(2, "current_best_path_margin_t2", "prior_relative_fisher_distance_total_t2", "prior_relative_distance_vs_choice_margin_t2.png")
    for dim in (0, 1):
        for trig in ("sin", "cos"):
            scatter_plot(
                1,
                "reward_t1",
                f"delta_halfplane_{trig}_angle_{dim}_t1",
                f"prior_relative_{trig}_angle_dim{dim}_vs_reward_t1.png",
            )
            scatter_plot(
                2,
                "reward_t2",
                f"delta_halfplane_{trig}_angle_{dim}_t2",
                f"prior_relative_{trig}_angle_dim{dim}_vs_reward_t2.png",
                color_col="reward_t1",
            )


def plot_prior_relative_vs_kl_comparison(
    comparison: pd.DataFrame,
    figdir: Path,
    filename_suffix: str = "",
):
    if comparison is None or len(comparison) == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = [
        "kl_metric",
        "prior_relative_fisher_distance_total",
        "prior_relative_halfplane_angle_dim0",
        "prior_relative_halfplane_angle_dim1",
        "prior_relative_halfplane_radius_dim0",
        "prior_relative_halfplane_radius_dim1",
    ]
    group = comparison.groupby(["target_variable", "timestep"], dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    if len(group) == 0:
        return
    labels = [f"{row.target_variable}\nt{row.timestep}" for row in group.itertuples()]
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / len(metric_cols)
    fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
    for i, col in enumerate(metric_cols):
        ax.bar(
            x + (i - (len(metric_cols) - 1) / 2) * width,
            pd.to_numeric(group[col], errors="coerce"),
            width=width,
            label=col,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("primary metric")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(figdir / f"prior_relative_vs_kl_comparison{filename_suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_prior_relative_geometry_summary(
    outdir: Path,
    decoding: pd.DataFrame,
    encoding: pd.DataFrame,
    comparison: pd.DataFrame,
):
    valid_dec = prior_relative_status_filter(decoding)
    valid_enc = prior_relative_status_filter(encoding)
    if len(valid_dec):
        valid_dec["metric_primary"] = pd.to_numeric(valid_dec["metric_primary"], errors="coerce")
    if len(valid_enc):
        valid_enc["variance_weighted_R2"] = pd.to_numeric(valid_enc["variance_weighted_R2"], errors="coerce")

    def best_rows(df: pd.DataFrame, group_col: str, value_col: str, n: int = 12) -> List[Dict]:
        if len(df) == 0 or value_col not in df.columns:
            return []
        grouped = (
            df.groupby(group_col, dropna=False)[value_col]
            .mean()
            .sort_values(ascending=False)
            .head(n)
        )
        return [
            {group_col: str(index), value_col: float(value)}
            for index, value in grouped.items()
            if np.isfinite(value)
        ]

    comp = comparison.copy() if comparison is not None else pd.DataFrame()
    if len(comp):
        comp["delta_best_prior_relative_minus_kl"] = pd.to_numeric(comp["delta_best_prior_relative_minus_kl"], errors="coerce")
    summary = {
        "analysis": "learned_prior_relative_gaussian_fisher_geometry",
        "prior_relative_decoding_best_predictors": best_rows(valid_dec, "predictor_family", "metric_primary"),
        "prior_relative_encoding_best_targets": best_rows(valid_enc, "target_geometry", "variance_weighted_R2"),
        "mean_delta_best_prior_relative_minus_kl": (
            float(comp["delta_best_prior_relative_minus_kl"].mean())
            if len(comp) and np.isfinite(comp["delta_best_prior_relative_minus_kl"]).any()
            else None
        ),
        "best_prior_relative_predictors_vs_kl": (
            comp.groupby("best_prior_relative_predictor", dropna=False)["delta_best_prior_relative_minus_kl"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .to_dict()
            if len(comp) and "best_prior_relative_predictor" in comp.columns
            else {}
        ),
        "interpretation_questions": {
            "tracks_reward_difference_switch_or_margin": "Inspect prior_relative_distance_vs_kl_results.csv and prior_relative_geometry_to_task_decoding_t*.png.",
            "direction_beyond_kl": "Positive delta_best_prior_relative_minus_kl indicates prior-relative geometry beats KL for that target/timestep.",
            "stronger_latent_dimension": "Compare dim0 versus dim1 Fisher/angle/radius rows in decoding and encoding outputs.",
            "seed_consistency": "Group prior_relative_distance_vs_kl_results.csv by seed and predictor family.",
            "canonical_vs_prior_relative": "Compare prior-relative heatmaps to geometry_to_task_decoding_heatmap_t*.png and canonical half-plane/Poincare outputs.",
        },
    }
    summary = json_safe_value(summary)
    with open(outdir / "prior_relative_geometry_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "prior_relative_geometry_summary.txt", "w") as handle:
        handle.write("Learned-prior-relative Gaussian/Fisher geometry summary\n")
        handle.write("=====================================================\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")


def plot_prior_relative_geometry_outputs(
    data: pd.DataFrame,
    decoding: pd.DataFrame,
    encoding: pd.DataFrame,
    comparison: pd.DataFrame,
    figdir: Path,
):
    figdir.mkdir(parents=True, exist_ok=True)
    scatter_specs = [
        (1, "reward_t1", "prior_relative_fisher_distance_total_t1", None),
        (2, "reward_t2", "prior_relative_fisher_distance_total_t2", None),
        (2, "reward_t2_minus_reward_t1", "prior_relative_fisher_distance_total_t2", None),
        (2, "current_best_path_margin_t2", "prior_relative_fisher_distance_total_t2", None),
    ]
    for dim in (0, 1):
        for trig in ("sin", "cos"):
            scatter_specs.append((1, "reward_t1", f"delta_halfplane_{trig}_angle_{dim}_t1", None))
            scatter_specs.append((2, "reward_t2", f"delta_halfplane_{trig}_angle_{dim}_t2", "reward_t1"))
    arrow_axis_limits = prior_posterior_halfplane_axis_limits(data)
    arrow_color_limits = prior_posterior_halfplane_color_limits(data)
    scatter_axis_limits = prior_relative_scatter_axis_limits(
        data,
        [(timestep, x_col, y_col) for timestep, x_col, y_col, _ in scatter_specs],
    )
    scatter_color_limits = prior_relative_scatter_color_limits(data, scatter_specs)
    prior_dec = prior_relative_status_filter(decoding)
    prior_enc = prior_relative_status_filter(encoding)
    if len(prior_dec):
        prior_dec = prior_dec[
            prior_dec["predictor_family"].astype(str).map(is_prior_relative_family)
            | prior_dec["predictor_family"].astype(str).str.startswith("posterior_prior_kl")
        ].copy()
        prior_dec["metric_primary"] = pd.to_numeric(prior_dec["metric_primary"], errors="coerce")
        plot_prior_relative_heatmap(
            prior_dec[prior_dec["timestep"] == 1],
            index="predictor_family",
            columns="target_variable",
            values="metric_primary",
            path=figdir / "prior_relative_geometry_to_task_decoding_t1.png",
            title="Prior-relative geometry to task t1",
        )
        plot_prior_relative_heatmap(
            prior_dec[prior_dec["timestep"] == 2],
            index="predictor_family",
            columns="target_variable",
            values="metric_primary",
            path=figdir / "prior_relative_geometry_to_task_decoding_t2.png",
            title="Prior-relative geometry to task t2",
        )
    if len(prior_enc):
        prior_enc = prior_enc[
            prior_enc["target_geometry"].astype(str).map(is_prior_relative_family)
            | prior_enc["target_geometry"].astype(str).str.startswith("posterior_prior_kl")
        ].copy()
        prior_enc["variance_weighted_R2"] = pd.to_numeric(prior_enc["variance_weighted_R2"], errors="coerce")
        plot_prior_relative_heatmap(
            prior_enc[prior_enc["timestep"] == 1],
            index="predictor_family",
            columns="target_geometry",
            values="variance_weighted_R2",
            path=figdir / "prior_relative_task_to_geometry_encoding_t1.png",
            title="Task to prior-relative geometry t1",
        )
        plot_prior_relative_heatmap(
            prior_enc[prior_enc["timestep"] == 2],
            index="predictor_family",
            columns="target_geometry",
            values="variance_weighted_R2",
            path=figdir / "prior_relative_task_to_geometry_encoding_t2.png",
            title="Task to prior-relative geometry t2",
        )
    plot_prior_posterior_halfplane_arrows(
        data,
        figdir,
        axis_limits=arrow_axis_limits,
        color_limits=arrow_color_limits,
    )
    plot_prior_relative_distance_and_angle(
        data,
        figdir,
        axis_limits=scatter_axis_limits,
        color_limits=scatter_color_limits,
    )
    plot_prior_relative_vs_kl_comparison(comparison, figdir)

    if "seed" in data.columns:
        for seed_value in sorted(pd.Series(data["seed"]).dropna().unique()):
            seed_suffix = f"_seed_{file_token(seed_value)}"
            seed_data = data[data["seed"] == seed_value].copy()
            seed_dec = prior_dec[prior_dec["seed"] == seed_value].copy() if len(prior_dec) and "seed" in prior_dec.columns else pd.DataFrame()
            seed_enc = prior_enc[prior_enc["seed"] == seed_value].copy() if len(prior_enc) and "seed" in prior_enc.columns else pd.DataFrame()
            seed_comp = comparison[comparison["seed"] == seed_value].copy() if comparison is not None and len(comparison) and "seed" in comparison.columns else pd.DataFrame()
            if len(seed_dec):
                plot_prior_relative_heatmap(
                    seed_dec[seed_dec["timestep"] == 1],
                    index="predictor_family",
                    columns="target_variable",
                    values="metric_primary",
                    path=figdir / f"prior_relative_geometry_to_task_decoding_t1{seed_suffix}.png",
                    title=f"Prior-relative geometry to task t1 seed {seed_value}",
                )
                plot_prior_relative_heatmap(
                    seed_dec[seed_dec["timestep"] == 2],
                    index="predictor_family",
                    columns="target_variable",
                    values="metric_primary",
                    path=figdir / f"prior_relative_geometry_to_task_decoding_t2{seed_suffix}.png",
                    title=f"Prior-relative geometry to task t2 seed {seed_value}",
                )
            if len(seed_enc):
                plot_prior_relative_heatmap(
                    seed_enc[seed_enc["timestep"] == 1],
                    index="predictor_family",
                    columns="target_geometry",
                    values="variance_weighted_R2",
                    path=figdir / f"prior_relative_task_to_geometry_encoding_t1{seed_suffix}.png",
                    title=f"Task to prior-relative geometry t1 seed {seed_value}",
                )
                plot_prior_relative_heatmap(
                    seed_enc[seed_enc["timestep"] == 2],
                    index="predictor_family",
                    columns="target_geometry",
                    values="variance_weighted_R2",
                    path=figdir / f"prior_relative_task_to_geometry_encoding_t2{seed_suffix}.png",
                    title=f"Task to prior-relative geometry t2 seed {seed_value}",
                )
            plot_prior_posterior_halfplane_arrows(
                seed_data,
                figdir,
                filename_suffix=seed_suffix,
                axis_limits=arrow_axis_limits,
                color_limits=arrow_color_limits,
            )
            plot_prior_relative_distance_and_angle(
                seed_data,
                figdir,
                filename_suffix=seed_suffix,
                axis_limits=scatter_axis_limits,
                color_limits=scatter_color_limits,
            )
            plot_prior_relative_vs_kl_comparison(
                seed_comp,
                figdir,
                filename_suffix=seed_suffix,
            )


def run_geometry_meaning_analyses(
    df: pd.DataFrame,
    outdir: Path,
    cv_folds: int,
    min_within_path_n: int,
    make_plots: bool = True,
):
    df = add_geometry_meaning_columns(df)
    group_cols = reward_encoding_group_columns(df)
    if not group_cols:
        group_iter = [((), df)]
    else:
        group_iter = df.groupby(group_cols, dropna=False)
    decoding_rows = []
    encoding_rows = []
    within_rows = []
    decoding_specs = make_geometry_to_task_specs()
    encoding_specs = make_task_to_geometry_specs()
    for values, group_df in group_iter:
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(group_cols, values))
        for spec in decoding_specs:
            decoding_rows.append(fit_geometry_to_task_spec(group_name, group_df, spec, cv_folds))
        for spec in encoding_specs:
            encoding_rows.append(fit_task_to_geometry_spec(group_name, group_df, spec, cv_folds))
        within_rows.extend(fit_within_path_geometry_decoding(group_name, group_df, cv_folds, min_within_path_n))
    decoding = pd.DataFrame(decoding_rows)
    encoding = pd.DataFrame(encoding_rows)
    within_path = pd.DataFrame(within_rows)
    summary, feature_summary = summarize_geometry_meaning(decoding)
    prior_relative_decoding = decoding[
        decoding["predictor_family"].astype(str).map(is_prior_relative_family)
        | decoding["predictor_family"].astype(str).str.startswith("posterior_prior_kl")
    ].copy() if len(decoding) else pd.DataFrame()
    prior_relative_encoding = encoding[
        encoding["target_geometry"].astype(str).map(is_prior_relative_family)
        | encoding["target_geometry"].astype(str).str.startswith("posterior_prior_kl")
    ].copy() if len(encoding) else pd.DataFrame()
    prior_relative_comparison = build_prior_relative_distance_vs_kl_results(decoding)
    decoding.to_csv(outdir / "geometry_to_task_decoding_results.csv", index=False)
    encoding.to_csv(outdir / "task_to_geometry_encoding_results.csv", index=False)
    prior_relative_decoding.to_csv(outdir / "prior_relative_geometry_to_task_decoding_results.csv", index=False)
    prior_relative_encoding.to_csv(outdir / "prior_relative_task_to_geometry_encoding_results.csv", index=False)
    prior_relative_comparison.to_csv(outdir / "prior_relative_distance_vs_kl_results.csv", index=False)
    within_path.to_csv(outdir / "within_path_geometry_decoding_results.csv", index=False)
    summary.to_csv(outdir / "geometry_meaning_summary.csv", index=False)
    feature_summary.to_csv(outdir / "geometry_feature_family_summary.csv", index=False)
    if make_plots:
        plot_geometry_meaning_outputs(decoding, encoding, within_path, summary, outdir / "figures")
        plot_prior_relative_geometry_outputs(df, decoding, encoding, prior_relative_comparison, outdir / "figures")
    write_prior_relative_geometry_summary(outdir, prior_relative_decoding, prior_relative_encoding, prior_relative_comparison)
    return decoding, encoding, within_path


HALFPLANE_REWARD_FEATURES = [
    ("radius", ["{space}_radius_{dim}_t{t}"]),
    ("angle_raw", ["{space}_angle_{dim}_t{t}"]),
    ("angle_sin", ["{space}_sin_angle_{dim}_t{t}"]),
    ("angle_cos", ["{space}_cos_angle_{dim}_t{t}"]),
    ("angle_sin_cos", ["{space}_sin_angle_{dim}_t{t}", "{space}_cos_angle_{dim}_t{t}"]),
]


def halfplane_reward_feature_columns(space: str, dim: int, timestep: int, feature: str) -> List[str]:
    for feature_name, templates in HALFPLANE_REWARD_FEATURES:
        if feature_name == feature:
            return [
                template.format(space=space, dim=dim, t=timestep)
                for template in templates
            ]
    return []


def base_halfplane_reward_row(
    group_name: Dict,
    *,
    analysis_name: str,
    latent_dimension: int,
    timestep: int,
    reward_variable: str,
    geometry_space: str,
    geometry_feature: str,
    predictor_columns: Sequence[str],
    status: str,
    error_message: str,
    n_trials: int = 0,
    reward_t1_value=np.nan,
) -> Dict:
    row = {
        **group_name,
        "analysis_name": analysis_name,
        "latent_dimension": int(latent_dimension),
        "timestep": int(timestep),
        "reward_variable": reward_variable,
        "geometry_space": geometry_space,
        "geometry_feature": geometry_feature,
        "predictor_columns": ",".join(predictor_columns),
        "n_trials": int(n_trials),
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "R2": np.nan,
        "slope_reward_t2_to_feature": np.nan,
        "slope_feature_to_reward_t2": np.nan,
        "status": status,
        "error_message": error_message,
    }
    if timestep == 2:
        row["reward_t1_value"] = reward_t1_value
    return row


def fit_halfplane_reward_relation(
    group_name: Dict,
    work: pd.DataFrame,
    *,
    analysis_name: str,
    latent_dimension: int,
    timestep: int,
    reward_variable: str,
    geometry_space: str,
    geometry_feature: str,
    predictor_columns: Sequence[str],
    reward_t1_value=np.nan,
    status: str = "ok",
    error_message: str = "",
) -> Dict:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    required = list(predictor_columns) + [reward_variable]
    missing = [col for col in required if col not in work.columns]
    if missing:
        return base_halfplane_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_missing_columns",
            error_message=f"missing columns: {','.join(missing)}",
            reward_t1_value=reward_t1_value,
        )
    numeric = work[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric).all(axis=1)
    numeric = numeric[valid]
    if len(numeric) < 3:
        return base_halfplane_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_too_few_rows",
            error_message="fewer than three valid rows",
            n_trials=len(numeric),
            reward_t1_value=reward_t1_value,
        )
    x = numeric[list(predictor_columns)].to_numpy(dtype=float)
    y = numeric[reward_variable].to_numpy(dtype=float)
    if np.nanstd(y) <= 1e-12:
        return base_halfplane_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_no_reward_variance",
            error_message="reward variable has no variance",
            n_trials=len(numeric),
            reward_t1_value=reward_t1_value,
        )
    if np.any(np.nanstd(x, axis=0) <= 1e-12):
        return base_halfplane_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_no_predictor_variance",
            error_message="one or more predictors have no variance",
            n_trials=len(numeric),
            reward_t1_value=reward_t1_value,
        )
    pred = LinearRegression().fit(x, y).predict(x)
    row = base_halfplane_reward_row(
        group_name,
        analysis_name=analysis_name,
        latent_dimension=latent_dimension,
        timestep=timestep,
        reward_variable=reward_variable,
        geometry_space=geometry_space,
        geometry_feature=geometry_feature,
        predictor_columns=predictor_columns,
        status=status,
        error_message=error_message,
        n_trials=len(numeric),
        reward_t1_value=reward_t1_value,
    )
    row["R2"] = float(r2_score(y, pred))
    if len(predictor_columns) == 1:
        xv = x[:, 0]
        row["pearson_r"] = safe_regression_corr(y, xv, "pearson")
        row["spearman_r"] = safe_regression_corr(y, xv, "spearman")
        row["slope_reward_t2_to_feature"] = float(LinearRegression().fit(y[:, None], xv).coef_[0])
        row["slope_feature_to_reward_t2"] = float(LinearRegression().fit(xv[:, None], y).coef_[0])
    return row


def halfplane_t2_slope_rows(t2_rows: pd.DataFrame, data: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    rows = []
    if t2_rows.empty or "status" not in t2_rows.columns:
        return pd.DataFrame()
    usable = t2_rows[
        t2_rows["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])
        & (t2_rows["geometry_feature"] != "angle_sin_cos")
    ].copy()
    if len(usable) == 0:
        return pd.DataFrame()
    slope_records = []
    for _, row in usable.iterrows():
        group_filter = np.ones(len(data), dtype=bool)
        for col in group_cols:
            if col in data.columns and col in row.index:
                group_filter &= data[col].astype(str).to_numpy() == str(row[col])
        work = rows_after_observed_reward_used_downstream(data[group_filter], timestep=2)
        work = work[pd.to_numeric(work["reward_t1"], errors="coerce") == pd.to_numeric(pd.Series([row["reward_t1_value"]]), errors="coerce").iloc[0]]
        predictor_cols = str(row["predictor_columns"]).split(",")
        if len(predictor_cols) != 1 or predictor_cols[0] not in work.columns:
            continue
        numeric = work[["reward_t2", predictor_cols[0]]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(numeric) < 3 or numeric["reward_t2"].std() <= 1e-12 or numeric[predictor_cols[0]].std() <= 1e-12:
            continue
        reward = numeric["reward_t2"].to_numpy(dtype=float)[:, None]
        feature = numeric[predictor_cols[0]].to_numpy(dtype=float)[:, None]
        slope_reward_to_feature = float(LinearRegression().fit(reward, feature[:, 0]).coef_[0])
        slope_feature_to_reward = float(LinearRegression().fit(feature, reward[:, 0]).coef_[0])
        record = row.to_dict()
        record.update({
            "slope_reward_t2_to_feature": slope_reward_to_feature,
            "slope_feature_to_reward_t2": slope_feature_to_reward,
        })
        slope_records.append(record)
    slopes = pd.DataFrame(slope_records)
    if len(slopes) == 0:
        return slopes
    summary_group_cols = [
        col for col in list(group_cols) + ["latent_dimension", "geometry_space", "geometry_feature"]
        if col in slopes.columns
    ]
    for values, group in slopes.groupby(summary_group_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(summary_group_cols, values))
        slope_values = pd.to_numeric(group["slope_feature_to_reward_t2"], errors="coerce")
        finite = slope_values[np.isfinite(slope_values)]
        signs = sorted(set(np.sign(finite[finite != 0]).astype(int).tolist()))
        rows.append({
            **group_name,
            "n_reward_t1_groups": int(group["reward_t1_value"].nunique()),
            "slope_min": float(finite.min()) if len(finite) else np.nan,
            "slope_max": float(finite.max()) if len(finite) else np.nan,
            "slope_signs": ",".join(str(sign) for sign in signs),
            "slope_sign_flip": bool((-1 in signs) and (1 in signs)),
            "mean_abs_slope": float(np.nanmean(np.abs(finite))) if len(finite) else np.nan,
            "mean_R2": float(pd.to_numeric(group["R2"], errors="coerce").mean()),
            "status": "ok" if len(finite) else "skipped_too_few_rows",
            "error_message": "" if len(finite) else "no finite slopes",
        })
    return pd.DataFrame(rows)


def plot_halfplane_reward_outputs(
    data: pd.DataFrame,
    t2_rows: pd.DataFrame,
    slope_summary: pd.DataFrame,
    figdir: Path,
    file_prefix: str = "",
    per_seed: bool = True,
    focused_only: bool = True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    valid_t1 = rows_after_observed_reward_used_downstream(data, timestep=1)
    valid_t2 = rows_after_observed_reward_used_downstream(data, timestep=2)
    plot_current = not (focused_only and per_seed and file_prefix == "")

    def scatter_feature_panels(df: pd.DataFrame, dim: int, space: str, reward_col: str, path: str, by_r1: bool = False):
        features = [
            (f"{space}_radius_{dim}_t{1 if reward_col == 'reward_t1' else 2}", "radius"),
            (f"{space}_sin_angle_{dim}_t{1 if reward_col == 'reward_t1' else 2}", "sin(angle)"),
            (f"{space}_cos_angle_{dim}_t{1 if reward_col == 'reward_t1' else 2}", "cos(angle)"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=panel_figsize(3, 1, title=False), squeeze=False)
        colors = None
        if by_r1 and "reward_t1" in df.columns:
            colors = pd.to_numeric(df["reward_t1"], errors="coerce")
        for ax, (col, label) in zip(axes[0], features):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[reward_col], errors="coerce")
            finite = np.isfinite(x) & np.isfinite(y)
            if colors is not None:
                c = colors[finite]
                sc = ax.scatter(x[finite], y[finite], c=c, cmap="viridis", s=8, alpha=0.45, linewidths=0)
            else:
                sc = ax.scatter(x[finite], y[finite], s=8, alpha=0.4, linewidths=0)
            ax.set_xlabel(label)
            ax.set_ylabel(reward_col)
            if colors is not None:
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="reward_t1")
        fig.tight_layout()
        fig.savefig(figdir / f"{file_prefix}{path}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def geometry_scatter(df: pd.DataFrame, dim: int, space: str, reward_col: str, path: str, by_r1: bool = False):
        t = 1 if reward_col == "reward_t1" else 2
        x_col = f"{space}_x_{dim}_t{t}" if space == "halfplane" else f"{space}_x_{dim}_t{t}"
        y_col = f"{space}_y_{dim}_t{t}" if space == "halfplane" else f"{space}_y_{dim}_t{t}"
        if x_col not in df.columns or y_col not in df.columns:
            return
        groups = [(None, df)]
        if by_r1 and "reward_t1" in df.columns:
            groups = list(df.groupby("reward_t1", dropna=True))
        n = len(groups)
        ncols = min(4, max(1, n))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=panel_figsize(ncols, nrows, title=False), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        for ax, (group_value, piece) in zip(axes.ravel(), groups):
            x = pd.to_numeric(piece[x_col], errors="coerce")
            y = pd.to_numeric(piece[y_col], errors="coerce")
            c = pd.to_numeric(piece[reward_col], errors="coerce")
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
            sc = ax.scatter(x[finite], y[finite], c=c[finite], cmap="viridis", s=8, alpha=0.5, linewidths=0)
            if space == "poincare_disk":
                circle = plt.Circle((0, 0), 1.0, fill=False, color="0.35", linewidth=0.8)
                ax.add_patch(circle)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if group_value is not None:
                ax.set_title(f"R1={group_value:g}", fontsize=PANEL_FONT_SIZE)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=reward_col)
        fig.tight_layout()
        fig.savefig(figdir / f"{file_prefix}{path}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if focused_only and plot_current:
        for space in ("halfplane", "poincare_disk"):
            scatter_feature_panels(
                valid_t1,
                0,
                space,
                "reward_t1",
                f"{space}_t1_reward_vs_radius_angle_dim0.png",
            )
            scatter_feature_panels(
                valid_t2,
                1,
                space,
                "reward_t2",
                f"{space}_t2_reward2_vs_geometry_by_R1_dim1.png",
                by_r1=True,
            )
    elif not focused_only:
        for dim in (0, 1):
            for space in ("halfplane", "poincare_disk"):
                scatter_feature_panels(valid_t1, dim, space, "reward_t1", f"{space}_t1_reward_vs_radius_angle_dim{dim}.png")
                scatter_feature_panels(valid_t2, dim, space, "reward_t2", f"{space}_t2_reward2_vs_geometry_by_R1_dim{dim}.png", by_r1=True)
                geometry_scatter(valid_t1, dim, space, "reward_t1", f"{space}_t1_scatter_reward_dim{dim}.png")
                geometry_scatter(valid_t2, dim, space, "reward_t2", f"{space}_t2_scatter_reward2_by_R1_dim{dim}.png", by_r1=True)

    if len(slope_summary) and plot_current and "status" in t2_rows.columns:
        detail = t2_rows[
            t2_rows["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])
            & t2_rows["geometry_feature"].isin(["radius", "angle_sin", "angle_cos"])
        ].copy()
        slope_spaces = ("halfplane", "poincare_disk")
        for space in slope_spaces:
            piece = detail[detail["geometry_space"] == space]
            if len(piece) == 0:
                continue
            fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
            for (dim, feature), group in piece.groupby(["latent_dimension", "geometry_feature"], dropna=False):
                group = group.sort_values("reward_t1_value")
                ax.plot(
                    pd.to_numeric(group["reward_t1_value"], errors="coerce"),
                    pd.to_numeric(group["pearson_r"], errors="coerce"),
                    marker="o",
                    linewidth=1.0,
                    label=f"dim{dim} {feature}",
                )
            ax.axhline(0, color="0.65", linewidth=0.8)
            ax.set_xlabel("reward_t1")
            ax.set_ylabel("Pearson r(reward_t2, feature)")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout(rect=[0, 0, 0.78, 1])
            fig.savefig(figdir / f"{file_prefix}{space}_t2_reward2_slope_by_R1.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    if per_seed and "seed" in data.columns:
        for seed_value in sorted(pd.Series(data["seed"]).dropna().unique()):
            seed_prefix = f"seed_{file_token(seed_value)}_"
            seed_data = data[data["seed"] == seed_value].copy()
            seed_t2_rows = t2_rows[t2_rows["seed"] == seed_value].copy() if "seed" in t2_rows.columns else t2_rows
            seed_slope = slope_summary[slope_summary["seed"] == seed_value].copy() if "seed" in slope_summary.columns else slope_summary
            plot_halfplane_reward_outputs(
                seed_data,
                seed_t2_rows,
                seed_slope,
                figdir,
                file_prefix=seed_prefix,
                per_seed=False,
                focused_only=focused_only,
            )


def write_halfplane_reward_summary(
    outdir: Path,
    t1_rows: pd.DataFrame,
    t2_rows: pd.DataFrame,
    slope_summary: pd.DataFrame,
    file_prefix: str = "",
):
    valid_t1 = t1_rows[t1_rows["status"] == "ok"].copy() if len(t1_rows) and "status" in t1_rows.columns else pd.DataFrame()
    valid_t2 = (
        t2_rows[t2_rows["status"].isin(["ok", "warning_qz2_downstream_usage_unknown"])].copy()
        if len(t2_rows) and "status" in t2_rows.columns
        else pd.DataFrame()
    )
    summary = {
        "analysis": "halfplane_reward_geometry",
        "t1_best_feature": None,
        "t1_best_latent_dimension": None,
        "t1_best_geometry_space": None,
        "t2_best_feature": None,
        "t2_best_latent_dimension": None,
        "t2_slope_sign_flip_rows": int(slope_summary["slope_sign_flip"].sum()) if len(slope_summary) and "slope_sign_flip" in slope_summary else 0,
        "t2_slope_sign_flip_summary": slope_summary[slope_summary.get("slope_sign_flip", False) == True].to_dict(orient="records") if len(slope_summary) and "slope_sign_flip" in slope_summary else [],
    }
    if len(valid_t1):
        row = valid_t1.sort_values("R2", ascending=False).iloc[0]
        summary.update({
            "t1_best_feature": row.get("geometry_feature"),
            "t1_best_latent_dimension": row.get("latent_dimension"),
            "t1_best_geometry_space": row.get("geometry_space"),
            "t1_best_R2": row.get("R2"),
            "t1_best_predictor_columns": row.get("predictor_columns"),
        })
    if len(valid_t2):
        row = valid_t2.sort_values("R2", ascending=False).iloc[0]
        summary.update({
            "t2_best_feature": row.get("geometry_feature"),
            "t2_best_latent_dimension": row.get("latent_dimension"),
            "t2_best_geometry_space": row.get("geometry_space"),
            "t2_best_R2": row.get("R2"),
            "t2_best_reward_t1_value": row.get("reward_t1_value"),
            "t2_best_predictor_columns": row.get("predictor_columns"),
        })
    summary = json_safe_value(summary)
    with open(outdir / f"{file_prefix}halfplane_reward_geometry_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / f"{file_prefix}halfplane_reward_geometry_summary.txt", "w") as handle:
        handle.write("Half-plane reward geometry summary\n")
        handle.write("==================================\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")


GEOMETRY_COORDINATE_SPACES = [
    {
        "space": "euclidean",
        "prefix": "euclidean",
        "x_label": "posterior mu",
        "y_label": "posterior sigma",
        "disk": False,
    },
    {
        "space": "poincare_plane",
        "prefix": "halfplane",
        "x_label": "mu / sqrt(2)",
        "y_label": "sigma",
        "disk": False,
    },
    {
        "space": "poincare_disk",
        "prefix": "poincare_disk",
        "x_label": "disk x",
        "y_label": "disk y",
        "disk": True,
    },
    {
        "space": "prior_centered_poincare_plane",
        "prefix": "prior_centered_halfplane",
        "x_label": "posterior-prior x",
        "y_label": "posterior-prior y",
        "disk": False,
    },
    {
        "space": "prior_centered_poincare_disk",
        "prefix": "prior_centered_disk",
        "x_label": "prior-centered disk x",
        "y_label": "prior-centered disk y",
        "disk": True,
    },
]

GEOMETRY_COORDINATE_FEATURES = ["radius", "sin_angle", "cos_angle"]
GEOMETRY_REWARD_FEATURE_SETS = [
    ("radius", ["radius"]),
    ("sin_angle", ["sin_angle"]),
    ("cos_angle", ["cos_angle"]),
    ("angle_sin_cos", ["sin_angle", "cos_angle"]),
    ("radius_plus_angle", ["radius", "sin_angle", "cos_angle"]),
]


def geometry_coordinate_space_lookup() -> Dict[str, Dict]:
    return {spec["space"]: spec for spec in GEOMETRY_COORDINATE_SPACES}


def build_geometry_coordinate_long(df: pd.DataFrame) -> pd.DataFrame:
    df = add_geometry_meaning_columns(df)
    meta_cols = [
        col for col in (
            MODEL_GROUP_COLUMNS
            + [
                "trial_id",
                "trial_uid",
                "timestep",
                "reward_t1",
                "reward_t2",
                "current_best_path",
                "current_best_path_value",
                "current_best_path_margin",
                "current_best_path_switch",
                "current_best_path_t1",
                "current_best_path_t2",
                "current_best_path_value_t1",
                "current_best_path_value_t2",
                "current_best_path_margin_t1",
                "current_best_path_margin_t2",
                "current_best_path_switch_t2",
                "observed_at_timestep",
                "stopped_at_timestep",
                "qz_used_downstream",
            ]
        )
        if col in df.columns
    ]
    dims = [dim for dim in available_latent_dims(df) if dim in (0, 1)]
    pieces = []
    for dim in dims:
        for space_spec in GEOMETRY_COORDINATE_SPACES:
            prefix = space_spec["prefix"]
            required = [f"{prefix}_x_{dim}", f"{prefix}_y_{dim}"]
            if not all(col in df.columns for col in required):
                continue
            piece = df[meta_cols].copy()
            piece["space"] = space_spec["space"]
            piece["latent_dimension"] = dim
            piece["x"] = pd.to_numeric(df[f"{prefix}_x_{dim}"], errors="coerce")
            piece["y"] = pd.to_numeric(df[f"{prefix}_y_{dim}"], errors="coerce")
            for feature in GEOMETRY_COORDINATE_FEATURES + ["angle"]:
                col = f"{prefix}_{feature}_{dim}"
                piece[feature] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
            distance_col = f"{prefix}_hyperbolic_distance_{dim}"
            piece["hyperbolic_distance"] = (
                pd.to_numeric(df[distance_col], errors="coerce")
                if distance_col in df.columns
                else np.nan
            )
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_geometry_coordinate_features_wide(coord: pd.DataFrame) -> pd.DataFrame:
    if coord is None or len(coord) == 0:
        return pd.DataFrame()
    index_cols = [
        col for col in MODEL_GROUP_COLUMNS + ["trial_id", "trial_uid"]
        if col in coord.columns
    ]
    if not index_cols:
        index_cols = ["trial_uid"] if "trial_uid" in coord.columns else []
    value_cols = ["x", "y", "radius", "sin_angle", "cos_angle"]
    value_cols = [col for col in value_cols if col in coord.columns]
    if not index_cols or not value_cols:
        return pd.DataFrame()
    wide = coord.pivot_table(
        index=index_cols,
        columns=["space", "latent_dimension", "timestep"],
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}_{space}_dim{int(dim)}_t{int(timestep)}"
        for metric, space, dim, timestep in wide.columns
    ]
    return wide.reset_index()


def geometry_rows_used_downstream(coord: pd.DataFrame, timestep: int) -> pd.DataFrame:
    if coord is None or len(coord) == 0:
        return pd.DataFrame()
    strict_mask = strict_qz_used_downstream_mask(coord)
    if strict_mask is not None:
        mask = strict_mask
    elif "qz_used_downstream" in coord.columns:
        mask = bool_series(coord["qz_used_downstream"])
    else:
        mask = pd.Series(True, index=coord.index)
        if "observed_at_timestep" in coord.columns:
            mask &= bool_series(coord["observed_at_timestep"])
        if "stopped_at_timestep" in coord.columns:
            mask &= ~bool_series(coord["stopped_at_timestep"])
    mask &= pd.to_numeric(coord["timestep"], errors="coerce") == timestep
    return coord[mask].copy()


def geometry_coordinate_axis_limits(coord: pd.DataFrame) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    limits = {}
    space_lookup = geometry_coordinate_space_lookup()
    plot_coord = pd.concat(
        [
            geometry_rows_used_downstream(coord, timestep=1),
            geometry_rows_used_downstream(coord, timestep=2),
        ],
        ignore_index=True,
    )
    if len(plot_coord) == 0 or "space" not in plot_coord.columns:
        return limits
    for space, piece in plot_coord.groupby("space", dropna=False):
        if space_lookup.get(space, {}).get("disk", False):
            limits[space] = ((-1.0, 1.0), (-1.0, 1.0))
            continue
        xlim = finite_plot_limits(piece["x"], pad=0.08)
        ylim = finite_plot_limits(piece["y"], pad=0.08)
        if xlim is not None and ylim is not None:
            limits[space] = (xlim, ylim)
    return limits


def geometry_feature_limits(coord: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    out = {}
    plot_coord = pd.concat(
        [
            geometry_rows_used_downstream(coord, timestep=1),
            geometry_rows_used_downstream(coord, timestep=2),
        ],
        ignore_index=True,
    )
    for space in sorted(plot_coord["space"].dropna().unique()) if "space" in plot_coord.columns else []:
        space_df = plot_coord[plot_coord["space"] == space]
        for feature in GEOMETRY_COORDINATE_FEATURES:
            if feature not in space_df.columns:
                continue
            limit = finite_plot_limits(space_df[feature], pad=0.10)
            if limit is not None:
                out[(space, feature)] = limit
    return out


def geometry_reward_norm(coord: pd.DataFrame, reward_col: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    values = pd.to_numeric(coord[reward_col], errors="coerce") if reward_col in coord.columns else pd.Series(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None, None
    vmin = float(finite.min())
    vmax = float(finite.max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    return Normalize(vmin=vmin, vmax=vmax), plt.get_cmap("viridis")


def draw_geometry_space_background(ax, space: str):
    if geometry_coordinate_space_lookup().get(space, {}).get("disk", False):
        import matplotlib.pyplot as plt

        circle = plt.Circle((0.0, 0.0), 1.0, fill=False, color="0.45", linewidth=0.8)
        ax.add_patch(circle)
        ax.set_aspect("equal", adjustable="box")
    if space.startswith("prior_centered"):
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)


def plot_geometry_coordinate_scatter_t1(coord: pd.DataFrame, figdir: Path):
    if coord is None or len(coord) == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    axis_limits = geometry_coordinate_axis_limits(coord)
    t1 = geometry_rows_used_downstream(coord, timestep=1)
    norm, cmap = geometry_reward_norm(t1, "reward_t1")
    if norm is None:
        return
    seed_groups = t1.groupby("seed", dropna=False) if "seed" in t1.columns else [("all", t1)]
    for seed, seed_df in seed_groups:
        for space_spec in GEOMETRY_COORDINATE_SPACES:
            space = space_spec["space"]
            piece = seed_df[seed_df["space"] == space].copy()
            dims = [dim for dim in (0, 1) if (piece["latent_dimension"] == dim).any()]
            if not dims:
                continue
            fig, axes = plt.subplots(1, len(dims), figsize=panel_figsize(len(dims), 1, colorbar=True, title=True), squeeze=False)
            scatter = None
            for col_i, dim in enumerate(dims):
                ax = axes[0, col_i]
                dim_df = piece[piece["latent_dimension"] == dim].copy()
                numeric = dim_df[["x", "y", "reward_t1"]].apply(pd.to_numeric, errors="coerce")
                dim_df = dim_df[np.isfinite(numeric).all(axis=1)].copy()
                if len(dim_df) > 8000:
                    dim_df = dim_df.sample(8000, random_state=301 + dim)
                scatter = ax.scatter(
                    dim_df["x"],
                    dim_df["y"],
                    c=pd.to_numeric(dim_df["reward_t1"], errors="coerce"),
                    cmap=cmap,
                    norm=norm,
                    s=6,
                    alpha=0.42,
                    linewidths=0,
                )
                draw_geometry_space_background(ax, space)
                if space in axis_limits:
                    ax.set_xlim(*axis_limits[space][0])
                    ax.set_ylim(*axis_limits[space][1])
                ax.set_title(f"dim {dim}", fontsize=PANEL_FONT_SIZE)
                ax.set_xlabel(space_spec["x_label"], fontsize=PANEL_FONT_SIZE)
                ax.set_ylabel(space_spec["y_label"], fontsize=PANEL_FONT_SIZE)
                ax.tick_params(labelsize=PANEL_FONT_SIZE)
            fig.suptitle(f"{space}, t1, seed {seed}", fontsize=PANEL_FONT_SIZE)
            fig.subplots_adjust(right=0.80, wspace=0.82, top=0.84)
            if scatter is not None:
                add_first_row_colorbar(fig, axes, scatter, "reward_t1", width=0.035)
            fig.savefig(
                figdir / f"geometry_scatter_t1_reward_seed_{file_token(seed)}_space_{space}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_geometry_coordinate_scatter_t2_by_r1(coord: pd.DataFrame, figdir: Path):
    if coord is None or len(coord) == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_7pt_plot_style(plt)

    figdir.mkdir(parents=True, exist_ok=True)
    axis_limits = geometry_coordinate_axis_limits(coord)
    t2 = geometry_rows_used_downstream(coord, timestep=2)
    if "reward_t1" not in t2.columns or "reward_t2" not in t2.columns:
        return
    norm, cmap = geometry_reward_norm(t2, "reward_t2")
    if norm is None:
        return
    seed_groups = t2.groupby("seed", dropna=False) if "seed" in t2.columns else [("all", t2)]
    for seed, seed_df in seed_groups:
        for space_spec in GEOMETRY_COORDINATE_SPACES:
            space = space_spec["space"]
            piece = seed_df[seed_df["space"] == space].copy()
            dims = [dim for dim in (0, 1) if (piece["latent_dimension"] == dim).any()]
            reward_t1_values = sorted(pd.to_numeric(piece["reward_t1"], errors="coerce").dropna().unique())
            if not dims or not reward_t1_values:
                continue
            fig, axes = plt.subplots(
                len(reward_t1_values),
                len(dims),
                figsize=panel_figsize(len(dims), len(reward_t1_values), colorbar=True, title=True),
                squeeze=False,
            )
            scatter = None
            for row_i, reward_t1_value in enumerate(reward_t1_values):
                row_df = piece[np.isclose(pd.to_numeric(piece["reward_t1"], errors="coerce"), reward_t1_value)].copy()
                for col_i, dim in enumerate(dims):
                    ax = axes[row_i, col_i]
                    dim_df = row_df[row_df["latent_dimension"] == dim].copy()
                    numeric = dim_df[["x", "y", "reward_t2"]].apply(pd.to_numeric, errors="coerce")
                    dim_df = dim_df[np.isfinite(numeric).all(axis=1)].copy()
                    if len(dim_df) > 5000:
                        dim_df = dim_df.sample(5000, random_state=401 + row_i * 10 + dim)
                    scatter = ax.scatter(
                        dim_df["x"],
                        dim_df["y"],
                        c=pd.to_numeric(dim_df["reward_t2"], errors="coerce"),
                        cmap=cmap,
                        norm=norm,
                        s=6,
                        alpha=0.42,
                        linewidths=0,
                    )
                    draw_geometry_space_background(ax, space)
                    if space in axis_limits:
                        ax.set_xlim(*axis_limits[space][0])
                        ax.set_ylim(*axis_limits[space][1])
                    ax.set_title(f"R1={reward_t1_value:g}, dim {dim}", fontsize=PANEL_FONT_SIZE)
                    ax.set_xlabel(space_spec["x_label"], fontsize=PANEL_FONT_SIZE)
                    ax.set_ylabel(space_spec["y_label"], fontsize=PANEL_FONT_SIZE)
                    ax.tick_params(labelsize=PANEL_FONT_SIZE)
            fig.suptitle(f"{space}, t2, seed {seed}", fontsize=PANEL_FONT_SIZE)
            fig.subplots_adjust(right=0.80, wspace=0.82, hspace=0.46, top=0.94)
            if scatter is not None:
                add_first_row_colorbar(fig, axes, scatter, "reward_t2")
            fig.savefig(
                figdir / f"geometry_scatter_t2_reward2_by_R1_seed_{file_token(seed)}_space_{space}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_geometry_feature_reward_relationships(coord: pd.DataFrame, figdir: Path):
    if coord is None or len(coord) == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_7pt_plot_style(plt)

    figdir.mkdir(parents=True, exist_ok=True)
    feature_limits = geometry_feature_limits(coord)
    t1 = geometry_rows_used_downstream(coord, timestep=1)
    t2 = geometry_rows_used_downstream(coord, timestep=2)
    t2_norm, t2_cmap = geometry_reward_norm(t2, "reward_t1")
    seed_values = (
        sorted(pd.Series(coord["seed"]).dropna().unique())
        if "seed" in coord.columns
        else ["all"]
    )
    for seed in seed_values:
        seed_t1 = t1[t1["seed"] == seed].copy() if "seed" in t1.columns else t1.copy()
        seed_t2 = t2[t2["seed"] == seed].copy() if "seed" in t2.columns else t2.copy()
        for space_spec in GEOMETRY_COORDINATE_SPACES:
            space = space_spec["space"]
            dims = [
                dim for dim in (0, 1)
                if ((seed_t1["space"] == space) & (seed_t1["latent_dimension"] == dim)).any()
                or ((seed_t2["space"] == space) & (seed_t2["latent_dimension"] == dim)).any()
            ]
            if not dims:
                continue
            fig, axes = plt.subplots(
                len(GEOMETRY_COORDINATE_FEATURES),
                len(dims),
                figsize=panel_figsize(len(dims), len(GEOMETRY_COORDINATE_FEATURES), title=True),
                squeeze=False,
            )
            for row_i, feature in enumerate(GEOMETRY_COORDINATE_FEATURES):
                for col_i, dim in enumerate(dims):
                    ax = axes[row_i, col_i]
                    piece = seed_t1[
                        (seed_t1["space"] == space)
                        & (seed_t1["latent_dimension"] == dim)
                    ].copy()
                    numeric = piece[["reward_t1", feature]].apply(pd.to_numeric, errors="coerce")
                    piece = piece[np.isfinite(numeric).all(axis=1)].copy()
                    ax.scatter(
                        pd.to_numeric(piece["reward_t1"], errors="coerce"),
                        pd.to_numeric(piece[feature], errors="coerce"),
                        s=7,
                        alpha=0.38,
                        linewidths=0,
                        color="#2b6cb0",
                    )
                    if (space, feature) in feature_limits:
                        ax.set_ylim(*feature_limits[(space, feature)])
                    ax.set_title(f"{feature}, dim {dim}", fontsize=PANEL_FONT_SIZE)
                    ax.set_xlabel("reward_t1", fontsize=PANEL_FONT_SIZE)
                    ax.set_ylabel(feature, fontsize=PANEL_FONT_SIZE)
                    ax.tick_params(labelsize=PANEL_FONT_SIZE)
            fig.suptitle(f"{space}, t1, seed {seed}", fontsize=PANEL_FONT_SIZE)
            fig.subplots_adjust(wspace=0.35, hspace=0.58, top=0.92)
            fig.savefig(
                figdir / f"geometry_features_t1_by_reward_seed_{file_token(seed)}_space_{space}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

            if t2_norm is None:
                continue
            fig, axes = plt.subplots(
                len(GEOMETRY_COORDINATE_FEATURES),
                len(dims),
                figsize=panel_figsize(len(dims), len(GEOMETRY_COORDINATE_FEATURES), colorbar=True, title=True),
                squeeze=False,
            )
            scatter = None
            for row_i, feature in enumerate(GEOMETRY_COORDINATE_FEATURES):
                for col_i, dim in enumerate(dims):
                    ax = axes[row_i, col_i]
                    piece = seed_t2[
                        (seed_t2["space"] == space)
                        & (seed_t2["latent_dimension"] == dim)
                    ].copy()
                    numeric = piece[["reward_t1", "reward_t2", feature]].apply(pd.to_numeric, errors="coerce")
                    piece = piece[np.isfinite(numeric).all(axis=1)].copy()
                    scatter = ax.scatter(
                        pd.to_numeric(piece["reward_t2"], errors="coerce"),
                        pd.to_numeric(piece[feature], errors="coerce"),
                        c=pd.to_numeric(piece["reward_t1"], errors="coerce"),
                        cmap=t2_cmap,
                        norm=t2_norm,
                        s=7,
                        alpha=0.38,
                        linewidths=0,
                    )
                    if (space, feature) in feature_limits:
                        ax.set_ylim(*feature_limits[(space, feature)])
                    ax.set_title(f"{feature}, dim {dim}", fontsize=PANEL_FONT_SIZE)
                    ax.set_xlabel("reward_t2", fontsize=PANEL_FONT_SIZE)
                    ax.set_ylabel(feature, fontsize=PANEL_FONT_SIZE)
                    ax.tick_params(labelsize=PANEL_FONT_SIZE)
            fig.suptitle(f"{space}, t2, seed {seed}", fontsize=PANEL_FONT_SIZE)
            fig.subplots_adjust(right=0.84, wspace=0.35, hspace=0.58, top=0.92)
            if scatter is not None:
                cax = fig.add_axes([0.88, 0.16, 0.025, 0.70])
                cbar = fig.colorbar(scatter, cax=cax)
                cbar.set_label("reward_t1", fontsize=PANEL_FONT_SIZE)
                cbar.ax.tick_params(labelsize=PANEL_FONT_SIZE)
            fig.savefig(
                figdir / f"geometry_features_t2_by_reward2_seed_{file_token(seed)}_space_{space}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def base_geometry_reward_row(
    group_name: Dict,
    *,
    analysis_name: str,
    latent_dimension: int,
    timestep: int,
    reward_variable: str,
    geometry_space: str,
    geometry_feature: str,
    predictor_columns: Sequence[str],
    status: str,
    error_message: str,
    n_trials: int = 0,
    reward_t1_value=np.nan,
) -> Dict:
    row = {
        **group_name,
        "analysis_name": analysis_name,
        "latent_dimension": int(latent_dimension),
        "timestep": int(timestep),
        "reward_variable": reward_variable,
        "geometry_space": geometry_space,
        "geometry_feature": geometry_feature,
        "predictor_columns": ",".join(predictor_columns),
        "n_trials": int(n_trials),
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "slope_feature_to_reward": np.nan,
        "slope_reward_to_feature": np.nan,
        "status": status,
        "error_message": error_message,
    }
    if timestep == 2:
        row["reward_t1_value"] = reward_t1_value
    return row


def fit_geometry_reward_relation(
    group_name: Dict,
    work: pd.DataFrame,
    *,
    analysis_name: str,
    latent_dimension: int,
    timestep: int,
    reward_variable: str,
    geometry_space: str,
    geometry_feature: str,
    predictor_columns: Sequence[str],
    reward_t1_value=np.nan,
) -> Dict:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    required = list(predictor_columns) + [reward_variable]
    missing = [col for col in required if col not in work.columns]
    if missing:
        return base_geometry_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_missing_columns",
            error_message=f"missing columns: {','.join(missing)}",
            reward_t1_value=reward_t1_value,
        )
    numeric = work[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 3:
        return base_geometry_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_too_few_rows",
            error_message="fewer than three valid rows",
            n_trials=len(numeric),
            reward_t1_value=reward_t1_value,
        )
    x = numeric[list(predictor_columns)].to_numpy(dtype=float)
    y = numeric[reward_variable].to_numpy(dtype=float)
    if np.nanstd(y) <= 1e-12:
        return base_geometry_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_no_reward_variance",
            error_message="reward variable has no variance",
            n_trials=len(numeric),
            reward_t1_value=reward_t1_value,
        )
    if np.any(np.nanstd(x, axis=0) <= 1e-12):
        return base_geometry_reward_row(
            group_name,
            analysis_name=analysis_name,
            latent_dimension=latent_dimension,
            timestep=timestep,
            reward_variable=reward_variable,
            geometry_space=geometry_space,
            geometry_feature=geometry_feature,
            predictor_columns=predictor_columns,
            status="skipped_no_predictor_variance",
            error_message="one or more predictors have no variance",
            n_trials=len(numeric),
            reward_t1_value=reward_t1_value,
        )
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    row = base_geometry_reward_row(
        group_name,
        analysis_name=analysis_name,
        latent_dimension=latent_dimension,
        timestep=timestep,
        reward_variable=reward_variable,
        geometry_space=geometry_space,
        geometry_feature=geometry_feature,
        predictor_columns=predictor_columns,
        status="ok",
        error_message="",
        n_trials=len(numeric),
        reward_t1_value=reward_t1_value,
    )
    row["R2"] = float(r2_score(y, pred))
    row["RMSE"] = float(np.sqrt(mean_squared_error(y, pred)))
    row["MAE"] = float(mean_absolute_error(y, pred))
    if len(predictor_columns) == 1:
        xv = x[:, 0]
        row["pearson_r"] = safe_regression_corr(y, xv, "pearson")
        row["spearman_r"] = safe_regression_corr(y, xv, "spearman")
        row["slope_feature_to_reward"] = float(LinearRegression().fit(xv[:, None], y).coef_[0])
        row["slope_reward_to_feature"] = float(LinearRegression().fit(y[:, None], xv).coef_[0])
    return row


def build_geometry_reward_correlations(coord: pd.DataFrame, min_reward_group_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = [col for col in MODEL_GROUP_COLUMNS if col in coord.columns]
    group_iter = coord.groupby(group_cols, dropna=False) if group_cols else [((), coord)]
    t1_rows = []
    t2_rows = []
    for values, group_df in group_iter:
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(group_cols, values))
        for space in sorted(group_df["space"].dropna().unique()):
            for dim in sorted(pd.to_numeric(group_df["latent_dimension"], errors="coerce").dropna().unique()):
                dim = int(dim)
                t1_work = geometry_rows_used_downstream(group_df, timestep=1)
                t1_work = t1_work[(t1_work["space"] == space) & (t1_work["latent_dimension"] == dim)].copy()
                for feature_name, feature_cols in GEOMETRY_REWARD_FEATURE_SETS:
                    t1_rows.append(fit_geometry_reward_relation(
                        group_name,
                        t1_work,
                        analysis_name="t1_reward_vs_geometry",
                        latent_dimension=dim,
                        timestep=1,
                        reward_variable="reward_t1",
                        geometry_space=space,
                        geometry_feature=feature_name,
                        predictor_columns=feature_cols,
                    ))
                t2_work = geometry_rows_used_downstream(group_df, timestep=2)
                t2_work = t2_work[(t2_work["space"] == space) & (t2_work["latent_dimension"] == dim)].copy()
                if "reward_t1" not in t2_work.columns:
                    continue
                for reward_t1_value, piece in t2_work.groupby("reward_t1", dropna=True):
                    for feature_name, feature_cols in GEOMETRY_REWARD_FEATURE_SETS:
                        if len(piece) < min_reward_group_n:
                            t2_rows.append(base_geometry_reward_row(
                                group_name,
                                analysis_name="t2_reward_vs_geometry_by_R1",
                                latent_dimension=dim,
                                timestep=2,
                                reward_variable="reward_t2",
                                geometry_space=space,
                                geometry_feature=feature_name,
                                predictor_columns=feature_cols,
                                status="skipped_too_few_rows",
                                error_message=f"fewer than min-reward-group-n={min_reward_group_n} rows",
                                n_trials=len(piece),
                                reward_t1_value=reward_t1_value,
                            ))
                            continue
                        t2_rows.append(fit_geometry_reward_relation(
                            group_name,
                            piece,
                            analysis_name="t2_reward_vs_geometry_by_R1",
                            latent_dimension=dim,
                            timestep=2,
                            reward_variable="reward_t2",
                            geometry_space=space,
                            geometry_feature=feature_name,
                            predictor_columns=feature_cols,
                            reward_t1_value=reward_t1_value,
                        ))
    t1_df = pd.DataFrame(t1_rows)
    t2_df = pd.DataFrame(t2_rows)
    slope_summary = geometry_t2_slope_summary(t2_df, group_cols)
    t1_comparison = geometry_space_comparison(t1_df, group_cols, by_r1=False)
    t2_comparison = geometry_space_comparison(t2_df, group_cols, by_r1=True)
    return t1_df, t2_df, slope_summary, t1_comparison, t2_comparison


def geometry_t2_slope_summary(t2_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if t2_df is None or len(t2_df) == 0:
        return pd.DataFrame()
    usable = t2_df[
        (t2_df["status"] == "ok")
        & t2_df["geometry_feature"].isin(GEOMETRY_COORDINATE_FEATURES)
    ].copy()
    usable["slope_feature_to_reward"] = pd.to_numeric(usable["slope_feature_to_reward"], errors="coerce")
    usable = usable[np.isfinite(usable["slope_feature_to_reward"])].copy()
    if len(usable) == 0:
        return pd.DataFrame()
    rows = []
    summary_cols = [
        col for col in list(group_cols) + ["latent_dimension", "geometry_space", "geometry_feature"]
        if col in usable.columns
    ]
    for values, piece in usable.groupby(summary_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(summary_cols, values))
        slopes = pd.to_numeric(piece["slope_feature_to_reward"], errors="coerce").dropna()
        if len(slopes) == 0:
            continue
        row.update({
            "n_reward_t1_groups": int(len(slopes)),
            "mean_slope_feature_to_reward": float(slopes.mean()),
            "mean_abs_slope_feature_to_reward": float(np.abs(slopes).mean()),
            "min_slope_feature_to_reward": float(slopes.min()),
            "max_slope_feature_to_reward": float(slopes.max()),
            "slope_sign_flip": bool((slopes.min() < 0) and (slopes.max() > 0)),
            "mean_R2": float(pd.to_numeric(piece["R2"], errors="coerce").mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_space_comparison(rows_df: pd.DataFrame, group_cols: Sequence[str], by_r1: bool) -> pd.DataFrame:
    if rows_df is None or len(rows_df) == 0:
        return pd.DataFrame()
    usable = rows_df[rows_df["status"] == "ok"].copy()
    usable["R2"] = pd.to_numeric(usable["R2"], errors="coerce")
    usable = usable[np.isfinite(usable["R2"])].copy()
    if len(usable) == 0:
        return pd.DataFrame()
    compare_cols = [
        col for col in list(group_cols) + ["latent_dimension", "geometry_feature"]
        if col in usable.columns
    ]
    if by_r1 and "reward_t1_value" in usable.columns:
        compare_cols.append("reward_t1_value")
    grouped = (
        usable.groupby(compare_cols + ["geometry_space"], dropna=False)["R2"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=compare_cols,
        columns="geometry_space",
        values="R2",
        aggfunc="first",
    ).reset_index()
    metric_cols = [col for col in pivot.columns if col not in compare_cols]
    if metric_cols:
        pivot["best_geometry_space"] = pivot[metric_cols].idxmax(axis=1)
        pivot["best_R2"] = pivot[metric_cols].max(axis=1)
        if "poincare_disk" in pivot.columns and "prior_centered_poincare_disk" in pivot.columns:
            pivot["delta_prior_centered_disk_minus_disk_R2"] = (
                pivot["prior_centered_poincare_disk"] - pivot["poincare_disk"]
            )
        if "poincare_plane" in pivot.columns and "prior_centered_poincare_plane" in pivot.columns:
            pivot["delta_prior_centered_plane_minus_plane_R2"] = (
                pivot["prior_centered_poincare_plane"] - pivot["poincare_plane"]
            )
    return pivot


def write_prior_centered_geometry_reward_summary(
    outdir: Path,
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    slope_summary: pd.DataFrame,
    t1_comparison: pd.DataFrame,
    t2_comparison: pd.DataFrame,
):
    valid_t1 = t1_df[t1_df["status"] == "ok"].copy() if len(t1_df) else pd.DataFrame()
    valid_t2 = t2_df[t2_df["status"] == "ok"].copy() if len(t2_df) else pd.DataFrame()
    summary = {
        "analysis": "prior_centered_geometry_reward",
        "coordinate_spaces": [spec["space"] for spec in GEOMETRY_COORDINATE_SPACES],
        "plotted_outputs": [
            "geometry_scatter_t1_reward_seed_{seed}_space_{space}.png",
            "geometry_scatter_t2_reward2_by_R1_seed_{seed}_space_{space}.png",
            "geometry_features_t1_by_reward_seed_{seed}_space_{space}.png",
            "geometry_features_t2_by_reward2_seed_{seed}_space_{space}.png",
        ],
        "t1_best": {},
        "t2_best": {},
        "slope_sign_flip_rows": int(slope_summary["slope_sign_flip"].sum()) if len(slope_summary) and "slope_sign_flip" in slope_summary else 0,
        "t1_space_win_counts": t1_comparison["best_geometry_space"].value_counts().to_dict() if len(t1_comparison) and "best_geometry_space" in t1_comparison else {},
        "t2_space_win_counts": t2_comparison["best_geometry_space"].value_counts().to_dict() if len(t2_comparison) and "best_geometry_space" in t2_comparison else {},
    }
    if len(valid_t1):
        row = valid_t1.sort_values("R2", ascending=False).iloc[0]
        summary["t1_best"] = {
            "geometry_space": row.get("geometry_space"),
            "geometry_feature": row.get("geometry_feature"),
            "latent_dimension": row.get("latent_dimension"),
            "R2": row.get("R2"),
            "seed": row.get("seed"),
        }
    if len(valid_t2):
        row = valid_t2.sort_values("R2", ascending=False).iloc[0]
        summary["t2_best"] = {
            "geometry_space": row.get("geometry_space"),
            "geometry_feature": row.get("geometry_feature"),
            "latent_dimension": row.get("latent_dimension"),
            "reward_t1_value": row.get("reward_t1_value"),
            "R2": row.get("R2"),
            "seed": row.get("seed"),
        }
    summary = json_safe_value(summary)
    with open(outdir / "prior_centered_geometry_reward_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "prior_centered_geometry_reward_summary.txt", "w") as handle:
        handle.write("Prior-centered geometry reward summary\n")
        handle.write("======================================\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")


def run_prior_centered_geometry_reward_analysis(
    df: pd.DataFrame,
    outdir: Path,
    min_reward_group_n: int,
):
    coord = build_geometry_coordinate_long(df)
    coord.to_csv(outdir / "geometry_coordinate_long.csv", index=False)
    wide = build_geometry_coordinate_features_wide(coord)
    wide.to_csv(outdir / "geometry_coordinate_features_wide.csv", index=False)
    if len(coord) == 0:
        return coord, pd.DataFrame(), pd.DataFrame()
    t1_df, t2_df, slope_summary, t1_comparison, t2_comparison = build_geometry_reward_correlations(
        coord,
        min_reward_group_n=min_reward_group_n,
    )
    t1_df.to_csv(outdir / "geometry_reward_correlation_t1_by_space.csv", index=False)
    t2_df.to_csv(outdir / "geometry_reward_correlation_t2_by_R1_by_space.csv", index=False)
    slope_summary.to_csv(outdir / "geometry_t2_reward2_slope_by_R1_summary_by_space.csv", index=False)
    t1_comparison.to_csv(outdir / "geometry_space_comparison_t1.csv", index=False)
    t2_comparison.to_csv(outdir / "geometry_space_comparison_t2_by_R1.csv", index=False)
    write_prior_centered_geometry_reward_summary(
        outdir,
        t1_df,
        t2_df,
        slope_summary,
        t1_comparison,
        t2_comparison,
    )
    figdir = outdir / "figures"
    plot_geometry_coordinate_scatter_t1(coord, figdir)
    plot_geometry_coordinate_scatter_t2_by_r1(coord, figdir)
    plot_geometry_feature_reward_relationships(coord, figdir)
    return coord, t1_df, t2_df


def run_halfplane_reward_geometry_analysis(
    df: pd.DataFrame,
    outdir: Path,
    min_reward_group_n: int,
    make_plots: bool = True,
):
    df = add_geometry_meaning_columns(df)
    group_cols = reward_encoding_group_columns(df)
    if not group_cols:
        group_iter = [((), df)]
    else:
        group_iter = df.groupby(group_cols, dropna=False)
    t1_rows = []
    t2_rows = []
    for values, group_df in group_iter:
        if not isinstance(values, tuple):
            values = (values,)
        group_name = dict(zip(group_cols, values))
        for dim in (0, 1):
            for space in ("halfplane", "poincare_disk"):
                for feature, _ in HALFPLANE_REWARD_FEATURES:
                    cols = halfplane_reward_feature_columns(space, dim, 1, feature)
                    t1_work = rows_after_observed_reward_used_downstream(group_df, timestep=1)
                    t1_rows.append(fit_halfplane_reward_relation(
                        group_name,
                        t1_work,
                        analysis_name="t1_reward_vs_halfplane_disk_geometry",
                        latent_dimension=dim,
                        timestep=1,
                        reward_variable="reward_t1",
                        geometry_space=space,
                        geometry_feature=feature,
                        predictor_columns=cols,
                    ))
                    t2_work = rows_after_observed_reward_used_downstream(group_df, timestep=2)
                    status = "ok"
                    error_message = ""
                    if "qz_used_downstream" not in group_df.columns:
                        status = "warning_qz2_downstream_usage_unknown"
                        error_message = "qz_used_downstream column is unavailable"
                    if "reward_t1" not in t2_work.columns:
                        continue
                    for reward_t1_value, piece in t2_work.groupby("reward_t1", dropna=True):
                        if len(piece) < min_reward_group_n:
                            t2_rows.append(base_halfplane_reward_row(
                                group_name,
                                analysis_name="t2_reward_vs_halfplane_disk_geometry_by_R1",
                                latent_dimension=dim,
                                timestep=2,
                                reward_variable="reward_t2",
                                geometry_space=space,
                                geometry_feature=feature,
                                predictor_columns=halfplane_reward_feature_columns(space, dim, 2, feature),
                                status="skipped_too_few_rows",
                                error_message=f"fewer than min-reward-group-n={min_reward_group_n} rows",
                                n_trials=len(piece),
                                reward_t1_value=reward_t1_value,
                            ))
                            continue
                        t2_rows.append(fit_halfplane_reward_relation(
                            group_name,
                            piece,
                            analysis_name="t2_reward_vs_halfplane_disk_geometry_by_R1",
                            latent_dimension=dim,
                            timestep=2,
                            reward_variable="reward_t2",
                            geometry_space=space,
                            geometry_feature=feature,
                            predictor_columns=halfplane_reward_feature_columns(space, dim, 2, feature),
                            reward_t1_value=reward_t1_value,
                            status=status,
                            error_message=error_message,
                        ))
    t1_df = pd.DataFrame(t1_rows)
    t2_df = pd.DataFrame(t2_rows)
    slope_summary = halfplane_t2_slope_rows(t2_df, df, group_cols)
    t1_df.to_csv(outdir / "halfplane_reward_correlation_t1.csv", index=False)
    t2_df.to_csv(outdir / "halfplane_reward_correlation_t2_by_R1.csv", index=False)
    slope_summary.to_csv(outdir / "halfplane_t2_slope_by_R1_summary.csv", index=False)
    if make_plots:
        plot_halfplane_reward_outputs(df, t2_df, slope_summary, outdir / "figures")
    write_halfplane_reward_summary(outdir, t1_df, t2_df, slope_summary)
    if "seed" in t1_df.columns:
        for seed_value in sorted(pd.Series(t1_df["seed"]).dropna().unique()):
            seed_prefix = f"seed_{file_token(seed_value)}_"
            seed_t1 = t1_df[t1_df["seed"] == seed_value].copy()
            seed_t2 = t2_df[t2_df["seed"] == seed_value].copy() if "seed" in t2_df.columns else t2_df
            seed_slope = (
                slope_summary[slope_summary["seed"] == seed_value].copy()
                if "seed" in slope_summary.columns
                else slope_summary
            )
            seed_t1.to_csv(outdir / f"{seed_prefix}halfplane_reward_correlation_t1.csv", index=False)
            seed_t2.to_csv(outdir / f"{seed_prefix}halfplane_reward_correlation_t2_by_R1.csv", index=False)
            seed_slope.to_csv(outdir / f"{seed_prefix}halfplane_t2_slope_by_R1_summary.csv", index=False)
            write_halfplane_reward_summary(outdir, seed_t1, seed_t2, seed_slope, file_prefix=seed_prefix)
    return t1_df, t2_df, slope_summary


def plot_requested_temporal_best_path_value_dim0_by_seed(
    transition_df: pd.DataFrame,
    figdir: Path,
    limits: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
):
    if transition_df is None or len(transition_df) == 0:
        return
    required = {
        "transition",
        "current_best_path_value_t2",
        "halfplane_x_0_t1",
        "halfplane_y_0_t1",
        "temporal_delta_halfplane_x_0",
        "temporal_delta_halfplane_y_0",
    }
    if not required.issubset(transition_df.columns):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    figdir.mkdir(parents=True, exist_ok=True)
    seed_groups = (
        transition_df.groupby("seed", dropna=False)
        if "seed" in transition_df.columns
        else [("all", transition_df)]
    )
    for seed, seed_df in seed_groups:
        seed_df = transitions_after_observed_rewards_used_downstream(seed_df)
        if len(seed_df) == 0:
            continue
        numeric_cols = [
            "halfplane_x_0_t1",
            "halfplane_y_0_t1",
            "temporal_delta_halfplane_x_0",
            "temporal_delta_halfplane_y_0",
        ]
        finite_numeric = seed_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        piece = seed_df[
            (seed_df["transition"] == "t1_to_t2")
            & seed_df["current_best_path_value_t2"].notna()
            & np.isfinite(finite_numeric).all(axis=1)
        ].copy()
        if len(piece) == 0:
            continue
        if len(piece) > 1500:
            piece = piece.sample(1500, random_state=101)
        values = pd.to_numeric(piece["current_best_path_value_t2"], errors="coerce")
        vmin = float(values.min())
        vmax = float(values.max())
        if math.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = plt.get_cmap("coolwarm")(norm(values))
        fig, ax = plt.subplots(figsize=panel_figsize(1, 1, title=True))
        ax.quiver(
            piece["halfplane_x_0_t1"],
            piece["halfplane_y_0_t1"],
            piece["temporal_delta_halfplane_x_0"],
            piece["temporal_delta_halfplane_y_0"],
            color=colors,
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.45,
            width=0.002,
        )
        ax.set_xlabel("mu_0 at start / sqrt(2)")
        ax.set_ylabel("sigma_0 at start")
        ax.set_title("t1_to_t2 half-plane movement, dim 0")
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
        if limits and 0 in limits:
            ax.set_xlim(*limits[0][0])
            ax.set_ylim(*limits[0][1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
        sm.set_array([])
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        cax = fig.add_axes([0.86, 0.18, 0.035, 0.64])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("current_best_path_value_t2")
        fig.savefig(
            figdir / (
                "gaussian_halfplane_temporal_arrows_by_best_path_value_"
                f"t1_to_t2_dim0_seed_{file_token(seed)}.png"
            ),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def make_requested_halfplane_plots(
    df: pd.DataFrame,
    transition_df: Optional[pd.DataFrame],
    outdir: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = add_halfplane_coordinate_columns(df)
    plot_df = rows_after_observed_reward_used_downstream(df)
    halfplane_limits = halfplane_axis_limits(plot_df)
    filtered_transition_df = transitions_after_observed_rewards_used_downstream(transition_df)
    transition_limits = temporal_halfplane_axis_limits(filtered_transition_df)
    if "seed" in plot_df.columns:
        for seed, seed_df in plot_df.groupby("seed", dropna=False):
            plot_t1_halfplane_observed_reward_by_dim(
                seed_df,
                figdir,
                filename_suffix=f"_seed_{file_token(seed)}",
                limits=halfplane_limits,
            )
            plot_poincare_disk_t1_t2_rewards_by_dim(
                seed_df,
                figdir,
                filename_suffix=f"_seed_{file_token(seed)}",
            )
    else:
        plot_t1_halfplane_observed_reward_by_dim(
            plot_df,
            figdir,
            limits=halfplane_limits,
        )
        plot_poincare_disk_t1_t2_rewards_by_dim(
            plot_df,
            figdir,
        )
    plot_requested_temporal_best_path_value_dim0_by_seed(
        filtered_transition_df,
        figdir,
        limits=transition_limits,
    )


def compute_gaussian_mixture_density(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mu0_col: str,
    mu1_col: str,
    sigma0_col: str,
    sigma1_col: str,
    sigma_floor: float = 1e-3,
    chunk_size: int = 256,
) -> np.ndarray:
    """Average trial-level diagonal 2D Gaussian posteriors in latent sample space."""
    required = [mu0_col, mu1_col, sigma0_col, sigma1_col]
    work = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) == 0:
        return np.zeros((len(y_grid), len(x_grid)), dtype=float)
    mu0 = work[mu0_col].to_numpy(dtype=float)
    mu1 = work[mu1_col].to_numpy(dtype=float)
    sigma0 = np.maximum(work[sigma0_col].to_numpy(dtype=float), sigma_floor)
    sigma1 = np.maximum(work[sigma1_col].to_numpy(dtype=float), sigma_floor)
    xx, yy = np.meshgrid(x_grid, y_grid)
    log_total = np.full(xx.shape, -np.inf, dtype=float)
    log_2pi = math.log(2.0 * math.pi)
    for start in range(0, len(work), chunk_size):
        end = min(start + chunk_size, len(work))
        m0 = mu0[start:end, None, None]
        m1 = mu1[start:end, None, None]
        s0 = sigma0[start:end, None, None]
        s1 = sigma1[start:end, None, None]
        log_density = (
            -log_2pi
            - np.log(s0)
            - np.log(s1)
            - 0.5 * ((xx[None, :, :] - m0) / s0) ** 2
            - 0.5 * ((yy[None, :, :] - m1) / s1) ** 2
        )
        chunk_max = np.max(log_density, axis=0)
        chunk_sum = np.sum(np.exp(log_density - chunk_max[None, :, :]), axis=0)
        chunk_logsum = chunk_max + np.log(chunk_sum + 1e-300)
        log_total = np.logaddexp(log_total, chunk_logsum)
    return np.exp(log_total - math.log(len(work)))


def latent_density_axis_limits(df: pd.DataFrame) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    required = {"z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1"}
    if not required.issubset(df.columns) or len(df) == 0:
        return None
    work = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) == 0:
        return None
    x_low = np.nanquantile(work["z_mu_0"] - 2.0 * work["z_sigma_0"], 0.01)
    x_high = np.nanquantile(work["z_mu_0"] + 2.0 * work["z_sigma_0"], 0.99)
    y_low = np.nanquantile(work["z_mu_1"] - 2.0 * work["z_sigma_1"], 0.01)
    y_high = np.nanquantile(work["z_mu_1"] + 2.0 * work["z_sigma_1"], 0.99)

    def pad_limits(lo: float, hi: float) -> Tuple[float, float]:
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (-1.0, 1.0)
        span = hi - lo
        if span <= 1e-9:
            span = max(abs(lo), 1.0)
            lo -= 0.5 * span
            hi += 0.5 * span
        else:
            lo -= 0.08 * span
            hi += 0.08 * span
        return float(lo), float(hi)

    return pad_limits(x_low, x_high), pad_limits(y_low, y_high)


def latent_density_reward_groups(df: pd.DataFrame, reward_col: str) -> Tuple[pd.Series, List[str], str]:
    reward = pd.to_numeric(df[reward_col], errors="coerce")
    unique = np.sort(reward[np.isfinite(reward)].unique())
    if len(unique) == 0:
        return pd.Series(np.nan, index=df.index, dtype=object), [], "missing"
    if len(unique) <= 12:
        labels = reward.map(lambda value: f"{float(value):g}" if np.isfinite(value) else np.nan)
        order = [f"{float(value):g}" for value in unique]
        return labels.astype(object), order, "exact"
    q = min(3, len(unique))
    bins = pd.qcut(reward, q=q, labels=False, duplicates="drop")
    bin_labels = ["low", "medium", "high"]
    observed_bins = sorted(int(value) for value in pd.Series(bins).dropna().unique())
    label_lookup = {
        bin_value: bin_labels[min(i, len(bin_labels) - 1)]
        for i, bin_value in enumerate(observed_bins)
    }
    labels = pd.Series(bins, index=df.index).map(label_lookup)
    order = [label_lookup[bin_value] for bin_value in observed_bins]
    return labels.astype(object), order, "quantile"


def ensure_latent_density_reward_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "reward_t1" not in df.columns and "t1_observed_value" in df.columns:
        df["reward_t1"] = df["t1_observed_value"]
    if "reward_t2" not in df.columns:
        if "t2_observed_value" in df.columns:
            df["reward_t2"] = df["t2_observed_value"]
        elif {"timestep", "observed_value"}.issubset(df.columns):
            key_cols = [
                col for col in ["model_id", "trial_uid", "trial_id"]
                if col in df.columns
            ]
            if key_cols:
                t2 = df[pd.to_numeric(df["timestep"], errors="coerce") == 2].copy()
                if len(t2) > 0:
                    lookup = t2.drop_duplicates(key_cols).set_index(key_cols)["observed_value"]
                    keys = pd.MultiIndex.from_frame(df[key_cols]) if len(key_cols) > 1 else df[key_cols[0]]
                    df["reward_t2"] = lookup.reindex(keys).to_numpy()
    if "observed_reward_t1" not in df.columns and "reward_t1" in df.columns:
        df["observed_reward_t1"] = df["reward_t1"]
    if "observed_reward_t2" not in df.columns and "reward_t2" in df.columns:
        df["observed_reward_t2"] = df["reward_t2"]
    return df


def latent_density_group_suffix(piece: pd.DataFrame) -> str:
    tokens = []
    for col in ["seed", "lambda_value", "beta", "alpha", "opportunity_cost", "rnn_dim", "latent_dim"]:
        if col in piece.columns and len(piece[col].dropna()) > 0:
            value = piece[col].dropna().iloc[0]
            label = "lambda" if col == "lambda_value" else col
            tokens.append(f"{label}_{file_token(value)}")
    return "_" + "_".join(tokens) if tokens else ""


def latent_density_title(piece: pd.DataFrame, prefix: str) -> str:
    fields = []
    for col, label in [("alpha", "alpha"), ("beta", "beta"), ("lambda_value", "lambda"), ("seed", "seed")]:
        if col in piece.columns and len(piece[col].dropna()) > 0:
            fields.append(f"{label}={piece[col].dropna().iloc[0]:g}")
    return prefix + ("\n" + ", ".join(fields) if fields else "")


def positive_contour_levels(density: np.ndarray, n_levels: int = 5) -> Optional[np.ndarray]:
    values = density[np.isfinite(density) & (density > 0.0)]
    if len(values) == 0:
        return None
    max_density = float(np.nanmax(values))
    min_density = float(np.nanmin(values))
    if not np.isfinite(max_density) or max_density <= 0.0:
        return None
    # Gaussian tails are positive everywhere, so quantiles over all positive grid
    # cells can select extremely low tail densities and make contours look much
    # wider than the posterior uncertainty. Use peak-relative levels instead.
    fractions = np.linspace(0.08, 0.72, n_levels)
    levels = max_density * fractions
    levels = np.unique(levels[np.isfinite(levels) & (levels > min_density) & (levels < max_density)])
    return levels if len(levels) > 0 else None


def plot_latent_2d_density_t1(
    piece: pd.DataFrame,
    figdir: Path,
    filename_suffix: str,
    grid_n: int,
    standard_name: bool = False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    t1 = rows_after_observed_reward_used_downstream(piece, timestep=1)
    required = {"z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1", "reward_t1"}
    if len(t1) == 0 or not required.issubset(t1.columns):
        return
    limits = latent_density_axis_limits(
        pd.concat(
            [
                rows_after_observed_reward_used_downstream(piece, timestep=1),
                rows_after_observed_reward_used_downstream(piece, timestep=2),
            ],
            ignore_index=True,
        )
    )
    if limits is None:
        return
    xlim, ylim = limits
    x_grid = np.linspace(xlim[0], xlim[1], grid_n)
    y_grid = np.linspace(ylim[0], ylim[1], grid_n)
    labels, order, _ = latent_density_reward_groups(t1, "reward_t1")
    if not order:
        return
    t1 = t1.copy()
    t1["_reward_group"] = labels
    reward_numeric = pd.to_numeric(t1["reward_t1"], errors="coerce")
    finite_reward = reward_numeric[np.isfinite(reward_numeric)]
    if len(finite_reward) == 0:
        return
    reward_min = float(finite_reward.min())
    reward_max = float(finite_reward.max())
    if abs(reward_max - reward_min) < 1e-12:
        reward_min -= 0.5
        reward_max += 0.5
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=reward_min, vmax=reward_max)
    fig, ax = plt.subplots(figsize=panel_figsize(1, 1, colorbar=True, title=True))
    for label in order:
        group = t1[t1["_reward_group"] == label]
        if len(group) == 0:
            continue
        group_rewards = pd.to_numeric(group["reward_t1"], errors="coerce")
        color = cmap(norm(float(group_rewards.mean())))
        ax.scatter(
            group["z_mu_0"],
            group["z_mu_1"],
            c=group_rewards,
            cmap=cmap,
            norm=norm,
            s=5,
            alpha=0.15,
            linewidths=0,
        )
        density = compute_gaussian_mixture_density(
            group, x_grid, y_grid, "z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1"
        )
        levels = positive_contour_levels(density)
        if levels is not None:
            ax.contour(x_grid, y_grid, density, levels=levels, colors=[color], linewidths=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z_0")
    ax.set_ylabel("z_1")
    ax.set_title(latent_density_title(piece, "Timestep 1: aggregate 2D posterior density by observed reward at t1"), fontsize=PANEL_FONT_SIZE)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    add_first_row_colorbar(fig, np.asarray([[ax]], dtype=object), sm, "reward_t1", width=0.03)
    name = "latent_2d_density_t1_by_reward_t1.png" if standard_name else f"latent_2d_density_t1_by_reward_t1{filename_suffix}.png"
    fig.savefig(figdir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latent_2d_density_t2(
    piece: pd.DataFrame,
    figdir: Path,
    filename_suffix: str,
    grid_n: int,
    standard_name: bool = False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    apply_7pt_plot_style(plt)

    t2 = rows_after_observed_reward_used_downstream(piece, timestep=2)
    required = {"z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1", "reward_t1", "reward_t2"}
    if len(t2) == 0 or not required.issubset(t2.columns):
        return
    limits = latent_density_axis_limits(
        pd.concat(
            [
                rows_after_observed_reward_used_downstream(piece, timestep=1),
                t2,
            ],
            ignore_index=True,
        )
    )
    if limits is None:
        return
    xlim, ylim = limits
    x_grid = np.linspace(xlim[0], xlim[1], grid_n)
    y_grid = np.linspace(ylim[0], ylim[1], grid_n)
    r1_labels, r1_order, _ = latent_density_reward_groups(t2, "reward_t1")
    r2_labels, r2_order, _ = latent_density_reward_groups(t2, "reward_t2")
    if not r1_order or not r2_order:
        return
    t2 = t2.copy()
    t2["_reward_t1_group"] = r1_labels
    t2["_reward_t2_group"] = r2_labels
    reward_numeric = pd.to_numeric(t2["reward_t2"], errors="coerce")
    finite_reward = reward_numeric[np.isfinite(reward_numeric)]
    if len(finite_reward) == 0:
        return
    reward_min = float(finite_reward.min())
    reward_max = float(finite_reward.max())
    if abs(reward_max - reward_min) < 1e-12:
        reward_min -= 0.5
        reward_max += 0.5
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=reward_min, vmax=reward_max)
    n_panels = len(r1_order)
    ncols = 2 if n_panels > 4 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=panel_figsize(ncols, nrows, colorbar=True, title=True), squeeze=False)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    for panel_i, r1_label in enumerate(r1_order):
        ax = axes.ravel()[panel_i]
        panel = t2[t2["_reward_t1_group"] == r1_label]
        for r2_label in r2_order:
            group = panel[panel["_reward_t2_group"] == r2_label]
            if len(group) == 0:
                continue
            group_rewards = pd.to_numeric(group["reward_t2"], errors="coerce")
            color = cmap(norm(float(group_rewards.mean())))
            ax.scatter(
                group["z_mu_0"],
                group["z_mu_1"],
                c=group_rewards,
                cmap=cmap,
                norm=norm,
                s=5,
                alpha=0.15,
                linewidths=0,
            )
            density = compute_gaussian_mixture_density(
                group, x_grid, y_grid, "z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1"
            )
            levels = positive_contour_levels(density)
            if levels is not None:
                ax.contour(x_grid, y_grid, density, levels=levels, colors=[color], linewidths=1.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z_0")
        ax.set_ylabel("z_1")
        ax.set_title(f"reward_t1={r1_label}", fontsize=PANEL_FONT_SIZE)
    fig.suptitle(
        latent_density_title(piece, "Timestep 2: aggregate 2D posterior density; panels = reward at t1, color = reward at t2"),
        fontsize=PANEL_FONT_SIZE,
    )
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    add_first_row_colorbar(fig, axes, sm, "reward_t2")
    name = (
        "latent_2d_density_t2_by_reward_t1_panels_reward_t2_color.png"
        if standard_name
        else f"latent_2d_density_t2_by_reward_t1_panels_reward_t2_color{filename_suffix}.png"
    )
    fig.savefig(figdir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def latent_density_base_row(piece: pd.DataFrame, analysis: str, timestep: int, conditioning_variable: str, conditioning_value) -> Dict:
    row = {
        "analysis": analysis,
        "timestep": timestep,
        "conditioning_variable": conditioning_variable,
        "conditioning_value": conditioning_value,
    }
    for col in MODEL_GROUP_COLUMNS:
        if col in piece.columns and len(piece[col].dropna()) > 0:
            row[col] = piece[col].dropna().iloc[0]
    return row


def append_metric_rows(base: Dict, rows: List[Dict], target: str, feature_set: str, n_trials: int, metrics: Dict[str, float]):
    for metric_name, metric_value in metrics.items():
        row = dict(base)
        row.update(
            {
                "target": target,
                "feature_set": feature_set,
                "n_trials": int(n_trials),
                "metric_name": metric_name,
                "metric_value": metric_value,
            }
        )
        rows.append(row)


def target_is_discrete(y: np.ndarray) -> bool:
    y = y[np.isfinite(y)]
    unique = np.unique(y)
    return len(unique) <= 12 and np.all(np.isclose(unique, np.round(unique)))


def fit_density_reward_separation(piece: pd.DataFrame, target_col: str, feature_cols: List[str], cv_folds: int) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    work = piece[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 5:
        return {"status_skipped_too_few_trials": np.nan}
    x = work[feature_cols].to_numpy(dtype=float)
    y = work[target_col].to_numpy(dtype=float)
    if target_is_discrete(y):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or np.min(counts) < 2:
            return {"status_skipped_too_few_classes": np.nan}
        n_splits = min(cv_folds, int(np.min(counts)))
        if n_splits < 2:
            return {"status_skipped_too_few_folds": np.nan}
        pred = np.full(len(y), np.nan)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
        y_class = pd.factorize(y)[0]
        for train_idx, test_idx in splitter.split(x, y_class):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver="lbfgs")
            )
            model.fit(x[train_idx], y_class[train_idx])
            pred[test_idx] = model.predict(x[test_idx])
        finite = np.isfinite(pred)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_class[finite], pred[finite])),
            "macro_f1": float(f1_score(y_class[finite], pred[finite], average="macro", zero_division=0)),
        }
    n_splits = min(cv_folds, len(y))
    if n_splits < 2:
        return {"status_skipped_too_few_trials": np.nan}
    pred = np.full(len(y), np.nan)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=17)
    for train_idx, test_idx in splitter.split(x):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    finite = np.isfinite(pred)
    metrics = {
        "R2": float(r2_score(y[finite], pred[finite])),
        "RMSE": float(math.sqrt(mean_squared_error(y[finite], pred[finite]))),
    }
    metrics["Pearson_r"] = safe_regression_corr(y[finite], pred[finite], "pearson")
    metrics["Spearman_r"] = safe_regression_corr(y[finite], pred[finite], "spearman")
    return metrics


def build_latent_2d_density_reward_separation_summary(df: pd.DataFrame, cv_folds: int) -> pd.DataFrame:
    rows = []
    feature_sets = {
        "mu_only": ["z_mu_0", "z_mu_1"],
        "mu_sigma": ["z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1"],
    }
    group_cols = [col for col in MODEL_GROUP_COLUMNS if col in df.columns]
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]
    for _, piece in groups:
        if "latent_dim" in piece.columns and not np.all(pd.to_numeric(piece["latent_dim"], errors="coerce") == 2):
            base = latent_density_base_row(piece, "latent_2d_density_reward_separation", 0, "none", "all")
            append_metric_rows(base, rows, "none", "none", len(piece), {"status_skipped_latent_dim_not_2": np.nan})
            continue
        t1 = rows_after_observed_reward_used_downstream(piece, timestep=1)
        if "reward_t1" in t1.columns:
            base = latent_density_base_row(piece, "t1_reward_from_2d_posterior", 1, "none", "all")
            for feature_set, feature_cols in feature_sets.items():
                if set(feature_cols + ["reward_t1"]).issubset(t1.columns):
                    metrics = fit_density_reward_separation(t1, "reward_t1", feature_cols, cv_folds)
                    append_metric_rows(base, rows, "reward_t1", feature_set, len(t1), metrics)
        t2 = rows_after_observed_reward_used_downstream(piece, timestep=2)
        if {"reward_t1", "reward_t2"}.issubset(t2.columns):
            r1_labels, r1_order, _ = latent_density_reward_groups(t2, "reward_t1")
            t2 = t2.copy()
            t2["_reward_t1_group"] = r1_labels
            for r1_label in r1_order:
                panel = t2[t2["_reward_t1_group"] == r1_label]
                base = latent_density_base_row(piece, "t2_reward_from_2d_posterior_by_R1", 2, "reward_t1", r1_label)
                for feature_set, feature_cols in feature_sets.items():
                    if set(feature_cols + ["reward_t2"]).issubset(panel.columns):
                        metrics = fit_density_reward_separation(panel, "reward_t2", feature_cols, cv_folds)
                        append_metric_rows(base, rows, "reward_t2", feature_set, len(panel), metrics)
    return pd.DataFrame(rows)


def plot_latent_2d_density_reward_outputs(df: pd.DataFrame, outdir: Path, grid_n: int, cv_folds: int):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = ensure_latent_density_reward_columns(df)
    required = {"z_mu_0", "z_mu_1", "z_sigma_0", "z_sigma_1"}
    if not required.issubset(df.columns):
        print("Skipping latent 2D density plots: required posterior columns are missing.")
        pd.DataFrame([
            {
                "analysis": "latent_2d_density",
                "status": "skipped_missing_posterior_columns",
                "error_message": "Missing z_mu_0/z_mu_1/z_sigma_0/z_sigma_1.",
            }
        ]).to_csv(outdir / "latent_2d_density_reward_separation_summary.csv", index=False)
        return
    if "latent_dim" in df.columns:
        valid_dim = pd.to_numeric(df["latent_dim"], errors="coerce") == 2
        if not bool(valid_dim.any()):
            print("Skipping latent 2D density plots: latent_dim is not 2.")
            pd.DataFrame([
                {
                    "analysis": "latent_2d_density",
                    "status": "skipped_latent_dim_not_2",
                    "error_message": "Aggregate 2D posterior density requires latent_dim == 2.",
                }
            ]).to_csv(outdir / "latent_2d_density_reward_separation_summary.csv", index=False)
            return
        df = df[valid_dim].copy()
    grid_n = max(40, min(int(grid_n), 250))
    plot_latent_2d_density_t1(df, figdir, "", grid_n, standard_name=True)
    plot_latent_2d_density_t2(df, figdir, "", grid_n, standard_name=True)
    group_cols = [col for col in MODEL_GROUP_COLUMNS if col in df.columns]
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]
    for _, piece in groups:
        suffix = latent_density_group_suffix(piece)
        plot_latent_2d_density_t1(piece, figdir, suffix, grid_n)
        plot_latent_2d_density_t2(piece, figdir, suffix, grid_n)
    summary = build_latent_2d_density_reward_separation_summary(df, cv_folds)
    summary.to_csv(outdir / "latent_2d_density_reward_separation_summary.csv", index=False)


def main():
    args = parse_args()
    tf = configure_tensorflow(args.device)
    helper, simulate = import_model_modules()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_7pt_plot_style(plt)
    except Exception:
        pass

    all_frames = []
    failures = []
    for alpha, beta, lambda_value, seed, rnn_dim, latent_dim, opportunity_cost in itertools.product(
        args.alphas,
        args.betas,
        args.lambda_values,
        args.seeds,
        args.rnn_dims,
        args.latent_dims,
        args.opportunity_costs,
    ):
        model_id = (
            f"lambda_{lambda_value}_alpha_{alpha}_beta_{beta}_seed_{seed}_"
            f"rnn_{rnn_dim}_latent_{latent_dim}_opp_{opportunity_cost}"
        )
        try:
            checkpoint_path, notes = find_checkpoint(
                args.checkpoint_root,
                lambda_value=lambda_value,
                alpha=alpha,
                beta=beta,
                seed=seed,
                tree_size=args.tree_size,
                tree_type=args.tree_type,
                rnn_dim=rnn_dim,
                latent_dim=latent_dim,
                opportunity_cost=opportunity_cost,
                expansion_decision_version=args.expansion_decision_version,
                model_variant=args.model_variant,
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
                model_variant=args.model_variant,
                checkpoint_root=args.checkpoint_root,
            )
            model = build_and_load_model(
                tf,
                simulate,
                config,
                alpha=alpha,
                beta=beta,
                lambda_value=lambda_value,
                opportunity_cost=opportunity_cost,
                checkpoint_path=checkpoint_path,
            )
            rewards = sample_rewards(
                args.n_trials,
                config.time_steps,
                args.input_type,
                seed=args.analysis_seed_offset + seed,
            )
            outputs = run_model_trials(tf, model, rewards, args.batch_size)
            metadata = {
                "model_id": model_id,
                "checkpoint_path": str(checkpoint_path),
                "alpha": alpha,
                "beta": beta,
                "lambda_value": lambda_value,
                "seed": seed,
                "rnn_dim": rnn_dim,
                "latent_dim": latent_dim,
                "opportunity_cost": opportunity_cost,
                "tree_size": args.tree_size,
                "tree_type": args.tree_type,
                "input_type": args.input_type,
                "expansion_decision_version": args.expansion_decision_version,
                "model_variant": args.model_variant,
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
            frame = trial_timestep_dataframe(
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
        raise SystemExit(f"No models were analyzed. See {failure_path}")

    data = pd.concat(all_frames, ignore_index=True)
    data = add_geometry_meaning_columns(data)
    csv_path = outdir / "latent_angle_trial_timestep_data.csv"
    data.to_csv(csv_path, index=False)
    try:
        data.to_parquet(outdir / "latent_angle_trial_timestep_data.parquet", index=False)
    except Exception as exc:
        print(f"Parquet output skipped: {exc}")

    transition_data = build_temporal_transition_features(data)
    transition_data.to_csv(outdir / "latent_temporal_transition_features.csv", index=False)
    try:
        if len(transition_data) > 0:
            transition_data.to_parquet(outdir / "latent_temporal_transition_features.parquet", index=False)
    except Exception as exc:
        print(f"Temporal transition parquet output skipped: {exc}")

    plot_latent_2d_density_reward_outputs(
        data,
        outdir,
        args.latent_density_grid_n,
        args.cv_folds,
    )

    run_reward_encoding_analyses(
        data,
        outdir,
        args.cv_folds,
        failures,
        args.n_trials,
        make_plots=False,
    )
    run_geometry_meaning_analyses(
        data,
        outdir,
        args.cv_folds,
        args.min_within_path_n,
        make_plots=False,
    )
    run_halfplane_reward_geometry_analysis(
        data,
        outdir,
        args.min_reward_group_n,
        make_plots=False,
    )
    try:
        run_prior_centered_geometry_reward_analysis(
            data,
            outdir,
            args.min_reward_group_n,
        )
    except Exception as exc:
        print(f"Plotting failed: {exc}")
    print(f"Saved focused reward-encoding analysis outputs to {outdir}")


if __name__ == "__main__":
    main()
