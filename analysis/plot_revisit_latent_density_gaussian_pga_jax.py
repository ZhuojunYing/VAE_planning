#!/usr/bin/env python
"""Gaussian-manifold PGA analysis for revisit-task JAX posterior states.

Example:
  python analysis/plot_revisit_latent_density_gaussian_pga_jax.py \
    --alphas 0.0 --betas 1000.0 --lambdas 100.0 \
    --opportunity-costs 0.0 --seeds 1 2 3 \
    --rnn-dims 32 --latent-dims 16 --tree-size 3 --tree-type bandit3 \
    --n-trials 2000 --n-components 2 --reduction-method both \
    --pga-fit-scope pooled --outdir analysis_outputs/revisit_latent_density_gaussian_pga

This script reuses the JAX model-loading, checkpoint selection, reward
sampling, task construction, and most plotting conventions from
analysis/plot_revisit_latent_density_jax.py. It does not change model training.

Replicated inputs:
  - revisit-enabled JAX checkpoints found with find_revisit_checkpoint()
  - model construction with build_model_and_params()
  - reward sampling with sample_rewards()
  - pre-stop observed-state filtering from the original rollout logic
  - task labels such as observed node/path, first observed path, rewards,
    timestep before stop, KL paid, and terminal-choice entropy

PGA analogue:
  A latent_dim=L diagonal Gaussian posterior is a point in a product of L
  univariate Gaussian manifolds, with 2L intrinsic parameter directions. Product
  Gaussian PGA computes a Frechet/Karcher mean in each univariate factor, maps
  states into a metric-orthonormal tangent space, then runs ordinary PCA there.
  Two PGA components are two scalar scores, not two new Gaussian factors.

Optional two-factor embedding:
  gaussian_product_mds fits two reduced univariate Gaussian factors per state by
  preserving product Fisher distances. This is a distance-preserving manifold
  embedding, not PCA/PGA, and is only used where explicit reduced Gaussian
  factors are useful for half-plane/disk style plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp

try:
    from analysis import plot_revisit_latent_density_jax as base
except ModuleNotFoundError:
    import plot_revisit_latent_density_jax as base

try:
    from model_jax import planning as jp
except ModuleNotFoundError:
    import planning as jp


SQRT2 = math.sqrt(2.0)
SIGMA_EPS = 1e-8
GEOM_EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument(
        "--scalers",
        "--betas",
        dest="beta_values",
        nargs="+",
        type=float,
        required=True,
        help="Reward/action/critic scalers; --betas is accepted as a legacy alias.",
    )
    parser.add_argument("--lambdas", dest="lambda_values", nargs="+", type=float, required=True)
    parser.add_argument("--opportunity-costs", "--opportunity-cost", dest="opportunity_costs", nargs="+", type=float, default=[0.0])
    parser.add_argument("--sigmas", "--observation-sigmas", dest="observation_sigmas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--rnn-dims", nargs="+", type=int, default=[32])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[16])
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", default="analysis_outputs/revisit_latent_density_gaussian_pga")
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--tree-size", type=int, default=2)
    parser.add_argument("--tree-type", default="default")
    parser.add_argument("--input-type", default="uniform", choices=["uniform", "binary"])
    parser.add_argument("--expansion-decision-version", default="lstm")
    parser.add_argument("--model-variant", default="vae", choices=["vae", "rnn"])
    parser.add_argument("--max-observations-before-stop", type=int, default=10)
    parser.add_argument("--analysis-seed-offset", type=int, default=300_000)
    parser.add_argument("--kl-start-multiplier", type=float, default=None)
    parser.add_argument("--kl-annealing-epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Accepted for CLI symmetry; plotting uses JAX default device.")
    parser.add_argument(
        "--min-density-samples",
        type=int,
        default=1,
        help="Minimum number of states required to draw each latent KDE contour.",
    )

    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--pga-fit-scope", choices=["pooled", "per_timestep"], default="pooled")
    parser.add_argument("--reduction-method", choices=["pga", "gaussian_product_mds", "both"], default="both")
    parser.add_argument("--max-states", type=int, default=8000)
    parser.add_argument("--max-pairs", type=int, default=25000)
    parser.add_argument("--train-pair-frac", type=float, default=0.8)
    parser.add_argument("--pga-tol", type=float, default=1e-7)
    parser.add_argument("--pga-max-iters", type=int, default=100)
    parser.add_argument("--embedding-restarts", type=int, default=3)
    parser.add_argument("--embedding-steps", type=int, default=1500)
    parser.add_argument("--embedding-lr", type=float, default=0.03)
    parser.add_argument("--embedding-grad-clip", type=float, default=10.0)
    parser.add_argument("--geometry-seed", "--seed", dest="geometry_seed", type=int, default=0)
    parser.add_argument("--run-synthetic-test", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_float_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf.")
    return arr


def validate_mu_sigma(mu: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = finite_float_array(mu, "mu")
    sigma = finite_float_array(sigma, "sigma")
    sigma = np.maximum(sigma, SIGMA_EPS)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive after clamping.")
    if mu.shape != sigma.shape:
        raise ValueError(f"mu and sigma shapes differ: {mu.shape} vs {sigma.shape}")
    if mu.ndim != 2:
        raise ValueError(f"mu and sigma must be [N, latent_dim], got {mu.shape}")
    return mu, sigma


def lorentz_dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1] + x[..., 2] * y[..., 2]


def halfplane_to_hyperboloid(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu, sigma = validate_mu_sigma(np.asarray(mu), np.asarray(sigma))
    x = mu / SQRT2
    y = np.maximum(sigma, SIGMA_EPS)
    x2_y2 = x * x + y * y
    x0 = (x2_y2 + 1.0) / (2.0 * y)
    x1 = x / y
    x2 = (x2_y2 - 1.0) / (2.0 * y)
    out = np.stack([x0, x1, x2], axis=-1)
    constraint = lorentz_dot(out, out)
    if not np.all(np.isfinite(out)) or np.nanmax(np.abs(constraint + 1.0)) > 1e-5:
        raise ValueError("Invalid hyperboloid conversion.")
    return out


def hyperboloid_to_halfplane(xh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xh = finite_float_array(xh, "hyperboloid")
    denom = np.maximum(xh[..., 0] - xh[..., 2], SIGMA_EPS)
    x = xh[..., 1] / denom
    y = 1.0 / denom
    return SQRT2 * x, np.maximum(y, SIGMA_EPS)


def normalize_hyperboloid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm2 = -lorentz_dot(x, x)
    if not np.isfinite(norm2) or norm2 <= 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    out = x / math.sqrt(norm2)
    if out[0] < 0:
        out = -out
    return out


def hyperbolic_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    alpha = np.maximum(-lorentz_dot(x, y), 1.0)
    return np.arccosh(alpha)


def gaussian_product_distance(mu_a: np.ndarray, sigma_a: np.ndarray, mu_b: np.ndarray, sigma_b: np.ndarray) -> np.ndarray:
    xa = halfplane_to_hyperboloid(mu_a, sigma_a)
    xb = halfplane_to_hyperboloid(mu_b, sigma_b)
    d_h = hyperbolic_distance(xa, xb)
    return np.sqrt(np.sum((SQRT2 * d_h) ** 2, axis=-1))


def hyperboloid_log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    alpha = np.maximum(-lorentz_dot(x, y), 1.0 + GEOM_EPS)
    denom = np.sqrt(np.maximum(alpha * alpha - 1.0, GEOM_EPS))
    coef = np.arccosh(alpha) / denom
    return coef[..., None] * (y - alpha[..., None] * x)


def hyperboloid_exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    norm_v = np.sqrt(np.maximum(lorentz_dot(v, v), 0.0))
    small = norm_v < 1e-10
    coef = np.empty_like(norm_v, dtype=np.float64)
    coef[small] = 1.0
    coef[~small] = np.sinh(norm_v[~small]) / norm_v[~small]
    out = np.cosh(norm_v)[..., None] * x + coef[..., None] * v
    flat = out.reshape((-1, 3))
    flat = np.asarray([normalize_hyperboloid(row) for row in flat])
    return flat.reshape(out.shape)


def karcher_mean_factor(x: np.ndarray, tol: float, max_iters: int) -> np.ndarray:
    extrinsic = np.mean(x, axis=0)
    mean = normalize_hyperboloid(extrinsic)
    if not np.isfinite(mean).all():
        mean = x[0].copy()
    for _ in range(int(max_iters)):
        logs = hyperboloid_log_map(mean[None, :], x)
        update = np.mean(logs, axis=0)
        update_norm = math.sqrt(max(float(lorentz_dot(update, update)), 0.0))
        if update_norm < tol:
            break
        mean = hyperboloid_exp_map(mean[None, :], update[None, :])[0]
    return mean


def tangent_basis_at(x: np.ndarray) -> np.ndarray:
    basis = []
    candidates = [
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    for cand in candidates:
        v = cand + lorentz_dot(x, cand) * x
        for b in basis:
            v = v - lorentz_dot(v, b) * b
        norm = math.sqrt(max(float(lorentz_dot(v, v)), 0.0))
        if norm > 1e-10:
            basis.append(v / norm)
        if len(basis) == 2:
            break
    if len(basis) != 2:
        raise ValueError("Could not construct Lorentz-orthonormal tangent basis.")
    out = np.stack(basis, axis=0)
    checks = [
        abs(float(lorentz_dot(x, out[0]))),
        abs(float(lorentz_dot(x, out[1]))),
        abs(float(lorentz_dot(out[0], out[1]))),
        abs(float(lorentz_dot(out[0], out[0])) - 1.0),
        abs(float(lorentz_dot(out[1], out[1])) - 1.0),
    ]
    if max(checks) > 1e-5:
        raise ValueError("Invalid tangent basis.")
    return out


@dataclass
class ProductGaussianPGA:
    n_components: int = 2
    tol: float = 1e-7
    max_iters: int = 100
    means_hyperboloid: Optional[np.ndarray] = None
    bases: Optional[np.ndarray] = None
    tangent_mean: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    singular_values: Optional[np.ndarray] = None

    def fit(self, mu: np.ndarray, sigma: np.ndarray) -> "ProductGaussianPGA":
        mu, sigma = validate_mu_sigma(mu, sigma)
        xh = halfplane_to_hyperboloid(mu, sigma)
        latent_dim = mu.shape[1]
        means = []
        bases = []
        tangent_parts = []
        for latent_i in range(latent_dim):
            mean_i = karcher_mean_factor(xh[:, latent_i, :], self.tol, self.max_iters)
            basis_i = tangent_basis_at(mean_i)
            logs = hyperboloid_log_map(mean_i[None, :], xh[:, latent_i, :])
            coords = np.stack(
                [
                    lorentz_dot(logs, basis_i[0]),
                    lorentz_dot(logs, basis_i[1]),
                ],
                axis=1,
            ) * SQRT2
            means.append(mean_i)
            bases.append(basis_i)
            tangent_parts.append(coords)
        tangent = np.concatenate(tangent_parts, axis=1)
        tangent_mean = tangent.mean(axis=0, keepdims=True)
        centered = tangent - tangent_mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        n = max(centered.shape[0] - 1, 1)
        eigenvalues = (s ** 2) / n
        total = float(np.sum(eigenvalues))
        self.means_hyperboloid = np.stack(means, axis=0)
        self.bases = np.stack(bases, axis=0)
        self.tangent_mean = tangent_mean[0]
        self.components = vt[: self.n_components]
        self.singular_values = s[: self.n_components]
        self.explained_variance = eigenvalues[: self.n_components]
        self.explained_variance_ratio = (
            eigenvalues[: self.n_components] / total if total > 0 else np.zeros(self.n_components)
        )
        return self

    def tangent_coordinates(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        mu, sigma = validate_mu_sigma(mu, sigma)
        if self.means_hyperboloid is None or self.bases is None:
            raise RuntimeError("ProductGaussianPGA is not fitted.")
        xh = halfplane_to_hyperboloid(mu, sigma)
        parts = []
        for latent_i in range(mu.shape[1]):
            logs = hyperboloid_log_map(self.means_hyperboloid[latent_i][None, :], xh[:, latent_i, :])
            basis_i = self.bases[latent_i]
            coords = np.stack(
                [lorentz_dot(logs, basis_i[0]), lorentz_dot(logs, basis_i[1])],
                axis=1,
            ) * SQRT2
            parts.append(coords)
        return np.concatenate(parts, axis=1)

    def transform(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        if self.components is None or self.tangent_mean is None:
            raise RuntimeError("ProductGaussianPGA is not fitted.")
        tangent = self.tangent_coordinates(mu, sigma)
        scores = (tangent - self.tangent_mean[None, :]) @ self.components.T
        if not np.all(np.isfinite(scores)):
            raise ValueError("PGA scores contain NaN or inf.")
        return scores

    def fit_transform(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        self.fit(mu, sigma)
        return self.transform(mu, sigma)

    def inverse_transform(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.components is None or self.tangent_mean is None or self.means_hyperboloid is None or self.bases is None:
            raise RuntimeError("ProductGaussianPGA is not fitted.")
        scores = np.asarray(scores, dtype=np.float64)
        tangent = scores @ self.components + self.tangent_mean[None, :]
        latent_dim = self.means_hyperboloid.shape[0]
        mu_parts = []
        sigma_parts = []
        for latent_i in range(latent_dim):
            coords = tangent[:, 2 * latent_i : 2 * latent_i + 2] / SQRT2
            basis_i = self.bases[latent_i]
            v = coords[:, 0, None] * basis_i[0][None, :] + coords[:, 1, None] * basis_i[1][None, :]
            rec_xh = hyperboloid_exp_map(self.means_hyperboloid[latent_i][None, :], v)
            rec_mu, rec_sigma = hyperboloid_to_halfplane(rec_xh)
            mu_parts.append(rec_mu)
            sigma_parts.append(rec_sigma)
        rec_mu = np.stack(mu_parts, axis=1)
        rec_sigma = np.stack(sigma_parts, axis=1)
        if np.any(rec_sigma <= 0) or not np.all(np.isfinite(rec_mu)) or not np.all(np.isfinite(rec_sigma)):
            raise ValueError("PGA inverse transform produced invalid posterior parameters.")
        return rec_mu, rec_sigma

    def reconstruction_metrics(self, mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator, max_pairs: int) -> Dict[str, float]:
        scores = self.transform(mu, sigma)
        rec_mu, rec_sigma = self.inverse_transform(scores)
        tangent = self.tangent_coordinates(mu, sigma)
        tangent_rec = (scores @ self.components) + self.tangent_mean[None, :]
        tangent_mse = float(np.mean((tangent - tangent_rec) ** 2))
        rec_err = gaussian_product_distance(mu, sigma, rec_mu, rec_sigma)
        pairs = sample_pair_indices(mu.shape[0], max_pairs, rng)
        pair_metrics = pairwise_preservation_metrics(mu, sigma, rec_mu, rec_sigma, pairs)
        return {
            "tangent_reconstruction_MSE": tangent_mse,
            "mean_product_Fisher_reconstruction_error": float(np.mean(rec_err)),
            "median_product_Fisher_reconstruction_error": float(np.median(rec_err)),
            **pair_metrics,
        }

    def save(self, path: Path) -> None:
        if self.components is None:
            raise RuntimeError("ProductGaussianPGA is not fitted.")
        np.savez_compressed(
            path,
            means_hyperboloid=self.means_hyperboloid,
            bases=self.bases,
            tangent_mean=self.tangent_mean,
            components=self.components,
            explained_variance=self.explained_variance,
            explained_variance_ratio=self.explained_variance_ratio,
            singular_values=self.singular_values,
            n_components=np.asarray(self.n_components),
        )

    @classmethod
    def load(cls, path: Path) -> "ProductGaussianPGA":
        data = np.load(path)
        obj = cls(n_components=int(np.asarray(data["n_components"])))
        obj.means_hyperboloid = data["means_hyperboloid"]
        obj.bases = data["bases"]
        obj.tangent_mean = data["tangent_mean"]
        obj.components = data["components"]
        obj.explained_variance = data["explained_variance"]
        obj.explained_variance_ratio = data["explained_variance_ratio"]
        obj.singular_values = data["singular_values"]
        return obj


def sample_pair_indices(n: int, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2 or max_pairs <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    total = n * (n - 1) // 2
    if total <= max_pairs and n <= 1500:
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
        return pairs
    i = rng.integers(0, n, size=max_pairs, endpoint=False)
    j = rng.integers(0, n - 1, size=max_pairs, endpoint=False)
    j = j + (j >= i)
    return np.stack([i, j], axis=1).astype(np.int64)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        stat = spearmanr(x, y, nan_policy="omit")
        return float(stat.correlation)
    except Exception:
        return float("nan")


def pair_distances(mu: np.ndarray, sigma: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.asarray([], dtype=np.float64)
    return gaussian_product_distance(mu[pairs[:, 0]], sigma[pairs[:, 0]], mu[pairs[:, 1]], sigma[pairs[:, 1]])


def pairwise_preservation_metrics(
    mu_orig: np.ndarray,
    sigma_orig: np.ndarray,
    mu_red: np.ndarray,
    sigma_red: np.ndarray,
    pairs: np.ndarray,
) -> Dict[str, float]:
    d0 = pair_distances(mu_orig, sigma_orig, pairs)
    d1 = pair_distances(mu_red, sigma_red, pairs)
    denom = float(np.sum(d0 ** 2))
    stress = float(math.sqrt(np.sum((d1 - d0) ** 2) / denom)) if denom > 0 else float("nan")
    return {
        "pairwise_distance_Pearson_r": pearson_corr(d0, d1),
        "pairwise_distance_Spearman_r": spearman_corr(d0, d1),
        "normalized_stress": stress,
        "n_pairs": int(len(d0)),
    }


def euclidean_pca_fit_transform(features: np.ndarray, n_components: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:n_components].T
    n = max(features.shape[0] - 1, 1)
    eigenvalues = (s ** 2) / n
    total = float(np.sum(eigenvalues))
    return scores, {
        "mean": mean[0],
        "components": vt[:n_components],
        "explained_variance": eigenvalues[:n_components],
        "explained_variance_ratio": eigenvalues[:n_components] / total if total > 0 else np.zeros(n_components),
        "singular_values": s[:n_components],
    }


def euclidean_reconstruction_to_gaussian(
    scores: np.ndarray,
    pca: Dict[str, np.ndarray],
    latent_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rec = scores @ pca["components"] + pca["mean"][None, :]
    rec_mu = rec[:, :latent_dim]
    rec_log_sigma = rec[:, latent_dim:]
    return rec_mu, np.maximum(np.exp(rec_log_sigma), SIGMA_EPS)


@dataclass
class GaussianProductMDSResult:
    reduced_mu: np.ndarray
    reduced_sigma: np.ndarray
    metrics: Dict[str, float]


def softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.logaddexp(x, 0.0)


def jax_halfplane_to_hyperboloid(mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    x = mu / SQRT2
    y = jnp.maximum(sigma, SIGMA_EPS)
    x2_y2 = x * x + y * y
    return jnp.stack(
        [
            (x2_y2 + 1.0) / (2.0 * y),
            x / y,
            (x2_y2 - 1.0) / (2.0 * y),
        ],
        axis=-1,
    )


def jax_lorentz_dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return -x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1] + x[..., 2] * y[..., 2]


def jax_product_distance(mu: jnp.ndarray, sigma: jnp.ndarray, pairs: jnp.ndarray) -> jnp.ndarray:
    x = jax_halfplane_to_hyperboloid(mu, sigma)
    xi = x[pairs[:, 0]]
    xj = x[pairs[:, 1]]
    alpha = jnp.maximum(-jax_lorentz_dot(xi, xj), 1.0)
    d = SQRT2 * jnp.arccosh(alpha)
    return jnp.sqrt(jnp.sum(d * d, axis=-1))


def fit_gaussian_product_mds(
    mu: np.ndarray,
    sigma: np.ndarray,
    pga_scores: Optional[np.ndarray],
    rng: np.random.Generator,
    *,
    max_pairs: int,
    train_pair_frac: float,
    restarts: int,
    steps: int,
    lr: float,
    grad_clip: float,
) -> GaussianProductMDSResult:
    mu, sigma = validate_mu_sigma(mu, sigma)
    n = mu.shape[0]
    pairs = sample_pair_indices(n, max_pairs, rng)
    if len(pairs) == 0:
        raise ValueError("Need at least two states for gaussian_product_mds.")
    perm = rng.permutation(len(pairs))
    train_n = max(1, int(round(len(pairs) * train_pair_frac)))
    train_pairs = pairs[perm[:train_n]]
    test_pairs = pairs[perm[train_n:]]
    if len(test_pairs) == 0:
        test_pairs = train_pairs.copy()
    train_targets = pair_distances(mu, sigma, train_pairs)
    test_targets = pair_distances(mu, sigma, test_pairs)
    train_pairs_j = jnp.asarray(train_pairs, dtype=jnp.int32)
    test_pairs_j = jnp.asarray(test_pairs, dtype=jnp.int32)
    train_targets_j = jnp.asarray(train_targets, dtype=jnp.float32)
    denom_train = jnp.maximum(jnp.mean(train_targets_j ** 2), 1e-8)

    if pga_scores is not None and pga_scores.shape[1] >= 2:
        init_mu = np.asarray(pga_scores[:, :2], dtype=np.float64)
        init_mu = (init_mu - init_mu.mean(axis=0, keepdims=True)) / np.maximum(init_mu.std(axis=0, keepdims=True), 1e-6)
    else:
        init_mu = rng.normal(size=(n, 2))
    init_rho = np.full((n, 2), math.log(math.exp(1.0 - SIGMA_EPS) - 1.0), dtype=np.float64)

    @jax.jit
    def loss_and_grad(mu_param, rho_param):
        def loss_fn(a, b):
            red_sigma = softplus(b) + SIGMA_EPS
            pred = jax_product_distance(a, red_sigma, train_pairs_j)
            return jnp.mean((pred - train_targets_j) ** 2) / denom_train

        value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(mu_param, rho_param)
        return value, grads

    best = None
    for restart in range(max(1, int(restarts))):
        jitter = 0.05 * rng.normal(size=init_mu.shape)
        mu_param = jnp.asarray(init_mu + jitter, dtype=jnp.float32)
        rho_param = jnp.asarray(init_rho + 0.05 * rng.normal(size=init_rho.shape), dtype=jnp.float32)
        best_restart_loss = float("inf")
        for step in range(max(1, int(steps))):
            loss_value, (g_mu, g_rho) = loss_and_grad(mu_param, rho_param)
            grad_norm = jnp.sqrt(jnp.sum(g_mu * g_mu) + jnp.sum(g_rho * g_rho))
            scale = jnp.minimum(1.0, grad_clip / jnp.maximum(grad_norm, 1e-8))
            step_lr = lr * 0.5 * (1.0 + math.cos(math.pi * step / max(int(steps), 1)))
            mu_param = mu_param - step_lr * scale * g_mu
            rho_param = rho_param - step_lr * scale * g_rho
            loss_float = float(loss_value)
            if loss_float + 1e-9 < best_restart_loss:
                best_restart_loss = loss_float
            if step > 50 and abs(loss_float - best_restart_loss) < 1e-10:
                pass
        red_mu = np.asarray(mu_param, dtype=np.float64)
        red_sigma = np.asarray(softplus(rho_param) + SIGMA_EPS, dtype=np.float64)
        test_pred = np.asarray(jax_product_distance(jnp.asarray(red_mu), jnp.asarray(red_sigma), test_pairs_j))
        train_pred = np.asarray(jax_product_distance(jnp.asarray(red_mu), jnp.asarray(red_sigma), train_pairs_j))
        train_stress = math.sqrt(float(np.sum((train_pred - train_targets) ** 2) / max(np.sum(train_targets ** 2), 1e-8)))
        test_stress = math.sqrt(float(np.sum((test_pred - test_targets) ** 2) / max(np.sum(test_targets ** 2), 1e-8)))
        metrics = {
            "train_stress": train_stress,
            "test_stress": test_stress,
            "Pearson_r_test": pearson_corr(test_targets, test_pred),
            "Spearman_r_test": spearman_corr(test_targets, test_pred),
            "optimization_restart": int(restart),
            "optimization_steps": int(steps),
            "status": "ok",
            "n_train_pairs": int(len(train_pairs)),
            "n_test_pairs": int(len(test_pairs)),
        }
        if best is None or test_stress < best.metrics["test_stress"]:
            best = GaussianProductMDSResult(red_mu, red_sigma, metrics)
    if best is None:
        raise RuntimeError("gaussian_product_mds failed to produce a result.")
    return best


def rollout_revisit_posterior_states(
    *,
    model: jp.PlanningVAE,
    params,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    batch_size: int,
    seed: int,
    beta: float,
    max_observations_before_stop: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    mu_rows = []
    logvar_rows = []
    sigma_rows = []
    prior_mu_rows = []
    prior_logvar_rows = []
    prior_sigma_rows = []
    path_map = np.asarray(task.path_map, dtype=np.float64)
    node_to_path = base.node_to_path_indices(path_map)
    reward_feature_dim = jp.reward_feature_dim_for_sigma(model.observation_sigma)
    num_steps = int(max_observations_before_stop) + 1
    schedule = jp.ScheduleValues(
        current_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        current_beta=jnp.asarray(1.0 / float(beta), dtype=jnp.float32),
        current_critic_coef=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        expansion_entropy_coef=jnp.asarray(0.0, dtype=jnp.float32),
        forced_continue_epsilon=jnp.asarray(0.0, dtype=jnp.float32),
        ppo_clip=jnp.asarray(0.3, dtype=jnp.float32),
    )
    has_two_one_node_paths = task.num_nodes == 2 and task.num_paths == 2
    for batch_i, start in enumerate(range(0, rewards.shape[0], batch_size)):
        batch_rewards = rewards[start : start + batch_size]
        path_rewards = batch_rewards @ path_map.T
        path_reward_sums = np.sum(path_rewards, axis=1)
        carry = jp.initial_carry(batch_rewards.shape[0], task, model.rnn_units, reward_feature_dim)
        carry = jp.reset_done_envs(carry, jnp.asarray(batch_rewards, dtype=jnp.float32))
        rng_key = jax.random.PRNGKey(seed + batch_i)
        stopped = np.zeros(batch_rewards.shape[0], dtype=bool)
        stop_decision_timestep = np.full(batch_rewards.shape[0], num_steps, dtype=int)
        first_observed_node = np.full(batch_rewards.shape[0], -1, dtype=int)
        first_observed_path = np.full(batch_rewards.shape[0], -1, dtype=int)
        path_visit_order = np.full((batch_rewards.shape[0], task.num_paths), -1, dtype=int)
        next_path_visit_order = np.ones(batch_rewards.shape[0], dtype=int)
        paid_kl_by_step = np.full((batch_rewards.shape[0], num_steps + 2), np.nan, dtype=float)
        batch_row_indices = []
        for step_i in range(num_steps):
            timestep = int(step_i + 1)
            rng_key, step_rng = jax.random.split(rng_key)
            carry, trans = model.apply(
                {"params": params},
                carry,
                step_rng,
                schedule,
                None,
                True,
                False,
                False,
                method=jp.PlanningVAE.__call__,
            )
            trans = jax.device_get(trans)
            node_index = np.asarray(trans.node_index, dtype=int)
            is_observe = np.asarray(trans.is_observe, dtype=float) > 0.5
            is_stop = np.asarray(trans.is_stop, dtype=float) > 0.5
            sampled_observed_reward = np.asarray(trans.expanded_reward, dtype=float)
            z_mu = np.asarray(trans.z_mu, dtype=np.float64)
            z_logvar = np.asarray(trans.z_logvar, dtype=np.float64)
            z_sigma = np.exp(0.5 * np.clip(z_logvar, -30.0, 30.0))
            prior_mu = np.asarray(trans.prior_mu, dtype=np.float64)
            prior_logvar = np.asarray(trans.prior_logvar, dtype=np.float64)
            prior_sigma = np.exp(0.5 * np.clip(prior_logvar, -30.0, 30.0))
            paid_kl = np.asarray(trans.paid_kl, dtype=float)
            observed_kl = np.asarray(trans.observed_kl, dtype=float)
            paid_kl_by_step[:, timestep] = paid_kl
            action_output = np.asarray(trans.action_output, dtype=float)
            prob_sums = np.nansum(np.where(np.isfinite(action_output), action_output, 0.0), axis=1)
            terminal_entropy = np.full(batch_rewards.shape[0], np.nan, dtype=float)
            valid_probs = np.isfinite(prob_sums) & (prob_sums > 0)
            if np.any(valid_probs):
                probs = np.zeros_like(action_output, dtype=float)
                probs[valid_probs] = np.where(np.isfinite(action_output[valid_probs]), action_output[valid_probs], 0.0) / prob_sums[valid_probs, None]
                terminal_entropy[valid_probs] = -np.sum(
                    np.where(probs[valid_probs] > 0, probs[valid_probs] * np.log(probs[valid_probs] + 1e-12), 0.0),
                    axis=1,
                )
            include = (~stopped) & is_observe & (node_index >= 0)
            for local_i in np.where(include)[0]:
                node_i = int(node_index[local_i])
                if node_i < 0 or node_i >= len(node_to_path):
                    continue
                path_i = int(node_to_path[node_i])
                if path_i < 0:
                    continue
                if first_observed_node[local_i] < 0:
                    first_observed_node[local_i] = node_i
                if first_observed_path[local_i] < 0:
                    first_observed_path[local_i] = path_i
                if path_visit_order[local_i, path_i] < 0:
                    path_visit_order[local_i, path_i] = next_path_visit_order[local_i]
                    next_path_visit_order[local_i] += 1
                observed_path_order = int(path_visit_order[local_i, path_i])
                first_path_i = int(first_observed_path[local_i])
                node_reward = float(batch_rewards[local_i, node_i])
                observed_path_reward = float(path_rewards[local_i, path_i])
                first_path_reward = float(path_rewards[local_i, first_path_i])
                mean_other = (float(path_reward_sums[local_i] - first_path_reward) / float(task.num_paths - 1)) if task.num_paths > 1 else np.nan
                observed_mean_other = (float(path_reward_sums[local_i] - observed_path_reward) / float(task.num_paths - 1)) if task.num_paths > 1 else np.nan
                row = {
                    "state_id": len(rows),
                    "trial_id": int(start + local_i),
                    "timestep": int(step_i + 1),
                    "observed_node": int(node_i + 1),
                    "observed_path": int(path_i + 1),
                    "observed_path_order": observed_path_order,
                    "observed_path_order_label": base.observed_path_order_label(observed_path_order, unit="path"),
                    "first_observed_node": int(first_observed_node[local_i] + 1),
                    "first_observed_path": int(first_path_i + 1),
                    "actual_node_reward": node_reward,
                    "node_reward": node_reward,
                    "observed_path_actual_reward": observed_path_reward,
                    "first_observed_path_actual_reward_raw": first_path_reward,
                    "mean_other_path_actual_reward_raw": mean_other,
                    "first_observed_path_actual_reward": first_path_reward,
                    "mean_other_path_actual_reward": base.default_mean_other_path_value(mean_other, task),
                    "first_observed_path_actual_reward_integer": base.bin_nearest_integer_value(first_path_reward),
                    "mean_other_path_actual_reward_integer": base.bin_nearest_integer_value(mean_other),
                    "first_observed_path_actual_reward_bin2": base.bin_width_two_away_from_zero_value(first_path_reward),
                    "mean_other_path_actual_reward_bin2": base.bin_width_two_away_from_zero_value(mean_other),
                    "observed_path_actual_reward_raw": observed_path_reward,
                    "mean_other_observed_path_actual_reward_raw": observed_mean_other,
                    "observed_path_actual_reward_integer": base.bin_nearest_integer_value(observed_path_reward),
                    "mean_other_observed_path_actual_reward_integer": base.bin_nearest_integer_value(observed_mean_other),
                    "observed_path_actual_reward_bin2": base.bin_width_two_away_from_zero_value(observed_path_reward),
                    "mean_other_observed_path_actual_reward_bin2": base.bin_width_two_away_from_zero_value(observed_mean_other),
                    "mean_other_observed_path_actual_reward": base.default_mean_other_path_value(observed_mean_other, task),
                    "sampled_observed_reward": float(sampled_observed_reward[local_i]),
                    "kl_paid_current_action": float(paid_kl[local_i]),
                    "observed_kl_at_timestep": float(observed_kl[local_i]),
                    "kl_paid_at_timestep": np.nan,
                    "terminal_choice_entropy_at_timestep": float(terminal_entropy[local_i]),
                }
                if has_two_one_node_paths:
                    other_i = 1 - node_i
                    other_reward = float(batch_rewards[local_i, other_i])
                    visit_order = "first_observed" if observed_path_order == 1 else "second_observed"
                    row.update(
                        {
                            "node_visit_order": visit_order,
                            "node_role": "better" if node_reward > other_reward else "worse",
                            "node1_actual_reward": float(batch_rewards[local_i, 0]),
                            "node2_actual_reward": float(batch_rewards[local_i, 1]),
                            "first_observed_actual_reward": node_reward if visit_order == "first_observed" else other_reward,
                            "second_observed_actual_reward": other_reward if visit_order == "first_observed" else node_reward,
                            "first_observed_minus_second_actual_reward": (node_reward - other_reward) if visit_order == "first_observed" else (other_reward - node_reward),
                            "other_node_reward": other_reward,
                        }
                    )
                rows.append(row)
                batch_row_indices.append((len(rows) - 1, local_i))
                mu_rows.append(z_mu[local_i])
                logvar_rows.append(z_logvar[local_i])
                sigma_rows.append(np.maximum(z_sigma[local_i], SIGMA_EPS))
                prior_mu_rows.append(prior_mu[local_i])
                prior_logvar_rows.append(prior_logvar[local_i])
                prior_sigma_rows.append(np.maximum(prior_sigma[local_i], SIGMA_EPS))
            new_stop = (~stopped) & is_stop
            stop_decision_timestep[new_stop] = int(step_i + 1)
            stopped |= is_stop
        for row_i, local_i in batch_row_indices:
            timestep_before_stop = int(max(stop_decision_timestep[local_i] - 1, 0))
            timestep = int(rows[row_i]["timestep"])
            next_timestep = timestep + 1
            next_paid_kl = (
                float(paid_kl_by_step[local_i, next_timestep])
                if next_timestep < paid_kl_by_step.shape[1]
                else np.nan
            )
            rows[row_i]["timestep_before_stop"] = timestep_before_stop
            rows[row_i]["continued_after_observation"] = int(timestep < timestep_before_stop)
            rows[row_i]["kl_paid_next_observation"] = next_paid_kl
            # Match the R revisit diagnostic convention: timestep t is paired
            # with kl_d_t{t+1}, i.e. the KL paid when the posterior generated
            # after this observation is carried into the next observation.
            rows[row_i]["kl_paid_at_timestep"] = next_paid_kl
    if len(rows) == 0:
        empty = np.zeros((0, model.latent_dim))
        return pd.DataFrame(), empty, empty, empty, empty, empty, empty
    keep = np.asarray(
        [int(row.get("continued_after_observation", 0)) == 1 for row in rows],
        dtype=bool,
    )
    if not np.any(keep):
        empty = np.zeros((0, model.latent_dim))
        return pd.DataFrame(), empty, empty, empty, empty, empty, empty
    rows = [row for row, keep_row in zip(rows, keep) if keep_row]
    mu_array = np.asarray(mu_rows, dtype=np.float64)[keep]
    logvar_array = np.asarray(logvar_rows, dtype=np.float64)[keep]
    sigma_array = np.asarray(sigma_rows, dtype=np.float64)[keep]
    prior_mu_array = np.asarray(prior_mu_rows, dtype=np.float64)[keep]
    prior_logvar_array = np.asarray(prior_logvar_rows, dtype=np.float64)[keep]
    prior_sigma_array = np.asarray(prior_sigma_rows, dtype=np.float64)[keep]
    return (
        pd.DataFrame(rows),
        mu_array,
        logvar_array,
        sigma_array,
        prior_mu_array,
        prior_logvar_array,
        prior_sigma_array,
    )


def subsample_states(metadata: pd.DataFrame, arrays: Tuple[np.ndarray, np.ndarray, np.ndarray], max_states: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if max_states <= 0 or len(metadata) <= max_states:
        return metadata.reset_index(drop=True), arrays
    idx = np.sort(rng.choice(len(metadata), size=max_states, replace=False))
    meta = metadata.iloc[idx].copy().reset_index(drop=True)
    meta["state_id"] = np.arange(len(meta))
    return meta, tuple(arr[idx] for arr in arrays)


def scope_iter(metadata: pd.DataFrame, scope: str) -> Iterable[Tuple[str, np.ndarray]]:
    if scope == "pooled":
        yield "pooled", np.arange(len(metadata), dtype=np.int64)
    else:
        for timestep in sorted(pd.to_numeric(metadata["timestep"], errors="coerce").dropna().unique()):
            idx = np.where(np.isclose(pd.to_numeric(metadata["timestep"], errors="coerce"), timestep))[0]
            if len(idx) >= 3:
                yield f"timestep_{int(timestep)}", idx


def combo_output_dir(
    outdir: Path,
    *,
    seed: int,
    beta: float,
    opportunity: float,
    lambda_value: float,
    sigma: float,
    rnn_dim: int,
    latent_dim: int,
    tree_type: str,
) -> Path:
    return outdir / (
        f"seed_{base.file_token(seed)}_beta_{base.file_token(beta)}_opp_{base.file_token(opportunity)}_"
        f"lambda_{base.file_token(lambda_value)}_sigma_{base.file_token(sigma)}_"
        f"rnn_{base.file_token(rnn_dim)}_latent_{base.file_token(latent_dim)}_tree_{base.file_token(tree_type)}"
    )


def add_score_columns(metadata: pd.DataFrame, scores: np.ndarray, prefix: str) -> pd.DataFrame:
    out = metadata.copy()
    for comp_i in range(scores.shape[1]):
        out[f"{prefix}_score_{comp_i}"] = scores[:, comp_i]
    return out


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_summary_text(path: Path, rows: List[str]) -> None:
    path.write_text("\n".join(rows) + "\n")


def plot_scatter(
    df: pd.DataFrame,
    outpath: Path,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    color_label: str,
) -> None:
    if df.empty or not {x_col, y_col, color_col}.issubset(df.columns):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.configure_plot_text(plt)

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    c = pd.to_numeric(df[color_col], errors="coerce")
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not keep.any():
        return
    fig, ax = plt.subplots(figsize=base.single_panel_figsize(colorbar=True))
    sc = ax.scatter(x[keep], y[keep], c=c[keep], s=7, alpha=0.45, cmap="viridis", linewidths=0)
    ax.set_xlabel(x_col, fontsize=7)
    ax.set_ylabel(y_col, fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(color_label, fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pga_score_reward_grid_distributions(
    df: pd.DataFrame,
    figdir: Path,
    *,
    combo_label: str,
    scope_suffix: str,
    grid_n: int = 120,
    max_density_points: int = 1500,
    min_density_samples: int = 1,
) -> None:
    required = {
        "pga_score_0",
        "pga_score_1",
        "observed_path_actual_reward",
        "mean_other_observed_path_actual_reward",
        "timestep",
        "observed_node",
        "observed_path",
    }
    if df.empty or not required.issubset(df.columns):
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.configure_plot_text(plt)
    from matplotlib.colors import Normalize

    plot_df = df.copy()
    plot_df["z_mu_0"] = pd.to_numeric(plot_df["pga_score_0"], errors="coerce")
    plot_df["z_mu_1"] = pd.to_numeric(plot_df["pga_score_1"], errors="coerce")
    reward_y_col = "observed_path_actual_reward"
    reward_x_col = "mean_other_observed_path_actual_reward"
    reward_y_label = "R(observed path)"
    reward_x_label = "Mean R(other paths)"
    for col in [
        reward_y_col,
        reward_x_col,
        "timestep",
        "observed_node",
        "observed_path",
    ]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(
        subset=[
            "z_mu_0",
            "z_mu_1",
            reward_y_col,
            reward_x_col,
            "timestep",
            "observed_node",
            "observed_path",
        ]
    )
    if plot_df.empty:
        return

    reward_y_values = base.coordinate_axis_values(plot_df, reward_y_col)
    reward_x_values = base.coordinate_axis_values(plot_df, reward_x_col)
    limits = base.axis_limits(plot_df)
    if reward_y_values is None or reward_x_values is None or limits is None:
        return
    xlim, ylim = base.square_axis_limits(*limits)
    x_grid = np.linspace(xlim[0], xlim[1], max(40, min(int(grid_n), 250)))
    y_grid = np.linspace(ylim[0], ylim[1], max(40, min(int(grid_n), 250)))

    color_values = sorted(plot_df["timestep"].dropna().unique())
    if len(color_values) == 0:
        return
    color_min = float(min(color_values))
    color_max = float(max(color_values))
    if math.isclose(color_min, color_max):
        color_min -= 0.5
        color_max += 0.5
    color_norm = Normalize(vmin=color_min, vmax=color_max)
    color_cmap = plt.get_cmap("plasma")

    scope_part = scope_suffix if scope_suffix else ""
    figdir.mkdir(parents=True, exist_ok=True)
    for old_plot in figdir.glob(f"revisit_pga_scores_density_by_node1_node2_reward_*{scope_part}.png"):
        old_plot.unlink()
    for old_plot in figdir.glob(f"revisit_pga_scores_density_by_first_observed_and_mean_other_path_*{scope_part}.png"):
        old_plot.unlink()
    for old_plot in figdir.glob(f"revisit_pga_scores_density_by_observed_and_mean_other_path_*{scope_part}.png"):
        old_plot.unlink()
    split_col, order_unit = base.observed_index_split(plot_df)
    split_values = sorted(
        int(v)
        for v in pd.to_numeric(plot_df[split_col], errors="coerce").dropna().unique()
        if int(v) >= 1
    )
    for split_value in split_values:
        split_df = plot_df[np.isclose(plot_df[split_col], split_value)].copy()
        if split_df.empty:
            continue
        split_stub = f"observed_{order_unit}_{split_value}"
        split_label = f"observed {order_unit} {split_value}"

        n_rows = len(reward_y_values)
        n_cols = len(reward_x_values)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=base.reward_density_grid_figsize(n_rows, n_cols),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for row_i, y_value in enumerate(reward_y_values):
            for col_i, x_value in enumerate(reward_x_values):
                ax = axes[row_i, col_i]
                reward_df = split_df[
                    np.isclose(split_df[reward_y_col], y_value)
                    & np.isclose(split_df[reward_x_col], x_value)
                ].copy()
                if reward_df.empty:
                    ax.set_axis_off()
                    continue
                for timestep in color_values:
                    timestep_df = reward_df[
                        np.isclose(reward_df["timestep"], timestep)
                    ].copy()
                    if timestep_df.empty:
                        continue
                    color = color_cmap(color_norm(float(timestep)))
                    density = base.empirical_mu_kde_density(
                        timestep_df,
                        x_grid,
                        y_grid,
                        max_points=max_density_points,
                        seed=base.value_contour_seed(
                            timestep,
                            y_value * 100.0 + x_value,
                            f"{split_stub}_pga_scores",
                        ),
                        min_samples=min_density_samples,
                    )
                    if density is not None:
                        levels = base.positive_contour_levels(density)
                        if levels is not None:
                            ax.contour(
                                x_grid,
                                y_grid,
                                density,
                                levels=levels,
                                colors=[color],
                                linewidths=0.65,
                                alpha=0.85,
                            )
                    z0 = pd.to_numeric(timestep_df["pga_score_0"], errors="coerce")
                    z1 = pd.to_numeric(timestep_df["pga_score_1"], errors="coerce")
                    finite = np.isfinite(z0) & np.isfinite(z1)
                    if int(np.sum(finite)) >= max(1, int(min_density_samples)):
                        ax.scatter(
                            float(z0[finite].mean()),
                            float(z1[finite].mean()),
                            c=[color],
                            s=12,
                            alpha=0.95,
                            edgecolors="black",
                            linewidths=0.25,
                        )
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=4, length=1.5, pad=1)
                if row_i == 0:
                    ax.set_title(f"Mean other={x_value:g}", fontsize=5, pad=2)
                if col_i == 0:
                    ax.set_ylabel(f"Current={y_value:g}", fontsize=5)
                else:
                    ax.set_ylabel("")
                if row_i == n_rows - 1:
                    ax.set_xlabel("PGA score 0", fontsize=5)
                else:
                    ax.set_xlabel("")

        sm = plt.cm.ScalarMappable(norm=color_norm, cmap=color_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
        cbar.set_label("timestep", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        fig.suptitle(
            f"{combo_label}\n{split_label}; rows={reward_y_label}, columns={reward_x_label}",
            fontsize=7,
            y=0.995,
        )
        fig.supxlabel("PGA score 0", fontsize=7, y=0.02)
        fig.supylabel("PGA score 1", fontsize=7, x=0.01)
        out_name = (
            "revisit_pga_scores_density_by_observed_and_mean_other_path_"
            f"{split_stub}{scope_part}.png"
        )
        fig.savefig(figdir / out_name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_component_loadings(pga: ProductGaussianPGA, outpath: Path) -> None:
    if pga.components is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.configure_plot_text(plt)

    components = np.asarray(pga.components)
    n_components, tangent_dim = components.shape
    latent_dim = tangent_dim // 2
    fig, axes = plt.subplots(
        n_components,
        1,
        figsize=base.stacked_panel_figsize(n_components, colorbar=False),
        squeeze=False,
    )
    x = np.arange(latent_dim)
    width = 0.38
    for comp_i in range(n_components):
        ax = axes[comp_i, 0]
        ax.bar(x - width / 2, components[comp_i, 0::2], width=width, label="mean tangent")
        ax.bar(x + width / 2, components[comp_i, 1::2], width=width, label="sigma tangent")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"component {comp_i}", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(latent_dim)], fontsize=7)
        if comp_i == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("original latent Gaussian factor", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(ratio: np.ndarray, outpath: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.configure_plot_text(plt)

    ratio = np.asarray(ratio, dtype=float)
    fig, ax = plt.subplots(figsize=base.single_panel_figsize())
    ax.bar(np.arange(len(ratio)), ratio)
    ax.plot(np.arange(len(ratio)), np.cumsum(ratio), color="black", marker="o", linewidth=0.8, markersize=3)
    ax.set_xlabel("component", fontsize=7)
    ax.set_ylabel("explained variance ratio", fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distance_comparison(d0: np.ndarray, d1: np.ndarray, outpath: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.configure_plot_text(plt)

    d0 = np.asarray(d0, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    keep = np.isfinite(d0) & np.isfinite(d1)
    if keep.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=base.single_panel_figsize())
    ax.scatter(d0[keep], d1[keep], s=5, alpha=0.25, linewidths=0)
    lim = [0, max(float(np.nanmax(d0[keep])), float(np.nanmax(d1[keep])))]
    ax.plot(lim, lim, color="black", linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("original product Fisher distance", fontsize=7)
    ax.set_ylabel("reduced/reconstructed distance", fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def disk_coords_from_mu_sigma(mu: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xh = halfplane_to_hyperboloid(mu, sigma)
    denom = np.maximum(xh[..., 0] + 1.0, SIGMA_EPS)
    return xh[..., 1] / denom, xh[..., 2] / denom


def plot_mds_factor_geometry(df: pd.DataFrame, outdir: Path, label_prefix: str = "gaussian_product_mds") -> None:
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base.configure_plot_text(plt)

    color_col = "actual_node_reward" if "actual_node_reward" in df.columns else "timestep"
    c = pd.to_numeric(df[color_col], errors="coerce")
    for factor_i in range(2):
        mu_col = f"reduced_mu_{factor_i}"
        sigma_col = f"reduced_sigma_{factor_i}"
        if not {mu_col, sigma_col}.issubset(df.columns):
            continue
        mu = pd.to_numeric(df[mu_col], errors="coerce").to_numpy(dtype=float)
        sigma = pd.to_numeric(df[sigma_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(c)
        if not keep.any():
            continue
        fig, ax = plt.subplots(figsize=base.single_panel_figsize(colorbar=True))
        sc = ax.scatter(mu[keep] / SQRT2, sigma[keep], c=c[keep], s=7, alpha=0.45, cmap="viridis", linewidths=0)
        ax.set_xlabel("x = reduced_mu / sqrt(2)", fontsize=7)
        ax.set_ylabel("y = reduced_sigma", fontsize=7)
        ax.set_title(f"{label_prefix} factor {factor_i} half-plane", fontsize=7)
        ax.tick_params(labelsize=7)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label(color_col, fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"{label_prefix}_factor{factor_i}_halfplane.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        disk_x, disk_y = disk_coords_from_mu_sigma(mu[:, None], sigma[:, None])
        fig, ax = plt.subplots(figsize=base.single_panel_figsize(colorbar=True))
        circle = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=0.8)
        ax.add_patch(circle)
        sc = ax.scatter(disk_x[:, 0][keep], disk_y[:, 0][keep], c=c[keep], s=7, alpha=0.45, cmap="viridis", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Poincare disk x", fontsize=7)
        ax.set_ylabel("Poincare disk y", fontsize=7)
        ax.set_title(f"{label_prefix} factor {factor_i} disk", fontsize=7)
        ax.tick_params(labelsize=7)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label(color_col, fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"{label_prefix}_factor{factor_i}_disk.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def task_decoding_metrics(df: pd.DataFrame, score_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
    rows = []
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import balanced_accuracy_score, r2_score
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return pd.DataFrame([{"status": "skipped_missing_sklearn", "error": str(exc)}])

    x = df[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
        keep = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        if keep.sum() < 10:
            continue
        xv = x[keep]
        yv = y[keep]
        unique = np.unique(yv)
        if len(unique) <= 1:
            continue
        if len(unique) <= min(12, max(2, len(yv) // 5)) and np.all(np.isclose(unique, np.round(unique))):
            labels, encoded = np.unique(yv, return_inverse=True)
            min_class_n = int(np.min(np.bincount(encoded)))
            n_splits = min(5, min_class_n)
            if n_splits < 2:
                continue
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            pred = np.zeros_like(yv)
            pred_encoded = np.zeros_like(encoded)
            for train_idx, test_idx in cv.split(xv, encoded):
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="auto"))
                model.fit(xv[train_idx], encoded[train_idx])
                pred_encoded[test_idx] = model.predict(xv[test_idx])
            rows.append(
                {
                    "target": target_col,
                    "metric": "balanced_accuracy",
                    "value": float(balanced_accuracy_score(encoded, pred_encoded)),
                    "n": int(len(yv)),
                    "status": "descriptive_cv",
                }
            )
        else:
            cv = KFold(n_splits=min(5, len(yv)), shuffle=True, random_state=0)
            pred = np.zeros_like(yv, dtype=float)
            for train_idx, test_idx in cv.split(xv):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(xv[train_idx], yv[train_idx])
                pred[test_idx] = model.predict(xv[test_idx])
            rows.append(
                {
                    "target": target_col,
                    "metric": "r2",
                    "value": float(r2_score(yv, pred)),
                    "n": int(len(yv)),
                    "status": "descriptive_cv",
                }
            )
    return pd.DataFrame(rows)


def run_geometry_self_tests() -> Dict[str, str]:
    rng = np.random.default_rng(123)
    mu = rng.normal(size=(12, 4))
    sigma = np.exp(rng.normal(scale=0.2, size=(12, 4)))
    xh = halfplane_to_hyperboloid(mu, sigma)
    constraint_err = float(np.max(np.abs(lorentz_dot(xh, xh) + 1.0)))
    mu2, sigma2 = hyperboloid_to_halfplane(xh)
    roundtrip_err = float(max(np.max(np.abs(mu - mu2)), np.max(np.abs(sigma - sigma2))))
    d_self = gaussian_product_distance(mu, sigma, mu, sigma)
    d_pair = gaussian_product_distance(mu[:5], sigma[:5], mu[5:10], sigma[5:10])
    log_self = hyperboloid_log_map(xh[0, 0][None, :], xh[0, 0][None, :])
    y = xh[1, 0][None, :]
    rec_y = hyperboloid_exp_map(xh[0, 0][None, :], hyperboloid_log_map(xh[0, 0][None, :], y))
    basis = tangent_basis_at(xh[0, 0])
    pga = ProductGaussianPGA(n_components=2, max_iters=20).fit(mu, sigma)
    scores = pga.transform(mu, sigma)
    rec_mu, rec_sigma = pga.inverse_transform(scores)
    checks = {
        "sigma_positive": "ok" if np.all(sigma > 0) else "fail",
        "hyperboloid_constraint": "ok" if constraint_err < 1e-6 else f"fail:{constraint_err}",
        "roundtrip": "ok" if roundtrip_err < 1e-6 else f"fail:{roundtrip_err}",
        "distance_symmetry": "ok" if np.allclose(d_pair, gaussian_product_distance(mu[5:10], sigma[5:10], mu[:5], sigma[:5])) else "fail",
        "zero_self_distance": "ok" if np.max(np.abs(d_self)) < 1e-6 else "fail",
        "nonnegative_distances": "ok" if np.all(d_pair >= -1e-9) else "fail",
        "log_self_zero": "ok" if np.max(np.abs(log_self)) < 1e-5 else "fail",
        "exp_log_roundtrip": "ok" if np.max(np.abs(rec_y - y)) < 1e-5 else "fail",
        "basis_orthonormal": "ok" if abs(lorentz_dot(basis[0], basis[1])) < 1e-6 else "fail",
        "pga_scores_finite": "ok" if np.all(np.isfinite(scores)) else "fail",
        "inverse_positive_sigma": "ok" if np.all(rec_sigma > 0) and np.all(np.isfinite(rec_mu)) else "fail",
    }
    return checks


def synthetic_pga_test(outdir: Path) -> Dict[str, float]:
    rng = np.random.default_rng(321)
    n = 400
    latent_dim = 16
    t = rng.normal(size=(n, 2))
    mu = np.zeros((n, latent_dim))
    log_sigma = np.zeros((n, latent_dim))
    for l in range(latent_dim):
        mu[:, l] = 0.3 * math.cos(l) * t[:, 0] + 0.2 * math.sin(l) * t[:, 1]
        log_sigma[:, l] = 0.15 * math.sin(l + 1) * t[:, 0] - 0.1 * math.cos(l + 2) * t[:, 1]
    sigma = np.exp(log_sigma)
    pga = ProductGaussianPGA(n_components=2, max_iters=50).fit(mu, sigma)
    metrics = pga.reconstruction_metrics(mu, sigma, rng, 5000)
    metrics["synthetic_cumulative_explained_variance_ratio"] = float(np.sum(pga.explained_variance_ratio))
    save_json(outdir / "synthetic_pga_test.json", metrics)
    return metrics


def build_manifest(combo_dir: Path, methods: List[str]) -> pd.DataFrame:
    rows = []
    original_outputs = [
        (
            "latent_z0_z1_density_by_observed_and_mean_other_path",
            "revisit_latent_z0_z1_density_by_observed_and_mean_other_path_*.png",
        ),
        (
            "latent_reward_grid_kl",
            "revisit_latent_reward_grid_kl_paid_at_timestep_by_observed_and_mean_other_path_*.png",
        ),
        (
            "latent_reward_grid_entropy",
            "revisit_latent_reward_grid_terminal_choice_entropy_by_observed_and_mean_other_path_*.png",
        ),
        ("path_context_heatmaps", "revisit_path_context_grid_*.png"),
    ]
    for name, pattern in original_outputs:
        rows.append(
            {
                "original_analysis_name": name,
                "original_output_filename": pattern,
                "new_representation": "pga",
                "new_output_filename": "pga/figures/pga_*.png",
                "status": "analogue_2d_scores" if "pga" in methods else "not_requested",
                "notes": "PGA scores are scalar geodesic coordinates, not Gaussian factors.",
            }
        )
        rows.append(
            {
                "original_analysis_name": name,
                "original_output_filename": pattern,
                "new_representation": "gaussian_product_mds",
                "new_output_filename": "gaussian_product_mds/figures/gaussian_product_mds_*.png",
                "status": "explicit_two_factor_embedding" if "gaussian_product_mds" in methods else "not_requested",
                "notes": "Two fitted Gaussian factors are distance-preserving MDS coordinates, not PGA.",
            }
        )
    rows.append(
        {
            "original_analysis_name": "prior_centered_gaussian_factor_plots",
            "original_output_filename": "prior-centered factor plots",
            "new_representation": "gaussian_product_mds",
            "new_output_filename": "",
            "status": "not_applicable_no_reduced_prior",
            "notes": "The fitted two-factor embedding has no learned reduced prior; no prior parameters are fabricated.",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(combo_dir / "analysis_reproduction_manifest.csv", index=False)
    return out


def method_list(reduction_method: str) -> List[str]:
    if reduction_method == "both":
        return ["pga", "gaussian_product_mds"]
    return [reduction_method]


def run_combo(
    args: argparse.Namespace,
    *,
    lambda_value: float,
    alpha: float,
    beta: float,
    opportunity: float,
    sigma: float,
    seed: int,
    rnn_dim: int,
    latent_dim: int,
    outdir: Path,
    checkpoint_root: Path,
    failures: List[dict],
) -> Optional[dict]:
    task = jp.build_task(args.tree_size, args.tree_type, args.input_type)
    checkpoint, note = base.find_revisit_checkpoint(
        checkpoint_root,
        lambda_value=lambda_value,
        alpha=alpha,
        beta=beta,
        opportunity_cost=opportunity,
        seed=seed,
        task=task,
        tree_size=args.tree_size,
        rnn_dim=rnn_dim,
        latent_dim=latent_dim,
        expansion_decision_version=args.expansion_decision_version,
        model_variant=args.model_variant,
        max_observations_before_stop=args.max_observations_before_stop,
        observation_sigma=sigma,
        kl_start_multiplier=args.kl_start_multiplier,
        kl_annealing_epochs=args.kl_annealing_epochs,
    )
    combo_label = f"seed={seed}, beta={beta:g}, lambda={lambda_value:g}, opp={opportunity:g}, sigma={sigma:g}"
    if checkpoint is None:
        failures.append({"combo": combo_label, "reason": note})
        print(f"Skipping {combo_label}: {note}")
        return None

    combo_dir = combo_output_dir(
        outdir,
        seed=seed,
        beta=beta,
        opportunity=opportunity,
        lambda_value=lambda_value,
        sigma=sigma,
        rnn_dim=rnn_dim,
        latent_dim=latent_dim,
        tree_type=args.tree_type,
    )
    for subdir in ["metadata", "posterior_parameters", "pga", "gaussian_product_mds", "euclidean_pca", "figures", "tables", "logs"]:
        ensure_dir(combo_dir / subdir)

    model, params = base.build_model_and_params(
        checkpoint,
        task=task,
        lambda_value=lambda_value,
        alpha=alpha,
        beta=beta,
        opportunity_cost=opportunity,
        rnn_dim=rnn_dim,
        latent_dim=latent_dim,
        expansion_decision_version=args.expansion_decision_version,
        model_variant=args.model_variant,
        max_observations_before_stop=args.max_observations_before_stop,
        observation_sigma=sigma,
    )
    rewards = base.sample_rewards(args.n_trials, task, args.analysis_seed_offset + seed)
    metadata, z_mu, z_logvar, z_sigma, prior_mu, prior_logvar, prior_sigma = rollout_revisit_posterior_states(
        model=model,
        params=params,
        task=task,
        rewards=rewards,
        batch_size=args.batch_size,
        seed=args.analysis_seed_offset + seed + 10_000,
        beta=beta,
        max_observations_before_stop=args.max_observations_before_stop,
    )
    if metadata.empty:
        failures.append({"combo": combo_label, "reason": "no retained posterior states"})
        return None

    rng = np.random.default_rng(args.geometry_seed + 1009 * seed + 17 * int(round(beta * 100)))
    metadata, (z_mu, z_logvar, z_sigma, prior_mu, prior_logvar, prior_sigma) = subsample_states(
        metadata,
        (z_mu, z_logvar, z_sigma, prior_mu, prior_logvar, prior_sigma),
        args.max_states,
        rng,
    )
    for latent_i in range(z_mu.shape[1]):
        metadata[f"z_mu_{latent_i}"] = z_mu[:, latent_i]
        metadata[f"z_logvar_{latent_i}"] = z_logvar[:, latent_i]
        metadata[f"z_sigma_{latent_i}"] = z_sigma[:, latent_i]
        metadata[f"prior_mu_{latent_i}"] = prior_mu[:, latent_i]
        metadata[f"prior_logvar_{latent_i}"] = prior_logvar[:, latent_i]
        metadata[f"prior_sigma_{latent_i}"] = prior_sigma[:, latent_i]
        metadata[f"prior_normalized_z_mu_{latent_i}"] = (
            z_mu[:, latent_i] - prior_mu[:, latent_i]
        ) / np.maximum(prior_sigma[:, latent_i], SIGMA_EPS)
    metadata["latent_dim"] = latent_dim
    metadata["rnn_dim"] = rnn_dim
    metadata["seed"] = seed
    metadata["alpha"] = alpha
    metadata["beta"] = beta
    metadata["lambda"] = lambda_value
    metadata["opportunity"] = opportunity
    metadata["observation_sigma"] = sigma
    metadata["checkpoint_path"] = str(checkpoint)
    metadata.to_csv(combo_dir / "posterior_parameters" / "posterior_state_metadata.csv", index=False)
    np.savez_compressed(
        combo_dir / "posterior_parameters" / "posterior_parameters.npz",
        z_mu=z_mu,
        z_logvar=z_logvar,
        z_sigma=z_sigma,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        prior_sigma=prior_sigma,
    )

    methods = method_list(args.reduction_method)
    manifest = build_manifest(combo_dir, methods)
    summary_rows = []
    metric_rows = []
    pga_scores_for_mds = None

    for scope_name, idx in scope_iter(metadata, args.pga_fit_scope):
        scope_meta = metadata.iloc[idx].copy().reset_index(drop=True)
        scope_mu = z_mu[idx]
        scope_sigma = z_sigma[idx]
        scope_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        scope_suffix = "" if scope_name == "pooled" else f"_{scope_name}"

        scores_e, pca = euclidean_pca_fit_transform(np.concatenate([scope_mu, np.log(scope_sigma)], axis=1), args.n_components)
        euclid_df = add_score_columns(scope_meta, scores_e, "euclidean_pca")
        euclid_df.to_csv(combo_dir / "euclidean_pca" / f"euclidean_pca_scores{scope_suffix}.csv", index=False)
        np.savez_compressed(combo_dir / "euclidean_pca" / f"euclidean_pca_components{scope_suffix}.npz", **pca)

        if "pga" in methods:
            pga = ProductGaussianPGA(n_components=args.n_components, tol=args.pga_tol, max_iters=args.pga_max_iters)
            scores = pga.fit_transform(scope_mu, scope_sigma)
            pga_scores_for_mds = scores if scope_name == "pooled" else pga_scores_for_mds
            pga_df = add_score_columns(scope_meta, scores, "pga")
            pga_df.to_csv(combo_dir / "pga" / f"pga_scores{scope_suffix}.csv", index=False)
            pga.save(combo_dir / "pga" / f"pga_components{scope_suffix}.npz")
            rec_mu, rec_sigma = pga.inverse_transform(scores)
            np.savez_compressed(
                combo_dir / "pga" / f"pga_reconstructed_posteriors{scope_suffix}.npz",
                reconstructed_mu=rec_mu,
                reconstructed_sigma=rec_sigma,
            )
            pga_metrics = pga.reconstruction_metrics(scope_mu, scope_sigma, scope_rng, args.max_pairs)
            pga_metric_row = {
                "representation": "pga",
                "fit_scope": scope_name,
                "n_states": int(len(scope_meta)),
                "latent_dim": int(latent_dim),
                "explained_variance_ratio_component_1": float(pga.explained_variance_ratio[0]) if len(pga.explained_variance_ratio) > 0 else np.nan,
                "explained_variance_ratio_component_2": float(pga.explained_variance_ratio[1]) if len(pga.explained_variance_ratio) > 1 else np.nan,
                "cumulative_explained_variance_ratio": float(np.sum(pga.explained_variance_ratio)),
                **pga_metrics,
            }
            metric_rows.append(pga_metric_row)
            pd.DataFrame([pga_metric_row]).to_csv(combo_dir / "pga" / f"pga_fit_metrics{scope_suffix}.csv", index=False)

            figdir = ensure_dir(combo_dir / "pga" / "figures")
            base.plot_combo_density(
                scope_meta.copy(),
                figdir,
                combo_label=f"{combo_label}, raw posterior {scope_name}",
                grid_n=120,
                max_density_points=1500,
                task_tree_type=task.tree_type,
                min_density_samples=args.min_density_samples,
            )
            if {"prior_normalized_z_mu_0", "prior_normalized_z_mu_1"}.issubset(scope_meta.columns):
                prior_norm_df = scope_meta.copy()
                prior_norm_df["z_mu_0"] = pd.to_numeric(
                    prior_norm_df["prior_normalized_z_mu_0"],
                    errors="coerce",
                )
                prior_norm_df["z_mu_1"] = pd.to_numeric(
                    prior_norm_df["prior_normalized_z_mu_1"],
                    errors="coerce",
                )
                base.plot_combo_density(
                    prior_norm_df,
                    figdir,
                    combo_label=f"{combo_label}, prior-normalized posterior {scope_name}",
                    grid_n=120,
                    max_density_points=1500,
                    task_tree_type=task.tree_type,
                    latent_file_prefix="revisit_prior_normalized_z0_z1_density",
                    plot_context_heatmaps=False,
                    x_axis_label="(z_mu_0 - prior_mu_0) / prior_sigma_0",
                    y_axis_label="(z_mu_1 - prior_mu_1) / prior_sigma_1",
                    x_panel_label="prior-norm z0",
                    min_density_samples=args.min_density_samples,
                )
            plot_pga_score_reward_grid_distributions(
                pga_df,
                figdir,
                combo_label=f"{combo_label}, PGA {scope_name}",
                scope_suffix=scope_suffix,
                grid_n=120,
                max_density_points=1500,
                min_density_samples=args.min_density_samples,
            )
            plot_scatter(pga_df, figdir / f"pga_scores_by_reward{scope_suffix}.png", x_col="pga_score_0", y_col="pga_score_1", color_col="actual_node_reward", title=f"PGA scores by reward ({scope_name})", color_label="actual node reward")
            plot_scatter(pga_df, figdir / f"pga_scores_by_timestep{scope_suffix}.png", x_col="pga_score_0", y_col="pga_score_1", color_col="timestep", title=f"PGA scores by timestep ({scope_name})", color_label="timestep")
            if "first_observed_path_actual_reward_raw" in pga_df:
                plot_scatter(pga_df, figdir / f"pga_scores_by_current_best_path{scope_suffix}.png", x_col="pga_score_0", y_col="pga_score_1", color_col="first_observed_path_actual_reward_raw", title=f"PGA scores by first observed path value ({scope_name})", color_label="first observed path value")
            if "observed_path" in pga_df and "first_observed_path" in pga_df:
                switch_df = pga_df.copy()
                switch_df["path_switch"] = (switch_df["observed_path"] != switch_df["first_observed_path"]).astype(float)
                plot_scatter(switch_df, figdir / f"pga_scores_by_path_switch{scope_suffix}.png", x_col="pga_score_0", y_col="pga_score_1", color_col="path_switch", title=f"PGA scores by path switch ({scope_name})", color_label="observed path != first path")
            plot_component_loadings(pga, figdir / f"pga_component_loadings{scope_suffix}.png")
            plot_explained_variance(pga.explained_variance_ratio, figdir / f"pga_explained_variance{scope_suffix}.png", f"PGA explained variance ({scope_name})")
            pairs = sample_pair_indices(len(scope_mu), args.max_pairs, scope_rng)
            plot_distance_comparison(
                pair_distances(scope_mu, scope_sigma, pairs),
                pair_distances(rec_mu, rec_sigma, pairs),
                figdir / f"pga_original_vs_reconstructed_distance{scope_suffix}.png",
                f"PGA distance preservation ({scope_name})",
            )
            decoding = task_decoding_metrics(
                pga_df,
                ["pga_score_0", "pga_score_1"],
                ["actual_node_reward", "observed_node", "observed_path", "timestep", "first_observed_path_actual_reward_raw"],
            )
            decoding.to_csv(combo_dir / "pga" / f"original_vs_pga_task_decoding{scope_suffix}.csv", index=False)

        if scope_name == "pooled":
            # Baseline comparison is only meaningful in the common pooled coordinate system.
            rec_mu_e, rec_sigma_e = euclidean_reconstruction_to_gaussian(scores_e, pca, latent_dim)
            pairs = sample_pair_indices(len(scope_mu), args.max_pairs, scope_rng)
            euclid_metrics = pairwise_preservation_metrics(scope_mu, scope_sigma, rec_mu_e, rec_sigma_e, pairs)
            euclid_row = {
                "representation": "euclidean_pca",
                "fit_scope": scope_name,
                "n_states": int(len(scope_meta)),
                "latent_dim": int(latent_dim),
                "explained_variance_ratio_component_1": float(pca["explained_variance_ratio"][0]) if len(pca["explained_variance_ratio"]) > 0 else np.nan,
                "explained_variance_ratio_component_2": float(pca["explained_variance_ratio"][1]) if len(pca["explained_variance_ratio"]) > 1 else np.nan,
                "cumulative_explained_variance_ratio": float(np.sum(pca["explained_variance_ratio"])),
                **euclid_metrics,
            }
            metric_rows.append(euclid_row)
            pd.DataFrame([euclid_row]).to_csv(combo_dir / "euclidean_pca" / "euclidean_pca_geometry_metrics.csv", index=False)

    if "gaussian_product_mds" in methods:
        mds_dir = ensure_dir(combo_dir / "gaussian_product_mds")
        mds_figdir = ensure_dir(mds_dir / "figures")
        pga_init = pga_scores_for_mds
        if pga_init is None:
            pga_tmp = ProductGaussianPGA(n_components=2, tol=args.pga_tol, max_iters=args.pga_max_iters).fit(z_mu, z_sigma)
            pga_init = pga_tmp.transform(z_mu, z_sigma)
        mds = fit_gaussian_product_mds(
            z_mu,
            z_sigma,
            pga_init,
            rng,
            max_pairs=args.max_pairs,
            train_pair_frac=args.train_pair_frac,
            restarts=args.embedding_restarts,
            steps=args.embedding_steps,
            lr=args.embedding_lr,
            grad_clip=args.embedding_grad_clip,
        )
        mds_df = metadata.copy()
        for factor_i in range(2):
            mds_df[f"reduced_mu_{factor_i}"] = mds.reduced_mu[:, factor_i]
            mds_df[f"reduced_sigma_{factor_i}"] = mds.reduced_sigma[:, factor_i]
        mds_df["representation"] = "gaussian_product_mds"
        mds_df.to_csv(mds_dir / "gaussian_product_mds_coordinates.csv", index=False)
        np.savez_compressed(mds_dir / "gaussian_product_mds_coordinates.npz", reduced_mu=mds.reduced_mu, reduced_sigma=mds.reduced_sigma)
        mds_metrics = {
            "representation": "gaussian_product_mds",
            "fit_scope": "pooled",
            "n_states": int(len(metadata)),
            "latent_dim": int(latent_dim),
            **mds.metrics,
        }
        metric_rows.append(mds_metrics)
        pd.DataFrame([mds_metrics]).to_csv(mds_dir / "gaussian_product_mds_metrics.csv", index=False)
        plot_mds_factor_geometry(mds_df, mds_figdir)
        plot_scatter(mds_df.assign(z0=mds.reduced_mu[:, 0], z1=mds.reduced_mu[:, 1]), mds_figdir / "gaussian_product_mds_2factor_reward_summary.png", x_col="z0", y_col="z1", color_col="actual_node_reward", title="Gaussian product MDS reduced means", color_label="actual node reward")
        decoding = task_decoding_metrics(
            mds_df.assign(mds_score_0=mds.reduced_mu[:, 0], mds_score_1=mds.reduced_mu[:, 1]),
            ["mds_score_0", "mds_score_1"],
            ["actual_node_reward", "observed_node", "observed_path", "timestep", "first_observed_path_actual_reward_raw"],
        )
        decoding.to_csv(mds_dir / "original_vs_gaussian_product_mds_task_decoding.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df.to_csv(combo_dir / "tables" / "geometry_reduction_metrics.csv", index=False)
        metrics_df.to_csv(combo_dir / "gaussian_pga_vs_euclidean_pca.csv", index=False)
    run_config = {
        **vars(args),
        "combo": {
            "lambda": lambda_value,
            "alpha": alpha,
            "beta": beta,
            "opportunity": opportunity,
            "sigma": sigma,
            "seed": seed,
            "rnn_dim": rnn_dim,
            "latent_dim": latent_dim,
            "checkpoint_path": str(checkpoint),
        },
        "interpretation_notes": [
            "PGA scores are scalar principal geodesic coordinates, not reduced Gaussian factors.",
            "gaussian_product_mds directly fits two Gaussian factors but is a stress embedding, not PCA/PGA.",
            "PGA axes may flip or rotate across seeds; compare invariant metrics or aligned summaries.",
        ],
    }
    save_json(combo_dir / "metadata" / "run_config.json", run_config)
    save_json(combo_dir / "gaussian_pga_analysis_summary.json", {"metrics": metric_rows, "manifest_rows": manifest.to_dict(orient="records")})
    summary_rows.append(f"Combo: {combo_label}")
    summary_rows.append(f"Checkpoint: {checkpoint}")
    summary_rows.append(f"Retained states: {len(metadata)}")
    summary_rows.append(f"Latent dim: {latent_dim}; product manifold dim: {2 * latent_dim}")
    summary_rows.append("PGA scores are scalar geodesic scores; they are not Gaussian mu/sigma factors.")
    summary_rows.append("gaussian_product_mds fits two reduced Gaussian factors as a distance-preserving embedding.")
    if not metrics_df.empty:
        summary_rows.append("")
        summary_rows.append(metrics_df.to_string(index=False))
    write_summary_text(combo_dir / "gaussian_pga_analysis_summary.txt", summary_rows)
    return {"combo": combo_label, "n_states": int(len(metadata)), "combo_dir": str(combo_dir)}


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))
    checkpoint_root = Path(args.checkpoint_root)
    save_json(outdir / "run_config.json", vars(args))
    self_tests = run_geometry_self_tests()
    save_json(outdir / "geometry_self_tests.json", self_tests)
    if args.run_synthetic_test:
        synthetic_pga_test(ensure_dir(outdir / "logs"))

    failures: List[dict] = []
    successes: List[dict] = []
    for lambda_value in args.lambda_values:
        for alpha in args.alphas:
            for beta in args.beta_values:
                for opportunity in args.opportunity_costs:
                    for sigma in args.observation_sigmas:
                        for seed in args.seeds:
                            for rnn_dim in args.rnn_dims:
                                for latent_dim in args.latent_dims:
                                    result = run_combo(
                                        args,
                                        lambda_value=lambda_value,
                                        alpha=alpha,
                                        beta=beta,
                                        opportunity=opportunity,
                                        sigma=sigma,
                                        seed=seed,
                                        rnn_dim=rnn_dim,
                                        latent_dim=latent_dim,
                                        outdir=outdir,
                                        checkpoint_root=checkpoint_root,
                                        failures=failures,
                                    )
                                    if result is not None:
                                        successes.append(result)
                                        print(f"Saved Gaussian PGA analysis for {result['combo']} to {result['combo_dir']}")
    if failures:
        pd.DataFrame(failures).to_csv(outdir / "gaussian_pga_failures.csv", index=False)
    if successes:
        pd.DataFrame(successes).to_csv(outdir / "gaussian_pga_successes.csv", index=False)
    aggregate_rows = []
    for success in successes:
        metrics_path = Path(success["combo_dir"]) / "tables" / "geometry_reduction_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            metrics["combo"] = success["combo"]
            metrics["combo_dir"] = success["combo_dir"]
            aggregate_rows.append(metrics)
    if aggregate_rows:
        pd.concat(aggregate_rows, ignore_index=True).to_csv(outdir / "gaussian_pga_seed_summary.csv", index=False)


if __name__ == "__main__":
    main()
