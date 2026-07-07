#!/usr/bin/env python3
"""Build an interactive latent-trace UI for sample-set revisit diagnostics.

The pairwise diagnostic CSVs are aggregate summaries, so they cannot show a
single sample-set trajectory.  This script creates a companion trace cache from
the same synthetic histories and checkpoints, then writes a standalone HTML UI
that loads one model/sigma/seed JSON file at a time.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import sample_set_pairwise_last_paid_kl_jax as diag  # noqa: E402
from model_jax import planning as jp  # noqa: E402


def parse_dims(raw: str) -> tuple[int, int]:
    values = [int(x.strip()) for x in str(raw).replace(",", " ").split() if x.strip()]
    if len(values) != 2:
        raise ValueError("--latent-dims-to-plot must contain exactly two indices.")
    return values[0], values[1]


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, 6)


def pack_pair(values, dims: tuple[int, int]) -> list[float | None]:
    arr = np.asarray(values, dtype=float)
    out = []
    for dim in dims:
        out.append(finite_float(arr[dim]) if dim < arr.shape[0] else None)
    return out


def action_label(action: int, task: jp.TaskSpec) -> str:
    if action < int(task.num_nodes):
        return f"observe_node_{action + 1}"
    path = action - int(task.num_nodes)
    if 0 <= path < int(task.num_paths):
        return f"stop_path_{path + 1}"
    return f"action_{action}"


def trace_rollout_with_streams(
    model: jp.PlanningVAE,
    params,
    config: jp.RunConfig,
    task: jp.TaskSpec,
    rewards: np.ndarray,
    streams: np.ndarray,
    metadata: pd.DataFrame,
    *,
    seed_offset: int,
    force_round_robin_observations: bool,
    force_first_observe_node: int,
    latent_dims: tuple[int, int],
) -> list[dict]:
    n_trials = rewards.shape[0]
    reward_feature_dim = int(model.reward_feature_dim_override) or jp.reward_feature_dim_for_sigma(
        config.observation_sigma
    )
    carry = jp.initial_carry(
        n_trials,
        task,
        config.rnn_units,
        reward_feature_dim,
        jp.visited_lstm_feature_dim_for_task(task),
    )
    carry = jp.reset_done_envs(carry, jnp.asarray(rewards, dtype=jnp.float32))
    sched = diag.schedule_for(config.beta)
    rng = jax.random.PRNGKey(config.seed + 910_000 + int(seed_offset))
    stream_counts = np.zeros((n_trials, task.num_nodes), dtype=np.int32)
    rows: list[dict] = []

    meta_records = metadata.reset_index(drop=True).to_dict("records")
    for timestep in range(1, int(config.num_steps) + 1):
        active_before = ~np.asarray(jax.device_get(carry.done), dtype=bool)
        rng, step_rng = jax.random.split(rng)
        if force_round_robin_observations:
            if timestep <= int(config.max_observations_before_stop):
                forced_node = (timestep - 1) % int(task.num_nodes)
                action = np.full(n_trials, forced_node, dtype=np.int32)
            else:
                action = np.full(n_trials, int(task.num_nodes), dtype=np.int32)
        elif timestep == 1 and 1 <= int(force_first_observe_node) <= int(task.num_nodes):
            action = np.full(n_trials, int(force_first_observe_node) - 1, dtype=np.int32)
        else:
            _, probe_trans = model.apply(
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
            action = np.asarray(jax.device_get(probe_trans.action), dtype=np.int32)

        forced_observation = np.full(n_trials, np.nan, dtype=np.float32)
        sample_position = np.full(n_trials, -1, dtype=np.int32)
        for trial in range(n_trials):
            if not active_before[trial]:
                continue
            if action[trial] < task.num_nodes:
                node = int(action[trial])
                sample_idx = min(stream_counts[trial, node], streams.shape[2] - 1)
                forced_observation[trial] = streams[trial, node, sample_idx]
                sample_position[trial] = sample_idx
                stream_counts[trial, node] += 1

        carry, trans = model.apply(
            {"params": params},
            carry,
            step_rng,
            sched,
            forced_action=jnp.asarray(action, dtype=jnp.int32),
            forced_observation=jnp.asarray(forced_observation, dtype=jnp.float32),
            training=True,
            use_posterior_mean=False,
            compute_targets=False,
            method=jp.PlanningVAE.__call__,
        )
        trans_np = jax.device_get(trans)
        z_mu = np.asarray(trans_np.z_mu, dtype=float)
        z_logvar = np.asarray(trans_np.z_logvar, dtype=float)
        z_sigma = np.exp(0.5 * np.clip(z_logvar, -20.0, 20.0))
        prior_mu = np.asarray(trans_np.prior_mu, dtype=float)
        prior_logvar = np.asarray(trans_np.prior_logvar, dtype=float)
        prior_sigma = np.exp(0.5 * np.clip(prior_logvar, -20.0, 20.0))
        norm_mu = (z_mu - prior_mu) / np.maximum(prior_sigma, 1e-8)
        norm_sigma = z_sigma / np.maximum(prior_sigma, 1e-8)
        is_observe = np.asarray(trans_np.is_observe) > 0.5
        is_stop = np.asarray(trans_np.is_stop) > 0.5
        node_index = np.asarray(trans_np.node_index, dtype=np.int32)
        terminal_path = np.asarray(trans_np.terminal_path_index, dtype=np.int32)
        terminal_probs = np.asarray(trans_np.action_output, dtype=float)

        for trial in range(n_trials):
            if not active_before[trial]:
                continue
            meta = meta_records[trial]
            act = int(action[trial])
            observed_node = int(node_index[trial]) + 1 if bool(is_observe[trial]) else None
            sample_idx = int(sample_position[trial]) if sample_position[trial] >= 0 else None
            rows.append(
                {
                    "trial_index": int(trial),
                    "condition_index": int(meta["condition_index"]),
                    "original_condition_index": int(meta.get("original_condition_index", meta["condition_index"])),
                    "sample_set": int(meta["sample_set"]),
                    "timestep": int(timestep),
                    "action": act,
                    "action_label": action_label(act, task),
                    "action_type": "observe" if bool(is_observe[trial]) else "stop" if bool(is_stop[trial]) else "inactive",
                    "observed_node": observed_node,
                    "terminal_path": int(terminal_path[trial]) + 1 if terminal_path[trial] >= 0 else None,
                    "sample_position": sample_idx,
                    "observed_reward": finite_float(forced_observation[trial]) if sample_idx is not None else None,
                    "actual_observed_reward": (
                        finite_float(rewards[trial, observed_node - 1])
                        if observed_node is not None
                        else None
                    ),
                    "paid_kl": finite_float(np.asarray(trans_np.paid_kl)[trial]),
                    "observed_kl": finite_float(np.asarray(trans_np.observed_kl)[trial]),
                    "terminal_expected_reward": finite_float(np.asarray(trans_np.terminal_expected_reward)[trial]),
                    "raw_mu": pack_pair(z_mu[trial], latent_dims),
                    "raw_sigma": pack_pair(z_sigma[trial], latent_dims),
                    "norm_mu": pack_pair(norm_mu[trial], latent_dims),
                    "norm_sigma": pack_pair(norm_sigma[trial], latent_dims),
                    "terminal_probs": [finite_float(x) for x in terminal_probs[trial].tolist()],
                }
            )
    return rows


def stream_records(metadata: pd.DataFrame, rewards: np.ndarray, streams: np.ndarray) -> list[dict]:
    rows = []
    meta_records = metadata.reset_index(drop=True).to_dict("records")
    for trial, meta in enumerate(meta_records):
        rows.append(
            {
                "trial_index": int(trial),
                "condition_index": int(meta["condition_index"]),
                "original_condition_index": int(meta.get("original_condition_index", meta["condition_index"])),
                "sample_set": int(meta["sample_set"]),
                "rewards": [finite_float(x) for x in rewards[trial].tolist()],
                "streams": [
                    [finite_float(x) for x in streams[trial, node].tolist()]
                    for node in range(streams.shape[1])
                ],
            }
        )
    return rows


def data_file_name(family: str, parameter_value: float, beta: float, opportunity: float, sigma: float, seed: int) -> str:
    return (
        f"{family}_param_{diag.value_token(parameter_value)}"
        f"_beta_{diag.value_token(beta)}"
        f"_opp_{diag.value_token(opportunity)}"
        f"_sigma_{diag.value_token(sigma)}"
        f"_seed_{seed}.json"
    )


def generate_cache(args: argparse.Namespace, ui_dir: Path, latent_dims: tuple[int, int]) -> list[dict]:
    data_dir = ui_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    task = jp.build_task(args.tree_size, args.tree_type, args.input_type)
    for family, parameter_name, parameter_value, beta, opportunity in args.parameter_combos:
        for sigma in args.sigmas:
            for seed in args.seeds:
                file_name = data_file_name(family, parameter_value, beta, opportunity, sigma, seed)
                out_file = data_dir / file_name
                if out_file.exists() and not args.overwrite:
                    manifest.append(
                        {
                            "family": family,
                            "parameter_name": parameter_name,
                            "parameter_value": float(parameter_value),
                            "beta": float(beta),
                            "opportunity": float(opportunity),
                            "sigma": float(sigma),
                            "seed": int(seed),
                            "file": f"data/{file_name}",
                        }
                    )
                    continue
                print(
                    f"Tracing {family}: beta={beta:g}, opp={opportunity:g}, sigma={sigma:g}, seed={seed}",
                    flush=True,
                )
                config = diag.make_config(args, seed=seed, beta=beta, opportunity=opportunity, sigma=sigma)
                model, params = jp.load_state_for_sim(config, task)
                rewards, streams, metadata = diag.build_reward_combination_trials(
                    np.asarray(task.reward_values, dtype=float),
                    num_nodes=int(task.num_nodes),
                    sigma=float(sigma),
                    n_sample_sets=int(args.n_sample_sets),
                    max_observations=int(args.max_observations_before_stop),
                    seed=int(seed + round(1000 * sigma) + round(17 * beta) + round(31 * opportunity)),
                    n_reward_combinations=int(args.n_reward_combinations),
                    reward_combination_seed=int(seed),
                )
                traces = trace_rollout_with_streams(
                    model,
                    params,
                    config,
                    task,
                    rewards,
                    streams,
                    metadata,
                    seed_offset=int(round(10_000 * sigma) + round(beta) + round(10_000 * opportunity)),
                    force_round_robin_observations=bool(args.force_round_robin_observations),
                    force_first_observe_node=int(args.force_first_observe_node),
                    latent_dims=latent_dims,
                )
                streams_payload = stream_records(metadata, rewards, streams)
                payload = {
                    "meta": {
                        "family": family,
                        "parameter_name": parameter_name,
                        "parameter_value": float(parameter_value),
                        "beta": float(beta),
                        "opportunity": float(opportunity),
                        "sigma": float(sigma),
                        "seed": int(seed),
                        "tree": args.output_tree_label,
                        "tree_size": int(args.tree_size),
                        "tree_type": args.tree_type,
                        "num_nodes": int(task.num_nodes),
                        "num_paths": int(task.num_paths),
                        "latent_dims": list(latent_dims),
                        "n_reward_combinations": int(metadata["condition_index"].nunique()),
                        "n_sample_sets": int(args.n_sample_sets),
                        "max_observations_before_stop": int(args.max_observations_before_stop),
                        "force_first_observe_node": int(args.force_first_observe_node),
                    },
                    "streams": streams_payload,
                    "traces": traces,
                }
                with out_file.open("w") as f:
                    json.dump(payload, f, separators=(",", ":"))
                manifest.append({**payload["meta"], "file": f"data/{file_name}"})
    return manifest


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sample-Set Latent Trace UI</title>
<style>
  :root { --ink:#202124; --muted:#5f6368; --line:#dadce0; --panel:#fff; --bg:#f6f7f9; --accent:#1a73e8; }
  body { margin:0; font-family: Arial, sans-serif; font-size:13px; color:var(--ink); background:var(--bg); }
  header { padding:14px 18px; background:#fff; border-bottom:1px solid var(--line); }
  h1 { margin:0; font-size:18px; font-weight:600; }
  main { padding:14px; display:grid; grid-template-columns: 360px 1fr; gap:14px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  label { display:block; margin:10px 0 4px; color:var(--muted); font-size:12px; }
  select, input { width:100%; box-sizing:border-box; padding:7px; border:1px solid var(--line); border-radius:5px; background:#fff; }
  .control-grid { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
  canvas { width:100%; height:360px; border:1px solid var(--line); border-radius:6px; background:#fff; }
  table { width:100%; border-collapse:collapse; margin-top:8px; }
  th, td { border-bottom:1px solid #eee; padding:5px 4px; text-align:left; vertical-align:top; }
  th { color:var(--muted); font-weight:600; }
  .small { color:var(--muted); font-size:12px; }
  .pill { display:inline-block; padding:2px 6px; border-radius:999px; background:#eef3fd; color:#174ea6; margin-right:4px; }
  .status { margin-top:8px; color:var(--muted); }
  @media (max-width: 960px) { main { grid-template-columns:1fr; } .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Sample-Set Latent Trace UI</h1>
  <div class="small">Select a model, actual rewards, timestep, and sample-set index. Ellipses show diagonal posterior contours in z0/z1 space.</div>
</header>
<main>
  <section>
    <div class="control-grid">
      <div><label>Vary family</label><select id="familySelect"></select></div>
      <div><label>Beta</label><select id="betaSelect"></select></div>
      <div><label>Opportunity</label><select id="opportunitySelect"></select></div>
      <div><label>Sigma</label><select id="sigmaSelect"></select></div>
      <div><label>Seed</label><select id="seedSelect"></select></div>
    </div>
    <label>Actual rewards of nodes</label>
    <select id="conditionSelect"></select>
    <label>Timestep</label>
    <select id="timestepSelect"></select>
    <label>Sample set index</label>
    <select id="sampleSetSelect"></select>
    <div id="status" class="status"></div>
    <h3>Selected Sample Set</h3>
    <div id="sampleSetInfo" class="small"></div>
    <div id="streamTable"></div>
    <h3>Action Timeline</h3>
    <div id="timelineTable"></div>
  </section>
  <section>
    <div id="summary"></div>
    <div class="grid">
      <div>
        <h3>Raw latent posterior</h3>
        <canvas id="rawCanvas" width="640" height="420"></canvas>
      </div>
      <div>
        <h3>Prior-normalized posterior</h3>
        <canvas id="normCanvas" width="640" height="420"></canvas>
      </div>
    </div>
  </section>
</main>
<script>
const manifest = __MANIFEST__;
let current = null;
let currentModelIndex = 0;

const familySelect = document.getElementById('familySelect');
const betaSelect = document.getElementById('betaSelect');
const opportunitySelect = document.getElementById('opportunitySelect');
const sigmaSelect = document.getElementById('sigmaSelect');
const seedSelect = document.getElementById('seedSelect');
const conditionSelect = document.getElementById('conditionSelect');
const timestepSelect = document.getElementById('timestepSelect');
const sampleSetSelect = document.getElementById('sampleSetSelect');
const statusEl = document.getElementById('status');

function fmt(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return '';
  if (typeof x === 'number') return Number.parseFloat(x.toFixed(5)).toString();
  return x;
}
function rewardKey(stream) { return stream.rewards.join(','); }
function modelLabel(m) {
  return `${m.family} ${m.parameter_name}=${fmt(m.parameter_value)} | beta=${fmt(m.beta)} opp=${fmt(m.opportunity)} sigma=${fmt(m.sigma)} seed=${m.seed}`;
}
function familyLabel(value) {
  if (value === 'vary_beta') return 'vary beta';
  if (value === 'vary_opportunity') return 'vary opportunity';
  return value;
}
function fillSelect(select, items, labelFn) {
  select.innerHTML = '';
  items.forEach((item, idx) => {
    const option = document.createElement('option');
    option.value = idx;
    option.textContent = labelFn(item, idx);
    select.appendChild(option);
  });
}
function uniqueSorted(rows, field) {
  const values = Array.from(new Set(rows.map(r => r[field])));
  return values.sort((a,b) => {
    if (typeof a === 'number' && typeof b === 'number') return a-b;
    return String(a).localeCompare(String(b));
  });
}
function fillValueSelect(select, values, labelFn, previousValue = null) {
  const previous = previousValue !== null ? String(previousValue) : select.value;
  select.innerHTML = '';
  values.forEach(value => {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = labelFn ? labelFn(value) : fmt(value);
    select.appendChild(option);
  });
  if (values.map(v => String(v)).includes(previous)) {
    select.value = previous;
  }
}
function syncModelSelectors() {
  const prevFamily = familySelect.value || 'vary_beta';
  fillValueSelect(familySelect, uniqueSorted(manifest, 'family'), familyLabel, prevFamily);
  const familyRows = manifest.filter(m => m.family === familySelect.value);
  fillValueSelect(betaSelect, uniqueSorted(familyRows, 'beta'), fmt);
  const betaRows = familyRows.filter(m => String(m.beta) === betaSelect.value);
  fillValueSelect(opportunitySelect, uniqueSorted(betaRows, 'opportunity'), fmt);
  const oppRows = betaRows.filter(m => String(m.opportunity) === opportunitySelect.value);
  fillValueSelect(sigmaSelect, uniqueSorted(oppRows, 'sigma'), fmt);
  const sigmaRows = oppRows.filter(m => String(m.sigma) === sigmaSelect.value);
  fillValueSelect(seedSelect, uniqueSorted(sigmaRows, 'seed'), fmt);
  const idx = manifest.findIndex(m =>
    m.family === familySelect.value &&
    String(m.beta) === betaSelect.value &&
    String(m.opportunity) === opportunitySelect.value &&
    String(m.sigma) === sigmaSelect.value &&
    String(m.seed) === seedSelect.value
  );
  return idx >= 0 ? idx : 0;
}
function handleModelSelectionChange() {
  const idx = syncModelSelectors();
  loadModel(idx);
}
function traceForSelection() {
  const condition = current.conditions[Number(conditionSelect.value)];
  const timestep = Number(current.timesteps[Number(timestepSelect.value)]);
  const sampleSet = Number(current.sampleSets[Number(sampleSetSelect.value)]);
  const traces = current.traces.filter(r => r.condition_index === condition.condition_index && r.timestep === timestep);
  const selected = traces.find(r => r.sample_set === sampleSet);
  const selectedStream = current.streams.find(s => s.condition_index === condition.condition_index && s.sample_set === sampleSet);
  const timeline = current.traces.filter(r => r.condition_index === condition.condition_index && r.sample_set === sampleSet)
    .sort((a,b) => a.timestep - b.timestep);
  return {condition, timestep, sampleSet, traces, selected, selectedStream, timeline};
}
async function loadModel(index) {
  currentModelIndex = index;
  const m = manifest[index];
  statusEl.textContent = `Loading ${m.file} ...`;
  const response = await fetch(m.file);
  current = await response.json();
  const conditionMap = new Map();
  current.streams.forEach(s => {
    if (!conditionMap.has(s.condition_index)) {
      conditionMap.set(s.condition_index, {condition_index:s.condition_index, rewards:s.rewards, key:rewardKey(s)});
    }
  });
  current.conditions = Array.from(conditionMap.values()).sort((a,b) => a.condition_index - b.condition_index);
  current.timesteps = Array.from(new Set(current.traces.map(r => r.timestep))).sort((a,b) => a-b);
  current.sampleSets = Array.from(new Set(current.streams.map(s => s.sample_set))).sort((a,b) => a-b);
  fillSelect(conditionSelect, current.conditions, c => `R=[${c.rewards.map(fmt).join(', ')}]`);
  fillSelect(timestepSelect, current.timesteps, t => `t=${t}`);
  fillSelect(sampleSetSelect, current.sampleSets, s => `sample set ${s}`);
  statusEl.textContent = `${current.streams.length} sample sets, ${current.traces.length} trace rows`;
  updateView();
}
function renderStreamTable(stream) {
  if (!stream) return '';
  let html = '<table><thead><tr><th>Node</th><th>Actual</th><th>Samples</th></tr></thead><tbody>';
  stream.streams.forEach((vals, idx) => {
    html += `<tr><td>${idx+1}</td><td>${fmt(stream.rewards[idx])}</td><td>${vals.map(fmt).join(', ')}</td></tr>`;
  });
  html += '</tbody></table>';
  return html;
}
function renderTimeline(rows) {
  let html = '<table><thead><tr><th>t</th><th>action</th><th>obs</th><th>paid KL</th><th>obs KL</th></tr></thead><tbody>';
  rows.forEach(r => {
    html += `<tr><td>${r.timestep}</td><td>${r.action_label}</td><td>${fmt(r.observed_reward)}</td><td>${fmt(r.paid_kl)}</td><td>${fmt(r.observed_kl)}</td></tr>`;
  });
  html += '</tbody></table>';
  return html;
}
function boundsFor(rows, prefix) {
  const pts = [];
  rows.forEach(r => {
    const mu = r[`${prefix}_mu`], sig = r[`${prefix}_sigma`];
    if (!mu || !sig || mu[0] === null || mu[1] === null) return;
    pts.push([mu[0] - 2.5*sig[0], mu[1] - 2.5*sig[1]]);
    pts.push([mu[0] + 2.5*sig[0], mu[1] + 2.5*sig[1]]);
  });
  if (!pts.length) return {xmin:-1, xmax:1, ymin:-1, ymax:1};
  let xmin = Math.min(...pts.map(p => p[0])), xmax = Math.max(...pts.map(p => p[0]));
  let ymin = Math.min(...pts.map(p => p[1])), ymax = Math.max(...pts.map(p => p[1]));
  const dx = Math.max(xmax-xmin, 1e-3), dy = Math.max(ymax-ymin, 1e-3);
  xmin -= 0.08*dx; xmax += 0.08*dx; ymin -= 0.08*dy; ymax += 0.08*dy;
  const xlim = Math.max(Math.abs(xmin), Math.abs(xmax), 1e-3);
  const ylim = Math.max(Math.abs(ymin), Math.abs(ymax), 1e-3);
  xmin = -xlim; xmax = xlim; ymin = -ylim; ymax = ylim;
  return {xmin,xmax,ymin,ymax};
}
function drawLatent(canvas, rows, selected, prefix, boundsRows = rows) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const pad = {l:48,r:14,t:14,b:36};
  const b = boundsFor(boundsRows, prefix);
  const xmap = x => pad.l + (x-b.xmin)/(b.xmax-b.xmin)*(canvas.width-pad.l-pad.r);
  const ymap = y => canvas.height-pad.b - (y-b.ymin)/(b.ymax-b.ymin)*(canvas.height-pad.t-pad.b);
  ctx.strokeStyle = '#d0d0d0'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, canvas.height-pad.b); ctx.lineTo(canvas.width-pad.r, canvas.height-pad.b); ctx.stroke();
  ctx.fillStyle = '#555'; ctx.font = '12px Arial';
  ctx.fillText(fmt(b.xmin), pad.l, canvas.height-12); ctx.fillText(fmt(b.xmax), canvas.width-58, canvas.height-12);
  ctx.save(); ctx.translate(14, canvas.height/2); ctx.rotate(-Math.PI/2); ctx.fillText(prefix === 'raw' ? 'z1' : '(z1-prior)/prior_sd', 0, 0); ctx.restore();
  ctx.fillText(prefix === 'raw' ? 'z0' : '(z0-prior)/prior_sd', canvas.width/2-30, canvas.height-6);
  const radii = [1.177, 2.146];
  rows.forEach(r => {
    const mu = r[`${prefix}_mu`], sig = r[`${prefix}_sigma`];
    if (!mu || !sig || mu[0] === null || mu[1] === null) return;
    const isSel = selected && r.sample_set === selected.sample_set;
    const cx = xmap(mu[0]), cy = ymap(mu[1]);
    const sx = Math.abs(xmap(mu[0]+sig[0])-xmap(mu[0]));
    const sy = Math.abs(ymap(mu[1]+sig[1])-ymap(mu[1]));
    ctx.strokeStyle = isSel ? '#d93025' : 'rgba(80,80,80,0.25)';
    ctx.lineWidth = isSel ? 2.2 : 0.8;
    radii.forEach(rad => { ctx.beginPath(); ctx.ellipse(cx, cy, sx*rad, sy*rad, 0, 0, 2*Math.PI); ctx.stroke(); });
    ctx.fillStyle = isSel ? '#d93025' : 'rgba(40,100,200,0.35)';
    ctx.beginPath(); ctx.arc(cx, cy, isSel ? 4 : 2, 0, 2*Math.PI); ctx.fill();
  });
}
function updateView() {
  if (!current) return;
  const s = traceForSelection();
  document.getElementById('sampleSetInfo').innerHTML =
    `<span class="pill">condition ${s.condition.condition_index}</span><span class="pill">sample ${s.sampleSet}</span><span class="pill">t=${s.timestep}</span>`;
  document.getElementById('streamTable').innerHTML = renderStreamTable(s.selectedStream);
  document.getElementById('timelineTable').innerHTML = renderTimeline(s.timeline);
  const detail = s.selected
    ? `${s.selected.action_label}, paid KL=${fmt(s.selected.paid_kl)}, observed KL=${fmt(s.selected.observed_kl)}`
    : 'No selected row at this timestep.';
  const observedLine = s.selected && s.selected.action_type === 'observe'
    ? `Observed node ${s.selected.observed_node}, observed value ${fmt(s.selected.observed_reward)} (actual ${fmt(s.selected.actual_observed_reward)})`
    : 'No node observation at this timestep';
  document.getElementById('summary').innerHTML =
    `<p><b>${modelLabel(manifest[currentModelIndex])}</b></p>` +
    `<p><span class="pill">${observedLine}</span></p>` +
    `<p>${detail}</p>`;
  const conditionRows = current.traces.filter(r => r.condition_index === s.condition.condition_index);
  drawLatent(document.getElementById('rawCanvas'), s.traces, s.selected, 'raw', conditionRows);
  drawLatent(document.getElementById('normCanvas'), s.traces, s.selected, 'norm', conditionRows);
}
familySelect.addEventListener('change', handleModelSelectionChange);
betaSelect.addEventListener('change', handleModelSelectionChange);
opportunitySelect.addEventListener('change', handleModelSelectionChange);
sigmaSelect.addEventListener('change', handleModelSelectionChange);
seedSelect.addEventListener('change', handleModelSelectionChange);
conditionSelect.addEventListener('change', updateView);
timestepSelect.addEventListener('change', updateView);
sampleSetSelect.addEventListener('change', updateView);
syncModelSelectors();
loadModel(syncModelSelectors());
</script>
</body>
</html>
"""


def write_html(ui_dir: Path, manifest: list[dict]) -> Path:
    html = HTML_TEMPLATE.replace("__MANIFEST__", json.dumps(manifest, separators=(",", ":")))
    out = ui_dir / "index.html"
    out.write_text(html)
    return out


def sem_finite(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def diag_gaussian_sym_kl_from_sigma(mu_a, sigma_a, mu_b, sigma_b) -> float:
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    sigma_a = np.maximum(np.asarray(sigma_a, dtype=float), 1e-12)
    sigma_b = np.maximum(np.asarray(sigma_b, dtype=float), 1e-12)
    var_a = sigma_a**2
    var_b = sigma_b**2
    diff2 = (mu_a - mu_b) ** 2
    logvar_a = np.log(var_a)
    logvar_b = np.log(var_b)
    kl_ab = 0.5 * (logvar_b - logvar_a + (var_a + diff2) / var_b - 1.0)
    kl_ba = 0.5 * (logvar_a - logvar_b + (var_b + diff2) / var_a - 1.0)
    return float(np.nanmean(0.5 * (kl_ab + kl_ba)))


def trace_payloads(ui_dir: Path, manifest: list[dict]) -> list[dict]:
    payloads = []
    for item in manifest:
        path = ui_dir / item["file"]
        if path.exists():
            payloads.append(json.loads(path.read_text()))
    return payloads


def family_param_label(row: pd.Series) -> str:
    if row["family"] == "vary_beta":
        return f"beta={float(row['parameter_value']):g}"
    if row["family"] == "vary_opportunity":
        return f"opp={float(row['parameter_value']):g}"
    return f"{row['parameter_name']}={float(row['parameter_value']):g}"


def family_colors_for_values(family: str, values: list[float]) -> dict[float, object]:
    ramp = diag.FAMILY_COLOR_RAMPS.get(family, ["#333333"])
    colors = diag.color_values(len(values), ramp)
    return {float(value): color for value, color in zip(values, colors)}


def build_current_vs_previous_latent_change(payloads: list[dict], bin_width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for payload in payloads:
        meta = payload["meta"]
        traces = pd.DataFrame(payload.get("traces", []))
        if traces.empty:
            continue
        traces = traces[traces["action_type"] == "observe"].copy()
        if traces.empty:
            continue
        for (_, sample_set), group in traces.groupby(["condition_index", "sample_set"], sort=False):
            group = group.sort_values("timestep").reset_index(drop=True)
            previous_by_node = {}
            for obs_i, row in group.iterrows():
                node = row.get("observed_node")
                if node is None or not np.isfinite(float(node)):
                    continue
                node = int(node)
                current_is_paid = obs_i < len(group) - 1
                previous_observation = group.iloc[obs_i - 1] if obs_i > 0 else None
                previous_same_node = previous_by_node.get(node)
                if current_is_paid and previous_observation is not None and previous_same_node is not None:
                    current_value = float(row["observed_reward"])
                    previous_node_value = float(previous_same_node["observed_reward"])
                    abs_delta = abs(current_value - previous_node_value)
                    bin_width = float(bin_width)
                    abs_delta_bin = (
                        round(abs_delta / bin_width) * bin_width
                        if bin_width > 0
                        else abs_delta
                    )
                    rows.append(
                        {
                            "family": meta["family"],
                            "parameter_name": meta["parameter_name"],
                            "parameter_value": float(meta["parameter_value"]),
                            "beta": float(meta["beta"]),
                            "opportunity": float(meta["opportunity"]),
                            "sigma": float(meta["sigma"]),
                            "seed": int(meta["seed"]),
                            "condition_index": int(row["condition_index"]),
                            "original_condition_index": int(row.get("original_condition_index", row["condition_index"])),
                            "sample_set": int(sample_set),
                            "timestep": int(row["timestep"]),
                            "observed_node": node,
                            "abs_current_minus_previous_same_node_value": float(abs_delta),
                            "abs_delta_bin": float(abs_delta_bin),
                            "latent_sym_kl_current_vs_previous_observation": diag_gaussian_sym_kl_from_sigma(
                                row["raw_mu"],
                                row["raw_sigma"],
                                previous_observation["raw_mu"],
                                previous_observation["raw_sigma"],
                            ),
                            "z0_mu_displacement_current_vs_previous_observation": abs(
                                float(row["raw_mu"][0]) - float(previous_observation["raw_mu"][0])
                            ),
                            "z1_mu_displacement_current_vs_previous_observation": abs(
                                float(row["raw_mu"][1]) - float(previous_observation["raw_mu"][1])
                            ),
                        }
                    )
                previous_by_node[node] = row
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    per_stream = (
        detail.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "seed",
                "timestep",
                "abs_delta_bin",
                "original_condition_index",
                "sample_set",
            ],
            dropna=False,
        )
        .agg(
            latent_sym_kl_current_vs_previous_observation="mean",
            z0_mu_displacement_current_vs_previous_observation=(
                "z0_mu_displacement_current_vs_previous_observation",
                "mean",
            ),
            z1_mu_displacement_current_vs_previous_observation=(
                "z1_mu_displacement_current_vs_previous_observation",
                "mean",
            ),
        )
        .reset_index()
    )
    summary = (
        per_stream.groupby(
            [
                "family",
                "parameter_name",
                "parameter_value",
                "beta",
                "opportunity",
                "sigma",
                "timestep",
                "abs_delta_bin",
            ],
            dropna=False,
        )
        .agg(
            mean_latent_sym_kl=("latent_sym_kl_current_vs_previous_observation", "mean"),
            sem_latent_sym_kl=("latent_sym_kl_current_vs_previous_observation", sem_finite),
            mean_z0_mu_displacement=("z0_mu_displacement_current_vs_previous_observation", "mean"),
            sem_z0_mu_displacement=("z0_mu_displacement_current_vs_previous_observation", sem_finite),
            mean_z1_mu_displacement=("z1_mu_displacement_current_vs_previous_observation", "mean"),
            sem_z1_mu_displacement=("z1_mu_displacement_current_vs_previous_observation", sem_finite),
            n_streams=("latent_sym_kl_current_vs_previous_observation", "count"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return detail, summary


def plot_current_vs_previous_latent_change(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    sigmas = sorted(summary["sigma"].unique())
    timesteps = sorted(summary["timestep"].unique())
    fig, axes = plt.subplots(
        len(timesteps),
        len(sigmas),
        figsize=(max(1, len(sigmas)) * 2.2, max(1, len(timesteps)) * 1.8),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    all_values = summary["mean_latent_sym_kl"].to_numpy(dtype=float)
    finite = all_values[np.isfinite(all_values)]
    y_max = float(np.nanmax(finite)) if finite.size else 1.0
    y_max = max(y_max * 1.08, 1e-6)
    color_lookup: dict[tuple[str, float], object] = {}
    for family, fam_data in summary.groupby("family"):
        params = sorted(float(x) for x in fam_data["parameter_value"].unique())
        color_lookup.update({(family, value): color for value, color in family_colors_for_values(family, params).items()})
    for row_i, timestep in enumerate(timesteps):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = summary[(summary["timestep"] == timestep) & (summary["sigma"] == sigma)]
            if panel.empty:
                ax.axis("off")
                continue
            for (family, parameter_value), sub in panel.groupby(["family", "parameter_value"]):
                sub = sub.sort_values("abs_delta_bin")
                linestyle = "-" if family == "vary_beta" else "--"
                marker = "o" if family == "vary_beta" else "^"
                ax.errorbar(
                    sub["abs_delta_bin"],
                    sub["mean_latent_sym_kl"],
                    yerr=sub["sem_latent_sym_kl"],
                    color=color_lookup.get((family, float(parameter_value)), "black"),
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.0,
                    markersize=2.2,
                    capsize=1.5,
                    label=family_param_label(sub.iloc[0]),
                )
            if row_i == 0:
                ax.set_title(f"sigma={sigma:g}", fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"t={timestep}\nKL", fontsize=8)
            if row_i == len(timesteps) - 1:
                ax.set_xlabel("|current - previous\nsame-node value|", fontsize=8)
            ax.set_ylim(0, y_max)
            ax.tick_params(labelsize=7, length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_current_vs_previous_mu_displacement(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    required = {"mean_z0_mu_displacement", "mean_z1_mu_displacement"}
    if not required.issubset(summary.columns):
        return
    sigmas = sorted(summary["sigma"].unique())
    timesteps = sorted(summary["timestep"].unique())
    metrics = [
        ("z0", "mean_z0_mu_displacement", "sem_z0_mu_displacement", "o", "-"),
        ("z1", "mean_z1_mu_displacement", "sem_z1_mu_displacement", "s", ":"),
    ]
    fig, axes = plt.subplots(
        len(timesteps),
        len(sigmas),
        figsize=(max(1, len(sigmas)) * 2.2, max(1, len(timesteps)) * 1.85),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    all_values = np.concatenate(
        [
            summary[col].to_numpy(dtype=float)
            for _, col, _, _, _ in metrics
            if col in summary
        ]
    )
    finite = all_values[np.isfinite(all_values)]
    y_max = float(np.nanmax(finite)) if finite.size else 1.0
    y_max = max(y_max * 1.08, 1e-6)
    color_lookup: dict[tuple[str, float], object] = {}
    for family, fam_data in summary.groupby("family"):
        params = sorted(float(x) for x in fam_data["parameter_value"].unique())
        color_lookup.update({(family, value): color for value, color in family_colors_for_values(family, params).items()})
    for row_i, timestep in enumerate(timesteps):
        for col_i, sigma in enumerate(sigmas):
            ax = axes[row_i, col_i]
            panel = summary[(summary["timestep"] == timestep) & (summary["sigma"] == sigma)]
            if panel.empty:
                ax.axis("off")
                continue
            for (family, parameter_value), sub in panel.groupby(["family", "parameter_value"]):
                sub = sub.sort_values("abs_delta_bin")
                base_color = color_lookup.get((family, float(parameter_value)), "black")
                family_line = "-" if family == "vary_beta" else "--"
                for dim_label, mean_col, sem_col, marker, dim_line in metrics:
                    label = f"{family_param_label(sub.iloc[0])} {dim_label}"
                    ax.errorbar(
                        sub["abs_delta_bin"],
                        sub[mean_col],
                        yerr=sub[sem_col],
                        color=base_color,
                        linestyle=family_line if dim_label == "z0" else dim_line,
                        marker=marker,
                        linewidth=1.0,
                        markersize=2.0,
                        capsize=1.3,
                        label=label,
                        alpha=0.95 if dim_label == "z0" else 0.72,
                    )
            if row_i == 0:
                ax.set_title(f"sigma={sigma:g}", fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"t={timestep}\n|delta mu|", fontsize=8)
            if row_i == len(timesteps) - 1:
                ax.set_xlabel("|current - previous\nsame-node value|", fontsize=8)
            ax.set_ylim(0, y_max)
            ax.tick_params(labelsize=7, length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_last_paid_latents(payloads: list[dict]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        meta = payload["meta"]
        traces = pd.DataFrame(payload.get("traces", []))
        if traces.empty:
            continue
        traces = traces[traces["action_type"] == "observe"].copy()
        for (_, sample_set), group in traces.groupby(["condition_index", "sample_set"], sort=False):
            group = group.sort_values("timestep").reset_index(drop=True)
            if len(group) < 2:
                continue
            row = group.iloc[-2]
            rows.append(
                {
                    "family": meta["family"],
                    "parameter_name": meta["parameter_name"],
                    "parameter_value": float(meta["parameter_value"]),
                    "beta": float(meta["beta"]),
                    "opportunity": float(meta["opportunity"]),
                    "sigma": float(meta["sigma"]),
                    "seed": int(meta["seed"]),
                    "condition_index": int(row["condition_index"]),
                    "original_condition_index": int(row.get("original_condition_index", row["condition_index"])),
                    "sample_set": int(sample_set),
                    "last_paid_timestep": int(row["timestep"]),
                    "raw_mu": row["raw_mu"],
                    "raw_sigma": row["raw_sigma"],
                }
            )
    return pd.DataFrame(rows)


def build_sigma_pairwise_last_paid_kl(last_paid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if last_paid.empty:
        return last_paid, pd.DataFrame()
    group_cols = [
        "family",
        "parameter_name",
        "parameter_value",
        "beta",
        "opportunity",
        "seed",
        "original_condition_index",
        "sample_set",
    ]
    for _, group in last_paid.groupby(group_cols, sort=False):
        by_sigma = {float(row["sigma"]): row for _, row in group.iterrows()}
        sigmas = sorted(by_sigma)
        for sigma_a in sigmas:
            for sigma_b in sigmas:
                row_a = by_sigma[sigma_a]
                row_b = by_sigma[sigma_b]
                rows.append(
                    {
                        "family": row_a["family"],
                        "parameter_name": row_a["parameter_name"],
                        "parameter_value": float(row_a["parameter_value"]),
                        "beta": float(row_a["beta"]),
                        "opportunity": float(row_a["opportunity"]),
                        "seed": int(row_a["seed"]),
                        "original_condition_index": int(row_a["original_condition_index"]),
                        "sample_set": int(row_a["sample_set"]),
                        "sigma_a": float(sigma_a),
                        "sigma_b": float(sigma_b),
                        "last_paid_sigma_pair_sym_kl": diag_gaussian_sym_kl_from_sigma(
                            row_a["raw_mu"],
                            row_a["raw_sigma"],
                            row_b["raw_mu"],
                            row_b["raw_sigma"],
                        ),
                    }
                )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary = (
        pairwise.groupby(
            ["family", "parameter_name", "parameter_value", "beta", "opportunity", "sigma_a", "sigma_b"],
            dropna=False,
        )
        .agg(
            mean_last_paid_sigma_pair_sym_kl=("last_paid_sigma_pair_sym_kl", "mean"),
            sem_last_paid_sigma_pair_sym_kl=("last_paid_sigma_pair_sym_kl", sem_finite),
            n_pairs=("last_paid_sigma_pair_sym_kl", "count"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return pairwise, summary


def plot_sigma_pair_heatmaps(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        return
    finite = summary["mean_last_paid_sigma_pair_sym_kl"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    vmax = max(vmax, 1e-9)
    for (family, parameter_value), data in summary.groupby(["family", "parameter_value"]):
        sigmas = sorted(set(data["sigma_a"]).union(set(data["sigma_b"])))
        matrix = np.full((len(sigmas), len(sigmas)), np.nan, dtype=float)
        for _, row in data.iterrows():
            i = sigmas.index(float(row["sigma_a"]))
            j = sigmas.index(float(row["sigma_b"]))
            matrix[i, j] = float(row["mean_last_paid_sigma_pair_sym_kl"])
        fig, ax = plt.subplots(figsize=(3.0, 2.7))
        im = ax.imshow(matrix, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas], fontsize=7)
        ax.set_yticks(range(len(sigmas)), [f"{s:g}" for s in sigmas], fontsize=7)
        ax.set_xlabel("sigma b", fontsize=8)
        ax.set_ylabel("sigma a", fontsize=8)
        title = family_param_label(data.iloc[0])
        ax.set_title(f"Last-paid latent KL\n{title}", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("mean symmetric KL", fontsize=8)
        for i in range(len(sigmas)):
            for j in range(len(sigmas)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.2g}", ha="center", va="center", fontsize=6, color="white" if matrix[i, j] > 0.55 * vmax else "black")
        fig.tight_layout()
        param_name = "beta" if family == "vary_beta" else "opp" if family == "vary_opportunity" else "param"
        outpath = outdir / (
            f"last_paid_latent_sigma_pairwise_kl_heatmap_"
            f"{family}_{param_name}_{diag.value_token(parameter_value)}.png"
        )
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_analysis_plots(ui_dir: Path, manifest: list[dict], diff_bin_width: float) -> None:
    payloads = trace_payloads(ui_dir, manifest)
    if not payloads:
        return
    outdir = ui_dir / "analyses"
    outdir.mkdir(parents=True, exist_ok=True)
    change_detail, change_summary = build_current_vs_previous_latent_change(payloads, diff_bin_width)
    change_detail.to_csv(outdir / "latent_current_vs_previous_observation_kl_detail.csv", index=False)
    change_summary.to_csv(outdir / "latent_current_vs_previous_observation_kl_summary.csv", index=False)
    plot_current_vs_previous_latent_change(
        change_summary,
        outdir / "latent_current_vs_previous_observation_kl_by_timestep_sigma.png",
    )
    plot_current_vs_previous_mu_displacement(
        change_summary,
        outdir / "latent_current_vs_previous_observation_z_mu_displacement_by_timestep_sigma.png",
    )
    last_paid = build_last_paid_latents(payloads)
    last_paid.to_csv(outdir / "last_paid_latents_for_sigma_pairwise.csv", index=False)
    sigma_pairwise, sigma_summary = build_sigma_pairwise_last_paid_kl(last_paid)
    sigma_pairwise.to_csv(outdir / "last_paid_latent_sigma_pairwise_kl_detail.csv", index=False)
    sigma_summary.to_csv(outdir / "last_paid_latent_sigma_pairwise_kl_summary.csv", index=False)
    plot_sigma_pair_heatmaps(sigma_summary, outdir)


class ReusableThreadingHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


def serve_ui(ui_dir: Path, port: int, bind: str = "0.0.0.0") -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ui_dir))
    with ReusableThreadingHTTPServer((bind, int(port)), handler) as httpd:
        print(f"Serving {ui_dir}", flush=True)
        print(f"HTTP server: http://{bind}:{int(port)}/", flush=True)
        print("In VS Code SSH, forward/open this port from the Ports panel.", flush=True)
        httpd.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", nargs="?", default="default")
    parser.add_argument("--preset-file", default=str(diag.preset_file_default()))
    parser.add_argument("--vary-beta-values", "--betas", dest="vary_beta_values", default=None)
    parser.add_argument("--vary-opportunity-values", "--opportunity-costs", dest="vary_opportunity_values", default=None)
    parser.add_argument("--sigmas", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--rnn-units", "--rnn-dims", dest="rnn_units", default=None)
    parser.add_argument("--latent-dim", "--latent-dims", dest="latent_dim", default=None)
    parser.add_argument("--lambda-value", "--lambdas", dest="lambda_value", default=None)
    parser.add_argument("--alpha", "--alphas", dest="alpha", default=None)
    parser.add_argument("--n-sample-sets", type=int, default=50)
    parser.add_argument("--n-reward-combinations", "--max-reward-combinations", type=int, default=0)
    parser.add_argument("--max-observations-before-stop", type=int, default=None)
    parser.add_argument("--checkpoint-root", default="outputs/jax_models")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--sampled-lambda-critic", choices=["q", "value"], default="q")
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--kl-start-multiplier", type=float, default=1.0)
    parser.add_argument("--kl-annealing-epochs", type=int, default=0)
    parser.add_argument("--force-round-robin-observations", action="store_true")
    parser.add_argument(
        "--force-first-observe-node",
        type=int,
        default=0,
        help=(
            "Force only the first action to observe this 1-indexed node, then "
            "let the model policy choose all later actions. Use 0 to disable."
        ),
    )
    parser.add_argument("--generate-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--latent-dims-to-plot", default="0,1")
    parser.add_argument("--ui-subdir", default="latent_trace_ui")
    parser.add_argument(
        "--analysis-diff-bin-width",
        type=float,
        default=0.5,
        help=(
            "Deprecated no-op. These static analysis plots are now generated by "
            "analysis/sample_set_pairwise_last_paid_kl_jax.py."
        ),
    )
    parser.add_argument(
        "--skip-analysis-plots",
        action="store_true",
        help=(
            "Deprecated no-op. The UI builder now only rebuilds the browser UI; "
            "static analysis plots live in sample_set_pairwise_last_paid_kl_jax.py."
        ),
    )
    parser.add_argument("--serve", action="store_true", help="Serve the generated UI from the computed UI directory.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve.")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address for --serve.")
    args = parser.parse_args()
    return diag.load_default_preset(args)


def main() -> None:
    args = parse_args()
    latent_dims = parse_dims(args.latent_dims_to_plot)
    outdir = diag.resolve_output_dir(args)
    ui_dir = outdir / args.ui_subdir
    ui_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ui_dir / "manifest.json"
    if args.generate_cache:
        manifest = generate_cache(args, ui_dir, latent_dims)
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        raise FileNotFoundError(
            f"No trace manifest found at {manifest_path}. Re-run with --generate-cache "
            "to create per-sample-set trace files."
        )
    html_path = write_html(ui_dir, manifest)
    print(f"Wrote UI to {html_path}", flush=True)
    if args.serve:
        serve_ui(ui_dir, int(args.port), str(args.bind))
    else:
        print(
            f"Serve it with: vae_env/bin/python {Path(__file__).as_posix()} "
            f"{args.tree} [same args] --serve --port 8765",
            flush=True,
        )


if __name__ == "__main__":
    main()
