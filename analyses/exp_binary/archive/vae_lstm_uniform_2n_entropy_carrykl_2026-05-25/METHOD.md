# Method: archived VAE-LSTM stopping model with annealed expansion entropy

Archived on 2026-05-25 from repository commit `a710652`. The working tree was clean before this archive directory was created.

## Purpose

This archive records the VAE-LSTM version that recovered the desired two-node uniform stopping behavior after increasing the annealed entropy pressure on the joint expansion policy.

The target qualitative policy is:

- before seeing any reward, usually observe one node unless information cost is high enough to stop immediately;
- after seeing an extreme negative reward, stop and choose the other node;
- after seeing an extreme positive reward, stop and choose the observed node;
- continue mainly for ambiguous rewards near the middle.

## Task

- Tree size: 2 nodes.
- Input condition: `uniform`.
- Rewards are sampled independently from `{-4, -3, -2, -1, 1, 2, 3, 4}`.
- Each node is also a terminal path, so choosing a terminal path is equivalent to choosing node 1 or node 2.
- At each expansion decision the model chooses one joint action:
  - observe node 1;
  - observe node 2;
  - stop and choose path 1;
  - stop and choose path 2.
- The action mask permits unobserved-node observe actions and both terminal choices. After observing node 1, for example, the available actions are observe node 2, stop choose path 1, and stop choose path 2.

## Archived Configuration

Main successful grid:

```text
lambda = 10.0
alpha = 0.0
beta = 0.8, 0.9
opportunity_cost = 0.0
seeds = 1
input_type = uniform
tree_size = 2
expansion_decision_version = lstm
model_variant = vae
epochs = 60
trials_per_epoch = 200
batch_size = 200
simulation_trials = 2000
```

A broader beta scan is also included:

```text
beta = 0.25, 0.5, 0.75, 1.0
all other settings as above
```

## Commands

Successful run:

```bash
bash model/run_grid.sh 0.8 0.9 2 0 0 1 0 0 1 10.0 1 1 train uniform 2 lstm vae
```

Broader beta scan:

```bash
bash model/run_grid.sh 0.25 1 4 0 0 1 0 0 1 10.0 1 1 train uniform 2 lstm vae
```

Plot successful run:

```bash
Rscript analyses/exp_binary/plot_beta_summaries.R \
  "0.8,0.9" \
  "10.0" \
  "0.0" \
  "0.0" \
  outputs/simulations \
  results \
  2 \
  uniform \
  lstm \
  vae
```

Training diagnostics for successful run:

```bash
Rscript analyses/exp_binary/diagnose_training_logs.R \
  "0.8,0.9" \
  "10.0" \
  "0.0" \
  "0.0" \
  outputs/models \
  results \
  2 \
  lstm \
  "1:1" \
  vae
```

## Model Details

The archived model is `VariationalRNN` with `model_variant = vae`.

Important implementation details in this archive:

- The expansion policy and terminal choice are combined into one joint head.
- Stop actions are legal before observing any reward.
- Rewards are passed to the recurrent state as categorical one-hot inputs rather than scalar inputs.
- The VAE encoder/decoder bottleneck is active.
- `alpha = 0.0`, so reconstruction is logged and probed but is not an optimization pressure.
- For `expansion_decision_version = lstm`, the expansion head reads the LSTM state.
- The model uses a carry-KL convention for the LSTM/pre-LSTM decision state: after observing a reward, stopping immediately does not pay the carried KL for that reward, but continuing to observe the second node does.
- The LSTM reward probe is diagnostic only and is trained to test how well reward identity is linearly recoverable from the LSTM state.

## Losses

The policy objective follows the current Dreamer-style rollout setup used in this repo:

- The joint expansion action is sampled along a full trajectory.
- The expansion policy is trained with sampled-action log probabilities and return-style targets.
- The critic is trained on return targets from the sampled trajectory.
- KL contributes as an information cost only when the sampled transition carries encoded information forward into another observation decision under the LSTM/pre-LSTM convention.

The expansion entropy schedule is the critical change that made this run work:

```text
expansion_entropy_start = 0.2
expansion_entropy_end = 0.01
expansion_entropy_annealing_epochs = 100
```

The entropy bonus is targeted to expansion decisions after at least one reward has been observed. This keeps exploration alive at the boundary decision without strongly encouraging random immediate stopping before any observation.

Epsilon-greedy exploration remains:

```text
epsilon_start = 0.5
epsilon_end = 0.0
epsilon_annealing_epochs = 100
```

## Observed Behavior

For the successful beta scan, initial stopping was nearly zero:

```text
beta 0.8: P(stop at timestep 1) = 0.0020
beta 0.9: P(stop at timestep 1) = 0.0020
```

Conditional stopping at timestep 2 after the first observed reward:

```text
beta 0.8:
  reward -4: 1.0000
  reward -3: 1.0000
  reward -2: 0.9962
  reward -1: 0.3591
  reward  1: 0.4080
  reward  2: 1.0000
  reward  3: 1.0000
  reward  4: 1.0000

beta 0.9:
  reward -4: 1.0000
  reward -3: 1.0000
  reward -2: 0.9925
  reward -1: 0.1853
  reward  1: 0.7559
  reward  2: 1.0000
  reward  3: 1.0000
  reward  4: 1.0000
```

This is the first VAE run in this debugging sequence that cleanly shows boundary stopping on both negative and positive extremes while still mostly continuing near the ambiguous middle.

## Archived Outputs

The archive includes:

- source snapshots for `model/` and the two exp-binary plotting scripts;
- model weights and training logs for beta `0.25, 0.5, 0.75, 0.8, 0.9, 1.0`;
- simulation CSVs for the same beta values;
- summary and training diagnostic PDFs for beta `0.8, 0.9`;
- summary and training diagnostic PDFs for beta `0.25, 0.5, 0.75, 1.0`.
