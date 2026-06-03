# Method: archived multi-rollout return-target model

Archived on 2026-05-27 from repository commit `13f657b`.

This archive records the version where the two-node RNN/LSTM behavior looked normal again after adding multiple sampled rollouts for the return target and aligning the critic input with the expansion-policy input.

## Purpose

The immediate debugging target was the asymmetric or unstable continue/stop behavior in the two-node uniform task. The model should observe initially, then stop at clear boundary rewards and continue mainly around ambiguous rewards.

This version changes the training target used by the sampled expansion-policy update:

- sample multiple on-policy trajectories for the same batch;
- replay those sampled expansion choices under the current model;
- average the return-target losses over the sampled rollouts;
- keep detailed reward-conditioned diagnostics from the first rollout so existing log columns remain interpretable.

The default number of rollout samples is:

```text
RETURN_TARGET_ROLLOUTS = 4
```

It can be overridden from the shell:

```bash
RETURN_TARGET_ROLLOUTS=8 bash model/run_grid.sh ...
RETURN_TARGET_ROLLOUTS=1 bash model/run_grid.sh ...
```

`RETURN_TARGET_ROLLOUTS=1` recovers the previous single-rollout behavior.

## Task

Representative successful task:

```text
input_type = uniform
tree_size = 2
tree_config = default two-node bandit
expansion_decision_version = lstm
model_variant = rnn
lambda = 10.0
alpha = 0.0
beta = 1.0
opportunity_cost = 0.1, 0.2, 0.3, 0.4
seed = 6
epochs = 120
simulation_trials = 2000
```

Rewards are sampled from:

```text
{-4, -3, -2, -1, 1, 2, 3, 4}
```

At each decision, the joint expansion head chooses among observe actions for available nodes and stop-and-choose terminal actions.

## Key Implementation Details

- `train_step()` accepts `return_target_rollouts`.
- `train_model()` reads `RETURN_TARGET_ROLLOUTS` from the environment and defaults to 4.
- The critic head takes the same state representation as the expansion head.
- The critic head is a small MLP rather than a single linear layer.
- Reward-conditioned diagnostics include:
  - continue probability after each reward;
  - critic value after each reward;
  - terminal best-path probability before and after observation;
  - return target;
  - advantage.
- The first observation is not charged opportunity cost in the current environment convention.

## Representative Command

```bash
bash model/run_grid.sh \
  1.0 1.0 1 \
  0 0 1 \
  0.1 0.4 4 \
  10.0 \
  6 6 \
  train \
  uniform \
  2 \
  lstm \
  rnn
```

Plot command:

```bash
Rscript analyses/exp_binary/plot_beta_summaries.R \
  "1.0" \
  "10.0" \
  "0.0" \
  "0.1,0.2,0.3,0.4" \
  outputs/simulations \
  results \
  2 \
  uniform \
  lstm \
  rnn
```

## Observed Behavior

The archived representative summary is:

```text
results/continue_probability_uniform_lambda_10.0_alpha_0.0_beta_1.0_opportunity_0.1_0.2_0.3_0.4_expansion_lstm_variant_rnn_2n.pdf
```

The behavior after the multi-rollout target is qualitatively cleaner: initial observation remains high, boundary rewards are much closer to stopping, and continue probability is concentrated around ambiguous rewards.

## Notes

The model source files matched commit `13f657b` when archived. `UNCOMMITTED_CHANGES.patch` is included for consistency with older archives, but it is empty for this snapshot.
