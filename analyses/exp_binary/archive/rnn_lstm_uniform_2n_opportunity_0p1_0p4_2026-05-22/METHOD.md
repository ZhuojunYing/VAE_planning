# Method: archived 2-node uniform RNN stopping model

Archived on 2026-05-22 from repository commit `cbb6838` plus the uncommitted changes saved in `UNCOMMITTED_CHANGES.patch`.

## Purpose

This archive records the first plain-RNN variant that produced the desired qualitative stopping behavior on the two-node uniform task. The task is intentionally simple: each trial has two reward nodes, the model decides at each timestep whether to observe an unobserved node or stop, and if it stops it chooses one node/path as the terminal action.

The desired policy is:

- If the opportunity cost is low enough, observe one node.
- After observing an extreme reward, stop at the next decision:
  - reward `-4`: stop and choose the other node.
  - reward `+4`: stop and choose the observed node.
- If the opportunity cost is high enough, stop immediately before observing.

## Task

For the archived run:

- Tree size: 2 nodes.
- Input condition: `uniform`.
- Rewards sampled independently from `{-4, -3, -2, -1, 1, 2, 3, 4}`.
- Paths: one path per node, so terminal path choice is equivalent to choosing node 1 or node 2.
- At each timestep the expansion policy chooses among `node_1`, `node_2`, and `stop`.
- A visited-node mask prevents observing the same node twice.
- Once `stop` is sampled, an active mask freezes subsequent state updates and later expansion steps are ignored.

## Archived Configuration

The archived model uses:

```text
lambda = 1.0
alpha = 0.0
beta = 1.0
opportunity_cost = 0.1, 0.2, 0.30000000000000004, 0.4
seeds = 1:5
input_type = uniform
tree_size = 2
expansion_decision_version = lstm
model_variant = rnn
epochs = 60
trials_per_epoch = 200
batch_size = 200
simulation_trials = 2000 per trained model
```

The grid was launched with:

```bash
bash model/run_grid.sh 0 0 1 0 0 1 0.1 0.4 4 1.0 1 5 train uniform 2 lstm rnn
```

The main summary plots were generated with:

```bash
Rscript analyses/exp_binary/plot_beta_summaries.R \
  "1.0" \
  "1.0" \
  "0.0" \
  "0.1,0.2,0.30000000000000004,0.4" \
  outputs/simulations \
  results \
  2 \
  uniform \
  lstm \
  rnn
```

## Architecture

The model class is `VariationalRNN`, but this archive uses `model_variant = rnn`, so the autoencoder pathway is disabled.

Active components:

- LSTM cell with 64 hidden units.
- Expansion head: dense layer from the LSTM context to logits over `time_steps + 1` actions, where the final index is the stop token.
- Action head: dense layer from the same representation family to logits over terminal paths.
- Critic head: dense layer used as a value baseline for expansion decisions.

Inactive in the archived RNN variant:

- Encoder.
- Decoder.
- Categorical reconstruction loss.
- Time-conditional KL loss.
- Learned prior.

The RNN receives, at each selected expansion step, a vector containing:

- one-hot selected expansion action over nodes plus stop token;
- observed reward for the selected node, or zero when the selected action is stop.

For `expansion_decision_version = lstm`, the expansion policy at each decision uses the post-LSTM hidden representation from the previous observation/action context.

## Losses

The total training update is split by parameter group in `model/train.py`, but the relevant objectives are defined in `model/model.py`.

### Terminal action loss

The action head is trained to maximize terminal expected path reward:

```python
expected_reward = sum(action_probs * path_rewards)
action_loss = 1.0 - expected_reward / reward_norm
```

For this two-node uniform task, `reward_norm = 1.5625`.

### Expansion policy loss

The expansion policy is trained with a policy-gradient/PPO-style loss. A rollout samples expansion decisions, then the training pass reuses those sampled actions and old log probabilities.

For each valid decision timestep:

```python
sampled_return =
    expected_action_reward / reward_norm
    - opportunity_cost * non_stop_expansion
    - current_beta * kl_d * non_stop_expansion

advantage = sampled_return - critic_baseline
```

In the archived RNN variant, `kl_d = 0`, so the expansion return is controlled by normalized action value and opportunity cost.

The expansion policy loss also includes an entropy bonus:

```python
loss = policy_loss - 0.01 * entropy
```

### Opportunity-cost signal

Opportunity cost enters the model in two places:

- Directly in the expansion return above, penalizing each non-stop expansion action.
- As an auxiliary opportunity policy loss with negative advantage for non-stop expansion actions, weighted by `opportunity_cost`.

Thus a second observation in the two-node task pays another opportunity cost. Stopping at a timestep avoids that timestep's non-stop penalty.

### Critic loss

The critic predicts the best currently observed partial path reward:

```python
value_target = max(partial_path_rewards_from_observed_inputs)
critic_loss = mean_squared_error(value_pred, value_target)
```

For the two-node task, the critic coefficient during this 60-epoch run is `0.1`.

## Training Schedule

Training uses `AdamW` with cosine decay:

- initial learning rate: `1e-5`;
- warmup target: `3e-4`;
- weight decay: `1e-4`;
- clip norm: `20.0`.

Expansion exploration is epsilon-greedy:

- starts at `0.5`;
- linearly anneals toward `0.0` over 100 epochs;
- because this archive trains for 60 epochs, epsilon remains nonzero at the end.

The training loop uses `ppo_epochs = 3`; after the full parameter update, additional PPO-style expansion-head updates are applied to the same sampled expansion actions.

## Evaluation and Plots

Simulation uses 2000 fresh uniform trials per trained model. The exported simulation CSV repeats trial-level expansion decisions across node rows and records:

- selected expansion node and reward at each timestep;
- stop decisions;
- terminal chosen path;
- achieved terminal reward;
- reconstruction estimates, which are uninformative for the RNN variant.

The key diagnostic plot is:

```text
results/stop_probability_uniform_lambda_1.0_alpha_0.0_beta_1.0_opportunity_0.1_0.2_0.30000000000000004_0.4_expansion_lstm_variant_rnn_2n.pdf
```

It contains:

- `P(stop at timestep 1)`: initial stopping before any reward is observed.
- `P(stop at timestep 2 | reward observed at timestep 1)`: conditional stopping after observing the first reward.

For a two-node task there is no meaningful `P(stop after reward at timestep 2)` panel, because the timestep-2 decision occurs before observing reward at timestep 2.

## Observed Behavior

Initial stopping:

```text
opportunity 0.1: 0.0185
opportunity 0.2: 0.1366
opportunity 0.3: 0.9289
opportunity 0.4: 0.9970
```

Conditional stopping at timestep 2 after observing the first reward:

```text
opportunity 0.1:
  reward -4: 0.9652
  reward -3: 0.9491
  reward -2: 0.7573
  reward -1: 0.0545
  reward  1: 0.0615
  reward  2: 0.5455
  reward  3: 0.8676
  reward  4: 0.9530

opportunity 0.2:
  rewards -4,-3,-2,-1,1,2,3,4: 1.0000

opportunity 0.3:
  rewards -4,-3,-2,-1,1,2,3,4: 1.0000 among trials that observed a first reward

opportunity 0.4:
  observed first rewards in the simulation all had stop probability 1.0000, but nearly all trials stopped at timestep 1.
```

This captures the useful operating range: `opportunity = 0.1` produces a graded boundary-sensitive stopping rule, `0.2` stops reliably after one observation, and `0.3`/`0.4` mostly choose not to observe at all.

## Archive Contents

The archive contains:

- `model/`: copied model and run scripts.
- `analyses/`: copied R diagnostic/plotting scripts.
- `results/`: copied PDF summaries for the archived opportunity grid.
- `outputs/models/`: copied weights and training logs for all 20 archived models.
- `outputs/simulations/`: copied simulation CSVs for all 20 archived models.
- `UNCOMMITTED_CHANGES.patch`: patch from the working tree at archive time.

