# Manifest: multi-rollout return-target archive

Archive directory:

```text
analyses/exp_binary/archive/multirollout_return_target_2026-05-27
```

Repository commit:

```text
13f657b
```

The archived model source files matched commit `13f657b`. The archive includes an empty patch file for consistency with older snapshots:

```text
UNCOMMITTED_CHANGES.patch
```

## Source Snapshots

```text
model/config.py
model/helper.py
model/main.py
model/model.py
model/run_grid.sh
model/run_model.sh
model/simulate.py
model/train.py
analyses/diagnose_training_logs.R
analyses/plot_beta_summaries.R
```

## Representative Result PDFs

```text
results/continue_probability_uniform_lambda_10.0_alpha_0.0_beta_1.0_opportunity_0.1_0.2_0.3_0.4_expansion_lstm_variant_rnn_2n.pdf
results/deep_probe_accuracy_by_reward_uniform_lambda_10.0_alpha_0.0_beta_1.0_opportunity_0.1_0.2_0.3_0.4_expansion_lstm_variant_rnn_2n.pdf
```

## Main Training Setting

```text
lambda = 10.0
alpha = 0.0
beta = 1.0
opportunity = 0.1, 0.2, 0.3, 0.4
seed = 6
input_type = uniform
tree_size = 2
expansion = lstm
variant = rnn
epochs = 120
RETURN_TARGET_ROLLOUTS = 4
```

## Reproduction Commands

Training:

```bash
bash model/run_grid.sh 1.0 1.0 1 0 0 1 0.1 0.4 4 10.0 6 6 train uniform 2 lstm rnn
```

Summary:

```bash
Rscript analyses/exp_binary/plot_beta_summaries.R "1.0" "10.0" "0.0" "0.1,0.2,0.3,0.4" outputs/simulations results 2 uniform lstm rnn
```

Training diagnostics:

```bash
Rscript analyses/exp_binary/diagnose_training_logs.R "1.0" "10.0" "0.0" "0.1,0.2,0.3,0.4" outputs/models results 2 lstm "6:6" rnn
```
