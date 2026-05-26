# Manifest: VAE-LSTM entropy/carry-KL archive

Archive directory:

```text
analyses/exp_binary/archive/vae_lstm_uniform_2n_entropy_carrykl_2026-05-25
```

Repository commit:

```text
a710652
```

The working tree was clean before the archive directory was created, so there is no separate uncommitted patch file for this archive.

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

## Archived Model Outputs

`outputs/models/` contains trained weights and training logs for:

```text
beta = 0.25, 0.5, 0.75, 0.8, 0.9, 1.0
lambda = 10.0
alpha = 0.0
opportunity = 0.0
expansion = lstm
variant = vae
seed = 1
tree_size = 2
```

`outputs/simulations/` contains the matching simulation CSVs.

## Archived Result PDFs

Successful beta scan:

```text
beta = 0.8, 0.9
```

Included summaries:

```text
average_V_vs_MI
average_kl_d
choice_probability_by_other_reward_stop_t2
chosen_reward_given_stop
kl_d_by_reward
kl_d_t1_by_reward
reconstruction_accuracy_by_reward
stop_probability
training_diagnostics
training_gradients
training_lstm_reward_probe
```

Broader beta scan:

```text
beta = 0.25, 0.5, 0.75, 1.0
```

Included summaries:

```text
average_V_vs_MI
average_kl_d
choice_probability_by_other_reward_stop_t2
chosen_reward_given_stop
kl_d_by_reward
kl_d_t1_by_reward
reconstruction_accuracy_by_reward
stop_probability
training_diagnostics
training_gradients
training_lstm_reward_probe
```

## Reproduction Commands

Successful run:

```bash
bash model/run_grid.sh 0.8 0.9 2 0 0 1 0 0 1 10.0 1 1 train uniform 2 lstm vae
```

Broader beta scan:

```bash
bash model/run_grid.sh 0.25 1 4 0 0 1 0 0 1 10.0 1 1 train uniform 2 lstm vae
```

Successful summary plot:

```bash
Rscript analyses/exp_binary/plot_beta_summaries.R "0.8,0.9" "10.0" "0.0" "0.0" outputs/simulations results 2 uniform lstm vae
```

Successful training diagnostics:

```bash
Rscript analyses/exp_binary/diagnose_training_logs.R "0.8,0.9" "10.0" "0.0" "0.0" outputs/models results 2 lstm "1:1" vae
```

Broader training diagnostics:

```bash
Rscript analyses/exp_binary/diagnose_training_logs.R "0.25,0.5,0.75,1.0" "10.0" "0.0" "0.0" outputs/models results 2 lstm "1:1" vae
```
