# Archive Manifest

Archive directory:

```text
analyses/exp_binary/archive/rnn_lstm_uniform_2n_opportunity_0p1_0p4_2026-05-22
```

Base git commit:

```text
cbb6838
```

Archived run:

```text
lambda = 1.0
alpha = 0.0
beta = 1.0
opportunity_cost = 0.1, 0.2, 0.30000000000000004, 0.4
seeds = 1, 2, 3, 4, 5
input_type = uniform
tree_size = 2
expansion_decision_version = lstm
model_variant = rnn
```

Copied source files:

```text
model/config.py
model/helper.py
model/main.py
model/model.py
model/train.py
model/simulate.py
model/run_grid.sh
model/run_model.sh
analyses/plot_beta_summaries.R
analyses/diagnose_training_logs.R
```

Copied outputs:

```text
results/: 8 PDF files
outputs/models/: 20 weight files and 20 training-log CSV files
outputs/simulations/: 20 simulation CSV files
```

Patch:

```text
UNCOMMITTED_CHANGES.patch
```

Notes:

- The repository working tree was dirty when this archive was created.
- The copied files in this archive are the authoritative snapshot for this archived behavior.
- `generate_commands.py` existed as an untracked generated helper in the repository root, but it is not part of this archive.

