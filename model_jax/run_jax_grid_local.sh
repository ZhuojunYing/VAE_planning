#!/bin/bash

set -euo pipefail

if [ "$#" -lt 18 ]; then
    echo "Usage beta_list alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version model_variant tree_config rnn_units latent_dim num_updates num_envs [extra_jax_args...]"
    echo "Example: $0 1000.0 0 0 1 0.0266666666667,0.106666666667 100.0 31 33 train uniform 3 lstm vae bandit3 32 16 24000 200 --backend cpu"
    exit 1
fi

beta_list=$1
alpha_min=$2
alpha_max=$3
alpha_steps=$4
opportunity_list=$5
lambda_list=$6
seed_min=$7
seed_max=$8
train=$9
input_type=${10}
tree_size=${11}
expansion_decision_version=${12}
model_variant=${13}
tree_config=${14}
rnn_units=${15}
latent_dim=${16}
num_updates=${17}
num_envs=${18}
extra_args=("${@:19}")

python - "$beta_list" "$alpha_min" "$alpha_max" "$alpha_steps" "$opportunity_list" "$lambda_list" \
    "$seed_min" "$seed_max" "$train" "$input_type" "$tree_size" "$expansion_decision_version" \
    "$model_variant" "$tree_config" "$rnn_units" "$latent_dim" "$num_updates" "$num_envs" "${extra_args[@]}" <<'PY' |
import itertools
import shlex
import sys
import numpy as np

(
    beta_list, alpha_min, alpha_max, alpha_steps, opportunity_list, lambda_list,
    seed_min, seed_max, train, input_type, tree_size, expansion_decision_version,
    model_variant, tree_config, rnn_units, latent_dim, num_updates, num_envs, *extra_args
) = sys.argv[1:]

def vals(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]

betas = vals(beta_list)
alphas = list(np.linspace(float(alpha_min), float(alpha_max), int(alpha_steps)))
opps = vals(opportunity_list)
lambdas = vals(lambda_list)
num_steps = int(tree_size)
steps_per_epoch = int(num_updates) * int(num_envs) * num_steps // 120
steps_per_epoch = max(steps_per_epoch, int(num_envs) * num_steps)

for seed, beta, alpha, lambda_, opp in itertools.product(
    range(int(seed_min), int(seed_max) + 1), betas, alphas, lambdas, opps
):
    cmd = [
        "bash", "model_jax/run_jax_model.sh",
        str(lambda_), str(alpha), str(beta),
        "outputs/jax_models/", "outputs/jax_simulations/", "2000",
        input_type, str(seed), train, str(opp), tree_size,
        expansion_decision_version, model_variant, tree_config, rnn_units, latent_dim,
        "--num-envs", num_envs,
        "--num-steps", str(num_steps),
        "--steps-per-epoch", str(steps_per_epoch),
        *extra_args,
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
PY
while read -r cmd; do
    echo "Executing: $cmd"
    eval "$cmd"
done
