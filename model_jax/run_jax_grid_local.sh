#!/bin/bash

set -euo pipefail

if [ "$#" -lt 18 ]; then
    echo "Usage beta_list alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version model_variant tree_config rnn_units latent_dim num_updates num_envs [n_sim_trials] [extra_jax_args...]"
    echo "Example: $0 1000.0 0 0 1 0.0266666666667,0.106666666667 100.0 31 33 train uniform 3 lstm vae bandit3 32 16 24000 200 10000 --backend cpu --allow-node-revisit"
    echo "Add --allow-node-revisit in extra_jax_args, or set ALLOW_NODE_REVISIT=1, to keep observed nodes legal."
    echo "Revisit runs default to --max-observations-before-stop 10 and --num-steps 11 unless overridden."
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
n_sim_trials=${19:-"2000"}
if [[ "$n_sim_trials" == -* ]]; then
    n_sim_trials="2000"
    extra_args=("${@:19}")
else
    extra_args=("${@:20}")
fi

has_extra_arg() {
    local needle="$1"
    local arg
    for arg in "${extra_args[@]:-}"; do
        if [ "$arg" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

case "${ALLOW_NODE_REVISIT:-}" in
    1|true|TRUE|yes|YES|on|ON)
        if ! has_extra_arg "--allow-node-revisit"; then
            extra_args+=("--allow-node-revisit")
        fi
        ;;
esac

python - "$beta_list" "$alpha_min" "$alpha_max" "$alpha_steps" "$opportunity_list" "$lambda_list" \
    "$seed_min" "$seed_max" "$train" "$input_type" "$tree_size" "$expansion_decision_version" \
    "$model_variant" "$tree_config" "$rnn_units" "$latent_dim" "$num_updates" "$num_envs" "$n_sim_trials" "${extra_args[@]}" <<'PY' |
import itertools
import os
import shlex
import sys
import numpy as np

(
    beta_list, alpha_min, alpha_max, alpha_steps, opportunity_list, lambda_list,
    seed_min, seed_max, train, input_type, tree_size, expansion_decision_version,
    model_variant, tree_config, rnn_units, latent_dim, num_updates, num_envs, n_sim_trials, *extra_args
) = sys.argv[1:]

def vals(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]

def extra_has(flag):
    return any(arg == flag or str(arg).startswith(flag + "=") for arg in extra_args)

def extra_int(flag, default):
    for i, arg in enumerate(extra_args):
        text = str(arg)
        if text == flag and i + 1 < len(extra_args):
            return int(extra_args[i + 1])
        if text.startswith(flag + "="):
            return int(text.split("=", 1)[1])
    return int(default)

betas = vals(beta_list)
alphas = list(np.linspace(float(alpha_min), float(alpha_max), int(alpha_steps)))
opps = vals(opportunity_list)
lambdas = vals(lambda_list)
allow_revisit = extra_has("--allow-node-revisit")
max_observations = extra_int(
    "--max-observations-before-stop",
    os.environ.get("MAX_OBSERVATIONS_BEFORE_STOP", "10"),
)
num_steps = extra_int(
    "--num-steps",
    (max_observations + 1) if allow_revisit else int(tree_size),
)
steps_per_epoch = int(num_updates) * int(num_envs) * num_steps // 120
steps_per_epoch = max(steps_per_epoch, int(num_envs) * num_steps)

for seed, beta, alpha, lambda_, opp in itertools.product(
    range(int(seed_min), int(seed_max) + 1), betas, alphas, lambdas, opps
):
    cmd = [
        "bash", "model_jax/run_jax_model.sh",
        str(lambda_), str(alpha), str(beta),
        "outputs/jax_models/", "outputs/jax_simulations/", n_sim_trials,
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
