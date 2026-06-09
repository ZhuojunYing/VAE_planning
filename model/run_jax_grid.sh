#!/bin/bash

is_train_arg() {
    case "$1" in
        train|simulate|simulation|inference|eval|evaluate) return 0 ;;
        *) return 1 ;;
    esac
}

if ! {
    { [ "$#" -ge 16 ] && [ "$#" -le 20 ] && is_train_arg "${13}"; } ||
    { [ "$#" -ge 14 ] && [ "$#" -le 18 ] && is_train_arg "${11}"; } ||
    { [ "$#" -ge 12 ] && [ "$#" -le 16 ] && is_train_arg "${9}"; }
}; then
    echo "Usage legacy: $0 beta_log10_min beta_log10_max beta_steps alpha_min alpha_max alpha_steps opportunity_min opportunity_max opportunity_steps lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim]"
    echo "Usage opp/lambda lists: $0 beta_log10_min beta_log10_max beta_steps alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim]"
    echo "Usage beta/opp/lambda lists: $0 beta_list alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim]"
    echo "Example: $0 1000.0 0 0 1 0.025,0.125,0.225 100.0 1 3 train uniform 3 lstm jax_ppo bandit3 64 32"
    exit 1
fi

if [ "$#" -ge 16 ] && is_train_arg "${13}"; then
    arg_mode="legacy"
elif [ "$#" -ge 14 ] && is_train_arg "${11}"; then
    arg_mode="opportunity_lambda_lists"
else
    arg_mode="beta_opportunity_lambda_lists"
fi

if [ "$arg_mode" = "legacy" ]; then
    beta_arg=$1
    beta_max=$2
    beta_steps=$3
    alpha_log_min=$4
    alpha_log_max=$5
    alpha_steps=$6
    opportunity_arg=$7
    opportunity_max=$8
    opportunity_steps=$9
    lambda_str=${10}
    seed_str=${11}
    seed_max=${12}
    train=${13}
    input_type=${14}
    tree_size=${15}
    expansion_decision_version=${16}
    model_variant=${17:-"jax_ppo"}
    tree_config=${18:-""}
    rnn_units=${19:-"64"}
    latent_dim=${20:-"32"}
elif [ "$arg_mode" = "opportunity_lambda_lists" ]; then
    beta_arg=$1
    beta_max=$2
    beta_steps=$3
    alpha_log_min=$4
    alpha_log_max=$5
    alpha_steps=$6
    opportunity_arg=$7
    opportunity_max=""
    opportunity_steps=""
    lambda_str=$8
    seed_str=$9
    seed_max=${10}
    train=${11}
    input_type=${12}
    tree_size=${13}
    expansion_decision_version=${14}
    model_variant=${15:-"jax_ppo"}
    tree_config=${16:-""}
    rnn_units=${17:-"64"}
    latent_dim=${18:-"32"}
else
    beta_arg=$1
    beta_max=""
    beta_steps=""
    alpha_log_min=$2
    alpha_log_max=$3
    alpha_steps=$4
    opportunity_arg=$5
    opportunity_max=""
    opportunity_steps=""
    lambda_str=$6
    seed_str=$7
    seed_max=$8
    train=$9
    input_type=${10}
    tree_size=${11}
    expansion_decision_version=${12}
    model_variant=${13:-"jax_ppo"}
    tree_config=${14:-""}
    rnn_units=${15:-"64"}
    latent_dim=${16:-"32"}
fi

cat > generate_jax_commands.py << 'PYGEN'
import numpy as np
import sys

beta_arg = sys.argv[1]
beta_max_arg = sys.argv[2]
beta_steps_arg = sys.argv[3]
alpha_min = float(sys.argv[4])
alpha_max = float(sys.argv[5])
alpha_steps = int(sys.argv[6])
opportunity_arg = sys.argv[7]
opportunity_max_arg = sys.argv[8]
opportunity_steps_arg = sys.argv[9]
lambda_str = sys.argv[10]
seed_min = int(sys.argv[11])
seed_max = int(sys.argv[12])
train = sys.argv[13]
input_type = sys.argv[14]
tree_size = sys.argv[15]
expansion_decision_version = sys.argv[16]
model_variant = sys.argv[17]
tree_config = sys.argv[18]
rnn_units = sys.argv[19]
latent_dim = sys.argv[20]

def parse_float_list(raw):
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected numeric list, got {raw!r}")
    return np.asarray(values, dtype=float)

if beta_max_arg.strip() == "" and beta_steps_arg.strip() == "":
    betas = parse_float_list(beta_arg)
else:
    betas = np.linspace(float(beta_arg), float(beta_max_arg), int(beta_steps_arg))

alphas = np.linspace(alpha_min, alpha_max, alpha_steps)

if opportunity_max_arg.strip() == "" and opportunity_steps_arg.strip() == "":
    opportunities = parse_float_list(opportunity_arg)
else:
    opportunities = np.linspace(float(opportunity_arg), float(opportunity_max_arg), int(opportunity_steps_arg))

lambdas = parse_float_list(lambda_str)
seeds = np.arange(seed_min, seed_max + 1, dtype=int)

for seed in seeds:
    for beta in betas:
        for alpha in alphas:
            for lambda_ in lambdas:
                for opportunity in opportunities:
                    print(
                        "sbatch -p general --time=10:00:00 --output=logs/slurm-%j.out "
                        f"model/run_jax_model.sh \"{lambda_}\" \"{alpha}\" \"{beta}\" "
                        "\"outputs/jax_models/\" \"outputs/simulations/\" \"2000\" "
                        f"\"{input_type}\" \"{seed}\" \"{train}\" \"{opportunity}\" "
                        f"{tree_size} \"{expansion_decision_version}\" \"{model_variant}\" "
                        f"\"{tree_config}\" \"{rnn_units}\" \"{latent_dim}\""
                    )
PYGEN

JID_FILE=$(mktemp)

python generate_jax_commands.py "$beta_arg" "$beta_max" "$beta_steps" "$alpha_log_min" "$alpha_log_max" "$alpha_steps" "$opportunity_arg" "$opportunity_max" "$opportunity_steps" "$lambda_str" "$seed_str" "$seed_max" "$train" "$input_type" "$tree_size" "$expansion_decision_version" "$model_variant" "$tree_config" "$rnn_units" "$latent_dim" | while read -r cmd; do
    echo "Executing: $cmd"
    job_output=$(eval "$cmd")
    echo "$job_output"
    job_id=$(echo "$job_output" | awk '{print $NF}')
    echo -n ":$job_id" >> "$JID_FILE"
    sleep 1
done

rm generate_jax_commands.py
job_dependency_list=$(cat "$JID_FILE")
rm "$JID_FILE"

if [ -z "$job_dependency_list" ]; then
    echo "No JAX jobs were submitted."
else
    echo "Submitted JAX jobs: ${job_dependency_list:1}"
fi
