#!/bin/bash

set -euo pipefail

is_train_arg() {
    case "$1" in
        train|sim|simulate|simulation|inference|eval|evaluate) return 0 ;;
        *) return 1 ;;
    esac
}

if ! {
    { [ "$#" -ge 16 ] && is_train_arg "${13}"; } ||
    { [ "$#" -ge 14 ] && is_train_arg "${11}"; } ||
    { [ "$#" -ge 12 ] && is_train_arg "${9}"; }
}; then
    echo "Usage legacy: $0 beta_min beta_max beta_steps alpha_min alpha_max alpha_steps opportunity_min opportunity_max opportunity_steps lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim] [num_updates] [num_envs] [n_sim_trials] [extra_evidence_args...]"
    echo "Usage opp/lambda lists: $0 beta_min beta_max beta_steps alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim] [num_updates] [num_envs] [n_sim_trials] [extra_evidence_args...]"
    echo "Usage beta/opp/lambda lists: $0 beta_list alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim] [num_updates] [num_envs] [n_sim_trials] [extra_evidence_args...]"
    echo "Evidence defaults: model_dir=outputs/jax_models_evi, sim_dir=outputs/jax_simulations_evi, input_type=evidence, tree_config=evidence, tree_size=2, expansion=lstm."
    echo "Pass --coherence-values 0,0.05,0.1,0.2,0.4,0.8 to train/evaluate across coherence magnitudes in one model."
    echo "Pass --observation-noise-std 1.0 to control fixed Gaussian evidence noise."
    echo "Pass --observation-noise-std-list 0.5,1.0,2.0 to submit one Slurm job per observation-noise std."
    echo "Pass --max-observations-before-stop N to control the forced terminal decision; --num-steps defaults to the same N."
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
    beta_arg=$1; beta_max=$2; beta_steps=$3
    alpha_min=$4; alpha_max=$5; alpha_steps=$6
    opportunity_arg=$7; opportunity_max=$8; opportunity_steps=$9
    lambda_str=${10}; seed_min=${11}; seed_max=${12}
    train=${13}; input_type=${14}; tree_size=${15}; expansion_decision_version=${16}
    model_variant=${17:-"vae"}; tree_config=${18:-"evidence"}; rnn_units=${19:-"32"}; latent_dim=${20:-"16"}
    num_updates=${21:-"24000"}; num_envs=${22:-"200"}
    n_sim_trials=${23:-"2000"}
    if [[ "$n_sim_trials" == -* ]]; then
        n_sim_trials="2000"
        extra_args=("${@:23}")
    else
        extra_args=("${@:24}")
    fi
elif [ "$arg_mode" = "opportunity_lambda_lists" ]; then
    beta_arg=$1; beta_max=$2; beta_steps=$3
    alpha_min=$4; alpha_max=$5; alpha_steps=$6
    opportunity_arg=$7; opportunity_max=""; opportunity_steps=""
    lambda_str=$8; seed_min=$9; seed_max=${10}
    train=${11}; input_type=${12}; tree_size=${13}; expansion_decision_version=${14}
    model_variant=${15:-"vae"}; tree_config=${16:-"evidence"}; rnn_units=${17:-"32"}; latent_dim=${18:-"16"}
    num_updates=${19:-"24000"}; num_envs=${20:-"200"}
    n_sim_trials=${21:-"2000"}
    if [[ "$n_sim_trials" == -* ]]; then
        n_sim_trials="2000"
        extra_args=("${@:21}")
    else
        extra_args=("${@:22}")
    fi
else
    beta_arg=$1; beta_max=""; beta_steps=""
    alpha_min=$2; alpha_max=$3; alpha_steps=$4
    opportunity_arg=$5; opportunity_max=""; opportunity_steps=""
    lambda_str=$6; seed_min=$7; seed_max=$8
    train=$9; input_type=${10}; tree_size=${11}; expansion_decision_version=${12}
    model_variant=${13:-"vae"}; tree_config=${14:-"evidence"}; rnn_units=${15:-"32"}; latent_dim=${16:-"16"}
    num_updates=${17:-"24000"}; num_envs=${18:-"200"}
    n_sim_trials=${19:-"2000"}
    if [[ "$n_sim_trials" == -* ]]; then
        n_sim_trials="2000"
        extra_args=("${@:19}")
    else
        extra_args=("${@:20}")
    fi
fi

has_any_extra_arg() {
    local arg flag
    for arg in "${extra_args[@]:-}"; do
        for flag in "$@"; do
            if [ "$arg" = "$flag" ] || [[ "$arg" == "$flag="* ]]; then
                return 0
            fi
        done
    done
    return 1
}

if ! has_any_extra_arg "--return-target-mode"; then
    extra_args+=("--return-target-mode" "${EXPANSION_RETURN_TARGET:-sampled_lambda}")
fi

if ! has_any_extra_arg "--sampled-lambda-critic" "--critic" "--critic-type" "--critic-mode"; then
    extra_args+=("--sampled-lambda-critic" "${SAMPLED_LAMBDA_CRITIC:-value}")
fi

mkdir -p logs

jax_cpus_per_task=${JAX_CPUS_PER_TASK:-8}
jax_mem=${JAX_MEM:-16G}
jax_slurm_time=${JAX_SLURM_TIME:-24:00:00}

echo "Evidence JAX Slurm resources: cpus-per-task=${jax_cpus_per_task}, mem=${jax_mem}, time=${jax_slurm_time}"

python - "$beta_arg" "$beta_max" "$beta_steps" "$alpha_min" "$alpha_max" "$alpha_steps" \
    "$opportunity_arg" "$opportunity_max" "$opportunity_steps" "$lambda_str" \
    "$seed_min" "$seed_max" "$train" "$input_type" "$tree_size" "$expansion_decision_version" \
    "$model_variant" "$tree_config" "$rnn_units" "$latent_dim" "$num_updates" "$num_envs" \
    "$n_sim_trials" "$jax_cpus_per_task" "$jax_mem" "$jax_slurm_time" "${extra_args[@]}" <<'PY' |
import itertools
import os
import shlex
import sys
import numpy as np

(
    beta_arg, beta_max, beta_steps,
    alpha_min, alpha_max, alpha_steps,
    opportunity_arg, opportunity_max, opportunity_steps,
    lambda_str, seed_min, seed_max, train, input_type, tree_size,
    expansion_decision_version, model_variant, tree_config, rnn_units, latent_dim,
    num_updates, num_envs, n_sim_trials, jax_cpus_per_task, jax_mem, jax_slurm_time, *extra_args
) = sys.argv[1:]

def parse_list(raw):
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected values in {raw!r}")
    return vals

def parse_range_or_list(first, last, steps):
    if str(last).strip() == "" and str(steps).strip() == "":
        return parse_list(first)
    return list(np.linspace(float(first), float(last), int(steps)))

def pop_extra_option(args, flags):
    cleaned = []
    value = None
    i = 0
    flags = set(flags)
    while i < len(args):
        text = str(args[i])
        matched = False
        for flag in flags:
            if text == flag:
                if i + 1 >= len(args):
                    raise ValueError(f"{flag} requires a value.")
                value = str(args[i + 1])
                i += 2
                matched = True
                break
            if text.startswith(flag + "="):
                value = text.split("=", 1)[1]
                i += 1
                matched = True
                break
        if not matched:
            cleaned.append(args[i])
            i += 1
    return value, cleaned

def extra_has(flag):
    return any(arg == flag or str(arg).startswith(flag + "=") for arg in extra_args)

def has_scalar_noise_arg(args):
    return any(
        arg == "--observation-noise-std"
        or str(arg).startswith("--observation-noise-std=")
        for arg in args
    )

def extra_int(flag, default):
    for i, arg in enumerate(extra_args):
        text = str(arg)
        if text == flag and i + 1 < len(extra_args):
            return int(extra_args[i + 1])
        if text.startswith(flag + "="):
            return int(text.split("=", 1)[1])
    return int(default)

betas = parse_range_or_list(beta_arg, beta_max, beta_steps)
alphas = list(np.linspace(float(alpha_min), float(alpha_max), int(alpha_steps)))
opps = parse_range_or_list(opportunity_arg, opportunity_max, opportunity_steps)
lambdas = parse_list(lambda_str)
seeds = range(int(seed_min), int(seed_max) + 1)
noise_list_raw, extra_args = pop_extra_option(
    extra_args,
    ["--observation-noise-std-list", "--noise-std-list", "--noise-list"],
)
if noise_list_raw is None:
    noise_list_raw = os.environ.get("OBSERVATION_NOISE_STD_LIST", "").strip() or None
if noise_list_raw is not None and has_scalar_noise_arg(extra_args):
    raise ValueError(
        "Use either --observation-noise-std-list/OBSERVATION_NOISE_STD_LIST "
        "or --observation-noise-std, not both."
    )
noise_stds = parse_list(noise_list_raw) if noise_list_raw is not None else [None]
max_observations = extra_int(
    "--max-observations-before-stop",
    os.environ.get("MAX_OBSERVATIONS_BEFORE_STOP", "10"),
)
num_steps = extra_int("--num-steps", max_observations)
steps_per_epoch = int(num_updates) * int(num_envs) * num_steps // 120
steps_per_epoch = max(steps_per_epoch, int(num_envs) * num_steps)
num_steps_args = [] if extra_has("--num-steps") else ["--num-steps", str(num_steps)]

for seed, beta, alpha, lambda_, opp, noise_std in itertools.product(
    seeds, betas, alphas, lambdas, opps, noise_stds
):
    noise_args = [] if noise_std is None else ["--observation-noise-std", str(noise_std)]
    cmd = [
        "sbatch",
        "-p", "general",
        "--time", jax_slurm_time,
        "--cpus-per-task", jax_cpus_per_task,
        "--mem", jax_mem,
        "--output=logs/slurm-%j.out",
        "model_jax/run_jax_model_evi.sh",
        str(lambda_), str(alpha), str(beta),
        "outputs/jax_models_evi/", "outputs/jax_simulations_evi/", n_sim_trials,
        input_type, str(seed), train, str(opp), tree_size,
        expansion_decision_version, model_variant, tree_config, rnn_units, latent_dim,
        "--num-envs", num_envs,
        *num_steps_args,
        "--steps-per-epoch", str(steps_per_epoch),
        *noise_args,
        *extra_args,
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
PY
while read -r cmd; do
    echo "Executing: $cmd"
    eval "$cmd"
    sleep 1
done
