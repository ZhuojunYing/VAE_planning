#!/bin/bash

is_train_arg() {
    case "$1" in
        train|simulate|simulation|inference|eval|evaluate) return 0 ;;
        *) return 1 ;;
    esac
}

# Check if all required arguments are provided
if ! {
    { [ "$#" -ge 16 ] && [ "$#" -le 20 ] && is_train_arg "${13}"; } ||
    { [ "$#" -ge 14 ] && [ "$#" -le 18 ] && is_train_arg "${11}"; } ||
    { [ "$#" -ge 12 ] && [ "$#" -le 16 ] && is_train_arg "${9}"; }
}; then
    echo "Usage legacy: $0 beta_log10_min beta_log10_max beta_steps alpha_min alpha_max alpha_steps opportunity_min opportunity_max opportunity_steps lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim]"
    echo "Usage opp/lambda lists: $0 beta_log10_min beta_log10_max beta_steps alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim]"
    echo "Usage beta/opp/lambda lists: $0 beta_list alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim]"
    echo "Example legacy VAE: $0 3 3 1 1 1 1 -2 -2 1 1.0 1 5 train uniform 2 lstm vae '' 64 32"
    echo "Example opp/lambda list VAE: $0 1000 1000 1 0 0 1 0.025,0.125,0.225 100.0,300.0 1 5 train uniform 3 lstm vae bandit3 64 32"
    echo "Example beta/opp/lambda list VAE: $0 20.0,40.0,60.0 0 0 1 0.025,0.125,0.225 100.0 1 5 train uniform 3 lstm vae bandit3 64 32"
    echo "Example RNN: $0 0 0 1 0 0 1 0.0 1.0 1 5 train uniform 2 lstm rnn '' 64 32"
    echo "Example disjoint 2x2: $0 1 1 1 0 0 1 0.0 10.0 1 1 train uniform 4 lstm vae disjoint2x2 64 32"
    echo "Expansion versions: decoder/1, lstm/2, pre_lstm/3"
    echo "Model variants: vae, rnn"
    echo "Tree configs: bandit3, bandit4, disjoint2x2, disjoint3x2"
    echo "Defaults: rnn_units=64, latent_dim=32"
    exit 1
fi

if [ "$#" -ge 16 ] && is_train_arg "${13}"; then
    arg_mode="legacy"
elif [ "$#" -ge 14 ] && is_train_arg "${11}"; then
    arg_mode="opportunity_lambda_lists"
else
    arg_mode="beta_opportunity_lambda_lists"
fi

echo "=== DEBUG: Arguments received ==="
echo "Mode: '$arg_mode'"
echo "Total args: $#"
echo "1: '$1'"
echo "2: '$2'" 
echo "3: '$3'"
echo "4: '$4'"
echo "5: '$5'"
echo "6: '$6'"
echo "7: '$7'"
echo "8: '$8'"
echo "9: '$9'"
echo "10: '${10}'"
echo "11: '${11}'"
echo "12: '${12}'"
echo "13: '${13}'"
echo "14: '${14}'"
echo "15: '${15}'"
echo "16: '${16}'"
echo "17: '${17}'"
echo "18: '${18}'"
echo "19: '${19}'"
echo "20: '${20}'"
echo "================================="

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
    model_variant=${17:-"vae"}
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
    model_variant=${15:-"vae"}
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
    model_variant=${13:-"vae"}
    tree_config=${14:-""}
    rnn_units=${15:-"64"}
    latent_dim=${16:-"32"}
fi

# Create a Python script to generate the commands
cat > generate_commands.py << 'EOL'
import numpy as np
import sys

# Get command line arguments
beta_arg = sys.argv[1]
beta_max_arg = sys.argv[2]
beta_steps_arg = sys.argv[3]
alpha_log_min, alpha_log_max, alpha_steps = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
opportunity_arg = sys.argv[7]
opportunity_max_arg = sys.argv[8]
opportunity_steps_arg = sys.argv[9]
lambda_str = sys.argv[10]
seed_min = int(sys.argv[11])
seed_max= int(sys.argv[12])
train = sys.argv[13]
input_type = sys.argv[14]
tree_size = sys.argv[15]
expansion_decision_version = sys.argv[16]
model_variant = sys.argv[17]
tree_config = sys.argv[18]
rnn_units = sys.argv[19]
latent_dim = sys.argv[20]
tf_cpus_per_task = sys.argv[21]
tf_mem = sys.argv[22]
tf_slurm_time = sys.argv[23]

def parse_float_list(raw):
    values = []
    for part in str(raw).split(','):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError(f"Expected at least one numeric value, got {raw!r}")
    return np.array(values, dtype=float)

if beta_max_arg.strip() == "" and beta_steps_arg.strip() == "":
    beta_exp_values = parse_float_list(beta_arg)
else:
    beta_log_min = float(beta_arg)
    beta_log_max = float(beta_max_arg)
    beta_steps = int(beta_steps_arg)
    if np.isinf(beta_log_min) or np.isinf(beta_log_max):
        beta_exp_values = np.zeros(beta_steps)
    else:
        beta_log_values = np.linspace(beta_log_min, beta_log_max, beta_steps)
        beta_exp_values=np.array( [  l for l in beta_log_values])

# Handle alpha values
if np.isinf(alpha_log_min) or np.isinf(alpha_log_max):
    alpha_exp_values = np.zeros(alpha_steps)
else:
    alpha_log_values = np.linspace(alpha_log_min, alpha_log_max, alpha_steps)
    alpha_exp_values =np.array( [   l for l in alpha_log_values])

seed_values = np.arange(seed_min, seed_max + 1, dtype=int)

if opportunity_max_arg.strip() == "" and opportunity_steps_arg.strip() == "":
    opportunity_values = parse_float_list(opportunity_arg)
else:
    opportunity_min = float(opportunity_arg)
    opportunity_max = float(opportunity_max_arg)
    opportunity_steps = int(opportunity_steps_arg)
    if np.isinf(opportunity_min) or np.isinf(opportunity_max):
        opportunity_values = np.zeros(opportunity_steps)
    else:
        opportunity_log_values = np.linspace(opportunity_min, opportunity_max, opportunity_steps)
        opportunity_values =np.array( [  l for l in opportunity_log_values])

lambda_values = parse_float_list(lambda_str)

# Split into lists of length 5
def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

beta_exp_values_split = split_list(beta_exp_values.tolist(), 1)
alpha_exp_values_split = split_list(alpha_exp_values.tolist(), 1)
opportunity_values_split = split_list(opportunity_values.tolist(), 1)
lambda_values_split = split_list(lambda_values.tolist(), 1)
seed_values_split = split_list(seed_values.tolist(), 1)

# Generate all combinations and create slurm commands
for seed_list in seed_values_split:
    for beta_list in beta_exp_values_split:
        for alpha_list in alpha_exp_values_split:
            for lambda_list in lambda_values_split:
                for opportunity_list in opportunity_values_split:
                    beta_str = ", ".join([f"{x}" for x in beta_list])
                    alpha_str = ", ".join([f"{x}" for x in alpha_list])
                    seed_str = ", ".join([f"{x}" for x in seed_list])
                    lambda_job_str = ", ".join([f"{x}" for x in lambda_list])
                    opportunity_str = ", ".join([f"{x}" for x in opportunity_list])
                    # NOTE: We removed the extra double quotes inside the python print to make parsing easier in bash
                    command = f'sbatch -p general --time={tf_slurm_time} --cpus-per-task={tf_cpus_per_task} --mem={tf_mem} --output=logs/slurm-%j.out model/run_model.sh "{lambda_job_str}" "{alpha_str}" "{beta_str}" "outputs/models/" "outputs/simulations/" "2000" "{input_type}" "{seed_str}" "{train}" "{opportunity_str}" {tree_size} "{expansion_decision_version}" "{model_variant}" "{tree_config}" "{rnn_units}" "{latent_dim}"'
                    print(command)
EOL

# Temporary file to store job IDs (needed because the while loop runs in a subshell)
JID_FILE=$(mktemp)

tf_cpus_per_task=${TF_CPUS_PER_TASK:-1}
tf_mem=${TF_MEM:-16G}
tf_slurm_time=${TF_SLURM_TIME:-10:00:00}

echo "TensorFlow Slurm resources: cpus-per-task=${tf_cpus_per_task}, mem=${tf_mem}, time=${tf_slurm_time}"

# Run the Python script and execute each command
python generate_commands.py "$beta_arg" "$beta_max" "$beta_steps" "$alpha_log_min" "$alpha_log_max" "$alpha_steps" "$opportunity_arg" "$opportunity_max" "$opportunity_steps" "$lambda_str" "$seed_str" "$seed_max" "$train" "$input_type" "$tree_size" "$expansion_decision_version" "$model_variant" "$tree_config" "$rnn_units" "$latent_dim" "$tf_cpus_per_task" "$tf_mem" "$tf_slurm_time"| while read -r cmd; do
    echo "Executing: $cmd"
    
    # 1. Execute the command and capture the output (e.g., "Submitted batch job 12345")
    job_output=$(eval "$cmd")
    echo "$job_output"
    
    # 2. Extract the Job ID (usually the last word in the output string)
    job_id=$(echo "$job_output" | awk '{print $NF}')
    
    # 3. Append the job ID to our temp file with a colon separator
    echo -n ":$job_id" >> "$JID_FILE"
    
    sleep 1
done

# Clean up the temporary Python script
rm generate_commands.py

# --- NEW SECTION: Submit the Email Notification Job ---

# Read the collected Job IDs
job_dependency_list=$(cat "$JID_FILE")
rm "$JID_FILE"

if [ -z "$job_dependency_list" ]; then
    echo "No jobs were submitted, skipping email trigger."
else
    # Remove the very first colon if it exists (formatting cleanup)
    job_dependency_list=${job_dependency_list:1}

    echo "Submitting email watcher job dependent on: $job_dependency_list"

    # Submit a dummy job that waits for ALL previous jobs (afterany) to finish
    # 'afterany' ensures you get the email even if some training jobs crash.
    sbatch --dependency=afterany:$job_dependency_list \
           --job-name="email_notify" \
           --time=00:05:00 \
           --mem=100M \
           --output=/dev/null \
           --error=/dev/null \
           --wrap="echo 'All Grid Search jobs finished.' | mail -s 'Batch Job Complete: Grid VRNN' z5ying@ucsd.edu"
fi
