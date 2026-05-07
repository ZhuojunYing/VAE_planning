#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 13 ]; then
    echo "Usage: $0 beta_log10_min beta_log10_max beta_steps alpha_min alpha_max alpha_steps opportunity_min opportunity_max opportunity_steps lambda_str seed_min seed_max train"
    echo "Example: $0 -2 2 5 0 0 1 0 0.2 5 1.0 1 5 train"
    exit 1
fi

echo "=== DEBUG: Arguments received ==="
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
echo "================================="

# Assign arguments to variables
beta_log_min=$1
beta_log_max=$2
beta_steps=$3
alpha_log_min=$4
alpha_log_max=$5
alpha_steps=$6
opportunity_min=$7
opportunity_max=$8
opportunity_steps=$9
lambda_str=${10}
seed_str=${11}
seed_max=${12}
train=${13}
# Create a Python script to generate the commands
cat > generate_commands.py << 'EOL'
import numpy as np
import sys

# Get command line arguments
beta_log_min, beta_log_max, beta_steps = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
alpha_log_min, alpha_log_max, alpha_steps = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
opportunity_min, opportunity_max, opportunity_steps = float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])
lambda_str = sys.argv[10]
seed_min = int(sys.argv[11])
seed_max= int(sys.argv[12])
train = sys.argv[13]
if np.isinf(beta_log_min) or np.isinf(beta_log_max):
    beta_exp_values = np.zeros(beta_steps)
else:
    beta_log_values = np.linspace(beta_log_min, beta_log_max, beta_steps)
    beta_exp_values=np.array( [ 10**  l for l in beta_log_values])

# Handle alpha values
if np.isinf(alpha_log_min) or np.isinf(alpha_log_max):
    alpha_exp_values = np.zeros(alpha_steps)
else:
    alpha_log_values = np.linspace(alpha_log_min, alpha_log_max, alpha_steps)
    alpha_exp_values =np.array( [   l for l in alpha_log_values])

seed_values = np.arange(seed_min, seed_max + 1, dtype=int)
if np.isinf(opportunity_min) or np.isinf(opportunity_max):
    opportunity_values = np.zeros(opportunity_steps)
else:
    opportunity_log_values = np.linspace(opportunity_min, opportunity_max, opportunity_steps)
    opportunity_values =np.array( [ 10**  l for l in opportunity_log_values])

# Split into lists of length 5
def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

beta_exp_values_split = split_list(beta_exp_values.tolist(), 1)
alpha_exp_values_split = split_list(alpha_exp_values.tolist(), 1)
opportunity_values_split = split_list(opportunity_values.tolist(), 1)
seed_values_split = split_list(seed_values.tolist(), 1)

# Generate all combinations and create slurm commands
for seed_list in seed_values_split:
    for beta_list in beta_exp_values_split:
        for alpha_list in alpha_exp_values_split:
            for opportunity_list in opportunity_values_split:
                beta_str = ", ".join([f"{x}" for x in beta_list])
                alpha_str = ", ".join([f"{x}" for x in alpha_list])
                seed_str = ", ".join([f"{x}" for x in seed_list])
                opportunity_str = ", ".join([f"{x}" for x in opportunity_list])
                # NOTE: We removed the extra double quotes inside the python print to make parsing easier in bash
                command = f'sbatch -p general --time=07:00:00 --output=logs/slurm-%j.out model/run_model_2n_binary.sh "{lambda_str}" "{alpha_str}" "{beta_str}" "outputs/models/" "outputs/simulations/" "2000" "binary" "{seed_str}" "{train}" "{opportunity_str}"'
                print(command)
EOL

# Temporary file to store job IDs (needed because the while loop runs in a subshell)
JID_FILE=$(mktemp)

# Run the Python script and execute each command
python generate_commands.py "$beta_log_min" "$beta_log_max" "$beta_steps" "$alpha_log_min" "$alpha_log_max" "$alpha_steps" "$opportunity_min" "$opportunity_max" "$opportunity_steps" "$lambda_str" "$seed_str" "$seed_max" "$train" | while read -r cmd; do
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
