#!/bin/bash

# This script activates the virtual environment and runs the model_vae.py script.
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
echo "================================="
lambda_string=$1
alpha_string=$2
beta_string=$3
model_dir_name=$4
sim_dir_name=$5
trial_n_string=$6
input_type=$7
seed=$8
train=$9
opportunity_cost_string=${10:-"0.0"}
tree_size=${11}
# Navigate to the directory containing the virtual environment if it's not in the current directory
# Uncomment and modify the next line if needed
# cd /path/to/your/project
module unload cuda/11.0
module load cuda/11.2
# Activate the virtual environment
source vae_env/bin/activate
python model/main.py "$lambda_string" "$alpha_string" "$beta_string" "$model_dir_name" "120" "$input_type" "$seed" "$tree_size" "$train" "" "$opportunity_cost_string"


# Deactivate the virtual environment when done
deactivate

EXIT_STATUS=$?

# Define email details
EMAIL="z5ying@ucsd.edu"
HOSTNAME=$(hostname)
JOB_ID=$SLURM_JOB_ID

# Create subject and message based on exit status
#if [ $EXIT_STATUS -eq 0 ]; then
#SUBJECT="[SUCCESS] Job $JOB_ID completed on $HOSTNAME"
#    MESSAGE="Your job $JOB_ID has successfully completed on $HOSTNAME."
#else
#    SUBJECT="[FAILED] Job $JOB_ID failed on $HOSTNAME"
#    MESSAGE="Your job $JOB_ID has failed on $HOSTNAME with exit status $EXIT_STATUS."
#fi

# Send the email
#echo "$MESSAGE" | mail -s "$SUBJECT" $EMAIL

exit $EXIT_STATUS
