"""
config.py
Handles system arguments, GPU setup, hyperparameter definition, 
and builds the decision tree matrices used across the project.
"""

import os
import sys
import random
import numpy as np
import tensorflow as tf

# Import your existing helper script
import helper

# ---------------------------------------------------------
# 1. COMMAND LINE ARGUMENTS
# ---------------------------------------------------------
try:
    lambda_string = sys.argv[1]
    lambda_values = [float(x) for x in lambda_string.split(',')]

    alpha_string = sys.argv[2]
    alpha_values = [float(x) for x in alpha_string.split(',')]

    beta_string = sys.argv[3]
    beta_values = [float(x) for x in beta_string.split(',')]

    dir_name = sys.argv[4]

    epochs_string = sys.argv[5]
    epochs = int(epochs_string)
    epochs_count = 0

    input_type = sys.argv[6]

    seed = int(sys.argv[7])

    tree_size = int(sys.argv[8])
    train_mode = sys.argv[9]  # Renamed from 'train' to avoid variable shadowing
    tree_type = str(sys.argv[10])
    opportunity_cost_string = sys.argv[11] if len(sys.argv) > 11 else "0.0"
    opportunity_cost_values = [float(x) for x in opportunity_cost_string.split(',')]
    expansion_decision_version = sys.argv[12] if len(sys.argv) > 12 else "decoder"
    model_variant = sys.argv[13] if len(sys.argv) > 13 else "vae"
except IndexError:
    print("Error: Missing command-line arguments.")
    print("Usage: python main.py <lambda> <alpha> <beta> <dir_name> <epochs> <input_type> <seed> <tree_size> <train/simulate> <tree_type> [opportunity_cost] [expansion_decision_version] [model_variant]")
    sys.exit(1)

def normalize_expansion_decision_version(version):
    aliases = {
        "1": "decoder",
        "decoder": "decoder",
        "after_decoder": "decoder",
        "2": "lstm",
        "lstm": "lstm",
        "after_lstm": "lstm",
        "3": "pre_lstm",
        "pre_lstm": "pre_lstm",
        "before_lstm": "pre_lstm",
    }
    version_key = str(version).strip().lower()
    if version_key not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"expansion_decision_version must be one of: {valid}. "
            f"Got {version!r}."
        )
    return aliases[version_key]

expansion_decision_version = normalize_expansion_decision_version(expansion_decision_version)

def normalize_model_variant(variant):
    aliases = {
        "vae": "vae",
        "autoencoder": "vae",
        "rnn": "rnn",
        "plain_rnn": "rnn",
        "no_autoencoder": "rnn",
        "no_ae": "rnn",
    }
    variant_key = str(variant).strip().lower()
    if variant_key not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"model_variant must be one of: {valid}. Got {variant!r}.")
    return aliases[variant_key]

model_variant = normalize_model_variant(model_variant)

sim_dir_name = dir_name.replace("model", "simulation")

# ---------------------------------------------------------
# 2. SEED & REPRODUCIBILITY
# ---------------------------------------------------------
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------
# 3. GPU CONFIGURATION
# ---------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if gpus:
    print("GPU Details:")
    for gpu in gpus:
        print(f"  - {gpu}")
# ---------------------------------------------------------
# 4. MODEL HYPERPARAMETERS
# ---------------------------------------------------------
latent_dim = 16
output_dim = 64
reward_output_dim = tree_size
rnn_units = output_dim
time_steps = reward_output_dim
input_dim = 1
num_categories = 9

trials_per_epoch = 200
batch_size = 200
kl_scaler = 5
# ---------------------------------------------------------
# 5. DECISION TREE SETUP
# ---------------------------------------------------------
if tree_size == 2:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '2']},
        '1': {},
        '2': {}
    }

elif tree_size == 6:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '4']},
        '1': {'right': [-1, '2'], 'up': [-1, '3']},
        '2': {},
        '3': {},
        '4': {'right': [-1, '5'], 'up': [-1, '6']},
        '5': {},
        '6': {}
    }
    
elif tree_size == 12:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '5'], 'left': [-1, "9"]},
        '1': {'right': [-1, '2'], 'up': [-1, '3'], 'left': [-1, "4"]},
        '2': {},
        '3': {},
        '4': {},
        '5': {'right': [-1, '6'], 'up': [-1, '7'], 'left': [-1, "8"]},
        '6': {},
        '7': {},
        '8': {},
        '9': {'right': [-1, '10'], 'up': [-1, '11'], 'left': [-1, "12"]},
        '10': {},
        '11': {},
        '12': {}
    }
else:
    if (tree_type == "deep_breadth"):
        
        tree = "deep"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '2']},
            '1': {'up': [-1, '3'], 'down': [2, '4']},
            '2': {'up': [-1, '5'], 'down': [2, '6']},
            '3': {'up': [-1, '7'], 'down': [2, '8']},
            '4': {'up': [-1, '9'], 'down': [2, '10']},
            '5': {'up': [-1, '11'], 'down': [2, '12']},
            '6': {'up': [-1, '13'], 'down': [2, '14']},
            '7': {'up': [-1, '15'], 'down': [2, '16']},
            '8': {'up': [-1, '17'], 'down': [2, '18']},
            '9': {'up': [-1, '19'], 'down': [2, '20']},
            '10': {'right': [-1, '21'], 'left': [3, '22']},
            '11': {'up': [-1, '23'], 'down': [2, '24']},
            '12': {'up': [-1, '25'], 'down': [2, '26']},
            '13': {'up': [-1, '27'], 'down': [2, '28']},
            '14': {'up': [-1, '29'], 'down': [2, '30']},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }
        

    elif(tree_type == "deep_depth"):
        
        tree = "deep"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '16']},
            '1': {'up': [-1, '2'], 'down': [2, '9']},
            '2': {'up': [-1, '3'], 'down': [2, '6']},
            '3': {'up': [-1, '4'], 'down': [2, '5']},
            '4': {},
            '5': {},
            '6': {'up': [-1, '7'], 'down': [2, '8']},
            '7': {},
            '8': {},
            '9': {'up': [-1, '10'], 'down': [2, '13']},
            '10': {'right': [-1, '11'], 'left': [3, '12']},
            '11': {},
            '12': {},
            '13': {'up': [-1, '14'], 'down': [2, '15']},
            '14': {},
            '15': {},
            '16': {'up': [-1, '17'], 'down': [2, '24']},
            '17': {'up': [-1, '18'], 'down': [2, '21']},
            '18': {'up': [-1, '19'], 'down': [2, '20']},
            '19': {},
            '20': {},
            '21': {'up': [-1, '22'], 'down': [2, '23']},
            '22': {},
            '23': {},
            '24': {'up': [-1, '25'], 'down': [2, '28']},
            '25': {'up': [-1, '26'], 'down': [2, '27']},
            '26': {},
            '27': {},
            '28': {'up': [-1, '29'], 'down': [2, '30']},
            '29': {},
            '30': {}
        }
        
    elif (tree_type == "wide_breadth"):
        tree = "wide"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '2'],'up': [-1, '3'], 'down': [2, '4'], 'up1': [-1, '5']},
            '1': {'up': [-1, '6'], 'down': [2, '7'], 'right': [-1, '8'], 'left': [3, '9'],'up1': [-1, '10']},
            '2': {'up': [-1, '11'], 'down': [2, '12'], 'right': [-1, '13'], 'left': [3, '14'],'up1': [-1, '15']},
            '3': {'up': [-1, '16'], 'down': [2, '17'], 'right': [-1, '18'], 'left': [3, '19'],'up1': [-1, '20']},
            '4': {'up': [-1, '21'], 'down': [2, '22'], 'right': [-1, '23'], 'left': [3, '24'],'up1': [-1, '25']},
            '5': {'up': [-1, '26'], 'down': [2, '27'], 'right': [-1, '28'], 'left': [3, '29'],'up1': [-1, '30']},
            '6': {},
            '7': {},
            '8': {},
            '9':{},
            '10':{},
            '11': {},
            '12':{},
            '13': {},
            '14': {},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }
    

    else:

        
        tree = "wide"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '7'],'up': [-1, '13'], 'down': [2, '19'], 'up1': [-1, '25']},
            '1': {'up': [-1, '2'], 'down': [2, '3'], 'right': [-1, '4'], 'left': [3, '5'],'up1': [-1, '6']},
            '2': {},
            '3': {},
            '4': {},
            '5': {},
            '6': {},
            '7': {'up': [-1, '8'], 'down': [2, '9'], 'right': [-1, '10'], 'left': [3, '11'],'up1': [-1, '12']},
            '8': {},
            '9':{},
            '10':{},
            '11': {},
            '12':{},
            '13': {'up': [-1, '14'], 'down': [2, '15'], 'right': [-1, '16'], 'left': [3, '17'],'up1': [-1, '18']},
            '14': {},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {'up': [-1, '20'], 'down': [2, '21'], 'right': [-1, '22'], 'left': [3, '23'],'up1': [-1, '24']},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {'up': [-1, '26'], 'down': [2, '27'], 'right': [-1, '28'], 'left': [3, '29'],'up1': [-1, '30']},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }
        



# ---------------------------------------------------------
# 6. PATH ANALYSIS & TENSOR MATRICES
# ---------------------------------------------------------
results = helper.analyze_tree_paths(decision_tree)

(path_names, path_leaf_dict, sibling_map, node_path_map, node_path_name, 
 path_indices, node_indices, est_best_path_map, path_node_map) = results

num_paths = len(path_names)


# Create path map
path_map_np = np.zeros((num_paths, time_steps), dtype=int)
for i in range(num_paths):
    path_map_np[i, :] = [1 if f"{n+1}" in path_names[i] else 0 for n in range(time_steps)]
path_map = tf.convert_to_tensor(path_map_np, dtype=tf.float32)

# Create path covariance matrix
path_cov_mat_np = np.zeros((num_paths, num_paths, time_steps), dtype=int)
for i in range(num_paths):
    for j in range(num_paths):
        path_cov_mat_np[i, j, :] = path_map_np[i, :] * path_map_np[j, :]
path_cov_mat = tf.convert_to_tensor(path_cov_mat_np, dtype=tf.float32)

# Create index path map
index_path_map = {path_indices[i]: node_indices[i] for i in range(len(path_indices))}
