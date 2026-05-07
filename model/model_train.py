import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from scipy.stats import multivariate_normal

import os

import helper_VRNN_39n as helper
from math import sqrt, log10
import random 

import json
import pandas as pd

import sys
import tensorflow as tf
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
num_rollouts = int(sys.argv[9])
train = sys.argv[10]
gamma_string = sys.argv[11]
gamma_values = [float(x) for x in gamma_string.split(',')]


omega_string = sys.argv[12]
omega_values = [float(x) for x in omega_string.split(',')]


np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
sim_dir_name = dir_name.replace("model", "simulation")

current_beta = 1.0
current_alpha = 1.0
current_phase = 1
current_gamma = 1.0
current_entropy=1.0
current_epsilon = 0
current_omega = 1.0

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        tf.print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        tf.print(e)


tf.print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.print("GPU Details:")
for gpu in gpus:
    tf.print(f"  - {gpu}")

tf.print("Is GPU available:", tf.config.list_physical_devices('GPU'))
tf.print("Is built with CUDA:", tf.test.is_built_with_cuda())

latent_dim = 2
output_dim = 4
reward_output_dim = tree_size
rnn_units = output_dim
time_steps = reward_output_dim
input_dim = 1

# --- MODIFIED SAMPLING FUNCTION ---
def sample_with_ste(logits, temperature=1.0):
    """
    Sample from Categorical distribution using logits with Straight-Through Estimator.
    Replaces Gumbel-Softmax.
    """
    # 1. Apply Temperature
    # logits: [Batch, Num_Classes]
    scaled_logits = logits / temperature
    probs = tf.nn.softmax(scaled_logits)
    
    # 2. Sample from the distribution (Hard Step)
    # tf.random.categorical expects logits (log-probs)
    # Returns [Batch, 1]
    drawn_indices = tf.random.categorical(scaled_logits, num_samples=1, dtype=tf.int32)
    drawn_indices = tf.squeeze(drawn_indices, axis=-1) # [Batch]
    
    # Create One-Hot
    depth = tf.shape(logits)[-1]
    y_hard = tf.one_hot(drawn_indices, depth, dtype=tf.float32)
    
    # 3. Straight-Through Estimator
    # Forward pass: Uses y_hard (discrete one-hot)
    # Backward pass: Uses probs (soft probabilities)
    # (y_hard - probs) is detached, so gradients flow through 'probs'
    y = tf.stop_gradient(y_hard - probs) + probs
    # y = y_hard
    return y

# Categorical conversion functions
def scalar_to_categorical(scalar_values, num_classes=9):
    shifted = 4.0 - scalar_values
    indices = tf.floor(shifted + 0.5)
    indices = tf.clip_by_value(indices, 0, num_classes - 1)
    category_indices = tf.cast(indices, tf.int32)
    categories_onehot = tf.one_hot(category_indices, num_classes, dtype=tf.float32)
    return categories_onehot

def categorical_to_scalar(category_probs):
    category_values = tf.constant([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=tf.float32)
    pred_indices = tf.argmax(category_probs, axis=-1) 
    pred_scalars = tf.gather(category_values, pred_indices)
    expected_values = tf.expand_dims(pred_scalars, axis=-1)
    return expected_values

# Decision tree setup 
if tree_size == 2:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '2']},
        '1': { },
        '2': {},
        
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
    # Your decision tree dictionary
    decision_tree = {
        '0': {'right': [-1, '1'], 'left': [3, '14'], 'up': [3, '27']},
        '1': {'up': [-1, '2'], 'down': [2, '6'], 'right': [2, '10']},
        '2': {'up': [-1, '3'], 'down': [2, '4'], 'right': [2, '5']},
        '3': {},
        '4': {},
        '5': {},
        '6': {'up': [-1, '7'], 'down': [2, '8'], 'right': [2, '9']},
        '7': {},
        '8': {},
        '9': {},
        '10': {'right': [-1, '11'], 'left': [3, '12'], 'up': [3, '13']},
        '11': {},
        '12': {},
        '13': {},
        '14': {'up': [-1, '15'], 'down': [2, '19'], 'right': [2, '23']},
        '15': {'up': [-1, '16'], 'down': [2, '17'], 'right': [2, '18']},
        '16': {},
        '17': {},
        '18': {},
        '19': {'up': [-1, '20'], 'down': [2, '21'], 'right': [2, '22']},
        '20': {},
        '21': {},
        '22': {},
        '23': {'up': [-1, '24'], 'down': [2, '25'], 'right': [2, '26']},
        '24': {},
        '25': {},
        '26': {},
        '27': {'up': [-1, '28'], 'down': [2, '32'], 'right': [2, '36']},
        '28': {'up': [-1, '29'], 'down': [2, '30'], 'right': [2, '31']},
        '29': {},
        '30': {},
        '31': {},
        '32': {'up': [-1, '33'], 'down': [2, '34'], 'right': [2, '35']},
        '33': {},
        '34': {},
        '35': {},
        '36': {'up': [-1, '37'], 'down': [2, '38'], 'right': [2, '39']},
        '37': {},
        '38': {},
        '39': {}
    }

path_scale = 9

# Generate all path analysis data
results = helper.analyze_tree_paths(decision_tree)
path_names, path_leaf_dict, sibling_map, node_path_map, node_path_name, path_indices, node_indices, est_best_path_map, path_node_map = results

N2 = len(path_names)
N = time_steps

path_map = np.zeros((N2, N), dtype=int)
for i in range(N2):
    path_map[i, :] = [1 if f"{n+1}" in path_names[i] else 0 for n in range(N)]
path_map = tf.convert_to_tensor(path_map, dtype=tf.float32)

path_cov_mat = np.zeros((N2, N2, N), dtype=int)
for i in range(N2):
    for j in range(N2):
        path_cov_mat[i, j, :] = path_map[i, :] * path_map[j, :]
path_cov_mat = tf.convert_to_tensor(path_cov_mat, dtype=tf.float32)

trials_per_epoch = 200
batch_size =600

index_path_map = {path_indices[i]: node_indices[i] for i in range(len(path_indices))}

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if input_type == "binary":
    mse_param = 5
else:
    if lambda_values[0]!= 0:
        mse_param = 50
    else:
        mse_param = 0

def build_encoder(input_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(rnn_units)(inputs)        
    x = layers.LayerNormalization()(x)         
    x = layers.Activation('relu')(x)           
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    # --- SAFETY CLAMP ---
    # Prevent exp(z_log_var) from becoming Inf or NaN
    z_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -5.0, 5.0))(z_log_var)
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    model = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return model

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_decoder(latent_dim, output_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(rnn_units)(latent_inputs) 
    x = layers.LayerNormalization()(x)         
    x = layers.Activation('relu')(x)           
    outputs = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(latent_inputs, outputs, name='decoder')
    return model

def scalar_to_soft_categorical(scalar_values, sharpness=10.0):
    bin_centers = tf.constant([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=tf.float32)
    bin_centers = tf.reshape(bin_centers, [1, 1, 9]) 
    dist = tf.square(scalar_values - bin_centers)
    return tf.nn.softmax(-dist * sharpness, axis=-1)
    
class VariationalRNN(tf.keras.Model):
    def __init__(self, encoder, decoder, rnn_units, tree_size, time_steps=6, alpha=0.0, beta=1.0, lambda_=1.0,gamma=1.0, omega=1.0):
        super(VariationalRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_units = rnn_units
        self.tree_size = tree_size  
        self.max_time_steps = time_steps 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.omega = omega
        self.num_categories = 9
        
        self.dense_category = tf.keras.layers.Dense(
            self.num_categories, 
            activation='linear',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros'
        )
        self.dense_a = tf.keras.layers.Dense(N2, activation='softmax', kernel_initializer='glorot_uniform')

        self.stop_index = self.tree_size 
        def stop_bias_initializer(shape, dtype=None):
            biases = tf.zeros(shape, dtype=dtype)
            stop_index = shape[0] - 1 
            indices = [[stop_index]]
            updates = [0.0] 
            return tf.tensor_scatter_nd_update(biases, indices, updates)
        self.expansion_head = tf.keras.layers.Dense(
            self.tree_size + 1, 
            activation='linear', 
            name='expansion_head',
            bias_initializer=stop_bias_initializer 
        )
        self.lstm_cell = tf.keras.layers.LSTMCell(
            self.rnn_units,
            kernel_initializer='orthogonal',
            recurrent_initializer='orthogonal' 
        )
        self.prior_mean_layer = tf.keras.layers.Dense(latent_dim)
        self.prior_log_var_layer = tf.keras.layers.Dense(latent_dim)
        self.prior_mu = self.add_weight(name="prior_mu", shape=(latent_dim,), initializer="zeros", trainable=False)
        self.prior_logvar = self.add_weight(name="prior_logvar", shape=(latent_dim,), initializer="zeros", trainable=False)
        # 1. Generate the matrix (Currently: Root, Node1, Node2)
        full_matrix = self.build_relational_matrix(decision_tree, tree_size)
        
        # 2. Slice off the Root (Index 0). Keep Node1, Node2, ...
        # Now Index 0 -> Node 1
        nodes_only_matrix = full_matrix[1:] 
        
        # 3. Create a "Stop" row (All zeros) to match the Stop Index
        feature_dim = tf.shape(nodes_only_matrix)[1]
        stop_row = tf.zeros([1, feature_dim], dtype=tf.float32)
        
        # 4. Concatenate: [Node1, Node2, Stop]
        self.structure_matrix = tf.concat([nodes_only_matrix, stop_row], axis=0)


    def build_relational_matrix(self, tree_dict, size):
        num_nodes = size + 1 
        parents = {i: [] for i in range(num_nodes)}
        children = {i: [] for i in range(num_nodes)}
        for node_str, edges in tree_dict.items():
            u = int(node_str)
            if u > size: continue
            for _, val in edges.items():
                v = int(val[1]) 
                if v < size:
                    children[u].append(v)
                    parents[v].append(u)
        
        def get_siblings(u):
            sibs = set()
            for p in parents[u]:
                for child in children[p]:
                    if child != u: sibs.add(child)
            return list(sibs)

        def get_aunts(u):
            aunts = set()
            for p in parents[u]:
                for sib in get_siblings(p):
                    aunts.add(sib)
            return list(aunts)
            
        def get_cousins(u):
            cousins = set()
            for a in get_aunts(u):
                for child in children[a]:
                    cousins.add(child)
            return list(cousins)

        matrix_rows = []
        for u in range(num_nodes):
            v_parent = np.zeros(num_nodes, dtype=np.float32)
            v_child  = np.zeros(num_nodes, dtype=np.float32)
            v_sib    = np.zeros(num_nodes, dtype=np.float32)
            v_cous   = np.zeros(num_nodes, dtype=np.float32)
            v_aunt   = np.zeros(num_nodes, dtype=np.float32)
            
            if u == size:
                row_dim = 1 + 5 * num_nodes
                matrix_rows.append(np.zeros(row_dim, dtype=np.float32))
                continue

            is_leaf_bool = (len(children[u]) == 0)
            is_leaf = 1.0 if is_leaf_bool else 0.0
            
            if is_leaf_bool:
                for p in parents[u]: v_parent[p] = 1.0
            if not is_leaf_bool:
                for c in children[u]: v_child[c] = 1.0
            for s in get_siblings(u): v_sib[s] = 1.0
            if not is_leaf_bool:
                for c in get_cousins(u): v_cous[c] = 1.0
            if is_leaf_bool:
                for a in get_aunts(u): v_aunt[a] = 1.0

            row = np.concatenate([[is_leaf], v_parent, v_child, v_sib, v_cous, v_aunt])
            matrix_rows.append(row)
        return tf.constant(np.stack(matrix_rows), dtype=tf.float32)

    def get_expected_scalar_reward(self, category_probs):
        values = tf.constant([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=tf.float32)
        values_broadcast = tf.reshape(values, [1, self.num_categories])
        expected_val = tf.reduce_sum(category_probs * values_broadcast, axis=-1)
        return tf.expand_dims(expected_val, -1)
    def call(self, inputs, training=True, current_beta=1.0, current_alpha=1.0, current_gamma=1.0,current_omega = 1.0, current_epsilon=0.0,current_entropy=1.0):
        batch_size = tf.shape(inputs)[0]
        
        # --- PRE-PROCESSING (Same as before) ---
        shifted = 4.0 - inputs
        indices = tf.floor(shifted + 0.5)
        indices = tf.clip_by_value(indices, 0, self.num_categories - 1)
        indices = tf.cast(indices, tf.int32)
        node_onehots = tf.one_hot(indices, self.num_categories, dtype=tf.float32)
        node_distributions = tf.squeeze(node_onehots, axis=2)
        stop_indices = tf.ones([batch_size, 1], dtype=tf.int32) * 4 
        stop_distribution = tf.one_hot(stop_indices, self.num_categories, dtype=tf.float32)
        reward_lookup_table = tf.concat([node_distributions, stop_distribution], axis=1)

        # --- STATE INIT ---
        active_mask = tf.ones([batch_size, 1], dtype=tf.float32)
        visited_mask = tf.zeros([batch_size, self.tree_size + 1], dtype=tf.float32)
        nodes_visited_count = tf.zeros([batch_size], dtype=tf.int32)
        state = self.lstm_cell.get_initial_state(batch_size=batch_size)
        kl_d = 0

        # --- COLLECTORS ---
        all_category_outputs = []
        all_expansion_logits = []
        all_node_selections = []
        all_action_outputs = [] 
        valid_step_mask = [] 
        all_accumulated_kl = [] 
        all_log_probs = []      # <--- NEW: For REINFORCE
        
        for t in range(self.max_time_steps):
            valid_step_mask.append(active_mask)
            h_state = state[0]
            
            # --- A. EXPANSION POLICY ---
            exp_logits = self.expansion_head(h_state) 
            # Mask visited nodes (big negative number)
            masked_exp_logits = exp_logits / 2 + (tf.cast(visited_mask, tf.float32) * -1e9)
            
            # --- B. SAMPLING (Vanilla Categorical - No STE) ---
            if training:
                # Epsilon Greedy
                do_explore = tf.cast(tf.random.uniform([batch_size, 1]) < current_epsilon, tf.float32)
                
                # Random choice
                random_logits = tf.random.uniform(shape=tf.shape(masked_exp_logits))
                random_indices = tf.argmax(random_logits + (visited_mask * -1e9), axis=-1, output_type=tf.int32)
                tf.debugging.check_numerics(masked_exp_logits, "Logits contain NaNs!")
                # Sample from distribution (tf.random.categorical inputs logits)
                sampled_indices = tf.random.categorical(masked_exp_logits, num_samples=1, dtype=tf.int32)
                sampled_indices = tf.squeeze(sampled_indices, axis=-1)
                
                # Mix Exploration vs Exploitation
                next_node_indices = (sampled_indices * tf.cast(1.0 - do_explore, tf.int32)[:,0]) + \
                                    (random_indices * tf.cast(do_explore, tf.int32)[:,0])
            else:
                next_node_indices = tf.argmax(masked_exp_logits, axis=-1, output_type=tf.int32)

            next_node_onehot = tf.one_hot(next_node_indices, self.tree_size + 1)
            
            # --- C. EXTRACT LOG PROBABILITY (Crucial for REINFORCE) ---
            # 1. Get Log Softmax of ALL actions
            log_probs_all = tf.nn.log_softmax(masked_exp_logits)
            
            # 2. Gather only the log_prob of the action we actually took
            indices_for_gather = tf.stack([tf.range(batch_size, dtype=tf.int32), next_node_indices], axis=1)
            chosen_log_prob = tf.gather_nd(log_probs_all, indices_for_gather)
            chosen_log_prob = tf.expand_dims(chosen_log_prob, -1) # [Batch, 1]
            
            all_log_probs.append(chosen_log_prob)
            all_node_selections.append(next_node_indices)
            all_expansion_logits.append(masked_exp_logits)

            # --- D. UPDATE MASKS & STATE (Standard Logic) ---
            is_stop_chosen = tf.cast(tf.equal(next_node_indices, self.stop_index), tf.float32) 
            next_active_mask = active_mask * (1.0 - tf.expand_dims(is_stop_chosen, -1))
            
            new_visit = tf.one_hot(next_node_indices, self.tree_size + 1)
            visited_mask = tf.minimum(visited_mask + new_visit, 1.0)
            
            selection_vector = tf.expand_dims(next_node_onehot, 1)
            category_t_dist = tf.matmul(selection_vector, reward_lookup_table)
            effective_dist = category_t_dist * tf.expand_dims(next_active_mask, -1)
            category_t = tf.squeeze(effective_dist, axis=1) 
            scalar_t_val = self.get_expected_scalar_reward(category_t)
            
            # LSTM Update
            one_hot_t = tf.one_hot(nodes_visited_count, self.max_time_steps)
            lstm_input = tf.concat([category_t, scalar_t_val, one_hot_t, next_node_onehot], axis=1)
            candidate_out, candidate_state = self.lstm_cell(lstm_input, states=state)
            
            new_h = (candidate_state[0] * active_mask) + (state[0] * (1.0 - active_mask))
            new_c = (candidate_state[1] * active_mask) + (state[1] * (1.0 - active_mask))
            state = (new_h, new_c)
            
            # --- E. HEADS (VAE & Action) ---
            encoder_input = tf.concat(state, axis=-1)
            z_mean, z_log_var, z = self.encoder(encoder_input)
            decoder_output = self.decoder(tf.concat([z], axis=1))
            
            pred_logits = self.dense_category(decoder_output)
            all_category_outputs.append(pred_logits)
            
            step_action_probs = self.dense_a(decoder_output) 
            all_action_outputs.append(step_action_probs)
            
            # KL Loss
            prior_mean, prior_var = self.compute_prior(batch_size)
            step_kl = self.calculate_kl_loss(z_mean, z_log_var, prior_mean, prior_var)
            kl_d += step_kl * tf.squeeze(active_mask, -1)
            all_accumulated_kl.append(step_kl * tf.squeeze(active_mask, -1))
            
            # Update Counters
            is_real_step = tf.cast(active_mask, tf.int32)[:, 0] * (1 - tf.cast(is_stop_chosen, tf.int32))
            nodes_visited_count += is_real_step
            active_mask = next_active_mask

        # --- F. STACK OUTPUTS ---
        category_probs = tf.nn.softmax(tf.stack(all_category_outputs, axis=1), axis=-1)
        expansion_outputs = tf.stack(all_expansion_logits, axis=1)
        node_selections = tf.stack(all_node_selections, axis=1)
        step_mask = tf.stack(valid_step_mask, axis=1) 
        action_outputs_sequence = tf.stack(all_action_outputs, axis=1)
        kl_history_sequence = tf.stack(all_accumulated_kl, axis=1)
        log_probs_sequence = tf.stack(all_log_probs, axis=1) # <--- NEW OUTPUT

        return category_probs, expansion_outputs, action_outputs_sequence, kl_d, node_selections, step_mask, kl_history_sequence, log_probs_sequence
    
    
    def compute_prior(self, batch_size):
        prior_mean = tf.broadcast_to(self.prior_mu, [batch_size, tf.shape(self.prior_mu)[0]])
        prior_var  = tf.exp(tf.broadcast_to(self.prior_logvar, [batch_size, tf.shape(self.prior_logvar)[0]]))
        return prior_mean, prior_var    
    def calculate_kl_loss(self, z_means, z_log_vars, prior_mean, prior_var, epsilon=1e-6):
        prior_var = prior_var + epsilon
        prior_log_var = tf.math.log(prior_var)
        z_var = tf.exp(z_log_vars) + epsilon
        log_var_ratio = z_log_vars - prior_log_var
        kl_loss = -0.5 * tf.reduce_sum(
            1 + log_var_ratio - ((tf.square(z_means - prior_mean) + z_var) / prior_var),
            axis=1
        )
        return tf.reduce_sum(kl_loss)
        
    def convert_categorical_to_scalar(self,category_probs):
        category_values = tf.constant([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=tf.float32)
        pred_indices = tf.argmax(category_probs, axis=-1) 
        pred_scalars = tf.gather(category_values, pred_indices)
        expected_values = tf.expand_dims(pred_scalars, axis=-1)
        return expected_values

    def compute_action_loss(self, inputs, action_outputs, scalar_outputs, step_mask):
        path_lengths = tf.reduce_sum(tf.squeeze(step_mask, -1), axis=1) 
        stop_indices = tf.cast(path_lengths - 1, tf.int32) 
        batch_size = tf.shape(action_outputs)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, stop_indices], axis=1)
        selected_action_policy = tf.gather_nd(action_outputs, gather_indices) 
        action_loss = -helper.calculate_V_3(
            inputs, 
            selected_action_policy, 
            N, N2, 
            index_path_map, 
            path_map, 
            path_cov_mat, 
            self
        )
        return action_loss
    
    def compute_categorical_cross_entropy_loss(self, target_categories, category_outputs):
        time_steps = self.time_steps
        batch_size = tf.shape(target_categories)[0]
        target_category_onehot = tf.squeeze(target_categories, axis=2) 
        stacked_preds = tf.stack(category_outputs, axis=0)
        mask = tf.linalg.band_part(tf.ones([time_steps, time_steps]), -1, 0)
        target_expanded = tf.tile(
            target_category_onehot[None, :, :, :], 
            [time_steps, 1, 1, 1]
        ) 
        epsilon = 1e-7
        safe_probs = tf.clip_by_value(stacked_preds, epsilon, 1.0 - epsilon)
        ce_raw = -tf.reduce_sum(target_expanded * tf.math.log(safe_probs), axis=-1) 
        ce_masked = ce_raw * mask[:, None, :] 
        total_loss = tf.reduce_sum(ce_masked)
        valid_count = tf.reduce_sum(mask) * tf.cast(batch_size, tf.float32)
        return total_loss / valid_count
@tf.function
def train_step(model, optimizer, current_alpha, current_beta, current_gamma, current_omega, current_epsilon, current_entropy, clip_value=1.0, num_rollouts=5):
    # Trainable Variables
    first_decoder_params = (
        model.encoder.trainable_variables +
        model.decoder.trainable_variables +
        model.prior_mean_layer.trainable_variables +
        model.prior_log_var_layer.trainable_variables +
        [model.prior_mu, model.prior_logvar] +
        model.lstm_cell.trainable_variables 
    )
    second_decoder_params = model.dense_category.trainable_variables
    action_head_params = model.dense_a.trainable_variables 
    policy_head_params = model.expansion_head.trainable_variables 
    
    accum_argmax_rewards = tf.zeros([time_steps], dtype=tf.float32)
    accum_kl_history = tf.zeros([time_steps], dtype=tf.float32)
    
    # NEW: Opportunity Cost Weight (Omega)
    # Penalizes the agent for taking too many steps.
    current_omega = 0.05 

    with tf.device('/GPU:0'):   
        # Generate Data
        values = np.array([[random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for i in range(time_steps)] for _ in range(batch_size)])
        input_data = tf.constant(values, dtype=tf.float32)
        input_data = tf.reshape(input_data, [batch_size, tree_size, 1])
        
        # Calculate True Rewards for Paths
        raw_path_rewards = helper.calculate_path_rewards(index_path_map, input_data)
        true_path_rewards = tf.squeeze(raw_path_rewards, axis=-1)
        true_path_rewards = tf.transpose(true_path_rewards) 

        total_backbone_loss = 0.0
        total_recon_head_loss = 0.0
        total_action_head_loss = 0.0
        total_policy_head_loss = 0.0
        total_entropy = 0.0 # <--- NEW: Tracker for debugging

        with tf.GradientTape(persistent=True) as tape:
            for r in range(num_rollouts):
                # Run Model
                cat_probs, exp_logits, action_output, kl_d, chosen_nodes, step_mask, kl_seq, log_probs_seq = model(
                    input_data, training=True, current_beta=current_beta, current_alpha=current_alpha, 
                    current_gamma=current_gamma, current_omega=current_omega, current_epsilon=current_epsilon, current_entropy=current_entropy
                )
                
                mask_flat = tf.squeeze(step_mask, -1) # [Batch, Time]
                is_stop_action = tf.cast(tf.equal(chosen_nodes, model.stop_index), tf.float32)
                
                # --- 1. CALCULATE & NORMALIZE TASK RETURNS ---
                true_rewards_expanded = tf.expand_dims(true_path_rewards, 1)
                chosen_path_values = tf.reduce_sum(action_output * true_rewards_expanded, axis=-1)
                task_value_per_step = tf.reduce_sum(action_output * true_rewards_expanded, axis=-1)
                
                raw_task_rewards = task_value_per_step * is_stop_action * model.lambda_
                
                task_returns_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
                future_task_ret = tf.zeros([batch_size], dtype=tf.float32)
                
                for t in tf.range(time_steps - 1, -1, -1):
                    r_t = raw_task_rewards[:, t]
                    future_task_ret = r_t + (model.gamma * future_task_ret)
                    task_returns_ta = task_returns_ta.write(t, future_task_ret)
                
                task_returns = tf.transpose(task_returns_ta.stack())
                
                # Normalize Task Returns
                tr_mean = tf.math.reduce_mean(task_returns, axis=0, keepdims=True)
                tr_std = tf.math.reduce_std(task_returns, axis=0, keepdims=True) + 1e-8
                norm_task_returns = (task_returns - tr_mean) / tr_std
                
                # --- 2. ADD SAFETY-CLIPPED PENALTIES ---
                raw_kl_penalty = kl_seq * model.alpha * 0.1
                safe_kl_penalty = tf.clip_by_value(raw_kl_penalty, 0.0, 10.0)
                
                raw_time_penalty = (1.0 - is_stop_action) * model.omega * current_omega
                safe_time_penalty = tf.clip_by_value(raw_time_penalty, 0.0, 8.0)
                
                # --- 3. FINAL GRADIENT SIGNAL ---
                # final_returns = norm_task_returns - safe_kl_penalty - safe_time_penalty
                final_returns = norm_task_returns - safe_kl_penalty - safe_time_penalty
                policy_loss_per_step = -log_probs_seq[:,:,0] * final_returns
                masked_policy_loss = policy_loss_per_step * mask_flat
                total_policy_loss = tf.reduce_sum(masked_policy_loss) / (tf.reduce_sum(mask_flat) + 1e-6)
                
                # --- UNPROTECTED ENTROPY CALCULATION ---
                policy_probs = tf.nn.softmax(exp_logits)
                policy_log_probs = tf.nn.log_softmax(exp_logits)
                step_entropy = -tf.reduce_sum(policy_probs * policy_log_probs, axis=-1)
                masked_entropy = step_entropy * mask_flat
                avg_entropy = tf.reduce_sum(masked_entropy) / (tf.reduce_sum(mask_flat) + 1e-6)
                
                total_policy_loss = total_policy_loss - (current_entropy * avg_entropy)
                
                path_lengths = tf.reduce_sum(mask_flat, axis=1)
                stop_indices_idxs = tf.cast(path_lengths - 1, tf.int32)
                batch_indices = tf.range(batch_size, dtype=tf.int32)
                gather_indices = tf.stack([batch_indices, stop_indices_idxs], axis=1)
                selected_action_policy = tf.gather_nd(action_output, gather_indices)
                
                raw_act_loss = -helper.calculate_V_3(input_data, selected_action_policy, N, N2, index_path_map, path_map, path_cov_mat, model)

                safe_chosen = tf.minimum(chosen_nodes, tree_size - 1)
                batch_indices_t = tf.expand_dims(tf.range(batch_size), 1)
                batch_indices_t = tf.tile(batch_indices_t, [1, time_steps])
                gather_indices_recon = tf.stack([batch_indices_t, safe_chosen], axis=-1)
                target_scalars = tf.gather_nd(input_data, gather_indices_recon)
                target_onehot = scalar_to_categorical(target_scalars, 9)
                target_flat = tf.squeeze(target_onehot, 2)
                
                recon_mask = mask_flat * (1.0 - is_stop_action)
                ce_loss = tf.keras.losses.categorical_crossentropy(target_flat, cat_probs)
                masked_recon_loss = tf.reduce_sum(ce_loss * recon_mask) / (tf.reduce_sum(recon_mask) + 1e-6) * model.beta
                
                avg_kl = tf.reduce_sum(kl_d) / (tf.reduce_sum(mask_flat) + 1e-6)
                accum_kl_history += avg_kl
                # D. Accumulate Losses
                loss_policy = total_policy_loss
                loss_action = raw_act_loss
                loss_backbone = masked_recon_loss + total_policy_loss + (raw_act_loss * model.lambda_)

                accum_argmax_rewards += tf.reduce_mean(chosen_path_values, axis=0)
                
                total_policy_head_loss += loss_policy / num_rollouts
                total_action_head_loss += loss_action / num_rollouts
                total_backbone_loss += loss_backbone / num_rollouts
                total_recon_head_loss += masked_recon_loss / num_rollouts
                total_entropy += avg_entropy / num_rollouts # <--- NEW: Accumulate entropy

    # --- 3. APPLY GRADIENTS ---
    grads_backbone = tape.gradient(total_backbone_loss, first_decoder_params)
    grads_recon = tape.gradient(total_recon_head_loss, second_decoder_params)
    grads_action = tape.gradient(total_action_head_loss, action_head_params)
    grads_policy = tape.gradient(total_policy_head_loss, policy_head_params)
    
    def clip_grads(gradients):
        clipped, _ = tf.clip_by_global_norm(gradients, clip_value)
        return clipped

    optimizer.apply_gradients(zip(clip_grads(grads_backbone), first_decoder_params))
    optimizer.apply_gradients(zip(clip_grads(grads_recon), second_decoder_params))
    optimizer.apply_gradients(zip(clip_grads(grads_action), action_head_params))
    optimizer.apply_gradients(zip(clip_grads(grads_policy), policy_head_params))
    del tape
    
    # NEW: Returned total_entropy at the very end
    return total_backbone_loss, accum_argmax_rewards / num_rollouts, accum_kl_history, tf.zeros_like(accum_kl_history), total_entropy
import pandas as pd
import os

def train_model(model, epochs, trials_per_epoch):
    global epochs_count 
    global input_type
    global current_beta
    global current_alpha
    global current_epsilon
    global current_gamma 
    global current_entropy
    global current_omega
    global current_phase
    global model_name
    global dir_name
    global sim_dir_name 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
    dummy_input = tf.zeros((1, time_steps, 1), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    
    all_trainables = (
        model.encoder.trainable_variables +
        model.decoder.trainable_variables +
        model.prior_mean_layer.trainable_variables +
        model.prior_log_var_layer.trainable_variables +
        [model.prior_mu, model.prior_logvar] +
        model.lstm_cell.trainable_variables +
        model.dense_category.trainable_variables +
        model.dense_a.trainable_variables + 
        model.expansion_head.trainable_variables
    )

    base_opt = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    base_opt.build(all_trainables)

    best_loss = float('inf')
    wait = 0
    patience = 30           
    min_delta = 0.01        
    warmup_epochs = 10  
    
    best_checkpoint_path =  dir_name + model_name + '_BEST.weights.h5'
    final_checkpoint_path =  dir_name + model_name + '.weights.h5'
    csv_log_path = sim_dir_name + model_name + "_train_history.csv"
    
    history_data = []

    for epoch in range(epochs):

        # Keeping Epsilon at 0 as requested to see pure behavior
        current_epsilon = 0.0
        decay_epochs = max(1, epochs // 2)
        if epoch < decay_epochs:
            current_entropy = 0.1 - (epoch * ((0.1 - 0.005) / decay_epochs))
        else:
            current_entropy = 0.005
            
        epoch_loss_accum = 0.0
        epoch_entropy_accum = 0.0 # <--- NEW TRACKER
        epochs_count += 1 
        
        epoch_reward_seq_accum = tf.zeros([time_steps], dtype=tf.float32)
        epoch_kl_seq_accum = tf.zeros([time_steps], dtype=tf.float32)
        epoch_opp_seq_accum = tf.zeros([time_steps], dtype=tf.float32)
        if epoch < warmup_epochs:
            progress = epoch / float(warmup_epochs)
            # Smoothly transition from 0.01 to 1.0
            current_beta = 0.01 + progress * (1.0 - 0.01)
            current_gamma = 0.01 + progress * (1.0 - 0.01)
            current_alpha =  0.01 + progress * (1.0 - 0.01)
            current_omega =  0.01 + progress * (1.0 - 0.01)
        else:
            current_beta = 1.0 
            current_alpha = 1.0
            current_gamma = 1.0
            current_omega = 1.0
             
        for i in range(trials_per_epoch):
            
            # --- NEW: Unpack the 5th variable (batch_entropy) ---
            loss, batch_rewards_seq, batch_kl_seq, batch_opp_seq, batch_entropy = train_step(
                model, optimizer, 
                current_alpha=tf.constant(current_alpha, dtype=tf.float32), 
                current_beta=tf.constant(current_beta, dtype=tf.float32),
                current_gamma=tf.constant(current_gamma, dtype=tf.float32),
                current_omega=tf.constant(current_omega, dtype=tf.float32),
                current_epsilon=tf.constant(current_epsilon, dtype=tf.float32),
                current_entropy=tf.constant(current_entropy, dtype=tf.float32)
            )
            
            epoch_loss_accum += loss
            epoch_entropy_accum += batch_entropy # Add to epoch tracking
            epoch_reward_seq_accum += batch_rewards_seq
            epoch_kl_seq_accum += batch_kl_seq
            epoch_opp_seq_accum += batch_opp_seq
        
        avg_epoch_loss = epoch_loss_accum / trials_per_epoch
        avg_epoch_entropy = epoch_entropy_accum / trials_per_epoch # Average for printing
        
        avg_epoch_rewards_seq = (epoch_reward_seq_accum / trials_per_epoch).numpy()
        avg_epoch_kl_seq = (epoch_kl_seq_accum / trials_per_epoch).numpy()
        avg_epoch_opp_seq = (epoch_opp_seq_accum / trials_per_epoch).numpy()
        
        row = {
            'epoch': epoch + 1,
            'phase': current_phase,
            'loss': avg_epoch_loss.numpy(),
            'avg_argmax_reward_by_step': avg_epoch_rewards_seq.tolist(), 
            'avg_cumulative_kl_by_step': avg_epoch_kl_seq.tolist(),
            'avg_cumulative_opp_by_step': avg_epoch_opp_seq.tolist()
        }
        history_data.append(row)
        
        df_history = pd.DataFrame(history_data)
        df_history.to_csv(csv_log_path, index=False)

        if (epoch + 1) % 1 == 0:
            # --- NEW: Print the Entropy side-by-side with Loss ---
            tf.print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f} | Avg Entropy = {avg_epoch_entropy:.4f}")

        if epoch >= warmup_epochs:
            if avg_epoch_loss < (best_loss - min_delta):
                best_loss = avg_epoch_loss
                wait = 0
                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                model.save_weights(best_checkpoint_path)
                print(f"   >>> New Best Loss: {best_loss:.4f}. Saved checkpoint.")
            else:
                wait += 1
                if wait >= patience:
                    print(f"\n🛑 CONVERGENCE REACHED.")
                    try:
                        model.load_weights(best_checkpoint_path)
                        print("   ✅ Successfully reloaded best weights.")
                    except:
                        print("   ⚠️ Warning: Could not reload best weights.")
                    break

    # if os.path.exists(best_checkpoint_path) and epoch == epochs - 1:
    #      model.load_weights(best_checkpoint_path)

    print(f"\n✅ Saving Final (Best) Model state")
    
    if os.path.exists(final_checkpoint_path):
        os.remove(final_checkpoint_path)
    model.save_weights(final_checkpoint_path)
    
    if os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
        
    print(f"History saved to: {csv_log_path}")


train_model_flag = True if sys.argv[10] == "train" else False

rewards_list = [[random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for i in range(time_steps)] for _ in range(20000)]

for beta in beta_values:
    for lambda_ in lambda_values:
        for alpha in alpha_values:
            for gamma in gamma_values:
                for omega in omega_values:
                    print("lambda: " + str(lambda_) + " alpha: " + str(alpha) + "gamma: " + str(gamma)+ "omega: " + str(omega))
                    model_name = f'stable_vrnn_model_lambda_{lambda_}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_omega_{omega}_seed_{seed}_{tree_size}n_final' 
                    file_path = dir_name + model_name + ".csv"
                    weights_path = dir_name + model_name + '.weights.h5'

                    encoder = build_encoder(rnn_units * 2, latent_dim)
                    decoder = build_decoder(latent_dim, 2 * rnn_units)

                    vrnn_model = VariationalRNN(encoder, decoder, rnn_units, tree_size=time_steps, time_steps=time_steps, alpha=alpha, beta=beta, lambda_=lambda_, gamma = gamma, omega = omega)
                    vrnn_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='mse',
                        run_eagerly=False 
                    )

                    if train_model_flag:
                        print(f"Training model: {model_name}...")
                        train_model(vrnn_model, epochs, trials_per_epoch)
                        
                        if os.path.exists(weights_path):
                            os.remove(weights_path)
                        vrnn_model.save_weights(weights_path)
                        print(f"Saved weights to {weights_path}")
                    
                    else:
                        print(f"Skipping training. Loading weights for: {model_name}...")
                        if os.path.exists(weights_path):
                            dummy_input = tf.zeros((1, time_steps, 1), dtype=tf.float32)
                            _ = vrnn_model(dummy_input, training=False)
                            
                            vrnn_model.load_weights(weights_path)
                            print("Weights loaded successfully.")
                        else:
                            print(f"Warning: Weights file not found at {weights_path}. Model is initialized with random weights.")

                    sibling_map_dict = {1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 5}
                    aunt_map_dict = {1: None, 2: 4, 3: 4, 4: None, 5: 1, 6: 1}
                    category_values = tf.constant([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=tf.float32)

                    columns = [
                        "graph", "node", "actual_reward", "precision", "visit_step", 
                        "sibling_reward", "aunt_reward",
                        "is_best", "is_second", "is_worst", 
                        "is_chosen_path", 
                        "path_rank", "max_path_reward", "path_reward", "MI_cost", "is_leaf", 
                        "cur_precision", "cur_best", "cur_worst", "V", "MI"
                    ]
                    
                    ts_int = int(time_steps) 
                    for t in range(1, ts_int + 1):
                        columns.append(f"dec_t{t}")
                        columns.append(f"act_t{t}")
                        columns.append(f"pred_t{t}")

                    df = pd.DataFrame(columns=columns)
                    # 1. Convert to Tensor  
                    rewards_tensor = tf.constant(rewards_list, dtype=tf.float32)
                    
                    # 2. Reshape correctly: [Batch_Size, Tree_Size, 1]
                    # -1 infers the number of trials (20000)
                    # time_steps (or tree_size) is the length of each trial
                    rewards_tensor = tf.reshape(rewards_tensor, [-1, int(time_steps), 1])

                    # 3. Predict (Keras will automatically handle the batching for 20k items)
                    outputs = vrnn_model.predict(rewards_tensor, batch_size=batch_size)
                    
                    # 4. Extract MI (KL Divergence)
                    # outputs[3] corresponds to 'kl_d' returned by call()
                    # This will be a vector of shape [20000]
                    MI_all = outputs[3]
                    
                    # If you need a single average scalar for the print/log:
                    MI = np.mean(MI_all)
                    print("Starting evaluation on 100 graphs...")
                    for g, rewards in enumerate(rewards_list[:1000]):
                        
                        rewards_tensor = tf.constant(rewards, dtype=tf.float32)
                        rewards_tensor = tf.reshape(rewards_tensor, [1, -1, 1])

                        outputs = vrnn_model.predict(rewards_tensor)
                        
                        category_probs = outputs[0]      
                        node_selections = outputs[4]     
                        action_policy_seq = outputs[2]  
                        
                        decisions_seq = node_selections[0] 
                        
                        stop_indices = np.where(decisions_seq == tree_size)[0]
                        if len(stop_indices) > 0:
                            stop_event_idx = stop_indices[0]
                        else:
                            stop_event_idx = len(decisions_seq)
                        
                        policy_idx = min(stop_event_idx, int(time_steps) - 1)
                        final_action_policy = action_policy_seq[0, policy_idx, :]
                        
                        chosen_path_idx = np.argmax(final_action_policy)
                        
                        path_rewards = helper.calculate_path_rewards(index_path_map, rewards_tensor)
                        path_rewards_list = path_rewards.numpy().tolist()
                        sorted_unique_path_rewards = np.unique(sorted(path_rewards_list))
                        
                        max_path_reward_val = max(path_rewards_list)[0][0]
                        chosen_path_reward = path_rewards_list[chosen_path_idx][0][0]
                        
                        print(f"Graph {g}: Stop Step {stop_event_idx} | "
                            f"Decision: {chosen_path_idx} | "
                            f"Reward: {chosen_path_reward:.1f} | "
                            f"Max Possible: {max_path_reward_val:.1f}")

                        pred_indices = tf.argmax(category_probs[0], axis=-1)
                        pred_scalars_seq = tf.gather(category_values, pred_indices).numpy()
                        
                        actual_scalars_seq = []
                        for node_idx in decisions_seq:
                            if node_idx < len(rewards):
                                actual_scalars_seq.append(rewards[node_idx])
                            else:
                                actual_scalars_seq.append(0.0) 
                        actual_scalars_seq = np.array(actual_scalars_seq)

                        for n in range(1, tree_size + 1):
                            node_idx = n - 1
                            
                            visits = np.where(decisions_seq == node_idx)[0]
                            if len(visits) > 0:
                                first_visit_step = visits[0] 
                                if first_visit_step > stop_event_idx:
                                    precision_val = -100.0
                                    visit_step_val = -100
                                    cur_precision_val = -100.0
                                else:
                                    precision_val = pred_scalars_seq[first_visit_step]
                                    visit_step_val = first_visit_step + 1 
                                    cur_precision_val = precision_val
                            else:
                                first_visit_step = -1
                                precision_val = np.nan
                                visit_step_val = -1
                                cur_precision_val = np.nan

                            sib_node = sibling_map_dict.get(n)
                            if sib_node is not None and 1 <= sib_node <= len(rewards):
                                sib_reward = rewards[sib_node - 1]
                            else:
                                sib_reward = np.nan
                                
                            aunt_node = aunt_map_dict.get(n)
                            if aunt_node is not None and 1 <= aunt_node <= len(rewards):
                                aunt_reward = rewards[aunt_node - 1]
                            else:
                                aunt_reward = np.nan

                            node_path_indices = node_path_map[str(n)] 
                            node_path_rewards = [path_rewards_list[i][0][0] for i in node_path_indices]
                            is_in_chosen_path = chosen_path_idx in node_path_indices

                            is_best = False
                            is_second = False
                            is_worst = False
                            
                            if max_path_reward_val in node_path_rewards:
                                is_best = True
                                path_rank = 0
                            elif sorted_unique_path_rewards[0] in node_path_rewards:
                                is_worst = True
                                path_rank = 3
                            elif len(sorted_unique_path_rewards) > 1 and sorted_unique_path_rewards[-2] in node_path_rewards:
                                is_second = True
                                path_rank = 1
                            else:
                                try:
                                    val = node_path_rewards[0]
                                    path_rank = N2 - list(sorted_unique_path_rewards).index(val)
                                except:
                                    path_rank = -1

                            is_leaf = n in list(path_leaf_dict.values())

                            row_data = {
                                "graph": g,
                                "node": n,
                                "actual_reward": rewards[node_idx],
                                "precision": precision_val,
                                "visit_step": visit_step_val,
                                "sibling_reward": sib_reward,
                                "aunt_reward": aunt_reward,
                                "is_best": is_best,
                                "is_second": is_second,
                                "is_worst": is_worst,
                                "is_chosen_path": is_in_chosen_path, 
                                "path_rank": path_rank,
                                "max_path_reward": max_path_reward_val,
                                "path_reward": np.mean(node_path_rewards),
                                "MI_cost": MI,
                                "is_leaf": is_leaf,
                                "cur_precision": cur_precision_val,
                                "cur_best": 0,
                                "cur_worst": 0,
                                "V": chosen_path_reward,
                                "MI": MI
                            }

                            for t in range(ts_int):
                                col_suffix = str(t + 1)
                                if t < len(decisions_seq) and t <= stop_event_idx:
                                    dec_idx = decisions_seq[t]
                                    if dec_idx == tree_size:
                                        row_data[f"dec_t{col_suffix}"] = "STOP"
                                        row_data[f"act_t{col_suffix}"] = np.nan
                                        row_data[f"pred_t{col_suffix}"] = np.nan
                                    else:
                                        row_data[f"dec_t{col_suffix}"] = dec_idx + 1 
                                        row_data[f"act_t{col_suffix}"] = actual_scalars_seq[t]
                                        row_data[f"pred_t{col_suffix}"] = pred_scalars_seq[t]
                                else:
                                    row_data[f"dec_t{col_suffix}"] = np.nan
                                    row_data[f"act_t{col_suffix}"] = np.nan
                                    row_data[f"pred_t{col_suffix}"] = np.nan

                            df.loc[len(df)] = row_data

                    if input_type == "normal":
                        df.to_csv(sim_dir_name + model_name + "_2000.csv", index=None)
                    elif input_type == "binary":
                        df.to_csv(sim_dir_name + model_name + "_binary.csv", index=None)
                    else:
                        df.to_csv(sim_dir_name + model_name + "_uniform.csv", index=None)