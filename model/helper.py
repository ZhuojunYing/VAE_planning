
import numpy as np  # Equivalent to Julia's LinearAlgebra and Random
from scipy.stats import multivariate_normal
import itertools
import copy
from math import sqrt
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import mvn
import tensorflow as tf
import random

def analyze_tree_paths(tree):
    def dfs(node, current_path):
        # Update node_path_map as we traverse
        path_index = len(paths)
        for n in current_path:
            if n not in node_path_map:
                node_path_map[n] = []
            if path_index not in node_path_map[n]:
                node_path_map[n].append(path_index)
        
        # If we're at a leaf node (empty dict), add the path to our results
        if not tree[node]:
            path_string = ', '.join(current_path)
            paths.append(path_string)
            path_leaf_dict[len(paths) - 1] = int(node)
            
            # Update node_path_name for all nodes in this path
            for n in current_path:
                node_path_name[n] = path_string
            
            # Add path_index and node_indices
            path_indices.append(int(node))
            current_nodes = [int(n) for n in current_path if n != '0']
            node_indices.append(current_nodes)
            
            # Update est_best_path_map and path_node_map
            current_path_index = len(paths) - 1
            est_best_path_map[path_string] = current_path_index
            path_node_map[current_path_index] = current_nodes
            
            return
        
        # Explore all possible directions (right, left, up, down)
        for direction, (_, next_node) in tree[node].items():
            # Record siblings
            find_siblings_for_node(node, next_node)
            dfs(next_node, current_path + [next_node])
    
    def find_siblings_for_node(parent, child):
        # Get all children of the parent
        children = [val[1] for val in tree[parent].values()]
        # For each child, find its first sibling (if any)
        for i, current_child in enumerate(children):
            if current_child == child:
                # If there are more children after this one, the next one is the sibling
                if i + 1 < len(children):
                    sibling_map[child] = children[i + 1]
                # If this is the last child, the first child is the sibling
                elif len(children) > 1:
                    sibling_map[child] = children[0]

    paths = []
    path_leaf_dict = {}
    sibling_map = {}
    node_path_map = {}
    node_path_name = {}
    path_indices = []
    node_indices = []
    est_best_path_map = {}
    path_node_map = {}
    
    dfs('0', ['0'])
    return (paths, path_leaf_dict, sibling_map, node_path_map, node_path_name, 
            path_indices, node_indices, est_best_path_map, path_node_map)

def pad_vector(vector, target_length, placeholder):
    # Calculate the number of placeholders to add
    padding_length = target_length - len(vector)

    # Check if padding is needed
    if padding_length > 0:
        # Create a new list with the desired length and fill it with the placeholder
        padded_vector = [placeholder] * target_length
        # Copy the original list's elements into the new list
        padded_vector[:len(vector)] = vector
    else:
        # No padding needed, return the original vector
        padded_vector = vector

    return padded_vector





def calculate_posterior_mean_variance(sigma_p, mu_B, sigma_B):
   
    # Assume inputs are already tensors of the same dtype
    sigma_p_squared_inv = 1 / tf.square(sigma_p)
    sigma_B_squared_inv = 1 / tf.square(sigma_B)

    posterior_variance = 1 / (sigma_p_squared_inv + sigma_B_squared_inv)
    posterior_mean = posterior_variance * (mu_B * sigma_B_squared_inv)

    # This line has been optimized to remove redundant computation
    posterior_mean_variance = tf.square(posterior_variance * sigma_B_squared_inv)

    return posterior_mean, posterior_mean_variance




def get_truncated_normal_samples(size=6, mean=0, sd=1, low=0, upp=10):
    dist = truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return dist.rvs(size)

def scalar_to_categorical(scalar_values, num_classes=9):
    shifted = 4.0 - scalar_values
    indices = tf.floor(shifted + 0.5)
    indices = tf.clip_by_value(indices, 0, num_classes - 1)
    category_indices = tf.cast(indices, tf.int32)
    categories_onehot = tf.one_hot(category_indices, num_classes, dtype=tf.float32)
    return categories_onehot

def categorical_to_scalar(category_probs):
    """
    Convert category probabilities back to scalar values using ARGMAX (Hard Classification).
    Logic is REVERSED: Index 0 -> +4.0, Index 8 -> -4.0
    """
    category_values = tf.constant([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=tf.float32)
    pred_indices = tf.argmax(category_probs, axis=-1)  
    pred_scalars = tf.gather(category_values, pred_indices)
    expected_values = tf.expand_dims(pred_scalars, axis=-1)
    return expected_values


def random_argmax(vector):
    max_indices = tf.argmax(vector, axis=0)
    return max_indices


def generate_batch_data(batch_size, time_steps, input_type):
    """Generates a fresh batch of data dynamically in standard Python."""
    feature_dim = 1
    
    # Using the original random choice logic
    if input_type == "binary":
        values = np.array([[random.choice([0, 1]) for _ in range(time_steps)] for _ in range(batch_size)])

    else:
        values = np.array([[random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(time_steps)] for _ in range(batch_size)])
    input_data = tf.constant(values, dtype=tf.float32)
    input_data = tf.reshape(input_data, [batch_size, time_steps, feature_dim])
    
    return input_data

def random_argmax_per_row(tensor):

    max_values_across_first_dimension = tf.reduce_max(tensor, axis=0)

    # Create a mask where the elements are equal to the maximum values
    equal_to_max_mask = tf.equal(tensor, max_values_across_first_dimension)

    # Convert boolean mask to 1s and 0s
    mask_as_integers = tf.cast(equal_to_max_mask, tf.float32)

    # Sum the mask along the first dimension (axis=0)
    sum_along_first_dimension = tf.reduce_sum(mask_as_integers, axis=0)

    result = mask_as_integers / sum_along_first_dimension

    return result


def policy(estimated_path_rewards):
    # Assuming estimated_path_rewards has shape [batch_size, num_paths]
    
    # Find the minimum path reward for each batch index
    min_rewards = tf.reduce_min(estimated_path_rewards, axis=1, keepdims=True)  # Shape: [batch_size, 1]
    
    # Find the maximum path reward for each batch index
    max_rewards = tf.reduce_max(estimated_path_rewards, axis=1, keepdims=True)  # Shape: [batch_size, 1]
    
    # Normalize the estimated path rewards
    normalized_vector = (estimated_path_rewards - min_rewards) / (max_rewards - min_rewards + 1e-8)  # Shape: [batch_size, num_paths]
    
    # Ensure the normalized vector sums to 1 to represent a valid policy
    normalized_vector = normalized_vector / tf.reduce_sum(normalized_vector, axis=1, keepdims=True)  # Shape: [batch_size, num_paths]
    
    return normalized_vector