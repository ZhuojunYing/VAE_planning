"""
simulate.py
Runs trained models in inference mode and exports expansion traces.

The output CSV has one row per graph/node. Trial-level expansion decisions are
repeated across node rows, while node-level reward estimates are stored per
timestep for the row's node.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from model import VariationalRNN, build_encoder, build_decoder


CATEGORY_VALUES = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])


def calculate_path_rewards_sim(index_path_map, trial_rewards):
    path_rewards = []
    for node_indices in index_path_map.values():
        path_rewards.append(sum(trial_rewards[node - 1] for node in node_indices))
    return np.array(path_rewards, dtype=float)


def build_model(config, alpha, beta, lambda_, opportunity_cost):
    encoder = build_encoder(config.rnn_units * 2, config.latent_dim, config.rnn_units)
    decoder = build_decoder(config.latent_dim, 2 * config.rnn_units, config.rnn_units)

    return VariationalRNN(
        encoder=encoder,
        decoder=decoder,
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        time_steps=config.time_steps,
        num_paths=config.num_paths,
        index_path_map=config.index_path_map,
        path_map=config.path_map,
        path_cov_mat=config.path_cov_mat,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        tree_type=getattr(config, "tree_type", "deep"),
        opportunity_cost=opportunity_cost,
        input_type=config.input_type,
        expansion_decision_version=config.expansion_decision_version,
        use_autoencoder=(config.model_variant == "vae"),
    )


def model_variant_label(variant):
    return f"variant_{variant}_"


def model_name_for(config, lambda_, alpha, beta, opportunity_cost):
    variant_label = model_variant_label(config.model_variant)
    if config.tree_size == 30:
        return (
            f"lambda_{lambda_}_alpha_{alpha}_beta_{beta}_opportunity_{opportunity_cost}_"
            f"expansion_{config.expansion_decision_version}_"
            f"{variant_label}"
            f"seed_{config.seed}_{config.tree_size}n_{config.tree_type}"
        )
    return (
        f"lambda_{lambda_}_alpha_{alpha}_beta_{beta}_opportunity_{opportunity_cost}_"
        f"expansion_{config.expansion_decision_version}_"
        f"{variant_label}"
        f"seed_{config.seed}_{config.tree_size}n"
    )


def sample_categorical_indices(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum()
    if total <= 0.0:
        probabilities = np.ones_like(probabilities, dtype=float) / len(probabilities)
    else:
        probabilities = probabilities / total
    return int(np.random.choice(len(probabilities), p=probabilities))


def scalar_estimates(category_probs):
    flat_probs = np.reshape(category_probs, [-1, category_probs.shape[-1]])
    sampled_indices = np.array([
        sample_categorical_indices(probabilities)
        for probabilities in flat_probs
    ])
    category_indices = np.reshape(sampled_indices, category_probs.shape[:-1])
    return CATEGORY_VALUES[category_indices]


def trial_rows(config, graph_index, rewards, outputs):
    category_outputs = np.asarray(outputs[0][0], dtype=float)
    action_policy = np.asarray(outputs[1][0], dtype=float)
    mi = float(np.asarray(outputs[10]))
    node_selections = np.asarray(outputs[12][0], dtype=int)
    stop_decisions = np.asarray(outputs[13][0, :, 0], dtype=bool)
    observed_masks = np.asarray(outputs[14][0], dtype=bool)
    kl_d_sequence = np.asarray(outputs[16][0, :, 0], dtype=float)
    terminal_path = int(np.asarray(outputs[-1][0])) if len(outputs) > 28 else -1

    time_steps = config.time_steps
    stop_seen = np.maximum.accumulate(stop_decisions)
    stopped_before = np.concatenate(([False], stop_seen[:-1]))

    estimated_rewards = scalar_estimates(category_outputs)
    estimated_rewards = np.where(observed_masks, estimated_rewards, np.nan)

    path_rewards = calculate_path_rewards_sim(config.index_path_map, rewards)
    if terminal_path >= 0:
        chosen_path = terminal_path
    else:
        chosen_path = sample_categorical_indices(action_policy)
    v = float(path_rewards[chosen_path])

    row_template = {
        "graph": int(graph_index),
        "chosen_path": chosen_path,
        "V": v,
        "MI": mi,
    }

    for t in range(time_steps):
        step = t + 1
        stopped_at_or_before_step = bool(stop_seen[t])
        selected_index = int(node_selections[t])
        stopped_at_step = bool(stop_decisions[t])

        if stopped_at_or_before_step or selected_index >= time_steps:
            expanded_node = np.nan
            expanded_reward = np.nan
        else:
            expanded_node = selected_index + 1
            expanded_reward = float(rewards[selected_index])

        row_template[f"expanded_node_t{step}"] = expanded_node
        row_template[f"expanded_reward_t{step}"] = expanded_reward
        row_template[f"stop_t{step}"] = stopped_at_step
        row_template[f"kl_d_t{step}"] = np.nan if stopped_before[t] else float(kl_d_sequence[t])

    rows = []
    for node_index in range(time_steps):
        row = dict(row_template)
        row["node"] = node_index + 1
        row["actual_reward"] = float(rewards[node_index])

        for t in range(time_steps):
            row[f"estimated_reward_t{t + 1}"] = estimated_rewards[t, node_index]

        rows.append(row)

    return rows


def run_simulation(config):
    print("--- Mode: INFERENCE ---")

    num_trials = 2000
    time_steps = config.time_steps

    if config.input_type == "binary":
        rewards_list = [
            [random.choice([0, 1]) for _ in range(time_steps)]
            for _ in range(num_trials)
        ]
    else:
        rewards_list = [
            [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(time_steps)]
            for _ in range(num_trials)
        ]

    for beta in config.beta_values:
        for lambda_ in config.lambda_values:
            for alpha in config.alpha_values:
                for opportunity_cost in config.opportunity_cost_values:
                    print(
                        f"\nEvaluating -> lambda: {lambda_}, alpha: {alpha}, "
                        f"beta: {beta}, opportunity_cost: {opportunity_cost}, "
                        f"expansion_decision_version: {config.expansion_decision_version}, "
                        f"model_variant: {config.model_variant}"
                    )

                    model_name = model_name_for(config, lambda_, alpha, beta, opportunity_cost)
                    weights_file_path = os.path.join(config.dir_name, model_name + ".weights.h5")

                    vrnn_model = build_model(config, alpha, beta, lambda_, opportunity_cost)

                    if not os.path.exists(weights_file_path):
                        print(f"⚠️ Weights file not found at: {weights_file_path}. Skipping combination.")
                        continue

                    dummy_input = tf.zeros((1, config.time_steps, 1), dtype=tf.float32)
                    _ = vrnn_model(dummy_input, training=False)

                    try:
                        vrnn_model.load_weights(weights_file_path)
                        print("✅ Weights loaded successfully.")
                    except Exception as e:
                        print(f"❌ Error loading weights: {e}")
                        continue

                    sim_data = []
                    print(f"Running {num_trials} trials...")

                    for graph_index, rewards in enumerate(rewards_list):
                        rewards_tensor = tf.constant(rewards, dtype=tf.float32)
                        rewards_tensor = tf.reshape(rewards_tensor, [1, -1, 1])
                        outputs = vrnn_model(
                            rewards_tensor,
                            training=True,
                            compute_losses=False,
                            expansion_epsilon=0.0
                        )
                        sim_data.extend(trial_rows(config, graph_index, rewards, outputs))

                    df = pd.DataFrame(sim_data)
                    df["opportunity_cost"] = opportunity_cost
                    df["expansion_decision_version"] = config.expansion_decision_version
                    output_file = os.path.join(config.sim_dir_name, model_name + "_"+ config.input_type+ ".csv")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    df.to_csv(output_file, index=False)
                    print(f"✅ Saved simulation results to: {output_file}")
