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
TERMINAL_PATH_OUTPUT_INDEX = 28
OBSERVATION_KL_OUTPUT_INDEX = 29
LSTM_STATE_OUTPUT_INDEX = 30
DECODER_STATE_OUTPUT_INDEX = 31


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
        reward_norm_value=getattr(config, "reward_norm_value", None),
    )


def model_variant_label(variant):
    return f"variant_{variant}_"


def architecture_file_label(config):
    return f"rnn_{config.rnn_units}_latent_{config.latent_dim}"


def model_name_for(config, lambda_, alpha, beta, opportunity_cost):
    variant_label = model_variant_label(config.model_variant)
    tree_label = f"{config.tree_size}n{getattr(config, 'tree_name_suffix', '')}"
    return (
        f"lambda_{lambda_}_alpha_{alpha}_beta_{beta}_opportunity_{opportunity_cost}_"
        f"expansion_{config.expansion_decision_version}_"
        f"{variant_label}"
        f"seed_{config.seed}_{tree_label}_{architecture_file_label(config)}"
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


def reward_category_indices(rewards):
    rewards = np.asarray(rewards, dtype=float)
    indices = np.floor(4.0 - rewards + 0.5).astype(int)
    return np.clip(indices, 0, len(CATEGORY_VALUES) - 1)


def build_probe_model(feature_dim, time_steps, num_categories):
    inputs = tf.keras.layers.Input(shape=(feature_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    logits = tf.keras.layers.Dense(time_steps * num_categories)(x)
    outputs = tf.keras.layers.Reshape((time_steps, num_categories))(logits)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    return model


def build_single_output_probe_model(feature_dim, output_dim):
    inputs = tf.keras.layers.Input(shape=(feature_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    logits = tf.keras.layers.Dense(output_dim)(x)
    model = tf.keras.Model(inputs, logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    return model


def make_probe_split(num_trials, seed):
    rng = np.random.default_rng(seed)
    trial_indices = np.arange(num_trials)
    rng.shuffle(trial_indices)
    if num_trials <= 1:
        train_trials = trial_indices
        test_trials = trial_indices
    else:
        train_count = max(1, int(0.8 * num_trials))
        train_count = min(train_count, num_trials - 1)
        train_trials = trial_indices[:train_count]
        test_trials = trial_indices[train_count:]

    split = np.full(num_trials, "test", dtype=object)
    split[train_trials] = "train"
    split[test_trials] = "test"
    return split


def train_single_deep_probe(features, observed_masks, rewards, seed, split):
    features = np.asarray(features, dtype=np.float32)
    observed_masks = np.asarray(observed_masks, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)

    num_trials, time_steps, feature_dim = features.shape
    num_categories = len(CATEGORY_VALUES)
    reward_indices = reward_category_indices(rewards)
    reward_onehot = np.eye(num_categories, dtype=np.float32)[reward_indices]
    y_all = reward_onehot
    pred_indices = np.full((num_trials, time_steps, time_steps), -1, dtype=int)

    # Train a separate frozen diagnostic probe for each state timestep. This
    # avoids letting easy early states dominate a single shared post-hoc probe.
    for state_t in range(time_steps):
        x_step = features[:, state_t, :]
        w_step = observed_masks[:, state_t, :]
        valid_samples = np.sum(w_step, axis=1) > 0

        if not np.any(valid_samples):
            continue

        train_samples = valid_samples & (split == "train")
        if not np.any(train_samples):
            train_samples = valid_samples

        tf.random.set_seed(seed + state_t)
        model = build_probe_model(feature_dim, time_steps, num_categories)
        model.fit(
            x_step[train_samples],
            y_all[train_samples],
            sample_weight=w_step[train_samples],
            epochs=80,
            batch_size=128,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=8,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            ],
        )

        logits = model.predict(x_step, batch_size=256, verbose=0)
        pred_indices[:, state_t, :] = np.argmax(logits, axis=-1)

    if np.all(pred_indices < 0):
        nan_pred = np.full((num_trials, time_steps, time_steps), np.nan, dtype=float)
        return {
            "pred_rewards": nan_pred,
            "correct": nan_pred.copy(),
            "split": np.full(num_trials, "test", dtype=object),
        }

    pred_indices = np.where(pred_indices < 0, 0, pred_indices)
    pred_rewards = CATEGORY_VALUES[pred_indices]
    target_indices = np.repeat(reward_indices[:, None, :], time_steps, axis=1)
    correct = (pred_indices == target_indices).astype(float)
    pred_rewards = np.where(observed_masks > 0, pred_rewards, np.nan)
    correct = np.where(observed_masks > 0, correct, np.nan)

    return {
        "pred_rewards": pred_rewards,
        "correct": correct,
        "split": split,
    }


def path_map_array_from_config(config):
    if hasattr(config, "path_map_np"):
        return np.asarray(config.path_map_np, dtype=np.float32)

    path_rows = []
    for node_indices in config.index_path_map.values():
        path_row = np.zeros(config.time_steps, dtype=np.float32)
        for node in node_indices:
            node_idx = int(node) - 1
            if 0 <= node_idx < config.time_steps:
                path_row[node_idx] = 1.0
        path_rows.append(path_row)
    return np.stack(path_rows, axis=0)


def compute_observed_path_targets(observed_masks, rewards, path_map):
    observed_masks = np.asarray(observed_masks, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    path_map = np.asarray(path_map, dtype=np.float32)

    observed_rewards = observed_masks * rewards[:, None, :]
    path_values = np.einsum("pt,nst->nsp", path_map, observed_rewards)
    best_path_values = np.max(path_values, axis=-1)
    best_path_mask = np.isclose(path_values, best_path_values[:, :, None])
    best_path_targets = best_path_mask.astype(np.float32)
    best_path_targets /= np.maximum(
        np.sum(best_path_targets, axis=-1, keepdims=True),
        1.0,
    )
    first_best_paths = np.argmax(path_values, axis=-1)
    return best_path_values, first_best_paths, best_path_targets


def train_single_best_value_probe(features, target_values, seed, split):
    features = np.asarray(features, dtype=np.float32)
    target_values = np.asarray(target_values, dtype=np.float32)

    num_trials, time_steps, feature_dim = features.shape
    finite_values = target_values[np.isfinite(target_values)]
    if finite_values.size == 0:
        nan_pred = np.full((num_trials, time_steps), np.nan, dtype=float)
        return {
            "pred_best_path_value": nan_pred,
            "correct_best_path_value": nan_pred.copy(),
            "target_best_path_value": target_values,
        }

    class_values = np.array(sorted(np.unique(finite_values)), dtype=np.float32)
    value_to_index = {float(value): idx for idx, value in enumerate(class_values)}
    target_indices = np.full(target_values.shape, -1, dtype=int)
    for value, idx in value_to_index.items():
        target_indices[np.isclose(target_values, value)] = idx

    num_classes = len(class_values)
    pred_indices = np.full((num_trials, time_steps), -1, dtype=int)

    for state_t in range(time_steps):
        valid_samples = target_indices[:, state_t] >= 0
        if not np.any(valid_samples):
            continue

        train_samples = valid_samples & (split == "train")
        if not np.any(train_samples):
            train_samples = valid_samples

        y_step = np.eye(num_classes, dtype=np.float32)[target_indices[:, state_t]]
        tf.random.set_seed(seed + state_t)
        model = build_single_output_probe_model(feature_dim, num_classes)
        model.fit(
            features[train_samples, state_t, :],
            y_step[train_samples],
            epochs=80,
            batch_size=128,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=8,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            ],
        )

        logits = model.predict(features[:, state_t, :], batch_size=256, verbose=0)
        pred_indices[:, state_t] = np.argmax(logits, axis=-1)

    pred_values = np.full((num_trials, time_steps), np.nan, dtype=float)
    valid_predictions = pred_indices >= 0
    pred_values[valid_predictions] = class_values[pred_indices[valid_predictions]]
    correct = np.where(
        valid_predictions & (target_indices >= 0),
        (pred_indices == target_indices).astype(float),
        np.nan,
    )
    return {
        "pred_best_path_value": pred_values,
        "correct_best_path_value": correct,
        "target_best_path_value": target_values,
    }


def train_single_best_path_probe(features, target_probs, first_best_paths, seed, split):
    features = np.asarray(features, dtype=np.float32)
    target_probs = np.asarray(target_probs, dtype=np.float32)
    first_best_paths = np.asarray(first_best_paths, dtype=int)

    num_trials, time_steps, feature_dim = features.shape
    num_paths = target_probs.shape[-1]
    pred_paths = np.full((num_trials, time_steps), -1, dtype=int)

    for state_t in range(time_steps):
        valid_samples = np.sum(target_probs[:, state_t, :], axis=1) > 0.0
        if not np.any(valid_samples):
            continue

        train_samples = valid_samples & (split == "train")
        if not np.any(train_samples):
            train_samples = valid_samples

        tf.random.set_seed(seed + state_t)
        model = build_single_output_probe_model(feature_dim, num_paths)
        model.fit(
            features[train_samples, state_t, :],
            target_probs[train_samples, state_t, :],
            epochs=80,
            batch_size=128,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=8,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            ],
        )

        logits = model.predict(features[:, state_t, :], batch_size=256, verbose=0)
        pred_paths[:, state_t] = np.argmax(logits, axis=-1)

    correct = np.full((num_trials, time_steps), np.nan, dtype=float)
    for state_t in range(time_steps):
        valid_predictions = pred_paths[:, state_t] >= 0
        if not np.any(valid_predictions):
            continue
        correct[valid_predictions, state_t] = target_probs[
            np.where(valid_predictions)[0],
            state_t,
            pred_paths[valid_predictions, state_t],
        ] > 0.0

    return {
        "pred_best_path": pred_paths.astype(float),
        "correct_best_path": correct,
        "target_best_path": first_best_paths.astype(float),
    }


def train_single_path_diagnostic_probes(features, best_path_values, best_path_targets, first_best_paths, seed, split):
    best_value_results = train_single_best_value_probe(
        features,
        best_path_values,
        seed=seed + 1_000,
        split=split,
    )
    best_path_results = train_single_best_path_probe(
        features,
        best_path_targets,
        first_best_paths,
        seed=seed + 2_000,
        split=split,
    )
    return {
        **best_value_results,
        **best_path_results,
    }


def train_deep_reward_probes(lstm_features, decoder_features, observed_masks, rewards, path_map, seed):
    shared_split = make_probe_split(np.asarray(rewards).shape[0], seed + 17)
    best_path_values, first_best_paths, best_path_targets = compute_observed_path_targets(
        observed_masks,
        rewards,
        path_map,
    )

    probe_inputs = {
        "lstm": (lstm_features, seed + 101),
        "decoder": (decoder_features, seed + 202),
    }
    results = {}
    for source_name, (source_features, source_seed) in probe_inputs.items():
        reward_results = train_single_deep_probe(
            source_features,
            observed_masks,
            rewards,
            seed=source_seed,
            split=shared_split,
        )
        path_results = train_single_path_diagnostic_probes(
            source_features,
            best_path_values,
            best_path_targets,
            first_best_paths,
            seed=source_seed,
            split=shared_split,
        )
        results[source_name] = {
            **reward_results,
            **path_results,
        }
    return results


def trial_rows(config, graph_index, rewards, outputs, probe_predictions=None, probe_split=None):
    category_outputs = np.asarray(outputs[0][0], dtype=float)
    action_policy = np.asarray(outputs[1][0], dtype=float)
    mi = float(np.asarray(outputs[10]))
    node_selections = np.asarray(outputs[12][0], dtype=int)
    stop_decisions = np.asarray(outputs[13][0, :, 0], dtype=bool)
    observed_masks = np.asarray(outputs[14][0], dtype=bool)
    kl_d_sequence = np.asarray(outputs[16][0, :, 0], dtype=float)
    observation_kl_d_sequence = (
        np.asarray(outputs[OBSERVATION_KL_OUTPUT_INDEX][0, :, 0], dtype=float)
        if len(outputs) > OBSERVATION_KL_OUTPUT_INDEX
        else kl_d_sequence
    )
    terminal_path = (
        int(np.asarray(outputs[TERMINAL_PATH_OUTPUT_INDEX][0]))
        if len(outputs) > TERMINAL_PATH_OUTPUT_INDEX
        else -1
    )

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
        row_template[f"kl_d_obs_t{step}"] = (
            np.nan
            if np.isnan(expanded_reward)
            else float(observation_kl_d_sequence[t])
        )

    rows = []
    for node_index in range(time_steps):
        row = dict(row_template)
        row["node"] = node_index + 1
        row["actual_reward"] = float(rewards[node_index])
        if probe_split is not None:
            row["deep_probe_split"] = probe_split

        for t in range(time_steps):
            row[f"estimated_reward_t{t + 1}"] = estimated_rewards[t, node_index]
            if probe_predictions is not None:
                for source_name, source_predictions in probe_predictions.items():
                    pred_reward = source_predictions["pred_rewards"][t, node_index]
                    correct = source_predictions["correct"][t, node_index]
                    row[f"{source_name}_deep_probe_pred_reward_t{t + 1}"] = pred_reward
                    row[f"{source_name}_deep_probe_correct_t{t + 1}"] = correct
                    if "pred_best_path_value" in source_predictions:
                        row[f"{source_name}_deep_probe_target_best_path_value_t{t + 1}"] = (
                            source_predictions["target_best_path_value"][t]
                        )
                        row[f"{source_name}_deep_probe_pred_best_path_value_t{t + 1}"] = (
                            source_predictions["pred_best_path_value"][t]
                        )
                        row[f"{source_name}_deep_probe_correct_best_path_value_t{t + 1}"] = (
                            source_predictions["correct_best_path_value"][t]
                        )
                    if "pred_best_path" in source_predictions:
                        row[f"{source_name}_deep_probe_target_best_path_t{t + 1}"] = (
                            source_predictions["target_best_path"][t]
                        )
                        row[f"{source_name}_deep_probe_pred_best_path_t{t + 1}"] = (
                            source_predictions["pred_best_path"][t]
                        )
                        row[f"{source_name}_deep_probe_correct_best_path_t{t + 1}"] = (
                            source_predictions["correct_best_path"][t]
                        )

        rows.append(row)

    return rows


def tensor_outputs_to_numpy(outputs):
    converted = []
    for output in outputs:
        if hasattr(output, "numpy"):
            converted.append(output.numpy())
        else:
            converted.append(np.asarray(output))
    return tuple(converted)


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
                    if hasattr(vrnn_model, "build_target_critic"):
                        vrnn_model.build_target_critic()

                    try:
                        vrnn_model.load_weights(weights_file_path)
                        print("✅ Weights loaded successfully.")
                    except Exception as e:
                        print(f"❌ Error loading weights: {e}")
                        continue

                    trial_outputs = []
                    trial_rewards = []
                    lstm_features = []
                    decoder_features = []
                    probe_observed_masks = []
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
                        outputs_np = tensor_outputs_to_numpy(outputs)
                        trial_outputs.append(outputs_np)
                        trial_rewards.append(np.asarray(rewards, dtype=np.float32))
                        probe_observed_masks.append(np.asarray(outputs_np[14][0], dtype=np.float32))
                        if len(outputs_np) > DECODER_STATE_OUTPUT_INDEX:
                            lstm_features.append(np.asarray(outputs_np[LSTM_STATE_OUTPUT_INDEX][0], dtype=np.float32))
                            decoder_features.append(np.asarray(outputs_np[DECODER_STATE_OUTPUT_INDEX][0], dtype=np.float32))

                    if len(lstm_features) == num_trials and len(decoder_features) == num_trials:
                        print("Training frozen deep reward/path probes for LSTM and decoder states...")
                        deep_probe_predictions = train_deep_reward_probes(
                            np.stack(lstm_features, axis=0),
                            np.stack(decoder_features, axis=0),
                            np.stack(probe_observed_masks, axis=0),
                            np.stack(trial_rewards, axis=0),
                            path_map=path_map_array_from_config(config),
                            seed=config.seed,
                        )
                    else:
                        print("⚠️ Deep probe features were unavailable; skipping probe columns.")
                        deep_probe_predictions = None

                    sim_data = []
                    for graph_index, (rewards, outputs_np) in enumerate(zip(trial_rewards, trial_outputs)):
                        if deep_probe_predictions is None:
                            graph_probe_predictions = None
                            graph_probe_split = None
                        else:
                            graph_probe_predictions = {
                                source_name: {
                                    "pred_rewards": source_predictions["pred_rewards"][graph_index],
                                    "correct": source_predictions["correct"][graph_index],
                                    "target_best_path_value": (
                                        source_predictions["target_best_path_value"][graph_index]
                                    ),
                                    "pred_best_path_value": (
                                        source_predictions["pred_best_path_value"][graph_index]
                                    ),
                                    "correct_best_path_value": (
                                        source_predictions["correct_best_path_value"][graph_index]
                                    ),
                                    "target_best_path": source_predictions["target_best_path"][graph_index],
                                    "pred_best_path": source_predictions["pred_best_path"][graph_index],
                                    "correct_best_path": source_predictions["correct_best_path"][graph_index],
                                }
                                for source_name, source_predictions in deep_probe_predictions.items()
                            }
                            graph_probe_split = deep_probe_predictions["lstm"]["split"][graph_index]

                        sim_data.extend(
                            trial_rows(
                                config,
                                graph_index,
                                rewards,
                                outputs_np,
                                probe_predictions=graph_probe_predictions,
                                probe_split=graph_probe_split,
                            )
                        )

                    df = pd.DataFrame(sim_data)
                    df["opportunity_cost"] = opportunity_cost
                    df["expansion_decision_version"] = config.expansion_decision_version
                    output_file = os.path.join(config.sim_dir_name, model_name + "_"+ config.input_type+ ".csv")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    df.to_csv(output_file, index=False)
                    print(f"✅ Saved simulation results to: {output_file}")
