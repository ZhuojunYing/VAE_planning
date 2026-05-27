
"""
model.py
Contains the neural network architecture, custom layers, and the VariationalRNN model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import helper


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_dim, latent_dim, rnn_units):
    """Build the encoder network with Layer Normalization."""
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(rnn_units)(inputs)        
    x = layers.LayerNormalization()(x)         
    x = layers.Activation('relu')(x)           
    
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    model = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return model

def build_decoder(latent_dim, output_dim, rnn_units):
    """Build the decoder network with Layer Normalization for stability."""
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    x = layers.Dense(rnn_units)(latent_inputs) 
    x = layers.LayerNormalization()(x)         
    x = layers.Activation('relu')(x)           
    
    outputs = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(latent_inputs, outputs, name='decoder')
    return model

class VariationalRNN(tf.keras.Model):
    def __init__(self, encoder, decoder, rnn_units, latent_dim, time_steps, num_paths, 
                 index_path_map, path_map, path_cov_mat, 
                 alpha=0.0, beta=1.0, lambda_=1.0, tree_type = "deep",
                 opportunity_cost=0.0, input_type="uniform",
                 expansion_decision_version="decoder",
                 use_autoencoder=True,
                 reward_norm_value=None):
        super(VariationalRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Output layer for categorical reconstruction
        # We need 2 outputs per timestep (Mean and Log-Scale) instead of 9
        self.reconstruction_head = tf.keras.layers.Dense(
            time_steps * 2, 
            activation='linear',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros'
        )
        
        self.stop_index = time_steps
        self.expansion_head = tf.keras.layers.Dense(
            time_steps + num_paths,
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name='expansion_head'
        )
        self.rnn_units = rnn_units
        self.lstm_cell = tf.keras.layers.LSTMCell(
            self.rnn_units,
            kernel_initializer='orthogonal',
            recurrent_initializer='orthogonal' 
        )
            
        self.time_steps = time_steps
        self.num_paths = num_paths
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.opportunity_cost = opportunity_cost
        self.input_type = input_type
        self.use_autoencoder = use_autoencoder
        self.expansion_decision_version = self.normalize_expansion_decision_version(
            expansion_decision_version
        )
        self.num_categories = 9 
        self.tree_type = tree_type
        self.reward_norm_value = reward_norm_value
        # Store the maps explicitly so they don't rely on global variables
        self.index_path_map = index_path_map
        self.path_map = path_map
        self.path_cov_mat = path_cov_mat
        self.joint_decision_dim = time_steps + num_paths
        # The decoder-policy condition cannot make a meaningful no-information
        # terminal decision: stopping before the first observation lets it avoid
        # the VAE bottleneck entirely. Keep that shortcut illegal only there.
        self.min_observations_before_stop = (
            1 if self.expansion_decision_version == "decoder" else 0
        )

        self.prior_mu = self.add_weight(
            name="prior_mu", 
            shape=(time_steps, latent_dim), 
            initializer="zeros", 
            trainable=True   
        )
        self.prior_logvar = self.add_weight(
            name="prior_logvar", 
            shape=(time_steps, latent_dim), 
            initializer="zeros", 
            trainable=True  
        )
        self.critic_head = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.rnn_units,
                activation=None,
                kernel_initializer='glorot_uniform'
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(
                max(self.rnn_units // 2, 16),
                activation='relu',
                kernel_initializer='glorot_uniform'
            ),
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer='glorot_uniform'
            )
        ], name='critic_head')
        self.lstm_reward_probe_head = tf.keras.layers.Dense(
            self.num_categories,
            activation=None,
            kernel_initializer='glorot_uniform',
            name='lstm_reward_probe_head'
        )

    def normalize_expansion_decision_version(self, version):
        version = str(version).strip().lower()
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
        if version not in aliases:
            raise ValueError(
                "expansion_decision_version must be one of: "
                "decoder/1, lstm/2, pre_lstm/3"
            )
        return aliases[version]

    def reward_norm(self):
        if self.reward_norm_value is not None:
            return self.reward_norm_value
        if self.time_steps == 6:
            return 3.58
        if self.time_steps == 2:
            if self.input_type == "binary":
                return 0.75
            return 1.5625
        if self.time_steps == 12:
            return 5.11
        return 8.1574

    def call(
        self,
        inputs,
        training=True,
        current_alpha=1.0,
        current_beta=1.0,
        current_critic_coef=1.0,
        forced_node_selections=None,
        old_expansion_log_probs=None,
        use_ppo_loss=False,
        compute_losses=True,
        ppo_clip=0.2,
        expansion_epsilon=0.0,
        expansion_entropy_coef=0.01,
        forced_continue_epsilon=0.0
    ):
        batch_size = tf.shape(inputs)[0]

        critic_loss = tf.constant(0.0, dtype=tf.float32)
        value_pred = None
        value_target = None

        categories_onehot = helper.scalar_to_categorical(inputs, self.num_categories)
        _, diagnostic_path_rewards = self._prepare_path_rewards(inputs)
        diagnostic_best_path_reward = tf.reduce_max(
            diagnostic_path_rewards,
            axis=1,
            keepdims=True
        )
        diagnostic_best_path_mask = tf.cast(
            tf.equal(diagnostic_path_rewards, diagnostic_best_path_reward),
            tf.float32
        )

        all_z_means = []
        all_z_log_vars = []
        all_category_outputs = []
        all_observed_masks = []
        all_expansion_log_probs = []
        all_expansion_entropies = []
        all_expansion_entropy_masks = []
        all_action_outputs = []
        all_node_selections = []
        all_stop_decisions = []
        all_terminal_path_selections = []
        all_stop_value_preds = []
        all_kl_d = []
        all_observation_kl_d = []
        all_lstm_states = []
        all_decoder_states = []
        all_lstm_probe_logits = []
        all_lstm_probe_targets = []
        all_lstm_probe_masks = []
        all_continue_after_reward_sums = []
        all_continue_after_reward_counts = []
        all_previous_reward_masks = []
        all_terminal_best_prob_pre = []
        all_terminal_best_prob_post = []
        valid_step_masks = []

        total_loss = tf.constant(0.0, dtype=tf.float32)
        first_decoder_loss = tf.constant(0.0, dtype=tf.float32)
        second_decoder_loss = tf.constant(0.0, dtype=tf.float32)
        action_head_loss = tf.constant(0.0, dtype=tf.float32)
        expansion_head_loss = tf.constant(0.0, dtype=tf.float32)

        z = tf.zeros([batch_size, self.latent_dim], dtype=tf.float32)
        information_cost = 0
        state = self.lstm_cell.get_initial_state(batch_size=batch_size)
        hidden_state_flat = state[0]
        c_flat = state[1]
        lstm_input_dim = self.time_steps + 1 + self.num_categories
        pre_lstm_expansion_context = tf.zeros(
            [batch_size, self.rnn_units + lstm_input_dim],
            dtype=tf.float32
        )
        lstm_expansion_context = hidden_state_flat
        active_mask = tf.ones([batch_size, 1], dtype=tf.float32)
        visited_node_mask = tf.zeros([batch_size, self.time_steps], dtype=tf.float32)
        observed_mask = tf.zeros([batch_size, self.time_steps, 1], dtype=tf.float32)
        observed_inputs = tf.zeros_like(inputs)
        pending_kl_per_sample = tf.zeros([batch_size], dtype=tf.float32)
        last_reward_onehot = tf.zeros([batch_size, self.num_categories], dtype=tf.float32)

        for t in range(self.time_steps):
            valid_step_masks.append(active_mask)
            if self.expansion_decision_version == "decoder":
                expansion_input = hidden_state_flat
            elif self.expansion_decision_version == "lstm":
                expansion_input = lstm_expansion_context
            else:
                expansion_input = pre_lstm_expansion_context
            all_stop_value_preds.append(self.critic_head(expansion_input))

            expansion_logits = self.expansion_head(expansion_input)
            # Joint action order: observe nodes first, then terminal path choices.
            # Observed nodes are masked; terminal choices are legal after the
            # condition-specific minimum number of observations.
            observed_count = tf.reduce_sum(visited_node_mask, axis=1, keepdims=True)
            has_observation = tf.cast(observed_count > 0.0, tf.float32)
            can_stop = tf.cast(
                observed_count >= tf.cast(self.min_observations_before_stop, tf.float32),
                tf.float32
            )
            terminal_action_mask = (
                (1.0 - can_stop) * tf.ones([batch_size, self.num_paths], dtype=tf.float32)
            )
            decision_mask = tf.concat([visited_node_mask, terminal_action_mask], axis=1)
            masked_expansion_logits = expansion_logits + (decision_mask * -1e9)
            masked_expansion_logits = tf.where(
                tf.math.is_finite(masked_expansion_logits),
                masked_expansion_logits,
                tf.fill(tf.shape(masked_expansion_logits), -1e9)
            )
            terminal_logits_pre = expansion_logits[:, self.time_steps:self.joint_decision_dim]
            terminal_probs_pre = tf.nn.softmax(terminal_logits_pre, axis=-1)

            if forced_node_selections is not None:
                next_node_indices = tf.cast(forced_node_selections[:, t], tf.int32)
            elif training:
                policy_sampled_indices = tf.squeeze(
                    tf.random.categorical(masked_expansion_logits, num_samples=1, dtype=tf.int32),
                    axis=-1
                )
                exploration_decision_mask = decision_mask
                if self.expansion_decision_version == "decoder":
                    terminal_exploration_mask = (
                        (1.0 - has_observation) *
                        tf.ones([batch_size, self.num_paths], dtype=tf.float32)
                    )
                    exploration_decision_mask = tf.concat(
                        [visited_node_mask, terminal_exploration_mask],
                        axis=1
                    )
                uniform_logits = exploration_decision_mask * -1e9
                uniform_logits = tf.where(
                    tf.math.is_finite(uniform_logits),
                    uniform_logits,
                    tf.fill(tf.shape(uniform_logits), -1e9)
                )
                uniform_sampled_indices = tf.squeeze(
                    tf.random.categorical(uniform_logits, num_samples=1, dtype=tf.int32),
                    axis=-1
                )
                legal_observe_mask = 1.0 - visited_node_mask
                has_observe_action = tf.cast(
                    tf.reduce_sum(legal_observe_mask, axis=1) > 0.0,
                    tf.float32
                )
                observe_logits = (1.0 - legal_observe_mask) * -1e9
                observe_sampled_indices = tf.squeeze(
                    tf.random.categorical(observe_logits, num_samples=1, dtype=tf.int32),
                    axis=-1
                )
                exploration_sampled_indices = uniform_sampled_indices
                expansion_epsilon = tf.clip_by_value(
                    tf.cast(expansion_epsilon, tf.float32),
                    0.0,
                    1.0
                )
                forced_continue_epsilon = tf.clip_by_value(
                    tf.cast(forced_continue_epsilon, tf.float32),
                    0.0,
                    1.0
                )
                active_decision_bool = tf.squeeze(active_mask, axis=-1) > 0.0
                explore_mask = (
                    (tf.random.uniform([batch_size], dtype=tf.float32) < expansion_epsilon)
                    & active_decision_bool
                )
                if self.time_steps > 2:
                    force_continue_mask = (
                        active_decision_bool
                        & (has_observe_action > 0.0)
                        & (
                            tf.random.uniform([batch_size], dtype=tf.float32)
                            < forced_continue_epsilon
                        )
                    )
                else:
                    force_continue_mask = tf.zeros([batch_size], dtype=tf.bool)
                next_node_indices = tf.where(
                    explore_mask,
                    exploration_sampled_indices,
                    policy_sampled_indices
                )
                next_node_indices = tf.where(
                    force_continue_mask,
                    observe_sampled_indices,
                    next_node_indices
                )
            else:
                next_node_indices = tf.argmax(masked_expansion_logits, axis=-1, output_type=tf.int32)

            next_node_indices = tf.clip_by_value(
                next_node_indices,
                0,
                self.joint_decision_dim - 1
            )
            chosen_log_probs = tf.gather_nd(
                tf.nn.log_softmax(masked_expansion_logits),
                tf.stack([tf.range(batch_size, dtype=tf.int32), next_node_indices], axis=1)
            )
            expansion_probs = tf.nn.softmax(masked_expansion_logits, axis=-1)
            expansion_entropy = -tf.reduce_sum(
                expansion_probs * tf.math.log(expansion_probs + 1e-8),
                axis=-1
            )
            all_expansion_log_probs.append(tf.expand_dims(chosen_log_probs, axis=-1))
            all_expansion_entropies.append(tf.expand_dims(expansion_entropy, axis=-1))
            if self.expansion_decision_version == "decoder":
                entropy_mask = active_mask
            else:
                entropy_mask = active_mask * has_observation
            all_expansion_entropy_masks.append(entropy_mask)

            is_stop_chosen = tf.cast(next_node_indices >= self.time_steps, tf.float32)
            is_observe_chosen = 1.0 - is_stop_chosen
            terminal_path_indices = next_node_indices - self.time_steps
            terminal_path_indices = tf.where(
                next_node_indices >= self.time_steps,
                terminal_path_indices,
                tf.fill([batch_size], -1)
            )
            all_node_selections.append(next_node_indices)
            all_terminal_path_selections.append(terminal_path_indices)
            all_stop_decisions.append(tf.expand_dims(is_stop_chosen, axis=-1) * active_mask)
            observe_active_mask = active_mask * tf.expand_dims(is_observe_chosen, axis=-1)
            next_active_mask = observe_active_mask

            active_decision_mask = tf.squeeze(active_mask, axis=-1)
            continue_decision = tf.squeeze(observe_active_mask, axis=-1)
            previous_reward_mask = last_reward_onehot * tf.expand_dims(active_decision_mask, axis=-1)
            all_previous_reward_masks.append(previous_reward_mask)
            all_continue_after_reward_sums.append(
                tf.reduce_sum(previous_reward_mask * tf.expand_dims(continue_decision, axis=-1), axis=0)
            )
            all_continue_after_reward_counts.append(
                tf.reduce_sum(previous_reward_mask, axis=0)
            )
            terminal_best_pre = tf.reduce_sum(
                terminal_probs_pre * diagnostic_best_path_mask,
                axis=1
            )
            all_terminal_best_prob_pre.append(terminal_best_pre)

            safe_node_indices = tf.minimum(next_node_indices, self.time_steps - 1)
            chosen_rewards = tf.gather(
                tf.squeeze(inputs, axis=-1),
                safe_node_indices,
                axis=1,
                batch_dims=1
            )
            chosen_rewards = tf.expand_dims(chosen_rewards, axis=-1)
            reward_onehot = helper.scalar_to_categorical(chosen_rewards, self.num_categories)
            reward_onehot = tf.squeeze(reward_onehot, axis=1) * observe_active_mask
            last_reward_onehot = reward_onehot + last_reward_onehot * (1.0 - observe_active_mask)

            expansion_token_indices = tf.where(
                next_node_indices < self.time_steps,
                safe_node_indices,
                tf.fill([batch_size], self.stop_index)
            )
            node_onehot = tf.one_hot(expansion_token_indices, self.time_steps + 1, dtype=tf.float32)
            lstm_input = tf.concat([node_onehot, reward_onehot], axis=1)

            prev_hidden_state = hidden_state_flat
            prev_c_state = c_flat
            pre_lstm_expansion_context = tf.concat([prev_hidden_state, lstm_input], axis=1)
            rnn_state = (hidden_state_flat, c_flat)
            _, candidate_state = self.lstm_cell(lstm_input, states=rnn_state)
            new_h = candidate_state[0] * observe_active_mask + hidden_state_flat * (1.0 - observe_active_mask)
            new_c = candidate_state[1] * observe_active_mask + c_flat * (1.0 - observe_active_mask)
            lstm_expansion_context = new_h
            all_lstm_probe_logits.append(
                self.lstm_reward_probe_head(tf.stop_gradient(new_h))
            )
            all_lstm_probe_targets.append(reward_onehot)
            all_lstm_probe_masks.append(observe_active_mask)
            all_lstm_states.append(new_h)
            state = (new_h, new_c)
            encoder_input = tf.concat(state, axis=-1)

            if self.use_autoencoder:
                z_mean, z_log_var, z_sampled = self.encoder(encoder_input)

                z = z_sampled
                all_z_means.append(z_mean)
                all_z_log_vars.append(z_log_var)

                decoder_input = tf.concat([z], axis=1)
                decoder_output = self.decoder(decoder_input)
                hidden_state_flat, c_flat = tf.split(decoder_output, num_or_size_splits=2, axis=-1)
                hidden_state_flat = hidden_state_flat * observe_active_mask + prev_hidden_state * (1.0 - observe_active_mask)
                c_flat = c_flat * observe_active_mask + prev_c_state * (1.0 - observe_active_mask)
            else:
                z_mean = tf.zeros([batch_size, self.latent_dim], dtype=tf.float32)
                z_log_var = tf.zeros([batch_size, self.latent_dim], dtype=tf.float32)
                z = z_mean
                all_z_means.append(z_mean)
                all_z_log_vars.append(z_log_var)
                hidden_state_flat = new_h
                c_flat = new_c
            all_decoder_states.append(hidden_state_flat)
            if self.expansion_decision_version == "decoder":
                action_input = hidden_state_flat
            elif self.expansion_decision_version == "lstm":
                action_input = lstm_expansion_context
            else:
                action_input = pre_lstm_expansion_context

            post_decision_logits = self.expansion_head(action_input)
            terminal_logits_post = post_decision_logits[:, self.time_steps:self.joint_decision_dim]
            terminal_probs_post = tf.nn.softmax(terminal_logits_post, axis=-1)
            step_action_output = (
                tf.expand_dims(is_stop_chosen, axis=-1) * terminal_probs_pre
                + tf.expand_dims(is_observe_chosen, axis=-1) * terminal_probs_post
            )
            terminal_best_post = tf.reduce_sum(
                step_action_output * diagnostic_best_path_mask,
                axis=1
            )
            all_terminal_best_prob_post.append(terminal_best_post)
            all_action_outputs.append(step_action_output)

            node_observation = tf.one_hot(safe_node_indices, self.time_steps, dtype=tf.float32)
            node_observation = tf.expand_dims(node_observation, axis=-1) * observe_active_mask[:, None, :]
            observed_mask = tf.minimum(observed_mask + node_observation, 1.0)
            observed_inputs = inputs * observed_mask
            all_observed_masks.append(tf.squeeze(observed_mask, axis=-1))

            if self.use_autoencoder:
                rec_decoder_input = tf.concat([hidden_state_flat], axis=1)

                dist_params = self.reconstruction_head(rec_decoder_input)
                dist_params = tf.reshape(dist_params, [batch_size, self.time_steps, 2])

                mu_raw = dist_params[:, :, 0:1]
                mu = 5*tf.math.tanh(mu_raw)
                scale = tf.nn.softplus(dist_params[:, :, 1:2]) + 1e-4

                bin_edges = tf.constant(
                    [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                    dtype=tf.float32
                )
                bin_edges = tf.reshape(bin_edges, [1, 1, 10])

                cdf_at_edges = tf.math.sigmoid((bin_edges - mu) / scale)
                category_slice_raw = cdf_at_edges[:, :, 1:10] - cdf_at_edges[:, :, 0:9]
                category_slice_raw = tf.reverse(category_slice_raw, axis=[-1])
                category_slice_raw = category_slice_raw + 1e-6
                category_sum = tf.reduce_sum(category_slice_raw, axis=-1, keepdims=True)
                category_slice = category_slice_raw / (category_sum + 1e-8)

                category_output = category_slice
            else:
                category_output = tf.ones(
                    [batch_size, self.time_steps, self.num_categories],
                    dtype=tf.float32
                ) / tf.cast(self.num_categories, tf.float32)
            all_category_outputs.append(category_output)

            if self.use_autoencoder:
                prior_mean, prior_var = self.compute_time_conditional_prior(t, batch_size)
                kl_per_sample = self.calculate_kl_per_sample(z_mean, z_log_var, prior_mean, prior_var)
                observe_mask = tf.squeeze(observe_active_mask, axis=-1)
                observed_kl_per_sample = kl_per_sample * observe_mask
                if self.expansion_decision_version in ("lstm", "pre_lstm"):
                    # For decisions made before the decoder bottleneck, KL is a carry cost:
                    # charge the previous observation only if this step continues observing.
                    masked_kl_per_sample = pending_kl_per_sample * observe_mask
                    pending_kl_per_sample = observed_kl_per_sample
                else:
                    masked_kl_per_sample = observed_kl_per_sample
                kl = tf.reduce_mean(masked_kl_per_sample)
            else:
                masked_kl_per_sample = tf.zeros([batch_size], dtype=tf.float32)
                observed_kl_per_sample = tf.zeros([batch_size], dtype=tf.float32)
                kl = tf.constant(0.0, dtype=tf.float32)
            all_kl_d.append(tf.expand_dims(masked_kl_per_sample, axis=-1))
            all_observation_kl_d.append(tf.expand_dims(observed_kl_per_sample, axis=-1))
            information_cost += kl

            visited_observation = tf.one_hot(safe_node_indices, self.time_steps, dtype=tf.float32)
            visited_observation = visited_observation * observe_active_mask
            visited_node_mask = tf.minimum(visited_node_mask + visited_observation, 1.0)
            active_mask = next_active_mask

        all_z_means = tf.stack(all_z_means, axis=1)
        all_z_log_vars = tf.stack(all_z_log_vars, axis=1)
        observed_masks = tf.stack(all_observed_masks, axis=1)
        action_outputs_sequence = tf.stack(all_action_outputs, axis=1)
        node_selections = tf.stack(all_node_selections, axis=1)
        terminal_path_selections = tf.stack(all_terminal_path_selections, axis=1)
        stop_decisions = tf.stack(all_stop_decisions, axis=1)
        stop_value_preds = tf.stack(all_stop_value_preds, axis=1)
        kl_d_sequence = tf.stack(all_kl_d, axis=1)
        observation_kl_d_sequence = tf.stack(all_observation_kl_d, axis=1)
        lstm_state_sequence = tf.stack(all_lstm_states, axis=1)
        decoder_state_sequence = tf.stack(all_decoder_states, axis=1)
        lstm_probe_logits = tf.stack(all_lstm_probe_logits, axis=1)
        lstm_probe_targets = tf.stack(all_lstm_probe_targets, axis=1)
        lstm_probe_masks = tf.stack(all_lstm_probe_masks, axis=1)
        continue_after_reward_sums = tf.stack(all_continue_after_reward_sums, axis=0)
        continue_after_reward_counts = tf.stack(all_continue_after_reward_counts, axis=0)
        previous_reward_masks = tf.stack(all_previous_reward_masks, axis=0)
        terminal_best_prob_pre = tf.stack(all_terminal_best_prob_pre, axis=0)
        terminal_best_prob_post = tf.stack(all_terminal_best_prob_post, axis=0)
        valid_step_masks = tf.stack(valid_step_masks, axis=1)

        stop_flags = tf.squeeze(stop_decisions > 0, axis=-1)
        has_stop = tf.reduce_any(stop_flags, axis=1)
        first_stop_index = tf.argmax(
            tf.cast(stop_flags, tf.int32),
            axis=1,
            output_type=tf.int32
        )
        selected_action_index = tf.where(
            has_stop,
            first_stop_index,
            tf.fill([batch_size], self.time_steps - 1)
        )
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, selected_action_index], axis=1)
        action_output = tf.gather_nd(action_outputs_sequence, gather_indices)
        terminal_path_output = tf.gather_nd(terminal_path_selections, gather_indices)
        terminal_path_output = tf.where(
            has_stop,
            terminal_path_output,
            tf.fill([batch_size], -1)
        )

        # Train the actor on the policy available at the model's stopping point.
        action_loss = self.compute_final_actor_loss(inputs, action_output)
        kl_scaler = 1
        information_loss = information_cost / self.time_steps / kl_scaler

        if self.use_autoencoder:
            reconstruction_loss = self.compute_categorical_cross_entropy_loss(
                categories_onehot,
                all_category_outputs,
                observed_masks
            )
        else:
            reconstruction_loss = tf.constant(0.0, dtype=tf.float32)
        expansion_log_probs = tf.stack(all_expansion_log_probs, axis=1)
        # Dreamer-style reward-to-go for each sampled observe/stop decision:
        # terminal choice value minus future opportunity and information costs.
        expansion_return_targets = self.compute_expansion_return_targets(
            inputs,
            action_outputs_sequence,
            terminal_path_selections,
            valid_step_masks,
            stop_decisions,
            kl_d_sequence,
            current_beta
        )
        critic_loss = self.compute_expansion_critic_loss(
            stop_value_preds,
            expansion_return_targets,
            valid_step_masks
        )
        value_pred = stop_value_preds
        value_target = expansion_return_targets
        diagnostic_reward_counts = tf.reduce_sum(previous_reward_masks, axis=1)
        stop_value_preds_t = tf.transpose(
            tf.squeeze(stop_value_preds, axis=-1),
            perm=[1, 0]
        )
        return_targets_t = tf.transpose(
            tf.squeeze(expansion_return_targets, axis=-1),
            perm=[1, 0]
        )
        advantages_t = return_targets_t - stop_value_preds_t
        critic_after_reward_sums = tf.reduce_sum(
            previous_reward_masks * stop_value_preds_t[:, :, None],
            axis=1
        )
        terminal_best_prob_pre_after_reward_sums = tf.reduce_sum(
            previous_reward_masks * terminal_best_prob_pre[:, :, None],
            axis=1
        )
        terminal_best_prob_post_after_reward_sums = tf.reduce_sum(
            previous_reward_masks * terminal_best_prob_post[:, :, None],
            axis=1
        )
        return_target_after_reward_sums = tf.reduce_sum(
            previous_reward_masks * return_targets_t[:, :, None],
            axis=1
        )
        advantage_after_reward_sums = tf.reduce_sum(
            previous_reward_masks * advantages_t[:, :, None],
            axis=1
        )
        expansion_loss = self.compute_expansion_policy_loss(
            expansion_log_probs,
            tf.stack(all_expansion_entropies, axis=1),
            tf.stack(all_expansion_entropy_masks, axis=1),
            stop_value_preds,
            valid_step_masks,
            expansion_return_targets,
            old_log_probs=old_expansion_log_probs,
            use_ppo_loss=use_ppo_loss,
            ppo_clip=ppo_clip,
            entropy_coef=expansion_entropy_coef
        )
        opportunity_loss = tf.constant(0.0, dtype=tf.float32)
        (
            lstm_probe_loss,
            lstm_probe_accuracy,
            lstm_probe_acc_by_category,
            lstm_probe_loss_by_category,
            lstm_probe_count_by_category
        ) = self.compute_lstm_reward_probe_metrics(
            lstm_probe_logits,
            lstm_probe_targets,
            lstm_probe_masks
        )
        valid_decision_count = tf.reduce_sum(valid_step_masks) + 1e-6
        expansion_stop_rate = tf.reduce_sum(stop_decisions) / valid_decision_count
        expansion_continue_rate = (
            tf.reduce_sum(valid_step_masks * (1.0 - tf.cast(stop_decisions, tf.float32))) /
            valid_decision_count
        )

        if training and compute_losses:
            tf.print("action loss:", action_loss)
            tf.print("critic loss:", critic_loss)
            tf.print("category loss:", reconstruction_loss)
            tf.print("expansion loss:", expansion_loss)
            tf.print("expansion stop rate:", expansion_stop_rate)
            tf.print("expansion continue rate:", expansion_continue_rate)
            tf.print("kl loss:", information_loss)
            tf.print(">> [CRITIC] Pred:", tf.reduce_mean(value_pred), "| Target:", tf.reduce_mean(value_target))

            self.add_loss(information_loss / self.beta)
            self.add_loss(action_loss* self.lambda_)
            self.add_loss(self.alpha * reconstruction_loss)
            self.add_loss(expansion_loss * self.lambda_)

            total_loss += sum(self.losses)

 
            first_decoder_loss += (
                information_loss * current_beta
                + action_loss * self.lambda_
                + expansion_loss * self.lambda_
                + critic_loss *  self.lambda_* current_critic_coef
                + reconstruction_loss * self.alpha
            )
            second_decoder_loss += reconstruction_loss
            expansion_head_loss += (
                 expansion_loss * self.lambda_
                 + critic_loss * self.lambda_ * current_critic_coef
                 + action_loss * self.lambda_
            )

        category_outputs = tf.stack(all_category_outputs, axis=1)

        return (
            category_outputs,
            action_output,
            total_loss,
            first_decoder_loss,
            second_decoder_loss,
            action_head_loss,
            critic_loss,
            information_loss,
            action_loss,
            reconstruction_loss,
            information_cost,
            all_z_means,
            node_selections,
            stop_decisions,
            observed_masks,
            action_outputs_sequence,
            kl_d_sequence,
            expansion_head_loss,
            expansion_log_probs,
            expansion_loss,
            expansion_stop_rate,
            expansion_continue_rate,
            opportunity_loss,
            lstm_probe_loss,
            lstm_probe_accuracy,
            lstm_probe_acc_by_category,
            lstm_probe_loss_by_category,
            lstm_probe_count_by_category,
            terminal_path_output,
            observation_kl_d_sequence,
            lstm_state_sequence,
            decoder_state_sequence,
            continue_after_reward_sums,
            continue_after_reward_counts,
            critic_after_reward_sums,
            diagnostic_reward_counts,
            terminal_best_prob_pre_after_reward_sums,
            terminal_best_prob_post_after_reward_sums,
            return_target_after_reward_sums,
            advantage_after_reward_sums
        )
    def compute_time_conditional_prior(self, t, batch_size):
        mu_t = self.prior_mu[t]           
        logvar_t = tf.clip_by_value(self.prior_logvar[t], -10.0, 10.0)
        
        prior_mean = tf.broadcast_to(mu_t, [batch_size, self.latent_dim])
        prior_var = tf.exp(tf.broadcast_to(logvar_t, [batch_size, self.latent_dim]))
        
        return prior_mean, prior_var

    def calculate_kl_per_sample(self, z_means, z_log_vars, prior_mean, prior_var, epsilon=1e-6):
        z_means = tf.where(tf.math.is_finite(z_means), z_means, tf.zeros_like(z_means))
        prior_mean = tf.where(tf.math.is_finite(prior_mean), prior_mean, tf.zeros_like(prior_mean))
        z_log_vars = tf.clip_by_value(
            tf.where(tf.math.is_finite(z_log_vars), z_log_vars, tf.zeros_like(z_log_vars)),
            -10.0,
            10.0
        )
        prior_var = tf.where(tf.math.is_finite(prior_var), prior_var, tf.ones_like(prior_var))
        prior_var = prior_var + epsilon
        prior_log_var = tf.math.log(prior_var)
        z_var = tf.exp(z_log_vars) + epsilon
        
        log_var_ratio = z_log_vars - prior_log_var
        
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var_ratio - ((tf.square(z_means - prior_mean) + z_var) / prior_var),
            axis=1
        )
        
        return kl_loss

    def calculate_kl_loss(self, z_means, z_log_vars, prior_mean, prior_var, epsilon=1e-6):
        kl_loss = self.calculate_kl_per_sample(
            z_means,
            z_log_vars,
            prior_mean,
            prior_var,
            epsilon=epsilon
        )
        return tf.reduce_mean(kl_loss)
    

    def calculate_path_rewards(self, rewards):
        num_paths = len(self.index_path_map)
        batch_size = tf.shape(rewards)[0]
        path_rewards = tf.TensorArray(dtype=tf.float32, size=num_paths, dynamic_size=False)

        for i, node_indices in enumerate(self.index_path_map.values()):
            try:
                # Convert node_indices to a tensor and adjust indexing
                node_indices_tensor = tf.convert_to_tensor(node_indices, dtype=tf.int32) - 1
                
                # Safety check for indices
                node_indices_tensor = tf.clip_by_value(node_indices_tensor, 0, tf.shape(rewards)[1] - 1)
                
                # Expand node_indices_tensor to match batch size
                node_indices_tensor = tf.tile(tf.expand_dims(node_indices_tensor, 0), [batch_size, 1])
                
                # Gather rewards for the current path across all batches
                gathered_rewards = tf.gather(rewards, node_indices_tensor, axis=1, batch_dims=1)
                path_reward = tf.reduce_sum(gathered_rewards, axis=1)
                
                # Check for NaN/Inf in path rewards
                path_reward = tf.where(tf.math.is_finite(path_reward), 
                                    path_reward, 
                                    tf.zeros_like(path_reward))
                
                path_rewards = path_rewards.write(i, path_reward)
                
            except Exception as e:
                tf.print(f"Error in path {i}: {e}")
                # Write zeros for this path if there's an error
                path_rewards = path_rewards.write(i, tf.zeros([batch_size], dtype=tf.float32))

        result = path_rewards.stack()  # [num_paths, batch_size]
        
        # Final safety check
        result = tf.where(tf.math.is_finite(result), result, tf.zeros_like(result))
        
        return result


    def compute_categorical_cross_entropy_loss(self, target_categories, category_outputs, observed_masks=None):
        time_steps = self.time_steps
        batch_size = tf.shape(target_categories)[0]
        
        target_category_onehot = tf.squeeze(target_categories, axis=2)  
        stacked_preds = tf.stack(category_outputs, axis=0)
        if observed_masks is None:
            mask = tf.linalg.band_part(tf.ones([time_steps, time_steps]), -1, 0)
            mask = mask[:, None, :]
        else:
            mask = tf.transpose(observed_masks, perm=[1, 0, 2])
        
        target_expanded = tf.tile(
            target_category_onehot[None, :, :, :], 
            [time_steps, 1, 1, 1]
        )  
        
        # THE FIX: Do not use clip_by_value. Add epsilon directly.
        # This preserves the derivative 1/(x+1e-7), ensuring massive 
        # corrective gradients flow backwards when the network is wrong.
        safe_probs = stacked_preds + 1e-7
        ce_raw = -tf.reduce_sum(target_expanded * tf.math.log(safe_probs), axis=-1)  
        ce_masked = ce_raw * mask
        
        total_loss = tf.reduce_sum(ce_masked) / tf.math.log(tf.cast(self.num_categories, tf.float32))
        valid_count = tf.reduce_sum(mask)
        
        return total_loss / (valid_count + 1e-6)

    def compute_expansion_return_targets(
        self,
        inputs,
        action_outputs_sequence,
        terminal_path_selections,
        valid_step_masks,
        stop_decisions,
        kl_d_sequence,
        current_beta
    ):
        _, path_rewards = self._prepare_path_rewards(inputs)

        stop_flags = tf.squeeze(tf.cast(stop_decisions > 0, tf.int32), axis=-1)
        has_stop = tf.reduce_any(tf.cast(stop_flags, tf.bool), axis=1)
        first_stop_index = tf.argmax(stop_flags, axis=1, output_type=tf.int32)
        selected_action_index = tf.where(
            has_stop,
            first_stop_index,
            tf.fill([tf.shape(action_outputs_sequence)[0]], self.time_steps - 1)
        )
        batch_indices = tf.range(tf.shape(action_outputs_sequence)[0], dtype=tf.int32)
        terminal_action_probs = tf.gather_nd(
            action_outputs_sequence,
            tf.stack([batch_indices, selected_action_index], axis=1)
        )
        terminal_path_indices = tf.gather_nd(
            terminal_path_selections,
            tf.stack([batch_indices, selected_action_index], axis=1)
        )
        terminal_path_is_sampled = terminal_path_indices >= 0
        sampled_terminal_probs = tf.one_hot(
            tf.maximum(terminal_path_indices, 0),
            self.num_paths,
            dtype=tf.float32
        )
        terminal_value_probs = tf.where(
            terminal_path_is_sampled[:, None],
            sampled_terminal_probs,
            terminal_action_probs
        )
        terminal_expected_reward = tf.reduce_sum(
            terminal_value_probs * path_rewards,
            axis=1,
            keepdims=True
        ) / self.reward_norm()

        mask = tf.cast(valid_step_masks, tf.float32)
        stop_decisions = tf.cast(stop_decisions, tf.float32)
        non_stop_expansion = mask * (1.0 - stop_decisions)
        step_costs = non_stop_expansion * (
            self.opportunity_cost + current_beta * kl_d_sequence
        )
        future_costs = tf.reverse(
            tf.cumsum(tf.reverse(step_costs, axis=[1]), axis=1),
            axis=[1]
        )
        return terminal_expected_reward[:, None, :] - future_costs

    def compute_expansion_critic_loss(
        self,
        stop_value_preds,
        return_targets,
        valid_step_masks
    ):
        mask = tf.cast(valid_step_masks, tf.float32)
        squared_error = tf.square(stop_value_preds - tf.stop_gradient(return_targets)) * mask
        loss = tf.reduce_sum(squared_error) / (tf.reduce_sum(mask) + 1e-6)
        return tf.where(tf.math.is_finite(loss), loss, tf.constant(0.0, dtype=tf.float32))

    def compute_lstm_reward_probe_metrics(
        self,
        logits,
        targets,
        masks
    ):
        mask = tf.squeeze(tf.cast(masks, tf.float32), axis=-1)
        target_idx = tf.argmax(targets, axis=-1, output_type=tf.int32)
        pred_idx = tf.argmax(logits, axis=-1, output_type=tf.int32)

        ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets,
            logits=logits
        )
        valid_count = tf.reduce_sum(mask) + 1e-6
        loss = tf.reduce_sum(ce * mask) / valid_count

        correct = tf.cast(tf.equal(pred_idx, target_idx), tf.float32)
        accuracy = tf.reduce_sum(correct * mask) / valid_count

        category_ids = tf.range(self.num_categories, dtype=tf.int32)
        category_mask = (
            tf.cast(target_idx[:, :, None] == category_ids[None, None, :], tf.float32)
            * mask[:, :, None]
        )
        category_counts = tf.reduce_sum(category_mask, axis=[0, 1])
        category_correct = tf.reduce_sum(correct[:, :, None] * category_mask, axis=[0, 1])
        category_loss_sum = tf.reduce_sum(ce[:, :, None] * category_mask, axis=[0, 1])
        nan_values = tf.fill([self.num_categories], tf.constant(float("nan"), dtype=tf.float32))
        acc_by_category = tf.where(
            category_counts > 0.0,
            category_correct / (category_counts + 1e-6),
            nan_values
        )
        loss_by_category = tf.where(
            category_counts > 0.0,
            category_loss_sum / (category_counts + 1e-6),
            nan_values
        )

        return (
            tf.where(tf.math.is_finite(loss), loss, tf.constant(0.0, dtype=tf.float32)),
            tf.where(tf.math.is_finite(accuracy), accuracy, tf.constant(0.0, dtype=tf.float32)),
            acc_by_category,
            loss_by_category,
            category_counts
        )

    def compute_expansion_policy_loss(
        self,
        log_probs,
        entropies,
        entropy_masks,
        stop_value_preds,
        valid_step_masks,
        return_targets,
        old_log_probs=None,
        use_ppo_loss=False,
        ppo_clip=0.2,
        entropy_coef=0.01
    ):
        mask = tf.cast(valid_step_masks, tf.float32)
        advantages = tf.stop_gradient(return_targets - stop_value_preds)

        if use_ppo_loss and old_log_probs is not None:
            old_log_probs = tf.stop_gradient(tf.cast(old_log_probs, tf.float32))
            log_ratio = tf.clip_by_value(log_probs - old_log_probs, -10.0, 10.0)
            ratios = tf.exp(log_ratio)
            clipped_ratios = tf.clip_by_value(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip)
            surrogate = tf.minimum(ratios * advantages, clipped_ratios * advantages)
            policy_loss = -surrogate * mask
        else:
            policy_loss = -log_probs * advantages * mask

        entropy_coef = tf.cast(entropy_coef, tf.float32)
        entropy_mask = mask * tf.cast(entropy_masks, tf.float32)
        entropy_bonus = entropies * entropy_mask
        loss = (
            tf.reduce_sum(policy_loss) / (tf.reduce_sum(mask) + 1e-6)
            - entropy_coef * tf.reduce_sum(entropy_bonus) / (tf.reduce_sum(entropy_mask) + 1e-6)
        )
        return tf.where(tf.math.is_finite(loss), loss, tf.constant(0.0, dtype=tf.float32))

    def compute_opportunity_policy_loss(
        self,
        log_probs,
        valid_step_masks,
        stop_decisions,
        old_log_probs=None,
        use_ppo_loss=False,
        ppo_clip=0.2
    ):
        mask = tf.cast(valid_step_masks, tf.float32)
        stop_decisions = tf.cast(stop_decisions, tf.float32)
        non_stop_expansion = mask * (1.0 - stop_decisions)
        opportunity_advantages = tf.stop_gradient(-non_stop_expansion)

        if use_ppo_loss and old_log_probs is not None:
            old_log_probs = tf.stop_gradient(tf.cast(old_log_probs, tf.float32))
            log_ratio = tf.clip_by_value(log_probs - old_log_probs, -10.0, 10.0)
            ratios = tf.exp(log_ratio)
            clipped_ratios = tf.clip_by_value(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip)
            surrogate = tf.minimum(
                ratios * opportunity_advantages,
                clipped_ratios * opportunity_advantages
            )
            policy_loss = -surrogate * mask
        else:
            policy_loss = -log_probs * opportunity_advantages * mask

        loss = tf.reduce_sum(policy_loss) / (tf.reduce_sum(mask) + 1e-6)
        return tf.where(tf.math.is_finite(loss), loss, tf.constant(0.0, dtype=tf.float32))


    # replace the current compute_action_loss function with these two functions

    def _prepare_path_rewards(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.where(tf.math.is_finite(inputs), inputs, tf.zeros_like(inputs))

        batch_size = tf.shape(inputs)[0]

        actual_path_rewards = self.calculate_path_rewards(inputs)
        actual_path_rewards = tf.where(
            tf.math.is_finite(actual_path_rewards),
            actual_path_rewards,
            tf.zeros_like(actual_path_rewards)
        )

        actual_path_rewards = tf.transpose(actual_path_rewards, perm=[1, 0, 2])
        actual_path_rewards = tf.reshape(actual_path_rewards, [batch_size, self.num_paths])

        epsilon = 1e-4
       
        normalized_path_rewards = actual_path_rewards 
        return actual_path_rewards, normalized_path_rewards

    def compute_final_actor_loss(self, inputs, action_probs, sample_mask=None):
        _, normalized_path_rewards = self._prepare_path_rewards(inputs)

        expected_reward = tf.reduce_sum(
            action_probs * normalized_path_rewards,
            axis=1,
            keepdims=True
        )

        entropy = -tf.reduce_sum(
            action_probs * tf.math.log(action_probs + 1e-8),
            axis=1,
            keepdims=True
        )

        if sample_mask is None:
            sample_mask = tf.ones_like(expected_reward)
        else:
            sample_mask = tf.cast(sample_mask, tf.float32)

        active_count = tf.reduce_sum(sample_mask)
        valid_count = active_count + 1e-6
        mean_expected_reward = tf.reduce_sum(expected_reward * sample_mask) / valid_count
        mean_entropy = tf.reduce_sum(entropy * sample_mask) / valid_count
        if self.time_steps == 12:
            entropy_beta = 0.05
        else:       
            entropy_beta = 0

        mean_expected_reward = mean_expected_reward / self.reward_norm()
        action_loss = (1.0 - mean_expected_reward) - entropy_beta * mean_entropy
        action_loss = tf.where(
            active_count > 0.0,
            action_loss,
            tf.constant(0.0, dtype=tf.float32)
        )

        action_loss = tf.where(
            tf.math.is_finite(action_loss),
            action_loss,
            tf.constant(0.0, dtype=tf.float32)
        )

        return action_loss
    def compute_best_achievable_value_target(self, inputs, t=None):
        """
        Critic target for the currently observed rewards.
    
        The normalization uses a precomputed per-timestep expected max partial
        reward, so the target is always scaled relative to what is achievable
        at that point in the sequence — not the fixed full-path maximum.
    
        Parameters
        ----------
        inputs : tf.Tensor [batch, time_steps, 1]
        t : unused, kept for compatibility with older callers.
    
        Returns
        -------
        value_target : tf.Tensor [batch, 1]  stop-gradient, values in ~[0, 1]
        """
        # Partial path rewards (unseen nodes contribute 0)
        actual_path_rewards, _ = self._prepare_path_rewards(inputs)
        # actual_path_rewards: [batch, num_paths]
    
        # Best partial path reward for each item in the batch
        best_partial = tf.reduce_max(actual_path_rewards, axis=1, keepdims=True)  # [batch, 1]
    
        # Normalize by the expected max achievable at this timestep
        norm = self.reward_norm()
        epsilon = 1e-4
        # normalized = (best_partial + epsilon) / (norm + epsilon)
        normalized =  best_partial  
        value_target = tf.stop_gradient(normalized)
        return value_target
 

    def compute_critic_loss(self, value_pred, value_target):
        critic_loss = tf.reduce_mean(tf.square(value_pred - value_target))
        critic_loss = tf.where(
            tf.math.is_finite(critic_loss),
            critic_loss,
            tf.constant(0.0, dtype=tf.float32)
        )
        return critic_loss
