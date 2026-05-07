
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
                 opportunity_cost=0.0):
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
        
        self.action_head = tf.keras.layers.Dense(
            num_paths, 
            activation=None,  
            kernel_initializer='glorot_uniform'
        )
        self.stop_index = time_steps
        self.expansion_head = tf.keras.layers.Dense(
            time_steps + 1,
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
        self.num_categories = 9 
        self.tree_type = tree_type
        # Store the maps explicitly so they don't rely on global variables
        self.index_path_map = index_path_map
        self.path_map = path_map
        self.path_cov_mat = path_cov_mat

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
        # add this inside __init__, right after self.action_head
        self.critic_head = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer='glorot_uniform',
            name='critic_head'
        )

    def call(self, inputs, training=True, current_alpha=1.0, current_beta=1.0, current_critic_coef=1.0):
        batch_size = tf.shape(inputs)[0]

        critic_loss = tf.constant(0.0, dtype=tf.float32)
        value_pred = None
        value_target = None

        categories_onehot = helper.scalar_to_categorical(inputs, self.num_categories)

        all_z_means = []
        all_z_log_vars = []
        all_category_outputs = []
        all_observed_masks = []
        all_expansion_log_probs = []
        all_expansion_entropies = []
        all_action_outputs = []
        all_node_selections = []
        all_stop_decisions = []
        all_kl_d = []
        valid_step_masks = []

        total_loss = 0
        first_decoder_loss = 0
        second_decoder_loss = 0
        action_head_loss = 0
        expansion_head_loss = 0

        z = tf.zeros([batch_size, self.latent_dim], dtype=tf.float32)
        information_cost = 0
        state = self.lstm_cell.get_initial_state(batch_size=batch_size)
        hidden_state_flat = state[0]
        c_flat = state[1]
        active_mask = tf.ones([batch_size, 1], dtype=tf.float32)
        visited_mask = tf.zeros([batch_size, self.time_steps + 1], dtype=tf.float32)
        observed_mask = tf.zeros([batch_size, self.time_steps, 1], dtype=tf.float32)
        observed_inputs = tf.zeros_like(inputs)

        cumulative_critic_loss = tf.constant(0.0, dtype=tf.float32)
        cumulative_action_loss = tf.constant(0.0, dtype=tf.float32)
        for t in range(self.time_steps):
            valid_step_masks.append(active_mask)
            expansion_logits = self.expansion_head(hidden_state_flat)
            masked_expansion_logits = expansion_logits + (visited_mask * -1e9)

            if training:
                sampled_indices = tf.random.categorical(masked_expansion_logits, num_samples=1, dtype=tf.int32)
                next_node_indices = tf.squeeze(sampled_indices, axis=-1)
            else:
                next_node_indices = tf.argmax(masked_expansion_logits, axis=-1, output_type=tf.int32)

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

            is_stop_chosen = tf.cast(tf.equal(next_node_indices, self.stop_index), tf.float32)
            all_node_selections.append(next_node_indices)
            all_stop_decisions.append(tf.expand_dims(is_stop_chosen, axis=-1) * active_mask)
            next_active_mask = active_mask * (1.0 - tf.expand_dims(is_stop_chosen, axis=-1))

            safe_node_indices = tf.minimum(next_node_indices, self.time_steps - 1)
            chosen_rewards = tf.gather(
                tf.squeeze(inputs, axis=-1),
                safe_node_indices,
                axis=1,
                batch_dims=1
            )
            chosen_rewards = tf.expand_dims(chosen_rewards, axis=-1) * next_active_mask

            node_onehot = tf.one_hot(next_node_indices, self.time_steps + 1, dtype=tf.float32)
            lstm_input = tf.concat([node_onehot, chosen_rewards], axis=1)

            prev_hidden_state = hidden_state_flat
            prev_c_state = c_flat
            rnn_state = (hidden_state_flat, c_flat)
            _, candidate_state = self.lstm_cell(lstm_input, states=rnn_state)
            new_h = candidate_state[0] * active_mask + hidden_state_flat * (1.0 - active_mask)
            new_c = candidate_state[1] * active_mask + c_flat * (1.0 - active_mask)
            state = (new_h, new_c)
            encoder_input = tf.concat(state, axis=-1)

            z_mean, z_log_var, z_sampled = self.encoder(encoder_input)

            z = z_sampled
            all_z_means.append(z_mean)
            all_z_log_vars.append(z_log_var)

            decoder_input = tf.concat([z], axis=1)
            decoder_output = self.decoder(decoder_input)
            hidden_state_flat, c_flat = tf.split(decoder_output, num_or_size_splits=2, axis=-1)
            hidden_state_flat = hidden_state_flat * active_mask + prev_hidden_state * (1.0 - active_mask)
            c_flat = c_flat * active_mask + prev_c_state * (1.0 - active_mask)
            # auxiliary action loss at timestep t based on observed info so far
            step_action_logits = self.action_head(hidden_state_flat)
            step_action_output = tf.nn.softmax(step_action_logits, axis=-1)
            all_action_outputs.append(step_action_output)

            node_observation = tf.one_hot(safe_node_indices, self.time_steps, dtype=tf.float32)
            node_observation = tf.expand_dims(node_observation, axis=-1) * next_active_mask[:, None, :]
            observed_mask = tf.minimum(observed_mask + node_observation, 1.0)
            observed_inputs = inputs * observed_mask
            all_observed_masks.append(tf.squeeze(observed_mask, axis=-1))

            step_action_loss = self.compute_final_actor_loss(observed_inputs, step_action_output)
            cumulative_action_loss += step_action_loss
            rec_decoder_input = tf.concat([hidden_state_flat], axis=1)

            dist_params = self.reconstruction_head(rec_decoder_input)
            dist_params = tf.reshape(dist_params, [batch_size, self.time_steps, 2])

            mu_raw = dist_params[:, :, 0:1]
            mu = 5*tf.math.tanh(mu_raw)
            scale = tf.exp(dist_params[:, :, 1:2]) + 1e-4

            bin_edges = tf.constant(
                [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                dtype=tf.float32
            )
            bin_edges = tf.reshape(bin_edges, [1, 1, 10])

            cdf_at_edges = tf.math.sigmoid((bin_edges - mu) / scale)
            category_slice_raw = cdf_at_edges[:, :, 1:10] - cdf_at_edges[:, :, 0:9]
            category_slice_raw = tf.reverse(category_slice_raw, axis=[-1])
            category_slice_raw = category_slice_raw + 1e-6
            category_slice = category_slice_raw / tf.reduce_sum(category_slice_raw, axis=-1, keepdims=True)

            category_output = category_slice
            all_category_outputs.append(category_output)

            # critic at every timestep
            step_value_pred = self.critic_head(hidden_state_flat)
            step_value_target = self.compute_best_achievable_value_target(observed_inputs)
            step_critic_loss = self.compute_critic_loss(step_value_pred, step_value_target)
            cumulative_critic_loss += step_critic_loss

            prior_mean, prior_var = self.compute_time_conditional_prior(t, batch_size)
            kl_per_sample = self.calculate_kl_per_sample(z_mean, z_log_var, prior_mean, prior_var)
            kl = tf.reduce_mean(kl_per_sample)
            all_kl_d.append(tf.expand_dims(kl_per_sample, axis=-1))
            information_cost += kl * tf.reduce_mean(active_mask)

            visited_mask = tf.minimum(visited_mask + node_onehot * active_mask, 1.0)
            active_mask = next_active_mask

        final_action_decoder_input = tf.concat([hidden_state_flat], axis=1)
        action_logits = self.action_head(final_action_decoder_input)
        action_output = tf.nn.softmax(action_logits, axis=-1)

        # final actor loss only
        action_loss = self.compute_final_actor_loss(inputs, action_output)
        aux_action_loss = cumulative_action_loss / tf.cast(self.time_steps, tf.float32)
        # average critic loss across time
        critic_loss = cumulative_critic_loss / tf.cast(self.time_steps, tf.float32)

        # final critic prediction/target for logging only
        value_pred = self.critic_head(final_action_decoder_input)
        value_target = self.compute_best_achievable_value_target(observed_inputs)

        all_z_means = tf.stack(all_z_means, axis=1)
        all_z_log_vars = tf.stack(all_z_log_vars, axis=1)
        observed_masks = tf.stack(all_observed_masks, axis=1)
        action_outputs_sequence = tf.stack(all_action_outputs, axis=1)
        node_selections = tf.stack(all_node_selections, axis=1)
        stop_decisions = tf.stack(all_stop_decisions, axis=1)
        kl_d_sequence = tf.stack(all_kl_d, axis=1)
        valid_step_masks = tf.stack(valid_step_masks, axis=1)

        kl_scaler = 1
        information_loss = information_cost / self.time_steps / kl_scaler

        reconstruction_loss = self.compute_categorical_cross_entropy_loss(
            categories_onehot,
            all_category_outputs,
            observed_masks
        )
        expansion_loss = self.compute_expansion_policy_loss(
            inputs,
            tf.stack(all_expansion_log_probs, axis=1),
            tf.stack(all_expansion_entropies, axis=1),
            action_outputs_sequence,
            valid_step_masks,
            stop_decisions
        )

        if training:
            tf.print("action loss:", action_loss)
            tf.print("critic loss:", critic_loss)
            tf.print("category loss:", reconstruction_loss)
            tf.print("expansion loss:", expansion_loss)
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
            action_head_loss += action_loss * self.lambda_
            expansion_head_loss += (
                information_loss * current_beta
                + action_loss * self.lambda_
                + expansion_loss * self.lambda_
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
            expansion_head_loss
        )
    def compute_time_conditional_prior(self, t, batch_size):
        mu_t = self.prior_mu[t]           
        logvar_t = self.prior_logvar[t]   
        
        prior_mean = tf.broadcast_to(mu_t, [batch_size, self.latent_dim])
        prior_var = tf.exp(tf.broadcast_to(logvar_t, [batch_size, self.latent_dim]))
        
        return prior_mean, prior_var

    def calculate_kl_per_sample(self, z_means, z_log_vars, prior_mean, prior_var, epsilon=1e-6):
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

    def compute_expansion_policy_loss(
        self,
        inputs,
        log_probs,
        entropies,
        action_outputs_sequence,
        valid_step_masks,
        stop_decisions
    ):
        _, path_rewards = self._prepare_path_rewards(inputs)
        step_expected_reward = tf.reduce_sum(
            action_outputs_sequence * path_rewards[:, None, :],
            axis=-1,
            keepdims=True
        )

        if self.time_steps == 6:
            reward_norm = 3.58
        elif self.time_steps == 2:
            reward_norm = 0.75
        elif self.time_steps == 12:
            reward_norm = 5.11
        else:
            reward_norm = 8.1574

        mask = tf.cast(valid_step_masks, tf.float32)
        stop_decisions = tf.cast(stop_decisions, tf.float32)
        non_stop_expansion = mask * (1.0 - stop_decisions)
        opportunity_penalty = self.opportunity_cost * non_stop_expansion

        returns = (step_expected_reward / reward_norm) - opportunity_penalty
        returns = tf.stop_gradient(returns)
        policy_loss = -log_probs * returns * mask

        entropy_beta = 0.01
        entropy_bonus = entropies * mask
        loss = (
            tf.reduce_sum(policy_loss) / (tf.reduce_sum(mask) + 1e-6)
            - entropy_beta * tf.reduce_sum(entropy_bonus) / (tf.reduce_sum(mask) + 1e-6)
        )
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

    def compute_final_actor_loss(self, inputs, action_probs):
        _, normalized_path_rewards = self._prepare_path_rewards(inputs)

        expected_reward = tf.reduce_sum(
            action_probs * normalized_path_rewards,
            axis=1,
            keepdims=True
        )

        mean_expected_reward = tf.reduce_mean(expected_reward)

        entropy = -tf.reduce_sum(
            action_probs * tf.math.log(action_probs + 1e-8),
            axis=1,
            keepdims=True
        )
        mean_entropy = tf.reduce_mean(entropy)
        if self.time_steps == 12:
            entropy_beta = 0.05
        else:       
            entropy_beta = 0

        if self.time_steps == 6:
            mean_expected_reward = mean_expected_reward/3.58
        elif self.time_steps == 2:
             mean_expected_reward = mean_expected_reward/0.75
        elif self.time_steps == 12:
             mean_expected_reward = mean_expected_reward/5.11
        else:
            mean_expected_reward = mean_expected_reward / (8.1574)
            # if self.tree_type == "deep":
            #     mean_expected_reward = mean_expected_reward / (8.1574)
            # else:
            #     mean_expected_reward = mean_expected_reward  / (6.4726)
        action_loss = (1.0 - mean_expected_reward) - entropy_beta * mean_entropy

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
        if self.time_steps == 6:
            norm = 3.58 # scalar
        elif self.time_steps == 2:
            norm = 0.75
        elif self.time_steps == 12:
            norm = 5.11
        else:
            norm=8.1574
            # if self.tree_type == "deep":
            #     norm=8.1574
            # else:
            #     norm = 6.4726
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
