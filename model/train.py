
"""
train.py
Handles data batch generation and the main training loop, including 
gradient calculation, backpropagation, and checkpointing.
"""

import os
import random
import numpy as np
import tensorflow as tf
import helper
import csv
import config
@tf.function
def train_step(
    model,
    optimizer,
    current_alpha,
    current_beta,
    current_critic_coef,
    current_expansion_epsilon,
    current_expansion_entropy_coef,
    current_forced_continue_epsilon,
    input_data,
    input_type_str,
    clip_value=10.0,
    ppo_epochs=3,
    return_target_rollouts=1
):
    """
    Executes one training step: forward pass, loss computation, and backpropagation.
    """
    if model.use_autoencoder:
        first_decoder_params = (
            model.decoder.trainable_variables + 
            model.lstm_cell.trainable_variables +
            model.encoder.trainable_variables + 
            [model.prior_mu] + 
            [model.prior_logvar]
        )
        second_decoder_params = model.reconstruction_head.trainable_variables
    else:
        first_decoder_params = model.lstm_cell.trainable_variables
        second_decoder_params = []
    action_head_params = []
    expansion_params = model.expansion_head.trainable_variables
    critic_params = model.critic_head.trainable_variables
    probe_params = model.lstm_reward_probe_head.trainable_variables
    return_target_rollouts = max(int(return_target_rollouts), 1)
    with tf.device('/GPU:0'):
        rollout_node_selections = []
        rollout_expansion_log_probs = []
        for _ in range(return_target_rollouts):
            rollout_outputs = model(
                input_data,
                training=True,
                current_alpha=current_alpha,
                current_beta=current_beta,
                current_critic_coef=current_critic_coef,
                expansion_epsilon=current_expansion_epsilon,
                expansion_entropy_coef=current_expansion_entropy_coef,
                forced_continue_epsilon=current_forced_continue_epsilon,
                compute_losses=False
            )
            rollout_node_selections.append(tf.stop_gradient(rollout_outputs[12]))
            rollout_expansion_log_probs.append(tf.stop_gradient(rollout_outputs[18]))

        with tf.GradientTape(persistent=True) as tape:
            time_steps = model.time_steps
            feature_dim = 1

            tf.print("DEBUG: input_type =", input_type_str)

            rollout_weight = 1.0 / float(return_target_rollouts)
            total_loss = tf.constant(0.0, dtype=tf.float32)
            first_decoder_loss = tf.constant(0.0, dtype=tf.float32)
            second_decoder_loss = tf.constant(0.0, dtype=tf.float32)
            action_head_loss = tf.constant(0.0, dtype=tf.float32)
            critic_loss = tf.constant(0.0, dtype=tf.float32)
            information_loss = tf.constant(0.0, dtype=tf.float32)
            action_loss = tf.constant(0.0, dtype=tf.float32)
            reconstruction_loss = tf.constant(0.0, dtype=tf.float32)
            information_cost = tf.constant(0.0, dtype=tf.float32)
            expansion_head_loss = tf.constant(0.0, dtype=tf.float32)
            expansion_policy_loss = tf.constant(0.0, dtype=tf.float32)
            expansion_stop_rate = tf.constant(0.0, dtype=tf.float32)
            expansion_continue_rate = tf.constant(0.0, dtype=tf.float32)
            opportunity_policy_loss = tf.constant(0.0, dtype=tf.float32)
            lstm_probe_loss = tf.constant(0.0, dtype=tf.float32)
            lstm_probe_accuracy = tf.constant(0.0, dtype=tf.float32)

            for rollout_idx in range(return_target_rollouts):
                # Unpack the forward pass exactly as defined in your model.
                (reconstructed, action, rollout_total_loss,
                 rollout_first_decoder_loss, rollout_second_decoder_loss,
                 rollout_action_head_loss, rollout_critic_loss,
                 rollout_information_loss, rollout_action_loss,
                 rollout_reconstruction_loss, rollout_information_cost,
                 all_z_means, *rollout_extra_outputs) = model(
                    input_data,
                    training=True,
                    current_alpha=current_alpha,
                    current_beta=current_beta,
                    current_critic_coef=current_critic_coef,
                    expansion_epsilon=current_expansion_epsilon,
                    expansion_entropy_coef=current_expansion_entropy_coef,
                    forced_continue_epsilon=current_forced_continue_epsilon,
                    forced_node_selections=rollout_node_selections[rollout_idx],
                    old_expansion_log_probs=rollout_expansion_log_probs[rollout_idx],
                    use_ppo_loss=True
                )

                total_loss += rollout_total_loss * rollout_weight
                first_decoder_loss += rollout_first_decoder_loss * rollout_weight
                second_decoder_loss += rollout_second_decoder_loss * rollout_weight
                action_head_loss += rollout_action_head_loss * rollout_weight
                critic_loss += rollout_critic_loss * rollout_weight
                information_loss += rollout_information_loss * rollout_weight
                action_loss += rollout_action_loss * rollout_weight
                reconstruction_loss += rollout_reconstruction_loss * rollout_weight
                information_cost += rollout_information_cost * rollout_weight
                expansion_head_loss += rollout_extra_outputs[5] * rollout_weight
                expansion_policy_loss += rollout_extra_outputs[7] * rollout_weight
                expansion_stop_rate += rollout_extra_outputs[8] * rollout_weight
                expansion_continue_rate += rollout_extra_outputs[9] * rollout_weight
                opportunity_policy_loss += rollout_extra_outputs[10] * rollout_weight
                lstm_probe_loss += rollout_extra_outputs[11] * rollout_weight
                lstm_probe_accuracy += rollout_extra_outputs[12] * rollout_weight

                if rollout_idx == 0:
                    # Keep the detailed reward-conditioned diagnostics on the
                    # first sampled trajectory so the log columns remain simple.
                    extra_outputs = rollout_extra_outputs
                    lstm_probe_acc_by_category = extra_outputs[13]
                    lstm_probe_loss_by_category = extra_outputs[14]
                    lstm_probe_count_by_category = extra_outputs[15]
                    continue_after_reward_sums = extra_outputs[20]
                    continue_after_reward_counts = extra_outputs[21]
                    critic_after_reward_sums = extra_outputs[22]
                    diagnostic_reward_counts = extra_outputs[23]
                    terminal_best_prob_pre_after_reward_sums = extra_outputs[24]
                    terminal_best_prob_post_after_reward_sums = extra_outputs[25]
                    return_target_after_reward_sums = extra_outputs[26]
                    advantage_after_reward_sums = extra_outputs[27]
             
            # --- THE FIX: Create the weighted tensors INSIDE the tape scope ---
            weighted_kl_for_logging = information_loss * current_beta
            weighted_rec_for_logging = reconstruction_loss * model.alpha
            weighted_exp_for_logging = expansion_policy_loss * model.lambda_
            weighted_opp_for_logging = opportunity_policy_loss * model.opportunity_cost
            weighted_critic_for_logging = (
                critic_loss * model.lambda_ * current_critic_coef
            )
            # ------------------------------------------------------------------
    
    # 1. Calculate Combined Gradients (for the actual optimizer update)
    first_decoder_gradients = tape.gradient(first_decoder_loss, first_decoder_params)
    second_decoder_gradients = tape.gradient(second_decoder_loss, second_decoder_params)
    action_head_gradients = tape.gradient(action_head_loss, action_head_params)
    expansion_gradients = tape.gradient(expansion_head_loss, expansion_params)
    critic_gradients = tape.gradient(critic_loss, critic_params)
    probe_gradients = tape.gradient(lstm_probe_loss, probe_params)
    
    # 2. Calculate Isolated Gradients (for logging only)
    kl_grads = tape.gradient(weighted_kl_for_logging, first_decoder_params)
    act_grads = tape.gradient(action_loss, first_decoder_params)
    rec_grads = tape.gradient(weighted_rec_for_logging, first_decoder_params)
    exp_grads = tape.gradient(weighted_exp_for_logging, first_decoder_params)
    opp_grads = tape.gradient(weighted_opp_for_logging, first_decoder_params)
    critic_backbone_grads = tape.gradient(weighted_critic_for_logging, first_decoder_params)
    exp_head_isolated_grads = tape.gradient(weighted_exp_for_logging, expansion_params)
    opp_head_isolated_grads = tape.gradient(weighted_opp_for_logging, expansion_params)
    act_head_isolated_grads = tape.gradient(action_loss, expansion_params)

    # --- Helper to slice and calculate global norms ---
    num_dec = len(model.decoder.trainable_variables)
    num_lstm = len(model.lstm_cell.trainable_variables)
    num_enc = len(model.encoder.trainable_variables)

    def global_norm_or_zero(grad_list):
        if grad_list is None:
            return tf.constant(0.0, dtype=tf.float32)
        filtered = [g for g in grad_list if g is not None]
        if not filtered:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.linalg.global_norm(filtered)

    def extract_norms(grad_list):
        if grad_list is None:
            # Fallback if a loss is totally disconnected
            return (
                tf.constant(0.0), tf.constant(0.0),
                tf.constant(0.0), tf.constant(0.0)
            )
        if not model.use_autoencoder:
            norm_lstm = global_norm_or_zero(grad_list)
            return tf.constant(0.0), norm_lstm, tf.constant(0.0), tf.constant(0.0)
            
        dec_g = grad_list[:num_dec]
        lstm_g = grad_list[num_dec : num_dec + num_lstm]
        enc_g = grad_list[num_dec + num_lstm : num_dec + num_lstm + num_enc]
        prior_g = grad_list[num_dec + num_lstm + num_enc :]

        norm_dec = global_norm_or_zero(dec_g)
        norm_lstm = global_norm_or_zero(lstm_g)
        norm_enc = global_norm_or_zero(enc_g)
        norm_prior = global_norm_or_zero(prior_g)
        
        return norm_enc, norm_lstm, norm_dec, norm_prior

    # Extract the 9 specific norms
    kl_norm_enc, kl_norm_lstm, kl_norm_dec, kl_norm_prior = extract_norms(kl_grads)
    act_norm_enc, act_norm_lstm, act_norm_dec, act_norm_prior = extract_norms(act_grads)
    rec_norm_enc, rec_norm_lstm, rec_norm_dec, rec_norm_prior = extract_norms(rec_grads)
    exp_norm_enc, exp_norm_lstm, exp_norm_dec, exp_norm_prior = extract_norms(exp_grads)
    opp_norm_enc, opp_norm_lstm, opp_norm_dec, opp_norm_prior = extract_norms(opp_grads)
    critic_norm_enc, critic_norm_lstm, critic_norm_dec, critic_norm_prior = extract_norms(
        critic_backbone_grads
    )
    update_norm_enc, update_norm_lstm, update_norm_dec, update_norm_prior = extract_norms(
        first_decoder_gradients
    )
    rec_head_norm = global_norm_or_zero(second_decoder_gradients)
    action_head_norm = global_norm_or_zero(act_head_isolated_grads)
    expansion_head_norm = global_norm_or_zero(expansion_gradients)
    critic_head_norm = global_norm_or_zero(critic_gradients)
    probe_head_norm = global_norm_or_zero(probe_gradients)
    exp_head_isolated_norm = global_norm_or_zero(exp_head_isolated_grads)
    opp_head_isolated_norm = global_norm_or_zero(opp_head_isolated_grads)
    # -----------------------------------------------------------------

    # --- COMBINED OPTIMIZER UPDATE: Ensures iterations match total_steps ---
    all_gradients = (
        first_decoder_gradients +
        second_decoder_gradients +
        action_head_gradients +
        expansion_gradients +
        critic_gradients +
        probe_gradients
    )
    all_params = (
        first_decoder_params + second_decoder_params + action_head_params +
        expansion_params + critic_params + probe_params
    )

    def sanitize_gradient(grad):
        if grad is None:
            return None
        if isinstance(grad, tf.IndexedSlices):
            safe_values = tf.where(
                tf.math.is_finite(grad.values),
                grad.values,
                tf.zeros_like(grad.values)
            )
            return tf.IndexedSlices(safe_values, grad.indices, grad.dense_shape)
        return tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))

    def prepare_grads_and_vars(gradients, params):
        pairs = [
            (sanitize_gradient(grad), var)
            for grad, var in zip(gradients, params)
            if grad is not None
        ]
        if not pairs:
            return []
        safe_grads, safe_vars = zip(*pairs)
        clipped_grads, _ = tf.clip_by_global_norm(list(safe_grads), clip_value)
        return list(zip(clipped_grads, safe_vars))

    grads_and_vars = prepare_grads_and_vars(all_gradients, all_params)
    optimizer.apply_gradients(grads_and_vars)
    
    # Must explicitly delete a persistent tape when done
    del tape

    for _ in range(max(ppo_epochs - 1, 0)):
        with tf.GradientTape() as ppo_tape:
            ppo_expansion_loss = tf.constant(0.0, dtype=tf.float32)
            rollout_weight = 1.0 / float(return_target_rollouts)
            for rollout_idx in range(return_target_rollouts):
                (reconstructed, action, total_loss,
                 first_decoder_loss, second_decoder_loss,
                 action_head_loss, critic_loss, information_loss, action_loss, reconstruction_loss,
                 information_cost, all_z_means, *extra_outputs) = model(
                    input_data,
                    training=True,
                    current_alpha=current_alpha,
                    current_beta=current_beta,
                    current_critic_coef=current_critic_coef,
                    expansion_epsilon=current_expansion_epsilon,
                    expansion_entropy_coef=current_expansion_entropy_coef,
                    forced_continue_epsilon=current_forced_continue_epsilon,
                    forced_node_selections=rollout_node_selections[rollout_idx],
                    old_expansion_log_probs=rollout_expansion_log_probs[rollout_idx],
                    use_ppo_loss=True
                )
                ppo_expansion_loss += extra_outputs[5] * rollout_weight

        ppo_expansion_gradients = ppo_tape.gradient(ppo_expansion_loss, expansion_params)
        ppo_grads_and_vars = prepare_grads_and_vars(
            ppo_expansion_gradients,
            expansion_params
        )
        optimizer.apply_gradients(ppo_grads_and_vars)
    
    # Return all 13 values
    return (
            total_loss, information_loss, action_loss, reconstruction_loss,
            expansion_policy_loss, expansion_stop_rate, expansion_continue_rate,
            kl_norm_enc, kl_norm_lstm, kl_norm_dec,
            act_norm_enc, act_norm_lstm, act_norm_dec,
            rec_norm_enc, rec_norm_lstm, rec_norm_dec,
            kl_norm_prior, act_norm_prior, rec_norm_prior,
            exp_norm_enc, exp_norm_lstm, exp_norm_dec, exp_norm_prior,
            opp_norm_enc, opp_norm_lstm, opp_norm_dec, opp_norm_prior,
            critic_norm_enc, critic_norm_lstm, critic_norm_dec, critic_norm_prior,
            rec_head_norm, action_head_norm, expansion_head_norm, critic_head_norm,
            probe_head_norm,
            exp_head_isolated_norm, opp_head_isolated_norm,
            update_norm_enc, update_norm_lstm, update_norm_dec, update_norm_prior,
            lstm_probe_loss, lstm_probe_accuracy, lstm_probe_acc_by_category,
            lstm_probe_loss_by_category, lstm_probe_count_by_category,
            continue_after_reward_sums, continue_after_reward_counts,
            critic_after_reward_sums, diagnostic_reward_counts,
            terminal_best_prob_pre_after_reward_sums,
            terminal_best_prob_post_after_reward_sums,
            return_target_after_reward_sums,
            advantage_after_reward_sums
    )

import os
import csv
import tensorflow as tf
import helper

def train_model(model, epochs, trials_per_epoch, batch_size, time_steps, input_type, dir_name, model_name):
    """
    The main training loop with early stopping, warmup scheduling, and checkpointing.
    """
   

    
    ppo_epochs = 3
    return_target_rollouts = max(
        int(os.environ.get("RETURN_TARGET_ROLLOUTS", "4")),
        1
    )

    # --- REVERTED: Standard Cosine Decay with Warmup ---
    total_steps = epochs * trials_per_epoch * ppo_epochs
    lr_warmup_epochs = 0
    warmup_steps = lr_warmup_epochs * trials_per_epoch

    # Let's keep a moderate warmup so the network doesn't panic
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,    
        decay_steps=total_steps,       
        alpha=0.01,                    
        warmup_target=0.0003,           
        warmup_steps=warmup_steps      
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=20.0)

    # --- NEW: Cosine Decay with Warm Restarts ---
    
    # We want to trigger a "restart" kick every 40 epochs.
    # epochs_per_restart = 40
    # first_decay_steps = epochs_per_restart * trials_per_epoch

    # lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #     initial_learning_rate=0.0003,  # Start right at the peak (skip the warmup)
    #     first_decay_steps=first_decay_steps, 
    #     t_mul=1.0,                     # 1.0 = Restart exactly every 40 epochs. (2.0 would double the time between restarts)
    #     m_mul=0.8,                     # 0.8 = Each time we restart, the peak LR is 20% lower than the last one
    #     alpha=0.01                     # The LR decays down to 1% of the peak before restarting
    # )
    # optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5, clipnorm=20.0)
        
    # --- BUILD OPTIMIZER VARIABLES ---
    # We pass a dummy input to build the model's graph and variables
    dummy_input = tf.zeros((1, time_steps, 1), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    
    all_trainables = (
        model.encoder.trainable_variables +
        model.decoder.trainable_variables +
        [model.prior_mu] + [model.prior_logvar] +
        model.lstm_cell.trainable_variables +
        model.reconstruction_head.trainable_variables +
        model.expansion_head.trainable_variables +
        model.critic_head.trainable_variables +
        model.lstm_reward_probe_head.trainable_variables
    )

    base_opt = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    base_opt.build(all_trainables)

    # --- CONVERGENCE / STOPPING VARIABLES ---
    best_loss = float('inf')
    wait = 0
    patience = 120           
    min_delta = 1e-10
    warmup_epochs = 80
    epochs_count = 0
    
    # Path for the "Best" model (separate from the final one)
    best_checkpoint_path = dir_name + model_name + '_BEST.weights.h5'

    # ------------------------------------------------------------------
    # TRAINING HISTORY DICTIONARY
    # ------------------------------------------------------------------
    probe_reward_values = [4, 3, 2, 1, 0, -1, -2, -3, -4]

    def probe_reward_label(value):
        if value > 0:
            return f"p{value}"
        if value < 0:
            return f"m{abs(value)}"
        return "z0"

    history = {
        'epoch': [], 'learning_rate': [], 'expansion_epsilon': [],
        'return_target_rollouts': [],
        'forced_continue_epsilon': [],
        'expansion_entropy_coef': [],
        'total_loss': [], 'kl_loss': [], 'action_loss': [], 'reconstruction_loss': [],
        'expansion_loss': [], 'expansion_stop_rate': [], 'expansion_continue_rate': [],
        'lstm_probe_loss': [], 'lstm_probe_accuracy': [],
        'kl_grad_norm_enc': [], 'kl_grad_norm_lstm': [], 'kl_grad_norm_dec': [],
        'act_grad_norm_enc': [], 'act_grad_norm_lstm': [], 'act_grad_norm_dec': [],
        'rec_grad_norm_enc': [], 'rec_grad_norm_lstm': [], 'rec_grad_norm_dec': [],
        'kl_grad_norm_prior': [], 'act_grad_norm_prior': [], 'rec_grad_norm_prior': [],
        'exp_grad_norm_enc': [], 'exp_grad_norm_lstm': [], 'exp_grad_norm_dec': [],
        'exp_grad_norm_prior': [],
        'opp_grad_norm_enc': [], 'opp_grad_norm_lstm': [], 'opp_grad_norm_dec': [],
        'opp_grad_norm_prior': [],
        'critic_grad_norm_enc': [], 'critic_grad_norm_lstm': [], 'critic_grad_norm_dec': [],
        'critic_grad_norm_prior': [],
        'rec_grad_norm_head': [], 'act_grad_norm_head': [],
        'exp_grad_norm_head': [], 'critic_grad_norm_head': [],
        'lstm_probe_grad_norm_head': [],
        'exp_policy_grad_norm_head': [], 'opp_grad_norm_head': [],
        'update_grad_norm_enc': [], 'update_grad_norm_lstm': [],
        'update_grad_norm_dec': [], 'update_grad_norm_prior': []
    }
    for reward_value in probe_reward_values:
        label = probe_reward_label(reward_value)
        history[f'lstm_probe_acc_reward_{label}'] = []
        history[f'lstm_probe_loss_reward_{label}'] = []
        history[f'lstm_probe_n_reward_{label}'] = []
    for decision_step in range(2, time_steps + 1):
        for reward_value in probe_reward_values:
            label = probe_reward_label(reward_value)
            history[f'exp_continue_t{decision_step}_after_reward_{label}'] = []
            history[f'exp_continue_n_t{decision_step}_after_reward_{label}'] = []
            history[f'exp_critic_t{decision_step}_after_reward_{label}'] = []
            history[f'exp_terminal_best_pre_t{decision_step}_after_reward_{label}'] = []
            history[f'exp_terminal_best_post_t{decision_step}_after_reward_{label}'] = []
            history[f'exp_return_target_t{decision_step}_after_reward_{label}'] = []
            history[f'exp_advantage_t{decision_step}_after_reward_{label}'] = []
    kl_warmup_epochs = 0   # Keep beta at 0.0 while learning rate warms up
    kl_annealing_epochs = 0 # How many epochs it takes to go from 0.0 to target_beta
    target_beta =1/model.beta
    critic_warmup_epochs = 80
    critic_annealing_epochs =120
    if model.time_steps  == 6:
        target_critic_coef = 1
    elif model.time_steps == 2:
        target_critic_coef = 0.1
    elif model.time_steps == 30:
        target_critic_coef =0
    else:
        target_critic_coef =0.1
    if model.time_steps > 2:
        expansion_epsilon_start = 0.0
        expansion_epsilon_end = 0.0
        expansion_epsilon_annealing_epochs = 0
        expansion_entropy_start = 2.0
        expansion_entropy_end = 0.0
        expansion_entropy_annealing_epochs = 50
        forced_continue_start = 0.0
        forced_continue_end = 0.0
        forced_continue_annealing_epochs = 0
    else:
        expansion_epsilon_start = 0.0
        expansion_epsilon_end = 0.0
        expansion_epsilon_annealing_epochs = 0
        expansion_entropy_start = 1.0
        expansion_entropy_end = 0.0
        expansion_entropy_annealing_epochs = 50
        forced_continue_start = 0.0
        forced_continue_end = 0.0
        forced_continue_annealing_epochs = 0
    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        # Accumulators for this epoch
        ep_total_loss, ep_kl, ep_act, ep_rec = 0.0, 0.0, 0.0, 0.0
        ep_expansion_loss, ep_stop_rate, ep_continue_rate = 0.0, 0.0, 0.0
        ep_lstm_probe_loss, ep_lstm_probe_acc = 0.0, 0.0
        ep_lstm_probe_acc_weighted = tf.zeros([model.num_categories], dtype=tf.float32)
        ep_lstm_probe_loss_weighted = tf.zeros([model.num_categories], dtype=tf.float32)
        ep_lstm_probe_counts = tf.zeros([model.num_categories], dtype=tf.float32)
        ep_continue_after_reward_sums = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_continue_after_reward_counts = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_critic_after_reward_sums = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_diagnostic_reward_counts = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_terminal_best_pre_after_reward_sums = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_terminal_best_post_after_reward_sums = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_return_target_after_reward_sums = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        ep_advantage_after_reward_sums = tf.zeros(
            [model.time_steps, model.num_categories],
            dtype=tf.float32
        )
        
        # Gradient accumulators
        ep_kl_gn_enc, ep_kl_gn_lstm, ep_kl_gn_dec = 0.0, 0.0, 0.0
        ep_act_gn_enc, ep_act_gn_lstm, ep_act_gn_dec = 0.0, 0.0, 0.0
        ep_rec_gn_enc, ep_rec_gn_lstm, ep_rec_gn_dec = 0.0, 0.0, 0.0
        ep_kl_gn_prior, ep_act_gn_prior, ep_rec_gn_prior = 0.0, 0.0, 0.0
        ep_exp_gn_enc, ep_exp_gn_lstm, ep_exp_gn_dec, ep_exp_gn_prior = 0.0, 0.0, 0.0, 0.0
        ep_opp_gn_enc, ep_opp_gn_lstm, ep_opp_gn_dec, ep_opp_gn_prior = 0.0, 0.0, 0.0, 0.0
        ep_critic_gn_enc, ep_critic_gn_lstm, ep_critic_gn_dec, ep_critic_gn_prior = 0.0, 0.0, 0.0, 0.0
        ep_rec_gn_head, ep_act_gn_head = 0.0, 0.0
        ep_exp_gn_head, ep_critic_gn_head, ep_lstm_probe_gn_head = 0.0, 0.0, 0.0
        ep_exp_policy_gn_head, ep_opp_gn_head = 0.0, 0.0
        ep_update_gn_enc, ep_update_gn_lstm, ep_update_gn_dec, ep_update_gn_prior = 0.0, 0.0, 0.0, 0.0
        
        epochs_count += 1 
        current_alpha = 1.0 
 
        
        # --- NEW: KL Annealing Math ---
        if epoch < kl_warmup_epochs:
            current_beta = 0.0
        elif epoch >= (kl_warmup_epochs + kl_annealing_epochs):
            current_beta = target_beta
        else:
            # Calculate how far along we are in the annealing phase (0.0 to 1.0)
            progress = (epoch - kl_warmup_epochs) / kl_annealing_epochs
            current_beta = target_beta * progress

        if epoch < critic_warmup_epochs:
            current_critic_coef = target_critic_coef
        elif epoch >= (critic_warmup_epochs + critic_annealing_epochs):
            current_critic_coef = 0.0
        else:
            progress = (epoch - critic_warmup_epochs) / critic_annealing_epochs
            current_critic_coef = target_critic_coef * (1.0 - progress)
        # current_critic_coef = target_critic_coef
        if expansion_epsilon_annealing_epochs <= 0:
            current_expansion_epsilon = expansion_epsilon_end
        else:
            epsilon_progress = min(epoch / expansion_epsilon_annealing_epochs, 1.0)
            current_expansion_epsilon = (
                expansion_epsilon_start
                + (expansion_epsilon_end - expansion_epsilon_start) * epsilon_progress
            )
        if expansion_entropy_annealing_epochs <= 0:
            current_expansion_entropy_coef = expansion_entropy_end
        else:
            entropy_progress = min(epoch / expansion_entropy_annealing_epochs, 1.0)
            current_expansion_entropy_coef = (
                expansion_entropy_start
                + (expansion_entropy_end - expansion_entropy_start) * entropy_progress
            )
        if forced_continue_annealing_epochs <= 0:
            current_forced_continue_epsilon = forced_continue_end
        else:
            forced_continue_progress = min(epoch / forced_continue_annealing_epochs, 1.0)
            current_forced_continue_epsilon = (
                forced_continue_start
                + (forced_continue_end - forced_continue_start) * forced_continue_progress
            )
        for i in range(trials_per_epoch):
            batch_input_data = helper.generate_batch_data(batch_size, time_steps, input_type)

            # Unpack metrics from train_step
            (loss, kl, act, rec, 
             exp_loss, stop_rate, continue_rate,
             kl_gn_enc, kl_gn_lstm, kl_gn_dec,
             act_gn_enc, act_gn_lstm, act_gn_dec,
             rec_gn_enc, rec_gn_lstm, rec_gn_dec,
             kl_gn_prior, act_gn_prior, rec_gn_prior,
             exp_gn_enc, exp_gn_lstm, exp_gn_dec, exp_gn_prior,
             opp_gn_enc, opp_gn_lstm, opp_gn_dec, opp_gn_prior,
             critic_gn_enc, critic_gn_lstm, critic_gn_dec, critic_gn_prior,
             rec_gn_head, act_gn_head, exp_gn_head, critic_gn_head,
             lstm_probe_gn_head,
             exp_policy_gn_head, opp_gn_head,
             update_gn_enc, update_gn_lstm, update_gn_dec, update_gn_prior,
             lstm_probe_loss, lstm_probe_acc, lstm_probe_acc_by_category,
             lstm_probe_loss_by_category, lstm_probe_count_by_category,
             continue_after_reward_sums, continue_after_reward_counts,
             critic_after_reward_sums, diagnostic_reward_counts,
             terminal_best_prob_pre_after_reward_sums,
             terminal_best_prob_post_after_reward_sums,
             return_target_after_reward_sums,
             advantage_after_reward_sums) = train_step(
                model=model, 
                optimizer=optimizer,
                current_alpha=tf.constant(current_alpha, dtype=tf.float32),
                current_beta=tf.constant(current_beta, dtype=tf.float32),
                current_critic_coef=tf.constant(current_critic_coef, dtype=tf.float32),
                current_expansion_epsilon=tf.constant(current_expansion_epsilon, dtype=tf.float32),
                current_expansion_entropy_coef=tf.constant(current_expansion_entropy_coef, dtype=tf.float32),
                current_forced_continue_epsilon=tf.constant(current_forced_continue_epsilon, dtype=tf.float32),
                input_data=batch_input_data,
                input_type_str=input_type,
                ppo_epochs=ppo_epochs,
                return_target_rollouts=return_target_rollouts
            )
            
            # Accumulate metrics
            ep_total_loss += loss
            ep_kl += kl
            ep_act += act
            ep_rec += rec
            ep_expansion_loss += exp_loss
            ep_stop_rate += stop_rate
            ep_continue_rate += continue_rate
            ep_lstm_probe_loss += lstm_probe_loss
            ep_lstm_probe_acc += lstm_probe_acc
            finite_probe_acc = tf.where(
                tf.math.is_finite(lstm_probe_acc_by_category),
                lstm_probe_acc_by_category,
                tf.zeros_like(lstm_probe_acc_by_category)
            )
            finite_probe_loss = tf.where(
                tf.math.is_finite(lstm_probe_loss_by_category),
                lstm_probe_loss_by_category,
                tf.zeros_like(lstm_probe_loss_by_category)
            )
            ep_lstm_probe_acc_weighted += finite_probe_acc * lstm_probe_count_by_category
            ep_lstm_probe_loss_weighted += finite_probe_loss * lstm_probe_count_by_category
            ep_lstm_probe_counts += lstm_probe_count_by_category
            ep_continue_after_reward_sums += continue_after_reward_sums
            ep_continue_after_reward_counts += continue_after_reward_counts
            ep_critic_after_reward_sums += critic_after_reward_sums
            ep_diagnostic_reward_counts += diagnostic_reward_counts
            ep_terminal_best_pre_after_reward_sums += terminal_best_prob_pre_after_reward_sums
            ep_terminal_best_post_after_reward_sums += terminal_best_prob_post_after_reward_sums
            ep_return_target_after_reward_sums += return_target_after_reward_sums
            ep_advantage_after_reward_sums += advantage_after_reward_sums
            
            ep_kl_gn_enc += kl_gn_enc
            ep_kl_gn_lstm += kl_gn_lstm
            ep_kl_gn_dec += kl_gn_dec
            
            ep_act_gn_enc += act_gn_enc
            ep_act_gn_lstm += act_gn_lstm
            ep_act_gn_dec += act_gn_dec
            
            ep_rec_gn_enc += rec_gn_enc
            ep_rec_gn_lstm += rec_gn_lstm
            ep_rec_gn_dec += rec_gn_dec

            ep_kl_gn_prior += kl_gn_prior
            ep_act_gn_prior += act_gn_prior
            ep_rec_gn_prior += rec_gn_prior

            ep_exp_gn_enc += exp_gn_enc
            ep_exp_gn_lstm += exp_gn_lstm
            ep_exp_gn_dec += exp_gn_dec
            ep_exp_gn_prior += exp_gn_prior

            ep_opp_gn_enc += opp_gn_enc
            ep_opp_gn_lstm += opp_gn_lstm
            ep_opp_gn_dec += opp_gn_dec
            ep_opp_gn_prior += opp_gn_prior

            ep_critic_gn_enc += critic_gn_enc
            ep_critic_gn_lstm += critic_gn_lstm
            ep_critic_gn_dec += critic_gn_dec
            ep_critic_gn_prior += critic_gn_prior

            ep_rec_gn_head += rec_gn_head
            ep_act_gn_head += act_gn_head
            ep_exp_gn_head += exp_gn_head
            ep_critic_gn_head += critic_gn_head
            ep_lstm_probe_gn_head += lstm_probe_gn_head
            ep_exp_policy_gn_head += exp_policy_gn_head
            ep_opp_gn_head += opp_gn_head

            ep_update_gn_enc += update_gn_enc
            ep_update_gn_lstm += update_gn_lstm
            ep_update_gn_dec += update_gn_dec
            ep_update_gn_prior += update_gn_prior
        
        # Calculate averages for the epoch
        avg_total_loss = ep_total_loss / trials_per_epoch
        current_lr = lr_schedule(optimizer.iterations).numpy()
        # Append to history
        history['epoch'].append(epoch + 1)
        history['learning_rate'].append(current_lr)
        history['expansion_epsilon'].append(current_expansion_epsilon)
        history['return_target_rollouts'].append(return_target_rollouts)
        history['forced_continue_epsilon'].append(current_forced_continue_epsilon)
        history['expansion_entropy_coef'].append(current_expansion_entropy_coef)
        history['total_loss'].append((ep_total_loss / trials_per_epoch).numpy())
        history['kl_loss'].append((ep_kl / trials_per_epoch).numpy())
        history['action_loss'].append((ep_act / trials_per_epoch).numpy())
        history['reconstruction_loss'].append((ep_rec / trials_per_epoch).numpy())
        history['expansion_loss'].append((ep_expansion_loss / trials_per_epoch).numpy())
        history['expansion_stop_rate'].append((ep_stop_rate / trials_per_epoch).numpy())
        history['expansion_continue_rate'].append((ep_continue_rate / trials_per_epoch).numpy())
        history['lstm_probe_loss'].append((ep_lstm_probe_loss / trials_per_epoch).numpy())
        history['lstm_probe_accuracy'].append((ep_lstm_probe_acc / trials_per_epoch).numpy())
        
        history['kl_grad_norm_enc'].append((ep_kl_gn_enc / trials_per_epoch).numpy())
        history['kl_grad_norm_lstm'].append((ep_kl_gn_lstm / trials_per_epoch).numpy())
        history['kl_grad_norm_dec'].append((ep_kl_gn_dec / trials_per_epoch).numpy())
        
        history['act_grad_norm_enc'].append((ep_act_gn_enc / trials_per_epoch).numpy())
        history['act_grad_norm_lstm'].append((ep_act_gn_lstm / trials_per_epoch).numpy())
        history['act_grad_norm_dec'].append((ep_act_gn_dec / trials_per_epoch).numpy())
        
        history['rec_grad_norm_enc'].append((ep_rec_gn_enc / trials_per_epoch).numpy())
        history['rec_grad_norm_lstm'].append((ep_rec_gn_lstm / trials_per_epoch).numpy())
        history['rec_grad_norm_dec'].append((ep_rec_gn_dec / trials_per_epoch).numpy())

        history['kl_grad_norm_prior'].append((ep_kl_gn_prior / trials_per_epoch).numpy())
        history['act_grad_norm_prior'].append((ep_act_gn_prior / trials_per_epoch).numpy())
        history['rec_grad_norm_prior'].append((ep_rec_gn_prior / trials_per_epoch).numpy())

        history['exp_grad_norm_enc'].append((ep_exp_gn_enc / trials_per_epoch).numpy())
        history['exp_grad_norm_lstm'].append((ep_exp_gn_lstm / trials_per_epoch).numpy())
        history['exp_grad_norm_dec'].append((ep_exp_gn_dec / trials_per_epoch).numpy())
        history['exp_grad_norm_prior'].append((ep_exp_gn_prior / trials_per_epoch).numpy())

        history['opp_grad_norm_enc'].append((ep_opp_gn_enc / trials_per_epoch).numpy())
        history['opp_grad_norm_lstm'].append((ep_opp_gn_lstm / trials_per_epoch).numpy())
        history['opp_grad_norm_dec'].append((ep_opp_gn_dec / trials_per_epoch).numpy())
        history['opp_grad_norm_prior'].append((ep_opp_gn_prior / trials_per_epoch).numpy())

        history['critic_grad_norm_enc'].append((ep_critic_gn_enc / trials_per_epoch).numpy())
        history['critic_grad_norm_lstm'].append((ep_critic_gn_lstm / trials_per_epoch).numpy())
        history['critic_grad_norm_dec'].append((ep_critic_gn_dec / trials_per_epoch).numpy())
        history['critic_grad_norm_prior'].append((ep_critic_gn_prior / trials_per_epoch).numpy())

        history['rec_grad_norm_head'].append((ep_rec_gn_head / trials_per_epoch).numpy())
        history['act_grad_norm_head'].append((ep_act_gn_head / trials_per_epoch).numpy())
        history['exp_grad_norm_head'].append((ep_exp_gn_head / trials_per_epoch).numpy())
        history['critic_grad_norm_head'].append((ep_critic_gn_head / trials_per_epoch).numpy())
        history['lstm_probe_grad_norm_head'].append((ep_lstm_probe_gn_head / trials_per_epoch).numpy())
        history['exp_policy_grad_norm_head'].append((ep_exp_policy_gn_head / trials_per_epoch).numpy())
        history['opp_grad_norm_head'].append((ep_opp_gn_head / trials_per_epoch).numpy())

        history['update_grad_norm_enc'].append((ep_update_gn_enc / trials_per_epoch).numpy())
        history['update_grad_norm_lstm'].append((ep_update_gn_lstm / trials_per_epoch).numpy())
        history['update_grad_norm_dec'].append((ep_update_gn_dec / trials_per_epoch).numpy())
        history['update_grad_norm_prior'].append((ep_update_gn_prior / trials_per_epoch).numpy())

        probe_acc_epoch = tf.where(
            ep_lstm_probe_counts > 0.0,
            ep_lstm_probe_acc_weighted / (ep_lstm_probe_counts + 1e-6),
            tf.fill([model.num_categories], tf.constant(float("nan"), dtype=tf.float32))
        )
        probe_loss_epoch = tf.where(
            ep_lstm_probe_counts > 0.0,
            ep_lstm_probe_loss_weighted / (ep_lstm_probe_counts + 1e-6),
            tf.fill([model.num_categories], tf.constant(float("nan"), dtype=tf.float32))
        )
        for idx, reward_value in enumerate(probe_reward_values):
            label = probe_reward_label(reward_value)
            history[f'lstm_probe_acc_reward_{label}'].append(probe_acc_epoch[idx].numpy())
            history[f'lstm_probe_loss_reward_{label}'].append(probe_loss_epoch[idx].numpy())
            history[f'lstm_probe_n_reward_{label}'].append(ep_lstm_probe_counts[idx].numpy())

        continue_after_reward_epoch = tf.where(
            ep_continue_after_reward_counts > 0.0,
            ep_continue_after_reward_sums / (ep_continue_after_reward_counts + 1e-6),
            tf.fill(
                [model.time_steps, model.num_categories],
                tf.constant(float("nan"), dtype=tf.float32)
            )
        )
        critic_after_reward_epoch = tf.where(
            ep_diagnostic_reward_counts > 0.0,
            ep_critic_after_reward_sums / (ep_diagnostic_reward_counts + 1e-6),
            tf.fill(
                [model.time_steps, model.num_categories],
                tf.constant(float("nan"), dtype=tf.float32)
            )
        )
        terminal_best_pre_after_reward_epoch = tf.where(
            ep_diagnostic_reward_counts > 0.0,
            ep_terminal_best_pre_after_reward_sums / (ep_diagnostic_reward_counts + 1e-6),
            tf.fill(
                [model.time_steps, model.num_categories],
                tf.constant(float("nan"), dtype=tf.float32)
            )
        )
        terminal_best_post_after_reward_epoch = tf.where(
            ep_diagnostic_reward_counts > 0.0,
            ep_terminal_best_post_after_reward_sums / (ep_diagnostic_reward_counts + 1e-6),
            tf.fill(
                [model.time_steps, model.num_categories],
                tf.constant(float("nan"), dtype=tf.float32)
            )
        )
        return_target_after_reward_epoch = tf.where(
            ep_diagnostic_reward_counts > 0.0,
            ep_return_target_after_reward_sums / (ep_diagnostic_reward_counts + 1e-6),
            tf.fill(
                [model.time_steps, model.num_categories],
                tf.constant(float("nan"), dtype=tf.float32)
            )
        )
        advantage_after_reward_epoch = tf.where(
            ep_diagnostic_reward_counts > 0.0,
            ep_advantage_after_reward_sums / (ep_diagnostic_reward_counts + 1e-6),
            tf.fill(
                [model.time_steps, model.num_categories],
                tf.constant(float("nan"), dtype=tf.float32)
            )
        )
        for decision_step in range(2, time_steps + 1):
            step_idx = decision_step - 1
            for idx, reward_value in enumerate(probe_reward_values):
                label = probe_reward_label(reward_value)
                history[f'exp_continue_t{decision_step}_after_reward_{label}'].append(
                    continue_after_reward_epoch[step_idx, idx].numpy()
                )
                history[f'exp_continue_n_t{decision_step}_after_reward_{label}'].append(
                    ep_continue_after_reward_counts[step_idx, idx].numpy()
                )
                history[f'exp_critic_t{decision_step}_after_reward_{label}'].append(
                    critic_after_reward_epoch[step_idx, idx].numpy()
                )
                history[f'exp_terminal_best_pre_t{decision_step}_after_reward_{label}'].append(
                    terminal_best_pre_after_reward_epoch[step_idx, idx].numpy()
                )
                history[f'exp_terminal_best_post_t{decision_step}_after_reward_{label}'].append(
                    terminal_best_post_after_reward_epoch[step_idx, idx].numpy()
                )
                history[f'exp_return_target_t{decision_step}_after_reward_{label}'].append(
                    return_target_after_reward_epoch[step_idx, idx].numpy()
                )
                history[f'exp_advantage_t{decision_step}_after_reward_{label}'].append(
                    advantage_after_reward_epoch[step_idx, idx].numpy()
                )

        t3_reward3_msg = ""
        if time_steps >= 3:
            t3_p3 = history['exp_continue_t3_after_reward_p3'][-1]
            t3_p3_n = history['exp_continue_n_t3_after_reward_p3'][-1]
            t3_reward3_msg = f" | Continue t3 after +3 = {t3_p3:.4f} (n={t3_p3_n:.0f})"

        tf.print(
            f"Epoch {epoch+1}/{epochs}: Loss = {avg_total_loss:.4f} | "
            f"KL = {history['kl_loss'][-1]:.4f} | "
            f"LSTM probe = {history['lstm_probe_accuracy'][-1]:.4f} | "
            f"Expansion epsilon = {current_expansion_epsilon:.4f} | "
            f"Forced continue = {current_forced_continue_epsilon:.4f} | "
            f"Entropy coef = {current_expansion_entropy_coef:.4f} | "
            f"Stop = {history['expansion_stop_rate'][-1]:.4f} | "
            f"Continue = {history['expansion_continue_rate'][-1]:.4f}"
            f"{t3_reward3_msg}"
        )
     
        # --- CHECKPOINTING LOGIC ---
        # We check improvement only after warmup to avoid saving "cheating" models
        if epoch >= warmup_epochs:
            if avg_total_loss < (best_loss - min_delta):
                best_loss = avg_total_loss
                wait = 0
                
                # SAVE THE BEST WEIGHTS IMMEDIATELY
                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                model.save_weights(best_checkpoint_path)
                print(f"   >>> New Best Loss: {best_loss:.4f}. Saved checkpoint.")
                
            else:
                wait += 1
                if wait >= patience:
                    print(f"\n🛑 CONVERGENCE REACHED: Loss stopped improving for {patience} epochs.")
                    print(f"   Best Loss was: {best_loss:.4f} (Reloading these weights now...)")
                    
                    # RESTORE BEST WEIGHTS
                    try:
                        model.load_weights(best_checkpoint_path)
                        print("   ✅ Successfully reloaded best weights.")
                    except:
                        print("   ⚠️ Warning: Could not reload best weights. Using final weights.")
                    break

    log_file_path = os.path.join(dir_name, f"{model_name}_training_logs.csv")
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for i in range(len(history['epoch'])):
            writer.writerow([history[key][i] for key in history.keys()])
    print(f"📝 Training logs saved to: {log_file_path}")
    # ------------------------------------------------------------------
    # FINAL SAVE
    # ------------------------------------------------------------------
    # if os.path.exists(best_checkpoint_path) and epoch == epochs - 1:
    #      model.load_weights(best_checkpoint_path)

    print(f"\n✅ Saving Final (Best) Model state")
    final_checkpoint_path = dir_name + model_name + '.weights.h5'
    if os.path.exists(final_checkpoint_path):
        os.remove(final_checkpoint_path)
    model.save_weights(final_checkpoint_path)
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    
    # Cleanup: Remove the temporary "BEST" file
    if os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
