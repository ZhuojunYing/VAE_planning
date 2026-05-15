
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
    input_data,
    input_type_str,
    clip_value=10.0,
    ppo_epochs=3
):
    """
    Executes one training step: forward pass, loss computation, and backpropagation.
    """
    first_decoder_params = (
        model.decoder.trainable_variables + 
        model.lstm_cell.trainable_variables +
        model.encoder.trainable_variables + 
        [model.prior_mu] + 
        [model.prior_logvar]
    )
    second_decoder_params = model.reconstruction_head.trainable_variables
    action_head_params = model.action_head.trainable_variables
    expansion_params = model.expansion_head.trainable_variables
    critic_params = model.critic_head.trainable_variables
    with tf.device('/GPU:0'):
        rollout_outputs = model(
            input_data,
            training=True,
            current_alpha=current_alpha,
            current_beta=current_beta,
            current_critic_coef=current_critic_coef,
            expansion_epsilon=current_expansion_epsilon,
            compute_losses=False
        )
        old_node_selections = tf.stop_gradient(rollout_outputs[12])
        old_expansion_log_probs = tf.stop_gradient(rollout_outputs[18])

        with tf.GradientTape(persistent=True) as tape:
            time_steps = model.time_steps
            feature_dim = 1
            
            tf.print("DEBUG: input_type =", input_type_str)
           
            # Unpack the forward pass exactly as defined in your model
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
                forced_node_selections=old_node_selections,
                old_expansion_log_probs=old_expansion_log_probs,
                use_ppo_loss=True
            )
            expansion_head_loss = extra_outputs[5]
            expansion_policy_loss = extra_outputs[7]
            expansion_stop_rate = extra_outputs[8]
            expansion_continue_rate = extra_outputs[9]
             
            # --- THE FIX: Create the weighted tensors INSIDE the tape scope ---
            weighted_kl_for_logging = information_loss * current_beta
            weighted_rec_for_logging = reconstruction_loss * model.alpha
            # ------------------------------------------------------------------
    
    # 1. Calculate Combined Gradients (for the actual optimizer update)
    first_decoder_gradients = tape.gradient(first_decoder_loss, first_decoder_params)
    second_decoder_gradients = tape.gradient(second_decoder_loss, second_decoder_params)
    action_head_gradients = tape.gradient(action_head_loss, action_head_params)
    expansion_gradients = tape.gradient(expansion_head_loss, expansion_params)
    critic_gradients = tape.gradient(critic_loss, critic_params)
    
    # 2. Calculate Isolated Gradients (for logging only)
    kl_grads = tape.gradient(weighted_kl_for_logging, first_decoder_params)
    act_grads = tape.gradient(action_loss, first_decoder_params)
    rec_grads = tape.gradient(weighted_rec_for_logging, first_decoder_params)

    # --- Helper to slice and calculate global norms ---
    num_dec = len(model.decoder.trainable_variables)
    num_lstm = len(model.lstm_cell.trainable_variables)
    num_enc = len(model.encoder.trainable_variables)

    def extract_norms(grad_list):
        if grad_list is None:
            # Fallback if a loss is totally disconnected
            return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
            
        dec_g = grad_list[:num_dec]
        lstm_g = grad_list[num_dec : num_dec + num_lstm]
        enc_g = grad_list[num_dec + num_lstm : num_dec + num_lstm + num_enc]

        norm_dec = tf.linalg.global_norm([g for g in dec_g if g is not None])
        norm_lstm = tf.linalg.global_norm([g for g in lstm_g if g is not None])
        norm_enc = tf.linalg.global_norm([g for g in enc_g if g is not None])
        
        return norm_enc, norm_lstm, norm_dec

    # Extract the 9 specific norms
    kl_norm_enc, kl_norm_lstm, kl_norm_dec = extract_norms(kl_grads)
    act_norm_enc, act_norm_lstm, act_norm_dec = extract_norms(act_grads)
    rec_norm_enc, rec_norm_lstm, rec_norm_dec = extract_norms(rec_grads)
    # -----------------------------------------------------------------


    # --- COMBINED OPTIMIZER UPDATE: Ensures iterations match total_steps ---
    all_gradients = (
        first_decoder_gradients +
        second_decoder_gradients +
        action_head_gradients +
        expansion_gradients +
        critic_gradients
    )
    all_params = first_decoder_params + second_decoder_params + action_head_params + expansion_params + critic_params
    
    optimizer.apply_gradients(zip(all_gradients, all_params))
    
    # Must explicitly delete a persistent tape when done
    del tape

    for _ in range(max(ppo_epochs - 1, 0)):
        with tf.GradientTape() as ppo_tape:
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
                forced_node_selections=old_node_selections,
                old_expansion_log_probs=old_expansion_log_probs,
                use_ppo_loss=True
            )
            ppo_expansion_loss = extra_outputs[5]

        ppo_expansion_gradients = ppo_tape.gradient(ppo_expansion_loss, expansion_params)
        optimizer.apply_gradients(zip(ppo_expansion_gradients, expansion_params))
    
    # Return all 13 values
    return (
            total_loss, information_loss, action_loss, reconstruction_loss,
            expansion_policy_loss, expansion_stop_rate, expansion_continue_rate,
            kl_norm_enc, kl_norm_lstm, kl_norm_dec,
            act_norm_enc, act_norm_lstm, act_norm_dec,
            rec_norm_enc, rec_norm_lstm, rec_norm_dec
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
        model.action_head.trainable_variables+
        model.expansion_head.trainable_variables +
        model.critic_head.trainable_variables
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
    history = {
        'epoch': [], 'learning_rate': [], 'expansion_epsilon': [],
        'total_loss': [], 'kl_loss': [], 'action_loss': [], 'reconstruction_loss': [],
        'expansion_loss': [], 'expansion_stop_rate': [], 'expansion_continue_rate': [],
        'kl_grad_norm_enc': [], 'kl_grad_norm_lstm': [], 'kl_grad_norm_dec': [],
        'act_grad_norm_enc': [], 'act_grad_norm_lstm': [], 'act_grad_norm_dec': [],
        'rec_grad_norm_enc': [], 'rec_grad_norm_lstm': [], 'rec_grad_norm_dec': []
    }
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
    expansion_epsilon_start = 0.5
    expansion_epsilon_end = 0.0
    expansion_epsilon_annealing_epochs = 100
    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        # Accumulators for this epoch
        ep_total_loss, ep_kl, ep_act, ep_rec = 0.0, 0.0, 0.0, 0.0
        ep_expansion_loss, ep_stop_rate, ep_continue_rate = 0.0, 0.0, 0.0
        
        # Gradient accumulators
        ep_kl_gn_enc, ep_kl_gn_lstm, ep_kl_gn_dec = 0.0, 0.0, 0.0
        ep_act_gn_enc, ep_act_gn_lstm, ep_act_gn_dec = 0.0, 0.0, 0.0
        ep_rec_gn_enc, ep_rec_gn_lstm, ep_rec_gn_dec = 0.0, 0.0, 0.0
        
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
        for i in range(trials_per_epoch):
            batch_input_data = helper.generate_batch_data(batch_size, time_steps, input_type)

            # Unpack metrics from train_step
            (loss, kl, act, rec, 
             exp_loss, stop_rate, continue_rate,
             kl_gn_enc, kl_gn_lstm, kl_gn_dec,
             act_gn_enc, act_gn_lstm, act_gn_dec,
             rec_gn_enc, rec_gn_lstm, rec_gn_dec) = train_step(
                model=model, 
                optimizer=optimizer,
                current_alpha=tf.constant(current_alpha, dtype=tf.float32),
                current_beta=tf.constant(current_beta, dtype=tf.float32),
                current_critic_coef=tf.constant(current_critic_coef, dtype=tf.float32),
                current_expansion_epsilon=tf.constant(current_expansion_epsilon, dtype=tf.float32),
                input_data=batch_input_data,
                input_type_str=input_type,
                ppo_epochs=ppo_epochs
            )
            
            # Accumulate metrics
            ep_total_loss += loss
            ep_kl += kl
            ep_act += act
            ep_rec += rec
            ep_expansion_loss += exp_loss
            ep_stop_rate += stop_rate
            ep_continue_rate += continue_rate
            
            ep_kl_gn_enc += kl_gn_enc
            ep_kl_gn_lstm += kl_gn_lstm
            ep_kl_gn_dec += kl_gn_dec
            
            ep_act_gn_enc += act_gn_enc
            ep_act_gn_lstm += act_gn_lstm
            ep_act_gn_dec += act_gn_dec
            
            ep_rec_gn_enc += rec_gn_enc
            ep_rec_gn_lstm += rec_gn_lstm
            ep_rec_gn_dec += rec_gn_dec
        
        # Calculate averages for the epoch
        avg_total_loss = ep_total_loss / trials_per_epoch
        current_lr = lr_schedule(optimizer.iterations).numpy()
        # Append to history
        history['epoch'].append(epoch + 1)
        history['learning_rate'].append(current_lr)
        history['expansion_epsilon'].append(current_expansion_epsilon)
        history['total_loss'].append((ep_total_loss / trials_per_epoch).numpy())
        history['kl_loss'].append((ep_kl / trials_per_epoch).numpy())
        history['action_loss'].append((ep_act / trials_per_epoch).numpy())
        history['reconstruction_loss'].append((ep_rec / trials_per_epoch).numpy())
        history['expansion_loss'].append((ep_expansion_loss / trials_per_epoch).numpy())
        history['expansion_stop_rate'].append((ep_stop_rate / trials_per_epoch).numpy())
        history['expansion_continue_rate'].append((ep_continue_rate / trials_per_epoch).numpy())
        
        history['kl_grad_norm_enc'].append((ep_kl_gn_enc / trials_per_epoch).numpy())
        history['kl_grad_norm_lstm'].append((ep_kl_gn_lstm / trials_per_epoch).numpy())
        history['kl_grad_norm_dec'].append((ep_kl_gn_dec / trials_per_epoch).numpy())
        
        history['act_grad_norm_enc'].append((ep_act_gn_enc / trials_per_epoch).numpy())
        history['act_grad_norm_lstm'].append((ep_act_gn_lstm / trials_per_epoch).numpy())
        history['act_grad_norm_dec'].append((ep_act_gn_dec / trials_per_epoch).numpy())
        
        history['rec_grad_norm_enc'].append((ep_rec_gn_enc / trials_per_epoch).numpy())
        history['rec_grad_norm_lstm'].append((ep_rec_gn_lstm / trials_per_epoch).numpy())
        history['rec_grad_norm_dec'].append((ep_rec_gn_dec / trials_per_epoch).numpy())

        tf.print(
            f"Epoch {epoch+1}/{epochs}: Loss = {avg_total_loss:.4f} | "
            f"KL = {history['kl_loss'][-1]:.4f} | "
            f"Expansion epsilon = {current_expansion_epsilon:.4f} | "
            f"Stop = {history['expansion_stop_rate'][-1]:.4f} | "
            f"Continue = {history['expansion_continue_rate'][-1]:.4f}"
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
