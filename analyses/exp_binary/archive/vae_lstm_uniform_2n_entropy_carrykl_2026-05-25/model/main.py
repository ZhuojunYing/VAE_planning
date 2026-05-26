"""
main.py
The central orchestrator for the project. Parses configurations and routes
execution to either the training phase or the simulation phase.
"""

import os
import tensorflow as tf

# Import our custom modules
import config
from model import VariationalRNN, build_encoder, build_decoder
from train import train_model
from simulate import run_simulation

def model_variant_label(variant):
    return f"variant_{variant}_"

def main():
    # Make sure the target directory exists
    os.makedirs(config.dir_name, exist_ok=True)

    if config.train_mode == "train":
        print("=== Starting Training Mode ===")
        
        # Loop over all hyperparameter combinations provided in command line
        for beta in config.beta_values:
            for lambda_ in config.lambda_values:
                for alpha in config.alpha_values:
                    for opportunity_cost in config.opportunity_cost_values:
                        print(
                            f"\n--- Training -> lambda: {lambda_}, alpha: {alpha}, "
                            f"beta: {beta}, opportunity_cost: {opportunity_cost}, "
                            f"expansion_decision_version: {config.expansion_decision_version}, "
                            f"model_variant: {config.model_variant} ---"
                        )
                        variant_label = model_variant_label(config.model_variant)
                        if config.tree_size == 30:
                            model_name = (
                                f'lambda_{lambda_}_alpha_{alpha}_beta_{beta}_'
                                f'opportunity_{opportunity_cost}_expansion_{config.expansion_decision_version}_'
                                f'{variant_label}'
                                f'seed_{config.seed}_'
                                f'{config.tree_size}n_{config.tree_type}'
                            )
                        else:
                            model_name = (
                                f'lambda_{lambda_}_alpha_{alpha}_beta_{beta}_'
                                f'opportunity_{opportunity_cost}_expansion_{config.expansion_decision_version}_'
                                f'{variant_label}'
                                f'seed_{config.seed}_{config.tree_size}n'
                            )
                        # 1. Initialize Model Architecture
                        encoder = build_encoder(config.rnn_units * 2, config.latent_dim, config.rnn_units)
                        decoder = build_decoder(config.latent_dim, 2 * config.rnn_units, config.rnn_units)
                        
                        if config.tree_size == 30:
                            vrnn_model = VariationalRNN(
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
                                tree_type = config.tree,
                                opportunity_cost=opportunity_cost,
                                input_type=config.input_type,
                                expansion_decision_version=config.expansion_decision_version,
                                use_autoencoder=(config.model_variant == "vae")
                            )
                        else:
                            
                            vrnn_model = VariationalRNN(
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
                                opportunity_cost=opportunity_cost,
                                input_type=config.input_type,
                                expansion_decision_version=config.expansion_decision_version,
                                use_autoencoder=(config.model_variant == "vae")
                            )
                        # 2. Run the Training Loop
                        train_model(
                            model=vrnn_model, 
                            epochs=config.epochs, 
                            trials_per_epoch=config.trials_per_epoch, 
                            batch_size=config.batch_size, 
                            time_steps=config.time_steps, 
                            input_type=config.input_type, 
                            dir_name=config.dir_name, 
                            model_name=model_name
                        )
  
 
    run_simulation(config)

if __name__ == "__main__":
    main()
