"""
This script is used to develop a baseline policy using only the observed patient data via Behavior Cloning.

This baseline policy is then used to truncate and guide evaluation of policies learned using dBCQ. It should only need to be
run once for each unique cohort that one looks to learn a better treatment policy for.

The patient cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

# IMPORTS
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dqn_utils import BehaviorCloning, prepare_bc_data
from utils import ReplayBuffer


def run(BC_network, train_dataloader, val_dataloader, num_epochs, storage_dir, loss_func, dem_context):
    # Construct training and validation loops
    validation_losses = []
    training_losses = []
    training_iters = 0
    eval_frequency = 100
	
    for i_epoch in range(num_epochs):
        
        train_loss = BC_network.train_epoch(train_dataloader, dem_context)
        training_losses.append(train_loss)

        if i_epoch % eval_frequency == 0:
            eval_errors = []
            BC_network.model.eval()
            with torch.no_grad():
                for dem, ob, ac, l, t, scores, _, _ in val_dataloader:
                    val_state, val_action = prepare_bc_data(dem, ob, ac, l, t, dem_context)
                    val_state = val_state.to(torch.device('cpu'))
                    val_action = val_action.to(torch.device('cpu'))
                    pred_actions = BC_network.model(val_state)
                    try:
                        eval_loss = loss_func(pred_actions, val_action.flatten())
                        eval_errors.append(eval_loss.item())
                    except:
                        print("LOL ERRORS")

            mean_val_loss = np.mean(eval_errors)
            validation_losses.append(mean_val_loss)
            np.save(storage_dir+'validation_losses.npy', validation_losses)
            np.save(storage_dir+'training_losses.npy', training_losses)

            print(f"Training iterations: {i_epoch}, Validation Loss: {mean_val_loss}")
            # Save off and store trained BC model
            torch.save(BC_network.model.state_dict(), storage_dir+'BC_model.pt')

            BC_network.model.train()
    
    print("Finished training Behavior Cloning model")
    print('+='*30)


if __name__ == '__main__':

    # Define input arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--demographics', dest='dem_context', default=False, action='store_true')
    parser.add_argument('--num_nodes', dest='num_nodes', default=128, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, type=float)
    parser.add_argument('--storage_folder', dest='storage_folder', default='test', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', default=5000, type=int)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.1, type=float)
    parser.add_argument('--optimizer_type', dest='optim_type', default='adam', type=str)

    args = parser.parse_args()

    device = torch.device('cpu')

    input_dim = 38 if args.dem_context else 33
    num_actions = 25

    train_data_file = '../data/train_set_tuples'
    val_data_file = '../data/val_set_tuples'
    minibatch_size = 128
    storage_dir = '/Users/huangyong/reinforcement-learning-for-sepsis/' + args.storage_folder + '/'

    train_demog, train_states, train_interventions, train_lengths, train_times, acuities, rewards = torch.load(train_data_file)
    train_idx = torch.arange(train_demog.shape[0])
    train_dataset = TensorDataset(train_demog, train_states, train_interventions,train_lengths,train_times, acuities, rewards, train_idx)
    train_loader = DataLoader(train_dataset, batch_size= minibatch_size, shuffle=True)

    val_demog, val_states, val_interventions, val_lengths, val_times, val_acuities, val_rewards = torch.load(val_data_file)
    val_idx = torch.arange(val_demog.shape[0])
    val_dataset = TensorDataset(val_demog, val_states, val_interventions, val_lengths, val_times, val_acuities, val_rewards, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False)

    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)


    # Initialize the BC network
    BC_network = BehaviorCloning(input_dim, num_actions, args.num_nodes, args.learning_rate, args.weight_decay, args.optim_type, device)

    loss_func = nn.CrossEntropyLoss()

    run(BC_network, train_loader, val_loader, args.num_epochs, storage_dir, loss_func, args.dem_context)