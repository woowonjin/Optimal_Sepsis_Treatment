"""
The classes and methods in this file are derived or pulled directly from https://github.com/sfujim/BCQ/tree/master/discrete_BCQ
which is a discrete implementation of BCQ by Scott Fujimoto, et al. and featured in the following 2019 DRL NeurIPS workshop paper:
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils import prepare_bc_data
from torch.utils.tensorboard import SummaryWriter

# Simple full-connected supervised network for Behavior Cloning of batch data
class FC_BC(nn.Module):
	def __init__(self, state_dim=33, num_actions=25, num_nodes=64):
		super(FC_BC, self).__init__()
		self.l1 = nn.Linear(state_dim, num_nodes)
		self.bn1 = nn.BatchNorm1d(num_nodes)
		self.l2 = nn.Linear(num_nodes, num_nodes)
		self.bn2 = nn.BatchNorm1d(num_nodes)
		self.l3 = nn.Linear(num_nodes, num_actions)

	def forward(self, state):
		out = F.relu(self.l1(state))
		out = self.bn1(out)
		out = F.relu(self.l2(out))
		out = self.bn2(out)
		return self.l3(out)

class BehaviorCloning(object):
	def __init__(self, input_dim, num_actions, num_nodes=256, learning_rate=1e-3, weight_decay=0.1, optimizer_type='adam', device='cpu'):
		'''Implement a fully-connected network that produces a supervised prediction of the actions
		preserved in the collected batch of data following observations of patient health.
		INPUTS:
		input_dim: int, the dimension of an input array. Default: 33
		num_actions: int, the number of actions available to choose from. Default: 25
		num_nodes: int, the number of nodes
		'''

		self.device = device
		self.state_shape = input_dim
		self.num_actions = num_actions
		self.lr = learning_rate

		# Initialize the network
		self.model = FC_BC(input_dim, num_actions, num_nodes).to(self.device)
		self.loss_func = nn.CrossEntropyLoss()
		if optimizer_type == 'adam':		
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
		else:
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)

		self.iterations = 0

	def train_epoch(self, train_dataloader, dem_context):
		'''Sample batches of data from training dataloader, predict actions using the network,
		Update the parameters of the network using CrossEntropyLoss.'''

		losses = []

		# Loop through the training data 
		for dem, ob, ac, l, t, scores, _, _ in train_dataloader:
			state, action = prepare_bc_data(dem, ob, ac, l, t, dem_context)
			state = state.to(self.device)
			action = action.to(self.device)
			
			# Predict the action with the network
			pred_actions = self.model(state)

			# Compute loss
			try:
				loss = self.loss_func(pred_actions, action.flatten())
			except:
				print("LOL ERRORS")

			# Optimize the network
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			losses.append(loss.item())

		self.iterations += 1

		return np.mean(losses)



# Simple fully-connected Q-network for the policy
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions, num_nodes=128):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, num_nodes)
		self.q2 = nn.Linear(num_nodes, num_nodes)
		self.q3 = nn.Linear(num_nodes, num_actions)

	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		return self.q3(q)


class DQN(object):
	def __init__(
		self, 
		num_actions,
		state_dim,
		device,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=1e3,
		tau=0.005
	):
	
		self.device = device

		# Determine network type
		self.Q = FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Evaluation hyper-parameters
		self.state_shape = (-1, state_dim)
		# self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done, obs_state, next_obs_state = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			q_curr = self.Q(next_state)

			# Use large negative number to mask actions from argmax
			next_action = q_curr.argmax(1, keepdim=True)

			q_target = self.Q_target(next_state)
			#target_Q = 10*reward + done * self.discount * q_target.gather(1, next_action).reshape(-1, 1)
			target_Q = reward + done * self.discount * q_target.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		current_Q = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		Q_loss = F.smooth_l1_loss(current_Q, target_Q)
		
		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()
		return Q_loss.item(), target_Q


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
				self.Q_target.load_state_dict(self.Q.state_dict())


def train_DQN(replay_buffer, num_actions, state_dim, device, parameters, pol_eval_dataloader, is_demog, writer):
	# For saving files
	pol_eval_file = parameters['pol_eval_file']
	pol_file = parameters['policy_file']
	buffer_dir = parameters['buffer_dir']

	# Initialize and load policy
	policy = DQN(
		num_actions,
		state_dim,
		device,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"]
	)

	# Load replay buffer
	replay_buffer.load(buffer_dir, bootstrap=True)

	evaluations = []
	episode_num = 0
	done = True
	training_iters = 0

	while training_iters < parameters["max_timesteps"]:
		for _ in range(int(parameters["eval_freq"])):
			l, targ_q = policy.train(replay_buffer)
			


		#evaluations.append(eval_policy(policy, behav_pol, pol_eval_dataloader, parameters["discount"], is_demog, device))  # TODO Run weighted importance sampling with learned policy and behavior policy
		#np.save(pol_eval_file, evaluations)
		torch.save({'policy_Q_function':policy.Q.state_dict(), 'policy_Q_target':policy.Q_target.state_dict()}, pol_file)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}")
		
		writer.add_scalar('Loss', l, training_iters)
		writer.add_scalar('Current Q value', torch.mean(targ_q), training_iters)

