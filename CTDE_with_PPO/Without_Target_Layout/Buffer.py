import os
import numpy as np

class PPOMemory:
	def __init__(self, batch_size):
		self.user_states = []
		self.asst_states = []
		self.layouts = []
		self.asst_outputs_one_hot = []

		self.asst_probs = [] #Log probs
		self.asst_actions = [] #Actions took
		
		self.user_probs = [] #Log probs
		self.user_actions = [] #Actions took
		
		self.vals = [] #Value of critics
		self.rewards = [] #Rewards recieved
		self.dones = []

		self.batch_size = batch_size

	def generate_batches(self):
		n_states = len(self.user_states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype = np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]

		return np.array(self.user_states), np.array(self.asst_states), np.array(self.layouts),\
		np.array(self.asst_outputs_one_hot), np.array(self.user_actions), np.array(self.user_probs),\
		np.array(self.asst_actions), np.array(self.asst_probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

	def store_memory(self, user_state, asst_state, layout, asst_output_one_hot,\
	 user_action, user_prob, asst_action, asst_prob, vals, reward, done):
		self.user_states.append(user_state)
		self.asst_states.append(asst_state)
		self.layouts.append(layout)
		self.asst_outputs_one_hot.append(asst_output_one_hot)

		self.user_probs.append(user_prob)
		self.user_actions.append(user_action)
		self.asst_probs.append(asst_prob)
		self.asst_actions.append(asst_action)
		
		self.vals.append(vals)
		self.rewards.append(reward)
		self.dones.append(done)

	def clear_memory(self):
		self.user_states = []
		self.asst_states = []
		self.layouts = []
		self.asst_outputs_one_hot = []

		self.asst_probs = [] #Log probs
		self.asst_actions = [] #Actions took
		
		self.user_probs = [] #Log probs
		self.user_actions = [] #Actions took
		
		self.vals = [] #Value of critics
		self.rewards = [] #Rewards recieved
		self.dones = []
