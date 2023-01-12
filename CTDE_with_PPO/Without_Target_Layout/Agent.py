import math

import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
import numpy as np

from Environment import Environment, make_one_hot, give_mapping
from Networks import UserActor, AsstActor, CentralizedCritic
from Buffer import PPOMemory

class Agent:
	def __init__(self, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2,
				 batch_size=64, N=20, n_epochs=10, memory_len = 6):
		#N is the horizon the number of steps after we do an update
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.n_epochs = n_epochs
		self.gae_lambda = gae_lambda

		self.user_actor = UserActor()
		self.user_actor = self.user_actor.model
		
		self.asst_actor = AsstActor(memory_len)
		self.asst_actor = self.asst_actor.model
		
		self.critic = CentralizedCritic(memory_len)
		self.critic = self.critic.model

		self.optimizer_critic = tf.keras.optimizers.Adam(lr = alpha)
		self.optimizer_user = tf.keras.optimizers.Adam(lr = alpha)
		self.optimizer_asst = tf.keras.optimizers.Adam(lr = alpha)
		self.memory = PPOMemory(batch_size)

	def remember(self, user_state, asst_state, layout, asst_output_one_hot,\
		user_action, user_prob, asst_action, asst_prob, vals, reward, done):

		self.memory.store_memory(user_state, asst_state, layout, asst_output_one_hot,\
			user_action, user_prob, asst_action, asst_prob, vals, reward, done)

	def save_models(self):
		print('Saving Models')
		save_model(self.user_actor, r'Models/user_actor.h5')
		save_model(self.asst_actor, r'Models/asst_actor.h5')
		save_model(self.critic, r'Models/critic.h5')

	def load_models(self):
		print('Loading Models')
		self.user_actor = load_model(r'Models/user_actor.h5')
		self.asst_actor = load_model(r'Models/asst_actor.h5')
		self.critic = load_model(r'Models/critic.h5')

	def choose_action(self, observation, layout, prev_steps_assist):
		#observation = curr_x, curr_y, target_x, target_y
		curr_loc = observation[:2]
		target_loc = observation[2:4]

		user_state = tf.convert_to_tensor([observation], dtype=tf.float32)
		user_probs = self.user_actor(user_state)
		user_action = np.random.choice(4, p=np.squeeze(user_probs))
		user_prob = np.squeeze(user_probs)[user_action]

		action_user_one_hot = make_one_hot(user_action, 4)

		ob_assist = [action_user_one_hot + curr_loc] 
		asst_state = tf.convert_to_tensor([prev_steps_assist + ob_assist], dtype=tf.float32)
		asst_probs = self.asst_actor([asst_state, layout])
		asst_action = np.random.choice(4, p=np.squeeze(asst_probs))
		asst_prob = np.squeeze(asst_probs)[asst_action]

		asst_output_one_hot = tf.convert_to_tensor([make_one_hot(asst_action, 4)], dtype=tf.float32)		
		value = np.squeeze(self.critic([user_state, asst_state, layout, asst_output_one_hot]))

		return user_action, user_prob, asst_action, asst_prob, value, prev_steps_assist+ob_assist, make_one_hot(asst_action, 4)

	def learn(self):
		for epoch in range(self.n_epochs):
			user_state_arr, asst_state_arr, layout_arr, asst_output_one_hot_arr, user_action_arr, old_probs_user_arr, asst_action_arr, old_probs_asst_arr,\
			vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

			values = vals_arr
			advantage = np.zeros(len(reward_arr), dtype=np.float32)

			for t in range(len(reward_arr)-1):
				discount = 1
				a_t = 0

				for k in range(t, len(reward_arr)-1):
					a_t += discount*(reward_arr[k] + self.gamma*values[k+1] *
									 (1-int(done_arr[k])) - values[k])

					discount *= self.gamma*self.gae_lambda

				advantage[t] = a_t

			advantage = tf.convert_to_tensor(advantage)
			values = tf.convert_to_tensor(values)

			for batch in batches:
				user_states = tf.convert_to_tensor(
					user_state_arr[batch], dtype=tf.float32)
	
				asst_states = tf.convert_to_tensor(
					asst_state_arr[batch], dtype=tf.float32)
	
				layouts = tf.convert_to_tensor(
					layout_arr[batch], dtype=tf.float32)

				asst_outputs_one_hot = tf.convert_to_tensor(
					asst_output_one_hot_arr[batch])

				old_user_probs = tf.convert_to_tensor(
					old_probs_user_arr[batch], dtype=tf.float32)

				old_asst_probs = tf.convert_to_tensor(
					old_probs_asst_arr[batch], dtype=tf.float32)

				actions_user = tf.convert_to_tensor(
					list(zip(range(len(batch)), user_action_arr[batch])))

				actions_asst = tf.convert_to_tensor(
					list(zip(range(len(batch)), asst_action_arr[batch])))

				with tf.GradientTape(persistent=True) as tape:
					# print(actions_user)
					# print(self.user_actor(user_states))
					new_user_probs = tf.gather_nd(self.user_actor(user_states), actions_user)
					new_asst_probs = tf.gather_nd(self.asst_actor([asst_states, layouts]), actions_asst)

					# print(new_user_probs)

					critic_val = self.critic([user_states, asst_states, layouts, asst_outputs_one_hot])
					critic_val = tf.squeeze(critic_val)

					probs_ratio_user = new_user_probs/old_user_probs
					probs_ratio_asst = new_asst_probs/old_asst_probs

					weighted_probs_user = tf.gather(advantage, batch)*probs_ratio_user
					weighted_probs_asst = tf.gather(advantage, batch)*probs_ratio_asst

					weighted_clipped_probs_user = tf.math.multiply(tf.clip_by_value
										(probs_ratio_user, 1-self.policy_clip, 1+self.policy_clip), tf.gather(advantage, batch))

					weighted_clipped_probs_asst = tf.math.multiply(tf.clip_by_value
										(probs_ratio_asst, 1-self.policy_clip, 1+self.policy_clip), tf.gather(advantage, batch))

					actor_loss_user = tf.reduce_mean(-tf.math.minimum(weighted_probs_user, weighted_clipped_probs_user))
					actor_loss_asst = tf.reduce_mean(-tf.math.minimum(weighted_probs_asst, weighted_clipped_probs_asst))

					returns = tf.gather(advantage, batch) + tf.gather(values, batch)
					critic_loss = tf.reduce_mean((returns-critic_val)**2)

					

				grads_user_actor = tape.gradient(actor_loss_user, self.user_actor.trainable_variables)
				self.optimizer_user.apply_gradients(zip(grads_user_actor, self.user_actor.trainable_variables))

				grads_asst_actor = tape.gradient(actor_loss_asst, self.asst_actor.trainable_variables)
				self.optimizer_asst.apply_gradients(zip(grads_asst_actor, self.asst_actor.trainable_variables))

				grads_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
				self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

		self.memory.clear_memory()