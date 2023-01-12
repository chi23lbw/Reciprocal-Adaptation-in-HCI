import numpy as np

class Environment:
	def __init__(self, loc_icons=None, num_icons=None, usage_prob=None):
		'''Initializes the location of icons and rescales btw 0 and 1'''
		if loc_icons is None:
			if num_icons is None:
				self.cells = np.random.randint(low=0, high=10, size=(6, 2))/10
			else:
				self.cells = np.random.randint(
					low=0, high=10, size=(num_icons, 2))/10

		else:
			self.cells = np.array(loc_icons)/10

		if usage_prob is not None:
			self.usage_prob = np.array(usage_prob)
		else:
			self.usage_prob = np.array(
				[1/self.cells.shape[0]]*self.cells.shape[0])

		# print('Icon Locations:')
		# print(self.cells)
		# print('Icon usage Probabilities')
		# print(self.usage_prob)

	def give_start_dest(self):
		dest_loc = self.cells[np.random.choice(
			self.cells.shape[0], p=self.usage_prob), :]
		start_loc = dest_loc[:]
		while np.equal(start_loc, dest_loc).all():
			start_loc = np.random.randint(low=0, high=10, size=(2,))/10

		return start_loc, dest_loc

	def step(self, action_user, action_mod, target_loc, curr_loc):
		'''Action of user : 0:left = [0, -1], 1:right = [0,1], 2:up = [-1,0], 3:down = [1, 0]
			Action of modulator = 1,2,3,4
		'''
		if action_user == 0:
			action_user = [0, -1]
		elif action_user == 1:
			action_user = [0, 1]
		elif action_user == 2:
			action_user = [-1, 0]
		elif action_user == 3:
			action_user = [1, 0]

		action_user = np.array(action_user)
		curr_loc = np.array(curr_loc)
		target_loc = np.array(target_loc)
		new_loc = curr_loc*10 + action_user*action_mod
		new_loc = new_loc/10
		new_loc = new_loc.round(1)

		new_loc[0] = min(new_loc[0], 1)
		new_loc[0] = max(new_loc[0], 0)
		new_loc[1] = min(new_loc[1], 1)
		new_loc[1] = max(new_loc[1], 0)
			
		reward_mod = -1
		reward_user = -1

		if np.allclose(new_loc, target_loc):
			done = 1
			reward_user = 10
			reward_mod = 10
		else:
			done = 0

		new_loc = [new_loc[0], new_loc[1]]

		return new_loc, reward_user, reward_mod, done

def make_one_hot(index, size_array):
	array = [0]*size_array
	array[index] = 1
	return array

def give_mapping(cell_locs):
	mapping = np.zeros((11,11))
	for cell in cell_locs:
		mapping[int(cell[0]*10), int(cell[1]*10)] = 1

	return mapping
	
if __name__ == '__main__':
	env = Environment()
	action_user = 1
	action_mod = 1

	curr_loc, target_loc = env.give_start_dest()
	print(curr_loc, target_loc)
	print(env.step(action_user, action_mod, target_loc, curr_loc))
