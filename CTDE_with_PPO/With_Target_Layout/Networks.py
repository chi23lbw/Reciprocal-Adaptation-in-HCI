import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Subtract, LSTM, Concatenate

class UserActor:
	def __init__(self):
		input_A = Input(shape = 4) #Input of user actor
		x = Subtract()([input_A[:, 2:], input_A[:, :2]])
		x = Dense(32, activation = 'relu')(x)
		x = Dense(64, activation = 'relu')(x)
		x = Dense(4, activation = 'softmax')(x)

		self.model = Model(input_A, x)
		self.model.summary()

class AsstActor:
	def __init__(self, memory_len):
		input_B = Input(shape = (memory_len, 6)) #Input of assistant actor/ Output of user actor 
		input_C = Input(shape = (11,11,1)) #Icon layout/ input of assistant actor
		a = Dense(32, activation = 'relu')(input_B)
		a = LSTM(32, activation = 'tanh')(a)

		# b = tf.keras.layers.Conv2D(filters = 2, kernel_size = 3, activation = 'relu')(input_C)
		# b = tf.keras.layers.MaxPooling2D()(b)
		b = tf.keras.layers.Flatten()(input_C)
		b = tf.keras.layers.Dense(64, activation = 'relu')(b)
		b = tf.keras.layers.Dense(32, activation = 'relu')(b)

		a = Concatenate()([a, b])
		a = Dense(32, activation = 'relu')(a)
		a = Dense(4, activation = 'softmax')(a)

		self.model = Model(inputs = [input_B, input_C], outputs = a)
		self.model.summary()


class CentralizedCritic:
	def __init__(self, memory_len):
		input_A = Input(shape = 4) #Input of user actor
		input_B = Input(shape = (memory_len, 6)) #Input of assistant actor/ Output of user actor 
		input_C = Input(shape = (11,11,1)) #Icon layout/ input of assistant actor
		input_D = Input(shape = 4) #Output of assistant actor

		x = Subtract()([input_A[:, 2:], input_A[:, :2]])
		x = Dense(32, activation = 'relu')(x)
		x = Dense(64, activation = 'relu')(x)
		x = Dense(32, activation = 'relu')(x)

		y = Dense(32, activation = 'relu')(input_B)
		y = LSTM(32, activation = 'tanh')(y)

		# z = tf.keras.layers.Conv2D(filters = 2, kernel_size = 3, activation = 'relu')(input_C)
		# z = tf.keras.layers.MaxPooling2D()(z)
		z = tf.keras.layers.Flatten()(input_C)
		z = tf.keras.layers.Dense(64, activation = 'relu')(z)
		z = tf.keras.layers.Dense(32, activation = 'relu')(z)

		w = Dense(32, activation = 'relu')(input_D)

		y = Concatenate()([x, y, z, w])
		y = Dense(32, activation = 'relu')(y)
		y = Dense(1)(y)

		self.model = Model(inputs = [input_A, input_B, input_C, input_D], outputs = y)
		self.model.summary()



