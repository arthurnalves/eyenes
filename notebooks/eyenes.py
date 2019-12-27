import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym import wrappers
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv


import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling1D, SeparableConv2D, Activation
from keras.layers import AveragePooling2D, MaxPooling2D, LSTM, Concatenate, Reshape, GRU, BatchNormalization
from keras.initializers import Constant
from keras.constraints import MaxNorm
from keras.applications.xception import Xception
from keras.applications.mobilenet_v2 import MobileNetV2

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

def make_env(version, movement_type):
    env = gym_super_mario_bros.make('SuperMarioBros-v' + str(version))
    env = BinarySpaceToDiscreteSpaceEnv(env, movement_type)
    return env


def get_mario_model(obs_shape, square_shape, strides, output_dim, hidden_size):
    model = Sequential()
    model.add(AveragePooling2D(batch_input_shape = np.concatenate(([1],obs_shape)),  pool_size = 16))
    model.add(Conv2D(activation='relu', kernel_size = (2,2), filters=3, strides = 2, padding = 'same'))
    model.add(Conv2D(activation='relu', kernel_size = (2,2), filters=6, strides = 2, padding = 'same'))
    model.add(Flatten())
    model.add(Reshape((1,96)))
    model.add(GRU(7, stateful = True))
    model.add(Activation('softmax'))
    model.compile(optimizer = 'adam', loss = 'mse')
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def action_translator(action_vec, movement_type, threshold = 0.1, verbose = False):
    
    actions = []
    if action_vec[0] > threshold:
        actions.append('right')
        if action_vec[3] > 0:
            actions.append('B')
    elif action_vec[0] < -threshold:
        actions.append('left')
        if action_vec[3] > 0:
            actions.append('B')
    if action_vec[2] > 0:
            actions.append('A')
    elif action_vec[0] < threshold and action_vec[0] > -threshold:
        if action_vec[1] > threshold:
            actions.append('up')
        elif action_vec[1] < -threshold:
            actions.append('down')
    
    if verbose:
        print(actions)
    for i, commands in enumerate(movement_type):
        if set(actions) == set(commands):
            return i
    return 0

def mutation(agents, progenitor_label, mutant_label, cancer_level = 1, chance = 0.1):
    progenitor = agents[progenitor_label]
    mutant = agents[mutant_label]
    for pro_layer, mut_layer in zip(progenitor.layers, mutant.layers):
        new_weights = []
        for sublayer in pro_layer.get_weights():
            mutated =  np.random.normal(0, cancer_level, np.shape(sublayer))
            for x in np.nditer(mutated, op_flags=['readwrite']):
                if np.random.random() > chance:
                    x[...] = 0
            new_weights.append(np.clip(sublayer + mutated, -1, 1))
        mut_layer.set_weights(new_weights)
    
    return mutant

def survival_of_the_fittest(agents, fitness_vec):
    top_agents = sorted(zip(agents, fitness_vec), key  = lambda agent: -agent[1])
    sorted_agents = list(np.array(top_agents).T[0])
    sorted_fitness = list(np.array(top_agents).T[1])
    return sorted_agents, sorted_fitness

def reproduction(agents, chance, cancer_level, num_survivors):
    offspring = []
    num_offspring = len(agents)//num_survivors
    for i in range(len(agents)):
        if i < num_survivors:
            offspring.append(agents[i])
        else:
            offspring.append(mutation(agents = agents, progenitor_label = (i-num_survivors)//num_offspring, 
                                  mutant_label = i, 
                                  chance = chance, cancer_level = cancer_level))
        
            
    return offspring

def get_fitness_vec(agents, env, max_frames, num_survivors, fitness_vec, buffer, first):
    for i, agent in enumerate(agents):
        if i >= num_survivors:
            fitness_vec[i] = gameplay(agent, env, max_frames, buffer = buffer)
        elif first:
            fitness_vec[i] = gameplay(agent, env, max_frames, buffer = buffer)
    return fitness_vec


def info_reward(info):
    #reward = info['x_pos']
    reward = info['score']/100
    if info['status'] == 'tall':
        reward += 10
    elif info['status'] == 'fireball':
        reward += 20
    reward += info['coins']
    reward += 15*info['life']
    reward += 1000*(info['stage']-1)
    reward += 5000*(info['world']-1)
    return reward
    
def gameplay(agent, env, max_frames, buffer, max_rest = 60, verbose = True, render = False):  
    
    agent.reset_states()
    reward_hist = []
    
    life = 2
    fitness = 0
    done = True
    x_pos = -1
    resting = 0
    score = 0
    action = 0
    reward = 0
    action_vec = np.ones((1,1,len(env.get_keys_to_action())))
    action = 0
    prev_reward = 0
    for step in range(max_frames):
        if done:
            state = env.reset()
        if step%buffer == 0:
            np_state = np.array(state).reshape((1,240,256,3))
            action_vec = agent.predict(np_state)
            action = np.argmax(action_vec)
        state, reward, done, info = env.step(action)
        if render:
        	env.render()
        fitness += info_reward(info) - prev_reward + reward
        prev_reward = info_reward(info)
        if abs(info['x_pos'] - x_pos) < 1:
            resting += 1
        else:
            x_pos = info['x_pos']
            resting = 0
        if resting > max_rest:
        	if verbose:
        		print(fitness)
        	return fitness
        if life > info['life']:
        	if verbose:
        		print(fitness)
        	return fitness
    if verbose:
    	print(fitness)
    return fitness

import time

def evolution_step(env, generation, num_agents_per_gen, num_survivors, chance, cancer_level, max_frames, fitness_vec, buffer, first):
    
    fitness_hist = []
    current = time.time()
    fitness_vec = get_fitness_vec(generation, env, max_frames, num_survivors, fitness_vec, buffer= buffer, first = first)
    fit_time = time.time() - current
    survivors, fitness_vec = survival_of_the_fittest(generation, fitness_vec)
    current = time.time()
    generation = reproduction(survivors, chance = chance, cancer_level = cancer_level, num_survivors = num_survivors)
    gen_time = time.time() - current
    times = (fit_time, gen_time)
    print('times: ', times)
    return generation, fitness_vec, times


import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym import wrappers
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv


import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Conv3D, Dense, Flatten, MaxPooling1D, SeparableConv2D, Activation, Lambda
from keras.layers import AveragePooling2D, MaxPooling2D, LSTM, Concatenate, Reshape, GRU, BatchNormalization, UpSampling2D
from keras.initializers import Constant
from keras.constraints import MaxNorm
from keras.applications.xception import Xception
from keras.applications.mobilenet_v2 import MobileNetV2


version = 0
movement_type = SIMPLE_MOVEMENT
env = make_env(version, movement_type)
obs_shape = env.observation_space.shape
square_shape = (16,16)
strides = int(square_shape[0]/2)




output_dim = len(env.get_action_meanings())

import tensorflow as tf

def get_mario_vision_model(obs_shape = obs_shape, square_shape = square_shape, strides = strides, output_dim = output_dim, hidden_size = 10):
    image = Input((240, 256, 3))
    encoded = Lambda(lambda x: K.spatial_2d_padding(x), output_shape=(256,256,3))(image)
    encoded = Lambda(function = lambda x: x/255.0)(encoded)
    encoded = Conv2D(kernel_size = (8,8), filters=3, padding = 'same', activation = 'relu')(encoded)
    encoded = MaxPooling2D((4,4))(encoded)
    encoded = Conv2D(kernel_size = (4,4), filters=6, padding = 'same', activation = 'relu')(encoded)
    encoded = MaxPooling2D((4,4))(encoded)
    encoded = Conv2D(kernel_size = (2,2), filters=12, padding = 'same', activation = 'relu')(encoded)
    encoded = MaxPooling2D((2,2))(encoded)
    encoded = Conv2D(kernel_size = (2,2), filters=24, padding = 'same', activation = 'relu')(encoded)
    encoded = MaxPooling2D((2,2))(encoded)
    
    encoder = Model(image, encoded)
 
    decoded = UpSampling2D((2,2))(encoded)
    decoded = Conv2D(kernel_size = (2,2), filters=12, activation = 'relu', padding = 'same')(decoded)
    decoded = UpSampling2D((2,2))(decoded)
    decoded = Conv2D(kernel_size = (2,2), filters=6, activation = 'relu', padding = 'same')(decoded)
    decoded = UpSampling2D((4,4))(decoded)
    decoded = Conv2D(kernel_size = (4,4), filters=3, activation = 'relu', padding = 'same')(decoded)
    decoded = UpSampling2D((4,4))(decoded)
    decoded = Conv2D(kernel_size = (8,8), filters=3, activation = 'relu', padding = 'same')(decoded)
    decoded = Lambda(lambda x: tf.image.resize_image_with_crop_or_pad(x, 240, 256), output_shape = (240, 256, 3))(decoded)
    autoencoder = Model(image, decoded)
    
    encoder.compile(optimizer = 'adam', loss = 'mse')
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    return encoder, autoencoder