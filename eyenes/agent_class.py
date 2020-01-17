from custom_joypad import CustomJoypad
from model_class import AgentModel

import gym_super_mario_bros
from gym import wrappers

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from IPython.display import Video, HTML, display
from collections import deque
import copy
import pickle
import numpy as np

def list_dist(list1, list2):
    dist = 0
    combined_list = list1 + list2
    
    for elem in combined_list:
        if elem not in list1:
            dist += 1
        elif elem not in list2:
            dist +=1
    return dist

class Agent:

    button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,
    }

    def __init__(self, ID = -1, black_and_white = None, rom_id = 'SuperMarioBros-v0', update = ['reward'], buffer = 3, patience = 5, max_steps = 500, freq = .25, intensity = .25, fps = 5):
        
        self.buffer = buffer
        self.freq = freq
        self.black_and_white = black_and_white
        self.intensity = intensity
        self.rom_id = rom_id
        self.env = self.make_env()
        self.start_model()
        self.update = update
        self.state = deque(maxlen = buffer*fps + 1)
        self.ID = ID
        self.max_steps = max_steps
        self.lineage = []
        self.fps = fps
        self.patience = patience
        self.lazy_penalty = -30
        self.death_penalty = -50
        self.total_reward = None
        self.video = None
        self.button_penalty = 5
        for _ in range(buffer*fps):
            self.state.append(np.zeros(self.env.observation_space.shape))
       
    def get_button_list(self, output_vec):
        button_list = []

        if round(output_vec[0]) == 1:
            button_list.append('right')
        elif round(output_vec[0]) == -1:
            button_list.append('left')

        if round(output_vec[1]) == 1:
            button_list.append('up')
        elif round(output_vec[1]) == -1:
            button_list.append('down')

        if round(output_vec[2]) == 1:
            button_list.append('A')

        if round(output_vec[3]) == 1:
            button_list.append('B')

        return button_list

    def play_video(self, width = 400, height = 300):
        
        if self.video == None:
            directory = 'pickled/top_models/videos/' + str(self.total_reward) + '/'
            self.run(mode = 'monitor', directory = directory)
        self.video.width = width
        self.video.height = height
        display(self.video)

    def make_env(self, mode = None, directory = None):
        env = gym_super_mario_bros.make(self.rom_id)
        env = CustomJoypad(env)
        if mode == 'monitor':
            env = wrappers.Monitor(env, directory, force = True)
        return env 
    
    def start_model(self):
        env = self.make_env()
        self.model = AgentModel(buffer = self.buffer, black_and_white = self.black_and_white, input_shape = env.observation_space.shape, output_dim = 4)
    
    def get_buffered_images(self):
        buffered_states = []
        for i in range(self.buffer):
            buffered_states.append(self.state[(i + 1)*self.fps - 1])
        return buffered_states
    
    def take_action(self):
        buffered_imgs = self.get_buffered_images()
        prediction = self.model.predict(np.expand_dims(np.concatenate(buffered_imgs, axis = 2), axis = 0))
        button_list = self.get_button_list(prediction[0])
        return button_list

    def reset_state(self):
        w, h, c = self.env.observation_space.shape
        if self.black_and_white:
            for i in range(self.buffer*self.fps + 1):
                self.state.append(np.zeros((w, h, 1)))
        else:
            for i in range(self.buffer*self.fps + 1):
                self.state.append(np.zeros((w, h, c)))
    
    def reset_data(self):
        self.reset_state()
        self.total_reward = 0
        self.reward = []
        self.done = []
        self.info = dict()
        self.next_state = []
    
    def to_gray(self, state):
        r, g, b = 0.299, 0.587, 0.114
        return np.expand_dims(np.dot(state[...,:3], [r,g,b]), axis = 2)

    def gather_data(self, step, state, reward, done, info, next_state):
        self.total_reward += reward
        if self.black_and_white:
            if step%self.fps == 0:
                self.state.append(np.array(self.to_gray(state)))
        else:
            if step%self.fps == 0:
                self.state.append(np.array(state))
            
    def run(self, mode = None, directory = None):  

        env = self.make_env(mode = mode, directory = directory)
        
        self.reset_data()
        self.model.model.reset_states()
        state = env.reset()

        resting = 0
        x_pos = 0
        prev_state = state
        done = False
        last_x_pos = 24
            
        prev_action = self.take_action()

        for step in range(self.max_steps):

            if step%self.fps == 0:
                action = self.take_action()
                
            next_state, reward, done, info = env.step(action)

            reward += self.button_penalty*list_dist(prev_action, action)
            prev_action = action

            #advancing check
            if info['x_pos'] > x_pos:
                x_pos = info['x_pos']
                resting = 0
            
            #sub-area entry check
            if abs(last_x_pos - info['x_pos']) > 10:
                resting = 0
                x_pos = info['x_pos']

            last_x_pos = info['x_pos']

            resting += 1

            if info['life'] < 2:
                break
                
            if resting > self.patience*60:
                break

            self.gather_data(step, state, reward, done, info, next_state)
            prev_state = state
            state = next_state

            if mode == 'render':
                env.render()

            if done:
                break
                
        if mode == 'monitor':
            file_name = directory + 'openaigym.video.%s.video000000.mp4'% env.file_infix
            mp4 = Video(file_name, width = 400, height = 300)
            self.video = mp4

        env.close()
                
    def get_reward(self):
        if self.total_reward == None:
            self.run()
        return self.total_reward

    def itsame(self):
        return 'Mario!'
    
    def copy_model(self, other, new_ID):
        self.ID = new_ID
        self.lineage = copy.copy(other.lineage)
        if other.ID not in self.lineage:
            self.lineage.append(other.ID)
        self.model.set_weights(copy.deepcopy(other.model.get_weights()))

    def mutate(self):
        self.model.mutate(freq = self.freq, intensity = self.intensity)
        self.total_reward = None
        
    def print_state(self):
        fig = plt.figure(figsize=(16., 12.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, self.buffer),  # creates 2x2 grid of axes
                         axes_pad=0.1,)  # pad between axes in inch.

        
        for ax, im in zip(grid, self.get_buffered_images()):
            # Iterating over the grid returns the Axes.
            if self.black_and_white:
                ax.imshow(im[:, :, 0], cmap = 'gray')
            else:
                ax.imshow(im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.show()

    def save_model(self):
        pickle.dump(self.model.model.get_weights(), open('pickled/top_models/weights/' + str(self.total_reward) + '_weights.pkl', 'wb'))