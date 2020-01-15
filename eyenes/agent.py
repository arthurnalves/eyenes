from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as MOVEMENT
from gym import wrappers
from IPython.display import Video
import io
import base64
from IPython.display import HTML, display
import numpy as np
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import pickle

from eyenes.agent_model import AgentModel

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

    def __init__(self, ID = -1, movement = MOVEMENT, black_and_white = None, rom_id = 'SuperMarioBros-v0', update = ['reward'], buffer = 3, patience = 5, max_steps = 500, freq = .25, intensity = .25, fps = 5):
        
        self.buffer = buffer
        self.freq = freq
        self.black_and_white = black_and_white
        self.intensity = intensity
        self.rom_id = rom_id
        self.movement = movement
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
        for _ in range(buffer*fps):
            self.state.append(np.zeros(self.env.observation_space.shape))
        
    def play_video(self, width = 400, height = 300):
        self.video.width = width
        self.video.height = height
        display(self.video)

    def make_env(self, mode = None, directory = None):
        env = gym_super_mario_bros.make(self.rom_id)
        env = JoypadSpace(env, self.movement)
        if mode == 'monitor':
            env = wrappers.Monitor(env, directory, force = True)
        return env 
    
    def start_model(self):
        env = self.make_env()
        self.model = AgentModel(buffer = self.buffer, black_and_white = self.black_and_white, input_shape = env.observation_space.shape, output_dim = env.action_space.n)
    
    def get_buffered_images(self):
        buffered_states = []
        for i in range(self.buffer):
            buffered_states.append(self.state[(i + 1)*self.fps - 1])
        return buffered_states
    
    def take_action(self):
        buffered_imgs = self.get_buffered_images()
        prediction = self.model.predict(np.expand_dims(np.concatenate(buffered_imgs, axis = 2), axis = 0))
        return np.argmax(prediction)

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
            
        for step in range(self.max_steps):

            if step%self.fps == 0:
                action = self.take_action()
                
            next_state, reward, done, info = env.step(action)
            
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
                
            if resting > self.patience*60:
                self.total_reward += self.lazy_penalty
                self.total_reward += info['score']/10
                break
                
            if info['life'] < 2: 
                self.total_reward += self.death_penalty
                self.total_reward += info['score']/10
                break

            self.gather_data(step, state, reward, done, info, next_state)
            prev_state = state
            state = next_state

            if mode == 'render':
                env.render()

        if mode == 'monitor':
            file_name = directory + 'openaigym.video.%s.video000000.mp4'% env.file_infix
            mp4 = Video(file_name, width = 400, height = 300)
            self.video = mp4

        if mode == 'render':    
            env.close()

        self.total_reward += info['score']/10
        #self.total_reward += info['time']/10
                
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