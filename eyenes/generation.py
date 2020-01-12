import subprocess
import ipyparallel as ipp
import time
from IPython.display import clear_output
import numpy as np
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from eyenes.agent import Agent
import pickle
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as MOVEMENT

class Generation:
    
    num_engines = None
    num_survivors = None
    agents_per_engine = None
    size = None
    agents = None
    history = None
    new_ID = None
    rc = None
    max_steps = None
    mode = None
    similar_penalty = False
    
    def __init__(self, size, num_survivors, buffer, movement = MOVEMENT, rom_id = 'SuperMarioBros-v2', patience = 5, similar_penalty = 1, fps = 5, freq = .25, layer_prob = .25, intensity = .25, wait = 10, max_steps = 500, mode = 'sequential'):
        self.size = size
        self.num_survivors = num_survivors
        self.num_engines = num_survivors
        self.agents = []
        self.history = dict()
        self.history['rewards'] = []
        self.history['runtime'] = []
        self.max_steps = max_steps
        self.mode = mode
        self.similar_penalty = similar_penalty
        self.rom_id = rom_id
        for ID in range(self.size):
            self.agents.append(Agent(ID = ID, movement= movement, rom_id = self.rom_id, buffer = buffer, patience = patience, max_steps = max_steps, freq = freq, intensity = intensity, fps = fps))
        self.new_ID = ID + 1
        
    def start_engines(self, num_engines):
        self.agents_per_engine = self.size//self.num_engines
        
        subprocess.Popen(["ipcluster", "stop"])
        time.sleep(5)
        subprocess.Popen(["ipcluster", "start", "-n={:d}".format(num_engines)])
        
        self.rc = ipp.Client()
     
    def same_as_parent(self, pos):
        ancestor_pos = pos//self.num_survivors*self.num_survivors
        return ancestor_pos != pos and self.agents[ancestor_pos].total_reward == self.agents[pos].total_reward
        
    def get_positions(self):
        positions = []
        for pos, agent in enumerate(self.agents):
            reward = agent.get_reward()
            if self.same_as_parent(pos):
                reward*=self.similar_penalty
            positions.append((pos, reward))
        return positions

    def get_survivors_pos(self):
        positions = self.get_positions()
        survivors_ranking = sorted(positions, reverse = True, key = lambda key: key[1])[:self.num_survivors]
        return sorted(list(zip(*survivors_ranking))[0])

    def replace(self):
        survivors_pos = self.get_survivors_pos()
        survivors = [self.agents.pop(old_pos) for old_pos in reversed(survivors_pos)]
        replaced = []
        
        factor = len(self.agents)//len(survivors)
        for i in range(self.num_survivors):
            replaced.append(survivors.pop())
            for j in range(factor):
                replaced.append(self.agents.pop())
    
        self.agents = replaced
        return replaced
    
    def derive(self, parent_pos, child_pos):
        self.agents[child_pos].copy_model(self.agents[parent_pos], new_ID = self.new_ID)
        self.new_ID += 1
        self.agents[child_pos].mutate()
   
    def sequential_run(self):
        for agent in self.agents:
            agent.get_reward()

    def parallel_run(self):
        dview = self.rc[:]
        dview.scatter('agents', self.agents)
        rewards = [agent.get_reward() for agent in agents]
        return dview.gather('rewards').get()    
    
    
    def replication(self):
        for e in range(self.num_survivors):
            for a in range(1, self.size//self.num_survivors):
                self.derive(e*self.size//self.num_survivors, e*self.size//self.num_survivors + a)

    def save(self):
        pickle.dump(self, open('generation.pkl', 'wb'))
    
    def evolution_step(self, max_steps = 500, chronometer = False, plot = False):
        start_time = time.time()
        
        if self.mode == 'parallel':
            rewards = self.parallel_run(max_steps = max_steps)
            for agent, reward in zip(self.agents, rewards):
                agent.reward = reward
            self.history['rewards'].append(sorted(rewards, reverse = True))
        
        
        elif self.mode == 'sequential':
            self.sequential_run()
            rewards = [agent.total_reward for agent in self.agents]
            self.history['rewards'].append(sorted(rewards, reverse = True))
                 
        end_time = time.time()
        self.history['runtime'].append(end_time - start_time)
        
        if chronometer:
            print('Run time:', end_time - start_time)
       
        start_time = time.time()
        self.replace()
        self.replication()
        
        if False:
            print('After replication')
            for agent_pos, agent in enumerate(self.agents):
                print(agent_pos, agent.total_reward)
        end_time = time.time()
        
        if chronometer:
            print('Replication time:', end_time - start_time)
            
        if plot:
            clear_output(wait = True)
            plt.title('History')
            plt.plot(self.history['rewards'])
            plt.show()
            plt.title('Run time')
            plt.plot(self.history['runtime'])
            plt.show()
            
