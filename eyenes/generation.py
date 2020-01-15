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
import os
import sys
import shutil
from mpl_toolkits.axes_grid1 import ImageGrid       
from IPython.display import display, HTML

class Generation:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.agents = []
        
        self.movement = MOVEMENT

        self.history = dict()
        self.history['total_rewards'] = []
        self.history['runtime'] = []
        
        for ID in range(self.size):
            self.agents.append(Agent(ID = ID, black_and_white = self.black_and_white, 
                movement= self.movement, rom_id = self.rom_id, buffer = self.buffer, patience = self.patience, 
                max_steps = self.max_steps, freq = self.freq, intensity = self.intensity, fps = self.fps))
        self.new_ID = ID + 1
        self.top_rewards = []

    def create_dir(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print('{} created'.format(dirname))

    def create_standard_folders(self):
        self.create_dir('pickled')
        self.create_dir('pickled/generation')
        self.create_dir('pickled/generation/weights')
        self.create_dir('pickled/generation/lineages')
        self.create_dir('pickled/top_models')
        self.create_dir('pickled/top_models/weights')
        self.create_dir('pickled/top_models/videos')

    def delete_standard_folders(self):
        #!/usr/bin/python
    

        # Get directory name
        mydir= 'pickled'

        ## Try to remove tree; if failed show an error using try...except on screen
        try:
            shutil.rmtree(mydir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

    def save_generation(self):
        pickle.dump(self.history, open('pickled/generation/history.pkl','wb'))
        for i, agent in enumerate(self.agents):
            pickle.dump(agent.model.model.get_weights(), open('pickled/generation/weights/weight_' + str(i) + '.pkl', 'wb'))
            pickle.dump(agent.lineage, open('pickled/generation/lineages/lineage_' + str(i) + '.pkl', 'wb'))
    
    def load_generation(self):
        current_id = 0
        for agent in self.agents:
            for lineage_id in agent.lineage:
                if lineage_id > current_id:
                    current_id = lineage_id
        self.new_ID = current_id
        self.history = pickle.load(open('pickled/generation/history.pkl','rb'))
        for i, agent in enumerate(self.agents):
            agent.model.model.set_weights(pickle.load(open('pickled/generation/weights/weight_' + str(i) + '.pkl', 'rb')))
            agent.lineage = pickle.load(open('pickled/generation/lineages/lineage_' + str(i) + '.pkl', 'rb'))


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

    def print_history(self):

        f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

        ax1.plot(self.history['total_rewards'])
        ax1.set_title('Reward History')
        ax1.set_ylabel('Total Reward')
        ax1.set_xlabel('Generation')

        ax2.plot(self.history['runtime'])
        ax2.set_title('Runtime History')
        ax2.set_ylabel('time (s)')
        ax2.set_xlabel('Generation')

        plt.show()

    def evolution_step(self, max_steps = 500, plot = False, monitor = False):
        start_time = time.time()

        if monitor or plot:
            display(HTML("""
            <style>
            .output {
                display: flex;
                align-items: center;
                text-align: center;
            }
            </style>
            """))

        
        if self.mode == 'parallel':
            rewards = self.parallel_run(max_steps = max_steps)
            for agent, reward in zip(self.agents, rewards):
                agent.reward = reward
            self.history['total_rewards'].append(sorted(rewards, reverse = True))
        
        
        elif self.mode == 'sequential':
            rewards = [agent.get_reward() for agent in self.agents]
            self.history['total_rewards'].append(sorted(rewards, reverse = True))
                 
        top_reward = np.max(rewards)
        agent_id = np.argmax(rewards)

        if top_reward not in self.top_rewards:
            self.top_rewards.append(top_reward)
            self.agents[agent_id].save_model()

            if monitor:
                directory = 'pickled/top_models/videos/' + str(len(self.top_rewards)) + '/'
                print(directory)
                self.agents[agent_id].run(mode = 'monitor', directory = directory)
        
        end_time = time.time()
        self.history['runtime'].append(end_time - start_time)
        
        start_time = time.time()
        self.replace()
        self.replication()
        
        if False:
            print('After replication')
            for agent_pos, agent in enumerate(self.agents):
                print(agent_pos, agent.total_reward)
        end_time = time.time()
        
            
        if plot:
            clear_output(wait = True)
            self.print_history()

        display(HTML("""
                <style>
                .output {
                    display: flex;
                    align-items: left;
                    text-align: left;
                }
                </style>
                """))