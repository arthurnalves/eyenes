import subprocess
import ipyparallel as ipp
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle
import os
import sys
import shutil     
from IPython.display import display, HTML

from agent_class import Agent

def printed_wait(wait_time, message = 'Wait is over'):
    clear_output(wait = True)
    for i in range(wait_time):
        print('Waiting {}s'.format(wait_time - i))
        time.sleep(1)
        clear_output(wait = True)
    print(message)
    time.sleep(1)


def flatten(list_of_lists):
    return [elem for sublist in list_of_lists for elem in sublist]
    
def printed_wait(wait_time, message):
    clear_output(wait = True)
    for i in range(wait_time):
        print('Waiting {}s'.format(wait_time - i))
        time.sleep(1)
        clear_output(wait = True)
    print(message)
    time.sleep(1)

class Generation:

    default_kwargs = {'size': 2, 'black_and_white': True, 'rom_id': 'SuperMarioBros-v0', 
        'max_steps': 9999, 'freq': .5, 'buffer': 3,  'num_engines': 4, 'actions': None,
        'layer_prob': .25, 'intensity': 1, 'fps': 3, 'patience': 5,
        'num_survivors': 1, 'similar_penalty': 1, 'rc': None, 'path_name': 'C://Users//arthu//git//eyenes//eyenes'}

    def __init__(self, **kwargs):

        #default kwargs (find out how to do this properly)
        for key, value in self.default_kwargs.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.save_kwargs(kwargs)

        self.history = dict()
        self.history['total_rewards'] = []
        self.history['runtime'] = []
            
        self.agents = []
        for ID in range(self.size):
            self.agents.append(Agent(ID = ID, black_and_white = self.black_and_white, 
                rom_id = self.rom_id, buffer = self.buffer, patience = self.patience, 
                max_steps = self.max_steps, freq = self.freq, intensity = self.intensity, fps = self.fps))

        self.new_ID = ID + 1
        self.top_rewards = []
        
        if 'restart' in self.actions:
            print('Deleting folders')
            self.delete_standard_folders()
            print('Creating folders')
            self.create_standard_folders()
        
        if 'resume' in self.actions:
            print('Loading models')
            self.load_generation()

    def start_engines(self):
        subprocess.Popen(["ipcluster", "stop"])
        printed_wait(10, 'Clusters stopped')
        subprocess.Popen(["ipcluster", "start", "-n={:d}".format(self.num_engines)])
        printed_wait(30, 'Clusters started')

    def remote_import(self):
        self.rc = ipp.Client()
        with self.rc[:].sync_imports():
            from gen_class import Generation
        self.rc.close()

    def create_dir(self, dirname, verbose = False):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            if verbose:
                print('{} created'.format(dirname))

    def save_kwargs(self, kwargs):
        pickle.dump(kwargs, open('pickled/hyper_parameters.pkl','wb'))

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
   
    def parallel_run(self):
        self.rc = ipp.Client()
        self.dview = self.rc[:]
        rewards = self.rc[:].map(lambda agent: agent.get_reward(), self.agents).get()
        self.rc.purge_everything()
        self.rc.close()
        self.dview.results.clear()
        self.rc.results.clear()
        self.rc.metadata.clear()
        del self.dview
        del self.rc
        return rewards

    def sequential_run(self):
        return [agent.get_reward() for agent in self.agents]
    
    def replication(self):
        for e in range(self.num_survivors):
            for a in range(1, self.size//self.num_survivors):
                self.derive(e*self.size//self.num_survivors, e*self.size//self.num_survivors + a)

    def print_history(self):

        f, (ax1, ax2) = plt.subplots(1,2,figsize=(16, 7))

        ax1.plot(self.history['total_rewards'])
        ax1.set_title('Reward History')
        ax1.set_ylabel('Total Reward')
        ax1.set_xlabel('Generation')

        ax2.plot(self.history['runtime'])
        ax2.set_title('Runtime History')
        ax2.set_ylabel('time (s)')
        ax2.set_xlabel('Generation')

        plt.show()

    def evolution_step(self, rewards = None, max_steps = 500, plot = False, monitor = False):
        
               
        start_time = time.time()
        
        if self.mode == 'sequential':
            rewards = self.sequential_run()
        else:
            rewards = self.parallel_run()

        for agent, reward in zip(self.agents, rewards):
            agent.total_reward = reward

        self.history['total_rewards'].append(sorted(rewards, reverse = True))

        top_reward = np.max(rewards)
        agent_id = np.argmax(rewards)
        best_agent = self.agents[agent_id]


        if top_reward not in self.top_rewards:
            self.top_rewards.append(top_reward)
            best_agent.save_model()

            if monitor:
                directory = 'pickled/top_models/videos/' + str(best_agent.total_reward) + '/'
                best_agent.run(mode = 'monitor', directory = directory)


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
    
        clear_output(wait = True)

        if monitor:
            best_agent.play_video(width = 500, height = 375)

        if plot:
            self.print_history()