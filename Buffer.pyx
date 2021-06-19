
"""
Created on Mon Feb 10 09:04:09 2020

@author: Administrator
"""

import numpy as np
class ReplayBuffer():
    def __init__(self, mem_size1, mem_size2, input_shape, n_actions):
        self.mem_size1= mem_size1
        self.mem_size2= mem_size2
        self.mem_cntr=0
        mm2= self.mem_size1 + self.mem_size2
       # print('shaped flattened ', *input_shape, ' original ', input_shape)
        self.state_memory = np.zeros((mm2, *input_shape), dtype= np.float32)
        self.new_state_memory = np.zeros((mm2, *input_shape), dtype= np.float32)
        self.action_memory = np.zeros(mm2, dtype= np.int64)
        self.reward_memory = np.zeros(mm2, dtype= np.float32)
        
    def store_transition(self, state, action, reward, state_):
        mm2= self.mem_size1 + self.mem_size2
        if self.mem_cntr < mm2:
            index= self.mem_cntr
        else:
            
            index = self.mem_size1 + (self.mem_cntr % self.mem_size2)
                
        
       # index= self.mem_cntr% self.mem_size
         
        self.state_memory[index]= state
        self.action_memory[index]= action
        self.reward_memory[index]= reward
        
        
        self.new_state_memory[index]= state_
        self.mem_cntr +=1
        
    def sample_buffer(self, batch_size):
        mm2= self.mem_size1 + self.mem_size2
        max_mem= min(self.mem_cntr, mm2)
        batch = np.random.choice(max_mem, batch_size, replace= False)
        states = self.state_memory[batch]
        actions= self.action_memory[batch]
        states_= self.new_state_memory[batch]
        rewards= self.reward_memory[batch]


        return states, actions, rewards, states_
    
        
        
    
        