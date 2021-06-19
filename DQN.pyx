# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:37:12 2020

@author: Administrator
"""

import os
import torch as T
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from pytorch_model_summary import summary

class DeepQNetwork(nn.Module): 
    def __init__(self, lr, n_actions, input_dims, type1):
        super(DeepQNetwork, self).__init__()
        #print('DepQnetworks ', n_actions, ' inout dims ', input_dims)
        
        #self.checkpoint_dir= chkpt_dir
        self.checkpoint_file= 'R_5'+type1 +'.dat' 
        #self.checkpoint_file_eval= os.path.join('q_eval.dat')
        #self.checkpoint_file_next= os.path.join('q_next.dat')
        
      #  self.conv1= nn.Conv2d(input_dims, 8, 4, stride= 4)
      #  self.conv2= nn.Conv2d(32, 54, 4, stride = 2)
     #   self.conv3= nn.Conv2d(64, 64, 3, stride= 1)
        
      #  fc_input_dims = self.calculate_conv_output_dims(input_dims)
       # print('input dims ', input_dims)
       
        #print('hidden units ', self.hidden_units)
        padding1= 1
        filter1= 2
        num_chan=1
        #self.conv1= nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4,2), stride=1, padding=0)
        #self.conv2= nn.Conv2d(6, 16, 2)
        
        #d1= num_chan * ((input_dims[0]+padding1-filter1+1) * (input_dims[1] + padding1 - filter1 +1))
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, padding=1)
        
        #fc_size=(I-F+2P)/S   
        #fc_size= 6
        
        #self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        #self.fc2 = nn.Linear(in_features=120, out_features=60)
        #self.out = nn.Linear(in_features=60, out_features=10)

        
        
        
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        print("fc input ", fc_input_dims);
        self.fc1= nn.Linear(fc_input_dims,128)
        self.fc2= nn.Linear(128, n_actions)
      #  self.checkpoint_file= 'DL_data2.data'
        
        self.optimizer = optim.RMSprop(self.parameters(), lr= lr)
        self.loss = nn.MSELoss() #T.device('cpu')
        self.device= T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    
    def forward(self, state):
        #print('forward ', state)
        #conv1= T.relu(self.conv1(state))
        #conv2= T.relu(self.conv1(conv1))
        #conv3= T.relu(self.conv3(conv2))
        #state1= state.view(64, -1)
        #conv_state = conv3.view(conv3.size()[0],-1)
        state1= state.reshape(state.size()[0],1,state.size()[1],state.size()[2])
        conv1= T.relu(self.conv1(state1)) #F.relu
        #conv2= T.relu(self.conv2(conv1))
        s1= conv1.size()[1]
        s2= conv1.size()[2]
        s3= s1 * s2 * conv1.size()[3]
        conv2= conv1.reshape(-1, s3)
        flat1= T.relu(self.fc1(conv2))
        actions= self.fc2(flat1)
        #actions= self.fc4(flat2)
        return actions
        
    def save_checkpoint(self):
        print('---saving checkpoint.....')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('....loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))
        
    def calculate_conv_output_dims(self, input_dims):
        #state= T.zeros(10,4)
        #print('cal state')
        #print(state)
        input1= Variable(T.ones(1,1,input_dims[0], input_dims[1]))
        dims= self.conv1(input1)
        
        return int(np.prod(dims.size()))
        
    
        
        