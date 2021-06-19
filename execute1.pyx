# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:42:04 2020

@author: ufarooq
"""


import sys
from pathlib import Path
import pickle
#import VLC2env
import numpy as np
import random
import math
from itertools import combinations
from model import basestation
from agentcode import DDQNAgent
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
#import multiprocessing


class runexp:
    def __init__(self, nusers, train_num, train_iter, num_trials1, rmin1, radius1, bw1):
        P_T= 10
        num_users=nusers
        R= radius1  # radius of the cell in m
        num_tests=train_num
        iter_per_test= train_iter
        mem_size1= num_tests * iter_per_test
        mem_size2= 50000
        batch_size1= 16
        hidden_units1= 64
        trace_en= True
    
        rmin= rmin1 # Minimum distance of any user from the BS
    # B : total bandwidth = 5 MHz
        num_exp= 1
    # x1, y1,fov1, area1, UAV1 
        BS= basestation(P_T, num_users, R, rmin, bw1)
        BS.setup_actions()
    
        num_trials= num_trials1
        slsqp_iter=1
        trust_iter=1 
        dqn_iter= 500
    
        lr= 0.0001
        gamma=0.99
        eps_ini= 0.6  #0.6
        eps_dec= 1e-3
        eps_min= 0.10
        target_replace=200 
    
        agent = DDQNAgent(gamma= gamma, epsilon= eps_ini, lr=lr, input_dims=(BS.state.shape),n_actions= BS.n_actions, mem_size1= mem_size1,mem_size2= mem_size2, eps_min= eps_min, batch_size= batch_size1, replace= target_replace, eps_dec=eps_dec)
        self.user_distances= np.zeros((num_trials,num_users ))
        
        BS.setup_actions()
        bk= np.zeros(BS.N)
        self.res_dqn=[]
        self.res_slsqp=[]
        self.res_trust=[]
        self.steps_dqn=[]
        print('generate preloadong data')
  #gen_pre_data(self, num_tests, iter_per_test, input_shape):      
        BS.gen_pre_data(num_tests, iter_per_test )
        
        ls1= num_tests * iter_per_test
        for i in range(ls1):
            obs= BS.state_memory[i]
            a1= BS.action_memory[i]
            r1= BS.reward_memory[i]
            obs_= BS.new_state_memory[i]
            
            agent.store_transition(obs, a1, r1, obs_ )
        print('preloading done')

            
    
    
    
#        for lp in range(num_tests):
#            BS.reset()
#            for j in range(BS.N):
#                BS.Users[j].resetDR()
#            BS.gen_channels()
#            BS.setup_actions()
#            BS.computeN_G()
#            pi, pi_inv= BS.computePi()
#            BS.gen_initial_solution();
#            agent.epsilon= 1
#            iter0=0
#            
#            
#            for lp2 in range(iter_per_test):
#                obs= BS.get_state();
#                action= random.choice([i for i in range(agent.n_actions)])
#                reward=  BS.apply_action(action)
#                obs_= BS.get_state() 
#                agent.store_transition(obs, action, reward, obs_ )
#            print('Number of tests done: ', lp)
        for test in range(num_trials):
            agent.epsilon= eps_ini
            BS.reset()
            for j in range(BS.N):
                BS.Users[j].resetDR()
            BS.gen_channels()
            BS.computeN_G()
            pi, pi_inv= BS.computePi()
                
            BS.gen_initial_solution()
            BS.fL()
            BS.fU()
            #s1= BS.applySLSQP(slsqp_iter)
            self.res_slsqp.append(s1)
              #print('SQSLP done ', s1)
            s1=0
            print('s1 ', s1);
            
            BS.gen_initial_solution()
            #s2= BS.applytrust1(slsqp_iter)#(slsqp_iter)
            s2=0
            self.res_trust.append(s2)
            BS.gen_initial_solution()
            BS.fL()
            BS.fU()
        
            iter2=0
            bcost=-1
            cnt_actionA=0
            cnt_actionB=0
            no_change=0
            self.dqn_trace=[]
            best_iter=0;
            max_no_change= 10000
            s3=0
            if s1 >0 or s2>0:
                s3= max(s1,s2)
            while bcost < s3 or no_change < dqn_iter:
                obs= BS.get_state();
                action = agent.choose_action(obs)
                reward=  BS.apply_action(action)
                if agent.agent_action==1 and reward>0:
                    cnt_actionA+=1
                elif agent.agent_action==1: 
                    cnt_actionB+=1
            
                obs_= BS.get_state() 
                agent.store_transition(obs, action, reward, obs_ )
                agent.learn()
                cs1= BS.min_datarate;
                self.dqn_trace.append(cs1)
                if cs1 > bcost:
                    bcost= cs1
                    no_change=0
                    best_iter= iter2
                elif agent.epsilon==eps_min:
                    no_change+=1
                iter2+=1
                if iter2%100 == 0:
                    print('Test #: ', test,' iter # ',  iter2, ', ',bcost, ' no improvement since ', no_change, ' epsilon ', agent.epsilon, ' s3 ', s3 )
                if no_change> max_no_change:
                    break
            self.res_dqn.append(bcost)
            self.steps_dqn.append(best_iter)
            for j in range(num_users):
                self.user_distances[test,j]= BS.user_distances[j]
            if trace_en==True:
                rn2= random.choice([i for i in range(10000)])
                fn1= "DQNtrace"+ str(num_users) +"_"+ str(test)+".txt"    
                file1 = open(fn1,"w+") 
                for x in self.dqn_trace:
                    str2= str(x)+  '\n'
                    file1.write(str2)
                file1.close()
                
            if test%10==0:
                nusers= num_users
                fn1= "resSLSQP"+ str(num_users) + ".txt"    
                file1 = open(fn1,"w+") 
                for x in self.res_slsqp:
                    str2= str(x)+  '\n'
                    file1.write(str2)
                file1.close()
                fn1= "restrust"+ str(num_users) + ".txt"    
        
                file1 = open(fn1,"w+") 
                for x in self.res_trust:
                    str2= str(x)+  '\n'
                    file1.write(str2)
                file1.close()
    
                fn1= "resDQN"+ str(nusers) + ".txt"    
        
                file1 = open(fn1,"w+") 
                for x in self.res_dqn:
                    str2= str(x)+  '\n'
                    file1.write(str2)
                file1.close()
                fn1= "stepsDQN"+ str(nusers) + ".txt"    
        
                file1 = open(fn1,"w+") 
                for x in self.steps_dqn:
                    str2= str(x)+  '\n'
                    file1.write(str2)
                file1.close()
                fn1= "distances"+ str(nusers) + ".csv"    
    
                np.savetxt(fn1, self.user_distances)
     


