# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:01:42 2020

@author: ufarooq
"""
import sys
from pathlib import Path
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy import save

from execute1 import runexp
#num_trials1

if __name__=='__main__':
    #self, lx1, ly1, h1, delta1, A1, B1, R1, angle_half_power1
    P_T= 10
    num_users=8
    train_num_test= 5
    train_iter_per_test= 10
    num_trials=0
    R= 1000  # radius of the cell in m
    rmin=35
    num_tests= 200
    BW= 5
    ex1= runexp(num_users, train_num_test, train_iter_per_test, num_tests, rmin, R, BW);
    nusers= num_users
    fn1= "resSLSQP"+ str(nusers) + ".txt"    
    file1 = open(fn1,"w+") 
    for x in ex1.res_slsqp:
        str2= str(x)+  '\n'
        file1.write(str2)
    file1.close()
    fn1= "restrust"+ str(nusers) + ".txt"    
        
    file1 = open(fn1,"w+") 
    for x in ex1.res_trust:
        str2= str(x)+  '\n'
        file1.write(str2)
    file1.close()
    
    fn1= "resDQN"+ str(nusers) + ".txt"    
        
    file1 = open(fn1,"w+") 
    for x in ex1.res_dqn:
        str2= str(x)+  '\n'
        file1.write(str2)
    file1.close()
    fn1= "stepsDQN"+ str(nusers) + ".txt"    
        
    file1 = open(fn1,"w+") 
    for x in ex1.steps_dqn:
        str2= str(x)+  '\n'
        file1.write(str2)
    file1.close()
    fn1= "distances"+ str(nusers) + ".csv"    
    
    np.savetxt(fn1, ex1.user_distances)
    
      
  