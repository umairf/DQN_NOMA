# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:25:53 2020

@author: Umair
"""

import numpy as np
#import random
import math
from numpy import sqrt
import scipy 
import random
from itertools import combinations
from scipy import optimize
#from mystic.models import rosen as my_model
#from mystic.termination import ChangeOverGeneration
#from mystic.solvers import NelderMeadSimplexSolver
#from mystic.solvers import DifferentialEvolutionSolver2
#from mystic.solvers import fmin_powell
#from mystic.tools import random_seed
#from mystic.solvers import NelderMeadSimplexSolver   
#from mystic.termination import CandidateRelativeTolerance as CRT
#from mystic.monitors import VerboseMonitor, Monitor
from scipy.optimize import minimize

#from mystic.termination import VTR

#from mystic.strategy import Best1Exp


#from mystic.tools import getch, random_seed

#from mystic.math import poly1d

import pylab

pylab.ion()

# draw the plot

def plot_frame(label=None):
    pylab.close()
    pylab.title("Objective function convergence")
    pylab.xlabel("Differential Evolution %s" % label)
    pylab.ylabel("Data-rate (Mbps)")
    pylab.draw()
# plot the polynomial trajectories
def plot_params(monitor):
    x = range(len(monitor))
    y = monitor.y
    pylab.plot(x,y,'b-')
    pylab.axis([1,0.5*x[-1],0,y[1]],'k-')
    pylab.draw()
    return


class user:
    def __init__(self, distance1, beta1):
        self.beta= beta1
        self.data_rate=0;
        self.snr1= 0;
        self.h=0
        self.distance=distance1
        self.Noise=0;
        self.N_G=0
        self.drates=[]
        self.reductionpossible=0
        self.increasepossible=0
        self.lower=-1
        self.upper=-1
        
    def computeavgdrate(self):
        ln1= len(self.drates)
        self.data_rate= sum(self.drates)/ln1 

    def resetDR(self):
        self.drates=[]


class basestation:
    def __init__(self, P_tot, num_users, R1, rmin1, bw1  ):
        self.P_T= P_tot
        self.N= num_users
        self.user_distances= [-1 for i in range(self.N)]
        self.user_channel_gains= [-1 for i in range(self.N)]
        self.B = bw1*10**6
        self.R= R1
        self.upper= np.zeros(self.N, dtype=float)
        self.lower= np.zeros(self.N, dtype=float)
        self.tol= 0.01
        
        self.rmin= rmin1
        Noise_Hz = 10**((-174-30)/10)
        self.pi = np.zeros(self.N,dtype=np.int8)
        self.pi_inv = np.zeros(self.N,dtype=np.int8)
        self.min_datarate=-1
        #self.delta= 0.0001       
        self.delta_large= 0.1/self.P_T
        self.delta_small= 0.01/self.P_T
        self.user_distances= self.rmin + math.sqrt(self.R**2-self.rmin**2)*np.sqrt(np.random.rand(self.N))
        self.psi= 1
        self.penalty= 1000
        self.Beta= 0.005
        self.state= np.zeros((4,self.N))
        self.reward= -1
        beta1= 1/self.N
        self.Users=[]
        self.pwr_coeff= np.zeros(self.N, dtype=float)
        self.iter=0
        self.cost_trials= 1000 #300
        for i in range(self.N):
            self.Users.append(user(self.user_distances[i],beta1 ))
        #user self.pi insteadd    
       # self.users_order= np.zeros(self.N, dtype=int)
        self.Noise = np.ones(self.N)*Noise_Hz*self.B
        for i in range(self.N):
            self.Users[i].Noise= self.Noise[i]
        
    def reset(self):
        self.user_distances= self.rmin + math.sqrt(self.R**2-self.rmin**2)*np.sqrt(np.random.rand(self.N))
        #mdd= min(self.user_distances)
        #mxx= max(self.user_distances)
       # print('Min distance ', mdd, ' max distance ', mxx)

        
        self.Users=[]
        self.pwr_coeff= np.zeros(self.N, dtype=float)
        self.iter=0
        beta1= 1/self.N
        Noise_Hz = 10**((-174-30)/10)
        
        self.cost_trials= 2000 #300
        for i in range(self.N):
            self.Users.append(user(self.user_distances[i],beta1 ))
        self.Noise = np.ones(self.N)*Noise_Hz*self.B
        for i in range(self.N):
            self.Users[i].Noise= self.Noise[i]
        self.pwr_coeff= np.zeros(self.N, dtype=float)
 
        
        

        
  #  def gen_rand_user_dist(self, R, rmin):# R is the radius of the cell and rmin the minimu distance of any user from the BS
  #      self.user_distances=  rmin + math.sqrt(R**2-rmin**2)*np.sqrt(np.random.rand(self.N))
                
    def gen_channels(self):
        rayleigh = np.random.randn(self.N,1) + 1j*np.random.randn(self.N,1)
        noise= np.zeros(self.N, dtype=float)
        gains_t= np.zeros(self.N, dtype=float)
        noise_mean=0
        noise_std= 1
        #path_loss= np.zeros(self.N, dtype=float)
        #for i in range(self.N):
        #    path_loss[i]= -(128.1+37.6*np.log10(self.Users[i]./1000))
        
        path_loss = -(128.1+37.6*np.log10(self.user_distances/1000))   # path loss model : BUdistance/1000 in km
        path_loss =  np.power(10,(path_loss/10))               # dB to scalar
        #print('path_loss ', path_loss)
        shadowing = -10*np.random.randn(self.N,1)          # lognormal distributed with SD 10
        shadowing = np.power(10,(shadowing/10))       # dB to scalar
        self.user_channel_gains =  np.array([[ path_loss[n] * np.power(np.absolute(rayleigh[n]),2) * shadowing[n]] for n in range(self.N)])
        #for i in range(self.N):
        #    gains_t[i] =sqrt(random.gauss(0,1)**2+random.gauss(0,1)**2)/sqrt(2) * pow(self.user_distances[i],-1)
        #for i in range(self.N):
        #    noise[i]= random.gauss(noise_mean, noise_std)
            
        
        #print('Channel gains')
        for i in range(self.N):
            self.Users[i].h= self.user_channel_gains[i]
        #    print(self.user_distances[i], '-',   self.Users[i].distance, '-', self.Users[i].h)
        
        
        #print(self.user_channel_gains)
                
                
            
#    def sort_users(self):
#        gain1= np.zeros(self.N, dtype=np.double)
#        for i in range(self.N):
#            gain1[i]= pow(self.user_channel_gains[i],2)
#        self.users_order= np.argsort(-gain1)
#        print(self.users_order)
#        print(self.pi)
            
    def gen_initial_solution(self):
        for i in range(self.N):
            self.Users[i].beta=0;
        total1= 0
        nn1= self.N +1
        for i in range(1,nn1):
            total1 = total1 + i
        part1= 1/total1;
        for i in range(self.N):
         #   user_index= self.pi[i]
            self.pwr_coeff[i]= (i+1)* part1
            #self.Users[user_index].beta= (i+1)* part1
            
        #print('solutiion ')
        #for i in range(self.N):
        #    user_id= self.pi[i]
        #    print(self.Users[user_id].beta)
        
        
        
    def computeN_G(self):
        self.N_G= np.zeros(self.N,dtype=np.double )
        for i in range(self.N):
            self.N_G[i]= -1 * self.user_channel_gains[i] #self.Noise[i] / self.user_channel_gains[i]
#            self.Users[i].computeN_G()
        return self.N_G

    def computePi(self):
        self.pi = np.zeros(self.N,dtype=np.int8)
        self.pi_inv = np.zeros(self.N,dtype=np.int8)
        #self.G_N = np.array([[self.user_channel_gains[i]/self.Noise[i] for i in range(self.N)]  ])
        
        self.G_N= np.zeros(self.N,dtype=np.double )
        for i in range(self.N):
            self.G_N[i]= self.user_channel_gains[i] #/ self.Noise[i]
        #self.G_N= np.divide(self.user_channel_gains, self.Noise)
       # self.G_N = np.array([[ self.gen_channels[i]/self.Noise[i] for i in range(self.N)]]) 
        self.pi= np.argsort(self.N_G)
   #     print('after arg sorting G_N')
   #     print(G_N_argsorted)
        
          
        for k in range(self.N):
            self.pi_inv[self.pi[k]] = k

        return self.pi,self.pi_inv

    def X2P(self, x):
        self.p = np.zeros(self.N)
        # p_{pi^n(i)} = x_i^l - x_{i+1}^l
        for i in range(self.N-1):
            self.p[self.pi[i]] = x[i] - x[i+1]

        # Last element 

        self.p[self.pi[self.N-1]] = x[self.N-1]

        return self.p

    def randomPower(self):
        self.p= np.random.random(self.N)
        for i in range(self.N):
            self.Users[i].beta= 1/self.N
        
        for i in range(self.N):
            v1= random.sample([i for i in range(self.N)], 2)
            b1= self.Users[v1[0]].beta
            b2= self.Users[v1[1]].beta
            b11= b1 + 0.1
            b22= b2 - 0.1
            if b11 <= 1 and b22>= 0:
                self.Users[v1[0]].beta= self.Users[v1[0]].beta+0.1
                self.Users[v1[1]].beta= self.Users[v1[1]].beta-0.1
            
            
        
    

   # def get_state(self):
   #     state_size= self.N * 
   #     self.state= np.zeros(self.N, type=float)
        
    def computelowerbound1(self, userindex):
        user_id= self.pi[userindex]
        after_users=[]
        before_users=[]
        ui= userindex+1
        for i in range(ui, self.N):
            ind1= self.pi[i]
            before_users.append(ind1)
        for i in range(userindex):
            ind1= self.pi[i]
            after_users.append(ind1)
        #print('P_T ', self.P_T)
        sum_beta1=0;
        sum_beta2=0
        
        for ui1 in before_users:
            sum_beta1= sum_beta1 + self.Users[ui1].beta 
        self.Users[user_id].lower= sum_beta1 + self.delta
        
    def getviolations(self):
        vio=0;
        nn1= self.N-1
        for i in range(0,self.N-1):
            df1= self.pwr_coeff[i+1]- self.pwr_coeff[i]
            if df1 <= 0:
                vio=vio+1
            
        sv1= sum(self.pwr_coeff)
        if sv1 > 1.001:
            vio = vio + sv1
        elif sv1 < 0.99:
            vio = vio + (1-sv1)
        
        #vio2=0
        #for i in range(self.N):
        #    vio2= vio2+self.pwr_coeff[i]
        #if vio2 <= 1:
        #    vio2=0;
        #vio= vio + vio2    
        return vio
    
    def computelowerbounds(self):
        for i in range(self.N):
            self.computelowerbound1(i)
            print(i, self.Users[i].lower)
            
        for i in range(self.N):
            print(i,  self.pi[i], ' ', self.Users[i].h,' ',self.Users[i].beta, ' ',     self.Users[i].lower)
    
#    def fL(self):
#        n1= self.N-1
#        for vi in self.pi:
#            if vi==0:
#                if self.Users[vi].beta==0:
#                    self.Users[vi].lower=0
#                else:
#                    self.Users[vi].lower= 1
#            else:
#                uv1= self.Users[vi].beta - self.delta
#                vii= vi-1
#                if uv1 > self.Users[vii].beta:
#                    self.Users[vi].lower= 1
#                else:
#                    self.Users[vi].lower=0
       
    
#    def fU(self):
#        n1= self.N-1
#        for vi in self.pi:
#            if vi==n1:
#                if self.Users[vi].beta ==1:
#                    self.Users[vi].upper=0
#                else:
#                    self.Users[vi].upper= 1
#            else:
#                vii= vi+1
#                df1= self.Users[vii].beta- self.Users[vi].beta
#                if df1 >= self.delta:
#                    self.Users[vi].upper=1
#                else:
#                    self.Users[vi].upper=0
    def fU(self):
        n1= self.N-1
        for i in range(self.N):
            if i==n1:
                if self.pwr_coeff[i] ==1:
                    self.upper[i]=0
                else:
                    self.upper[i]= 1
            else:
                ii= i+1
                df1= self.pwr_coeff[ii]- self.pwr_coeff[i]
                if df1 >= self.delta_large:
                    self.upper[i]= 1
                elif df1 >= self.delta_small:
                    self.upper[i]= 0.5
                else:
                    self.upper[i]=0

    def fL(self):
        n1= self.N-1
        for i in range(self.N):
            if i==0:
                if self.pwr_coeff[i]==0:
                    self.lower[i]=0
                else:
                    self.lower[i]= 1
            else:
                uv1= self.pwr_coeff[i] - self.delta_large
                uv2= self.pwr_coeff[i] - self.delta_small
                ii= i-1
                if uv1 > self.pwr_coeff[ii]:
                    self.lower[i]= 1
                elif uv2 > self.pwr_coeff[ii]:
                    self.lower[i]= 0.5
                else:
                    self.lower[i]=0


    def get_state(self):
        self.state= np.zeros((4,self.N))
        for i in range(self.N):
            self.state[0,i]= self.pwr_coeff[i]
        for i in range(self.N):
            self.state[1,i]= self.Users[i].data_rate;
        for i in range(self.N):
            self.state[2,i]= self.lower[i];
        for i in range(self.N):
            self.state[3,i]= self.upper[i];
        self.state= self.state
        return self.state
            
    def setup_actions(self):
        ulist= [i for i in range(self.N)]
        ucom1= list(combinations(ulist, 2))
        self.n_actions= (len(ucom1) * 4)
        
        self.actions_space= []
        
        for u,v in ucom1:
            self.actions_space.append((u,v, self.delta_large))
            self.actions_space.append((v,u, self.delta_large))
            self.actions_space.append((u,v, self.delta_small))
            self.actions_space.append((v,u, self.delta_small))
        
       # sz1= self.actions_space.count
       # print('as size ', sz1, ' n_actions ', self.n_actions)
            
 
    def apply_action(self, action_id):
        self.iter +=1
        #self.compute_DR_manytrials(10)
        self.fL()
        self.fU()
        cs1= self.min_datarate; #   getminrate()
        #vio1= self.getviolations();
        cs1= cs1 #- (self.penalty * vio1)
        self.reward=0
        #action_id=action_id
        sz1= len(self.actions_space)
      #  print('action id ', action_id, ' and n_actions ', self.n_actions, ' # of actions ', sz1 )
       # if action_id == self.n_actions-1:
       #     self.reward= 0 #-1 * self.delta_large
       #     return self.reward
       # else:
        act11= self.actions_space[action_id]
        incr_user= act11[0]
        dec_user= act11[1]
        delta_value= act11[2]
            #print('here 1')
        dec_v= self.pwr_coeff[dec_user] - delta_value;
        incr_v= self.pwr_coeff[incr_user] + delta_value
            
        bk_v1= self.pwr_coeff[dec_user]
        bk_v2= self.pwr_coeff[incr_user]
        self.pwr_coeff[dec_user]=  dec_v
        self.pwr_coeff[incr_user]= incr_v
            
        fs1=  self.optimconsA(self.pwr_coeff)
        fs2= self.optimconsB(self.pwr_coeff)
            
        self.pwr_coeff[dec_user]=  bk_v1
        self.pwr_coeff[incr_user]=  bk_v2
        if fs1 >0 or fs2 > 0:
            self.reward= 0;
            return self.reward
        else:
            self.pwr_coeff[dec_user] -=delta_value 
            self.pwr_coeff[incr_user] +=delta_value
            
            self.compute_DR_manytrials()
            self.fL()
            self.fU()
            cs2= self.getminrate()
            csum= cs2+cs1
            if csum> 0:
                csum2= (cs2-cs1)/csum
                self.reward= csum2;
            else:
                self.reward=0
            return(self.reward)
     
      
#    def computedatarate(self, userindex):
#        user_id= self.pi[userindex]
#        after_users=[]
#        before_users=[]
#        ui= userindex+1
#        for i in range(ui, self.N):
#            ind1= self.pi[i]
#            before_users.append(ind1)
#        for i in range(userindex):
#            ind1= self.pi[i]
#            after_users.append(ind1)
#        
#        #SIC error propagation interference
#        it1=0
#        #print('P_T ', self.P_T)
#        for ui1 in before_users:
#            d1= self.Users[ui1].beta * self.P_T * self.Users[user_id].h 
#            it1= it1 + d1
#        it1 = self.Beta * it1;
#        it2=0
#        for ui1 in after_users:
#            d1= self.Users[ui1].beta * self.P_T * self.Users[user_id].h
#            it2= it2 + d1
#        it3= self.Users[user_id].Noise + it1 + it2
#        signal1= self.Users[user_id].beta * self.P_T * self.Users[user_id].h
#        sn1= signal1 /it3
#        dr1= self.B * math.log2(1+ sn1)
#        dr1= dr1/1e6
#        self.Users[user_id].drates.append(dr1)
#        #self.Users[user_id].data_rate= dr1/1e6
#        #print('sinr ', sn1, ' dr1 ', dr1)
 
    def computedatarate(self, userindex):
        user_id= self.pi[userindex]
        after_users=[]
        before_users=[]
        ui= userindex+1
        for i in range(ui, self.N):
            ind1= self.pi[i]
            before_users.append(ind1)
        for i in range(userindex):
            ind1= self.pi[i]
            after_users.append(ind1)
        
        #SIC error propagation interference
        it1=0
#        print('P_T ', self.P_T)
        for ui1 in range(userindex):
            d1= self.pwr_coeff[ui1] * self.P_T * self.Users[user_id].h 
            it1= it1 + d1
        it1 = self.Beta * it1;
        it2=0
        udx= userindex+1
        for ui1 in range(udx, self.N):
            d1= self.pwr_coeff[ui1] * self.P_T * self.Users[user_id].h
            it2= it2 + d1
        it3= self.Users[user_id].Noise + it1 + it2
        signal1=  self.pwr_coeff[user_id]* self.P_T * self.Users[user_id].h
        sn1= signal1 /it3
        dr1=0
        if sn1 >0:
            dr1= self.B * math.log2(1+ sn1)
        dr1= dr1/1e6
        self.Users[user_id].drates.append(dr1)
        #self.Users[user_id].data_rate= dr1/1e6
        #print('sinr ', sn1, ' dr1 ', dr1)
    
    def compute_DR_manytrials(self):
        for i in range(self.N):
            self.Users[i].drates=[]
            self.Users[i].data_rate=0;
        for i in range(self.cost_trials):
            self.gen_channels()
            self.computeN_G()
            self.computePi()
            for j in range(self.N):
                self.computedatarate(j)
        for i in range(self.N):
            s1=0;
            cn=0
            for v1 in self.Users[i].drates:
                s1= s1 + v1
                cn= cn+1
            s1= s1 / cn
            self.Users[i].data_rate= s1
        
    def getminrate(self):
        min1= 1e6
        for i in range(self.N):
            if self.Users[i].data_rate < min1:
                min1= self.Users[i].data_rate
        self.min_datarate= min1
        return self.min_datarate
    
    def optimfn2(self, x):
        
        for i in range(self.N):
            self.pwr_coeff[i]= x[i]
        
        self.compute_DR_manytrials()
        self.fL()
        self.fU()
        cs2= self.getminrate()
        vio1= self.getviolations()
        cs3=   -1*(cs2 - (self.penalty*vio1))
        return cs2
    
    
    
    def optimfn(self, x):
        
        for i in range(self.N):
            self.pwr_coeff[i]= x[i]
        
        self.compute_DR_manytrials()
        self.fL()
        self.fU()
        cs2=  1 * self.getminrate()
        
        #vio1= self.getviolations()
        #cs3=   -1*(cs2 - (self.penalty*vio1))
        return cs2
    def optimcons(self, x):
        for i in range(self.N):
            self.pwr_coeff[i]= x[i]
        #self.fL()
        #self.fU()
        vio1= self.getviolations()
        return vio1
    
    def optimconsA(self, x):
        n1= self.N-1
        vio=0
        for i in range(n1):
            df1= x[i+1]-x[i]
            if df1 <=0:
                vio+=1
        return vio    
    def optimconsB(self, x):
        vio=0
        sum1= sum(x)
        df1= 1- sum1
        return(df1)
                

    def optimcons1(self, x):
        return(x[1] - x[0]-0.0001)
    def optimcons2(self, x):
        return(x[2] - x[1]-0.0001)
    def optimcons3(self, x):
        return(x[3] - x[2]-0.0001)
    def optimcons4(self, x):
        return(x[4] - x[3]-0.0001)
    def optimcons5(self, x):
        return(x[5] - x[4]-0.0001)
    def optimcons6(self, x):
        return(x[6] - x[5]-0.0001)
    def optimcons7(self, x):
        return(x[7] - x[6]-0.0001)
    def optimcons8(self, x):
        return(x[8] - x[7]-0.0001)
        
    
        
        
    def optimconsF3(self, x):
        return (x[0]+x[1]+x[2]-1)
    
    def optimconsF4(self, x):
        return (x[0]+x[1]+x[2]+x[3]-1)
    def optimconsF5(self, x):
        return (x[0]+x[1]+x[2]+x[3]+x[4]-1)
    def optimconsF6(self, x):
        return (x[0]+x[1]+x[2]+x[3]+x[4]+x[5]-1)
    def optimconsF7(self, x):
        return (x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]-1)
    def optimconsF8(self, x):
        return (x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]-1)
    
        
        
        #self.fL()
        #self.fU()
     #   vio1= self.getviolations()
     #   return vio1


    def applySLSQP(self, iters):
        # optimize
        b = (0,1.0)
        bnds = [(0,1) for i in range(self.N)]
        if self.N == 3:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            conF = {'type': 'eq', 'fun': self.optimconsF3}
            cons= ([con1,con2, conF])
        elif self.N==4:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            conF = {'type': 'eq', 'fun': self.optimconsF4}
            cons= ([con1,con2, con3,conF])
        elif self.N==5:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            con4 = {'type': 'ineq', 'fun': self.optimcons4}
            conF = {'type': 'eq', 'fun': self.optimconsF5}
            cons= ([con1,con2, con3, con4, conF])
        elif self.N==6:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            con4 = {'type': 'ineq', 'fun': self.optimcons4}
            con5 = {'type': 'ineq', 'fun': self.optimcons5}
            
            conF = {'type': 'eq', 'fun': self.optimconsF6}
            cons= ([con1,con2, con3, con4, con5, conF])

        elif self.N==8:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            con4 = {'type': 'ineq', 'fun': self.optimcons4}
            con5 = {'type': 'ineq', 'fun': self.optimcons5}
            con6 = {'type': 'ineq', 'fun': self.optimcons6}
            con7 = {'type': 'ineq', 'fun': self.optimcons7}
            
            conF = {'type': 'eq', 'fun': self.optimconsF8}
            cons= ([con1,con2, con3, con4, con5,con6, con7, conF])

        success1= False
        
        #while success1==False:
        s_res=[]
        for j in range(iters):
            self.gen_initial_solution()
            x0= np.zeros(self.N)
            for k in range(self.N):
                x0[k]= self.pwr_coeff[k]
            
            #x0 = [random.uniform(0.0,1) for i in range(self.N)]
            solution = minimize(self.optimfn ,x0,method='SLSQP',options={'maxiter': 20000, 'disp':True},  bounds=bnds,constraints=cons)
            x = solution.x
            success1= solution.success
            if success1==True:
                v1= self.optimfn(x)
                v1= v1 *1
                s_res.append(v1)
            print(x)
            #print(success1)
        sres= 0
        if len(s_res)> 0:
            sres= max(s_res)
        
        return sres
        
    def applytrust1(self, iters):
        # optimize
        b = (0,1.0)
        bnds = [(0,1) for i in range(self.N)]
        if self.N == 3:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            conF = {'type': 'eq', 'fun': self.optimconsF3}
            cons= ([con1,con2, conF])
        elif self.N==4:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            conF = {'type': 'eq', 'fun': self.optimconsF4}
            cons= ([con1,con2, con3,conF])
        elif self.N==5:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            con4 = {'type': 'ineq', 'fun': self.optimcons4}
            conF = {'type': 'eq', 'fun': self.optimconsF5}
            cons= ([con1,con2, con3, con4, conF])
        elif self.N==6:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            con4 = {'type': 'ineq', 'fun': self.optimcons4}
            con5 = {'type': 'ineq', 'fun': self.optimcons5}
            
            conF = {'type': 'eq', 'fun': self.optimconsF6}
            cons= ([con1,con2, con3, con4, con5, conF])
        elif self.N==8:
            con1 = {'type': 'ineq', 'fun': self.optimcons1}
            con2 = {'type': 'ineq', 'fun': self.optimcons2}
            con3 = {'type': 'ineq', 'fun': self.optimcons3}
            con4 = {'type': 'ineq', 'fun': self.optimcons4}
            con5 = {'type': 'ineq', 'fun': self.optimcons5}
            con6 = {'type': 'ineq', 'fun': self.optimcons6}
            con7 = {'type': 'ineq', 'fun': self.optimcons7}
            
            conF = {'type': 'eq', 'fun': self.optimconsF8}
            cons= ([con1,con2, con3, con4, con5,con6, con7, conF])


        success1= False
        
        #while success1==False:
        s_res=[]
        for j in range(iters):
            print('trust-iter ', j)
            self.gen_initial_solution()
            x0= np.zeros(self.N)
            for k in range(self.N):
                x0[k]= self.pwr_coeff[k]
            
            #x0 = [random.uniform(0.0,1) for i in range(self.N)]
            #sol1 = minimize(self.optimfn ,x0,method='trust-constr',options={'maxiter': 200},  bounds=bnds,constraints=cons)
            #sol2 = minimize(self.optimfn ,x0,method='trust-constr',options={'maxiter': 1000},  bounds=bnds,constraints=cons)
            #c0= self.optimfn(sol1.x)
            #c1= self.optimfn(sol2.x)
            #print('c0 ', c0, ' c1 ', c1)
                
            solution = minimize(self.optimfn ,x0,method='trust-constr',options={'maxiter': 20000, 'disp': True},  bounds=bnds,constraints=cons)
            x = solution.x
            print(x)
            #print(self.optimfn(x))
            success1= solution.success
            if success1==True:
                v1= self.optimfn(x)
                v1= v1 *1
                s_res.append(v1)
            #print(x)
            #print(success1)
        sres= 0
        if len(s_res)> 0:
            sres= max(s_res)

        #sres= max(s_res)
        return sres
        




#    def solveoptim(self):
#        cons = [{'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] -1 }]
#       # {'type': 'ineq', 'fun': lambda x: x[0] < x[2]}]
#       # {'type': 'ineq', 'fun': lambda x: x[2] < x[3]},
#       # {'type': 'ineq', 'fun': lambda x: x[3] < x[4]},
#       # {'type': 'ineq', 'fun': lambda x: x[4] < x[5]},
#       # {'type': 'ineq', 'fun': lambda x: x[5] < x[6]},
#       # {'type': 'ineq', 'fun': lambda x: x[6] < x[7]},
#       # {'type': 'ineq', 'fun': lambda x: x[7] < x[8]},
#       # {'type': 'ineq', 'fun': lambda x: x[8] < x[9]}]
#        
#       
#       #scipy optimiozation
#       
#       
#       
#       
#        bnds = [(0, 1) for i in range(self.N)] 
#        
#        lb = [0.0, 0.0, 0.0, 0.0] 
#        ub = [1.0, 1.0, 1.0, 1.0]
#        x0 = [random.uniform(0.0,1) for i in range(self.N)]
#        
#        # optimize
#        b = (0,1.0)
#        bnds = ((0,1), (0.0,1), (0.0, 1.0), (0.0, 1.0))
#        con1 = {'type': 'ineq', 'fun': self.optimcons1}
#        con2 = {'type': 'ineq', 'fun': self.optimcons2}
#        con3 = {'type': 'ineq', 'fun': self.optimcons3}
#        
#        con4 = {'type': 'eq', 'fun': self.optimcons4}
#      #  con2 = {'type': 'eq', 'fun': constraint2}
#        cons = ([con1,con2, con3, con4])
#        success1= False
#        
#        #while success1==False:
#        s_res=[]
#        for j in range(10):
#            self.gen_initial_solution()
#            x0= np.zeros(self.N)
#            for k in range(self.N):
#                x0[k]= self.pwr_coeff[k]
#            
#            #x0 = [random.uniform(0.0,1) for i in range(self.N)]
#            solution = minimize(self.optimfn ,x0,method='SLSQP',options={'maxiter': 5000},  bounds=bnds,constraints=cons)
#            x = solution.x
#            success1= solution.success
#            if success1==True:
#                v1= self.optimfn(x)
#                s_res.append(v1)
#            #print(x)
#            print(success1)
#        #solution.
#
## show final objective
#        print('Final Objective: ' + str(self.optimfn(x)))
#         
#        
#        
#        
#        ndim= self.N
#        npop= 10
#        stepmon = VerboseMonitor(50)
#        evalmon = Monitor()
#        
#        x0 = [random.uniform(0.0,1) for i in range(ndim)]
#        
#        solver = NelderMeadSimplexSolver(ndim)
#        solver.SetInitialPoints(x0)
#        solver.SetEvaluationLimits(generations=1000)
#        solver.SetEvaluationMonitor(evalmon)
#        solver.SetGenerationMonitor(stepmon)
#        solver.SetStrictRanges(lb,ub)
#  #      solver.SetConstraints(self.optimcons)
#  #      solver.enable_signal_handler()
#        solver.Solve(self.optimfn, termination=CRT(1e-4,1e-4))
#        #solver.SetConstraints         
#        solution1 = solver.bestSolution
#
#        
#    
#  #      print(solution, ' ', solver.generations)
#        
#  #      solver.solution_history
#        solver = DifferentialEvolutionSolver2(ndim,npop)
#        solver.SetStrictRanges(lb,ub)
#        solver.SetRandomInitialPoints(min=[0.0]*ndim, max=[1]*ndim)
#        solver.SetEvaluationLimits(generations=2000)
#        solver.SetEvaluationMonitor(evalmon)
#        solver.SetGenerationMonitor(stepmon)
#        #solver.SetConstraints(self.optimcons)
#        #solver.enable_signal_handler()
#        #plot_frame('iterations')
#        #plot_params(stepmon)
#        #getch()
##        solver.SetGenerationMonitor(stepmon)
#        solver.Solve(self.optimfn, strategy=Best1Exp, CrossProbability=1.0, ScalingFactor=0.9)
#        solution = solver.Solution()
#        #plot_frame('iterations')
#        #plot_params(stepmon)
#        #getch()
#        solver.evaluations
#        print(solution) 
#        solver.energy_history
#        solver.solution_history
##        COG = ChangeOverGeneration()
##        solver = NelderMeadSimplexSolver(self.N) 
##        solver.SetRandomInitialPoints(lb, ub)
##        solver.SetStrictRanges(lb, ub) 
##        #solver.SetConstraints(constrain) 
##        solver.Solve(self.optimfn, COG)
##        solution= solver.bestSolution
#
#        
#        #xinit= np.zeros(self.N); # = [(0, 1) for i in range(self.N)] 
#        #result = optimize.minimize(self.optimfn, x0= xinit) #, bounds=bnds, constraints=cons)
#        
#        return result
        
        
    def NormIplusN_DL(self):
    # Normalized interference plus noise vector
        v = np.zeros(self.N)
        for k in range(self.N):
            v[k] += self.N_G[k]
            rank = self.pi_inv[k]
            
            
    # Add the normalized interference from user pi[n][rank2] such that rank2 > rank  
            rkk= rank+1 
            if rkk < self.N:
                for rank2 in range(rkk,self.N):
                    k2 = self.pi[rank2]
                    v[k] += (self.p[k2]) # * self.user_channel_gains[k])
             
            #print('rank ', rank, ' v ', v[k])
           
                
            
        return v 
        
    def computeDR(self):
        v = self.NormIplusN_DL()
        print('SINRS ')
        #print(v)
        #print('vmin ', vmin)
        self.C = np.zeros(self.N)
        for u in range(self.N):
            s1= self.p[u]/v[u]
            print(' s1 ', s1)
            
            v1= self.B * math.log2(1+((self.p[u])/v[u] ))/1e6
            self.C[u] =v1
            self.Users[u].drates.append(v1)
            
        self.minR= min(self.C)
        return self.C
    
    def findminmax(self):
        drates1=[]
        for j in range(self.N):
            drates1.append(self.Users[j].data_rate)
            self.Users[j].computeavgdrate()
        drt1= min(drates1)
        self.minR= drt1
        return drt1

    def gen_pre_data(self, num_tests, iter_per_test):
        mm2= num_tests * iter_per_test
        input_shape= self.state.shape
        cntr=0
        self.state_memory = np.zeros((mm2, *input_shape), dtype= np.float32)
        self.new_state_memory = np.zeros((mm2, *input_shape), dtype= np.float32)
        self.action_memory = np.zeros(mm2, dtype= np.int64)
        self.reward_memory = np.zeros(mm2, dtype= np.float32)
        for i in range(num_tests):
            self.reset()
            for j in range(self.N):
                self.Users[j].resetDR()
            self.gen_channels()
            self.setup_actions()
            self.computeN_G()
            pi, pi_inv= self.computePi()
            self.gen_initial_solution();
            print('test # ', i)
            for j in range(iter_per_test):
                ar= random.choice( [k for k in range(self.n_actions)])
                st1= self.get_state();
                rw1= self.apply_action(ar)
                st2= self.get_state()
                
             #   print('ar ', ar, ' rw1 ', rw1)
                self.state_memory[cntr]= st1
                self.new_state_memory[cntr]= st2
                self.action_memory[cntr]= ar
                self.reward_memory[cntr]= rw1
                cntr+=1
            
                

