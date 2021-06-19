import numpy as np
import torch as T
from DQN import  DeepQNetwork
from Buffer import ReplayBuffer

class DDQNAgent():
    
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size1,mem_size2, batch_size, eps_min= 0.01, eps_dec= 5e-7, replace= 1000):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr= lr
        self.n_actions= n_actions
        self.input_dims= input_dims
        self.batch_size= batch_size
        self.eps_min= eps_min
        self.eps_dec= eps_dec
        self.replace_target_cnt= replace
       
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter= 0
    #    self.hidden_units= hidden_units
        self.memory = ReplayBuffer(mem_size1,mem_size2, input_dims, n_actions)
        self.q_eval= DeepQNetwork(self.lr, self.n_actions, input_dims= self.input_dims, type1= 'EVAL')
        self.q_next= DeepQNetwork(self.lr, self.n_actions, input_dims= self.input_dims, type1= 'NEXT') 
        print(sum([p.numel() for p in model.parameters()]))
        #self.q_eval.s 
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            self.agent_action=1
            #print('Choosing action using forward')
            state= T.tensor([observation],dtype= T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action= T.argmax(actions).item()
        else:
            self.agent_action=0
            #print('part 2')
            action= np.random.choice([i for i in range(self.n_actions)])
            #print('action ', action, self.n_actions, ' ', self.action_space)
            
        return action
    
    def choose_actionTest(self, observation):
        state= T.tensor([observation],dtype= T.float).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        action= T.argmax(actions).item()
        return action
            
    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)
        
    def sample_memory(self):
        state, action, reward, new_state= self.memory.sample_buffer(self.batch_size)
        states= T.tensor(state).to(self.q_eval.device)
        rewards= T.tensor(reward).to(self.q_eval.device)
        actions= T.tensor(action).to(self.q_eval.device)
        states_= T.tensor(new_state).to(self.q_eval.device)
        return states, actions, rewards, states_
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt ==0:
        #    print('replacing target network')
            self.q_next.load_state_dict(self.q_eval.state_dict())
        
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_ = self.sample_memory()
        indices = np.arange(self.batch_size)
        q_pred= self.q_eval.forward(states)[indices, actions]
        q_next= self.q_next.forward(states_)
        q_eval= self.q_eval.forward(states_)
        max_actions = T.argmax(q_eval, dim=1)
        q_target= rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter +=1
        self.decrement_epsilon()
        
        
        
        
        
        
        
        
        
        
        
        