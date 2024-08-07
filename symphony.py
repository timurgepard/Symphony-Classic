import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import random
import torch.nn.functional as F
import torch.jit as jit

#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# random seeds
#r1, r2, r3 = 2, 0, 2
r1, r2, r3 = random.randint(0,10), random.randint(0,10), random.randint(0,10)
#r1, r2, r3 = (r1+r1_), (r2+r2_), (r3+r3_)
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)
random.seed(r3)

class LogFile(object):
    def __init__(self, log_name):
        self.log_name= log_name
    def write(self, text):
        with open(self.log_name, 'a+') as file:
            file.write(text)

log_name = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_file = LogFile(log_name)



#Rectified Huber Symmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHSE(jit.ScriptModule):
    def __init__(self):
        super(ReHSE, self).__init__()

    @jit.script_method
    def forward(self, y1, y2):
        ae = torch.abs(y1-y2)
        ae = ae*torch.tanh(ae)
        return ae.mean()


#Rectified Huber Asymmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHAE(jit.ScriptModule):
    def __init__(self):
        super(ReHAE, self).__init__()

    @jit.script_method
    def forward(self, y1, y2):
        e = (y1-y2)
        e = torch.abs(e)*torch.tanh(e)
        return e.mean()


#Rectified Huber Symmetric Error Fractional Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHSEF(jit.ScriptModule):
    def __init__(self):
        super(ReHSEF, self).__init__()
        self.p = 0.95

    @jit.script_method
    def forward(self, y1, y2):
        ae = torch.abs(y1-y2) + 1e-6
        ae = ae**self.p*torch.tanh(self.p*ae/3)
        return ae.mean()


#Rectified Huber Asymmetric Error Fractional Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHAEF(jit.ScriptModule):
    def __init__(self):
        super(ReHAEF, self).__init__()
        self.p = 0.95

    @jit.script_method
    def forward(self, y1, y2):
        e = (y1-y2) + 1e-6
        e = torch.abs(e)**self.p*torch.tanh(self.p*e/3)
        return e.mean()


#Inplace Dropout function created with the help of ChatGPT
# nn.Module -> JIT C++ graph
class InplaceDropout(jit.ScriptModule):
    def __init__(self, prob=0.5):
        super(InplaceDropout, self).__init__()
        self.prob = prob

    # It is not recommended to use JIT compilation decorator with online random generator as Symphony updates seeds each time
    # We did exception only for this module as it is used inside neural networks.
    @jit.script_method
    def forward(self, x):
        mask = (torch.rand_like(x) > self.prob).float()
        return  mask * x + (1.0-mask) * x.detach()

#Linear followed by Inplace Dropout
# nn.Module -> JIT C++ graph
class LinearIDropout(jit.ScriptModule):
    def __init__(self, f_in, f_out, prob=0.5):
        super(LinearIDropout, self).__init__()
        self.ffw = nn.Linear(f_in, f_out)
        self.prob = prob

    @jit.script_method
    def forward(self, x):
        x = self.ffw(x)
        mask = (torch.rand_like(x) > self.prob).float()
        return  mask * x + (1.0-mask) * x.detach()
    
#ReSine Activation Function
# nn.Module -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()
        stdev = math.sqrt(1/hidden_dim)
        noise = torch.normal(mean=torch.zeros(hidden_dim), std=stdev).clamp(-3.0*stdev,3.0*stdev)
        self.s = nn.Parameter(data=noise, requires_grad=True)
        

    @jit.script_method
    def forward(self, x):
        k = torch.sigmoid(0.1*self.s)
        x = k*torch.sin(x/k)
        return F.prelu(x, 0.1*k)




#Shared Feed Forward Module
# nn.Module -> JIT C++ graph
class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, f_out, prob=0.5):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 320),
            nn.LayerNorm(320),
            nn.Linear(320, 256),
            ReSine(256),
            nn.Linear(256, 192),
            LinearIDropout(192, f_out, prob),
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)


# nn.Module -> JIT C++ graph
class Actor(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action=1.0, prob=0.15):
        super(Actor, self).__init__()

        hidden_dim = 320
        
        self.inA = LinearIDropout(state_dim, hidden_dim, prob=0.15)
        self.inB = LinearIDropout(state_dim, hidden_dim, prob=0.15)
        self.inC = LinearIDropout(state_dim, hidden_dim, prob=0.15)
        
        
        self.ffw = FeedForward(3*hidden_dim, action_dim, prob=0.5)
        self.tanh = nn.Tanh()


        self.max_action = torch.mean(max_action).item()
        self.scale = 0.1*self.max_action
        self.lim = 3.0*self.scale
    
    @jit.script_method
    def forward(self, state):
        x = torch.cat([self.inA(state), self.inB(state), self.inC(state)], dim=-1)
        return self.max_action*self.tanh(self.ffw(x))
    
    # Do not use any decorators with online random generators (Symphony updates seed each time)
    def soft(self, state):
        x = self.forward(state)
        x += self.scale*torch.randn_like(x).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)

 

# nn.Module -> JIT C++ graph
class Critic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, prob=0.75):
        super(Critic, self).__init__()


        qA = FeedForward(state_dim+action_dim, 128, prob=0.75)
        qB = FeedForward(state_dim+action_dim, 128, prob=0.75)
        qC = FeedForward(state_dim+action_dim, 128, prob=0.75)

        self.nets = nn.ModuleList([qA, qB, qC])

    @jit.script_method
    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return [net(x) for net in self.nets]
    
    # take means of 3 distributions and concatenate them
    @jit.script_method
    def cmin(self, state, action):
        xs = self.forward(state, action)
        xs = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in xs], dim=-1)
        xs = torch.sort(xs, dim=-1).values
        return (0.73*xs[:,0]+0.21*xs[:,1]+0.06*xs[:,2]).unsqueeze(1)
    



# Define the algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, device, max_action=1.0, tau=0.005, prob_a=0.15, prob_c = 0.75, capacity=300000, batch_lim = 768, fade_factor=7.0):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, capacity, batch_lim, fade_factor)

        self.actor = Actor(state_dim, action_dim, max_action=max_action, prob=prob_a).to(device)

        self.critic = Critic(state_dim, action_dim, prob=prob_c).to(device)
        self.critic_target = Critic(state_dim, action_dim, prob=prob_c).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-4)

        self.rehsef = ReHSEF()
        self.rehaef = ReHAEF()


        self.max_action = max_action
        self.tau = tau
        self.tau_ = 1.0 - tau
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_next_old_policy = [0.0, 0.0, 0.0]
        self.weights =  torch.FloatTensor([0.06, 0.21, 0.73])
        self.scaler = torch.cuda.amp.GradScaler()
        


    def select_action(self, state, mean=False):
        state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
        with torch.no_grad(): action = self.actor(state) if mean else self.actor.soft(state)
        return action.cpu().data.numpy().flatten()



    def train(self, tr_per_step=5):
        for _ in range(tr_per_step): self.update()
        

    def q_next_prev(self, q_next_target):
        with torch.no_grad():
            # cut list of the last 3 elements [Qn-3, Qn-2, Qn-1]
            self.q_next_old_policy = self.q_next_old_policy[-3:]
            # multiply last 3 elements with exp weights and sum, creating exponential weighted average
            ewa = (torch.FloatTensor(self.q_next_old_policy)*self.weights).sum() # [0.06 Qn-3 + 0.21 Qn-2 + 0.73 Qn-1]
            # append new q next target value to the list
            self.q_next_old_policy.append(q_next_target.mean().detach())
            # return exp weighted average
            return ewa

    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau_*target_param.data + self.tau*param)

        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            next_action = self.actor.soft(next_state)
            q_next_target = self.critic_target.cmin(next_state, next_action)
            actor_loss = -self.rehaef(q_next_target, 0.95*self.q_next_prev(q_next_target))

            q = reward + (1-done) * 0.99 * q_next_target.detach()
            qs = self.critic(state, action)
            critic_loss = self.rehsef(q, qs[0]) + self.rehsef(q, qs[1]) + self.rehsef(q, qs[2])

        #Actor Update
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)

        #Critic Update
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)
        
        self.scaler.update()




class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, capacity, batch_lim, fade_factor=7.0):

        self.capacity, self.length, self.device = capacity, 0, device
        self.batch_size = min(max(200, self.length//100), batch_lim) #in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs = [], np.array([]), np.array([])
        self.fade_factor = fade_factor
        self.batch_lim = batch_lim


        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.bfloat16, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.bfloat16, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.bfloat16, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.bfloat16, device=device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.bfloat16, device=device)


    #Normalized index conversion into fading probabilities
    def fade(self, norm_index):
        weights = np.tanh(self.fade_factor*norm_index**2.7) # linear / -> non-linear _/‾
        return weights/np.sum(weights) #probabilities



    def add(self, state, action, reward, next_state, done):
        idx = self.length-1
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)
            self.probs = self.fade(self.indexes/self.length) if self.length>1 else np.array([0.0])
            self.batch_size = min(max(200, self.length//100), self.batch_lim)
        

        self.states[idx,:] = torch.tensor(state, dtype=torch.bfloat16, device=self.device)
        self.actions[idx,:] = torch.tensor(action, dtype=torch.bfloat16, device=self.device)
        self.rewards[idx,:] = torch.tensor([reward], dtype=torch.bfloat16, device=self.device)
        self.next_states[idx,:] = torch.tensor(next_state, dtype=torch.bfloat16, device=self.device)
        self.dones[idx,:] = torch.tensor([done], dtype=torch.bfloat16, device=self.device)


        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)



   
    # Do not use any decorators with random generators (Symphony updates seed each time)
    def sample(self):
        indices = self.random.choice(self.indexes, p=self.probs, size=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


    def __len__(self):
        return self.length
