from symphony import Symphony
import gymnasium as gym

import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import numpy as np
import random, math
import pickle
import os, re

# global constants:
phi = (math.sqrt(5)+1)/2 #1.618...
phi_ = 1/phi #0.618...

#############################################
# ---------------Parametres-----------------#
#############################################

#global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(device)

learning_rate = 1e-4
explore_time, times = 20000, 50
capacity = explore_time * times
h_dim = 432
batch_size = q_dist = 432
alpha, tau = phi_, 0.001
num_episodes = 100000
limit_test = 1000
limit_step = 1000 #max steps per episode
start_episode = 1 #number for the identification of the current episode
episode_rewards, episode_steps, total_steps = [], [], 0
stage = {"exploration": True, "training": False, "testing": False}

# environment type.
env_name = 'Humanoid-v4'



pre_valid = True # testing models when loaded
env = gym.make(env_name)
env_test = gym.make(env_name)
env_valid = gym.make(env_name, render_mode="human")

state_low, state_high = env.observation_space.low, env.observation_space.high
state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

#max_action = torch.FloatTensor(env.action_space.high) if env.action_space.is_bounded() else torch.ones(action_dim)
max_action = torch.ones(action_dim)


algo = Symphony(capacity, state_dim, action_dim, h_dim, alpha, tau, q_dist, batch_size, max_action, state_high, state_low, learning_rate, device)


print("action_dim: ", action_dim, "state_dim: ", state_dim)
print("max_action:", max_action)
print("batch_size", batch_size)
print("q distribuion", q_dist)
print("Replay Buffer capacity", algo.nets.replay_buffer.capacity)



#############################################
# -----------Helper Functions---------------#
#############################################



# random seeds for reproducing the experiment
def seed_reset():
    r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
    torch.manual_seed(r1)
    np.random.seed(r2)
    random.seed(r3)
    return r1, r2, r3


def extract_r1_r2_r3():
    pattern = r'history_(\d+)_(\d+)_(\d+)\.csv'

    # Iterate through the files in the given directory
    for filename in os.listdir():
        # Match the filename with the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the numbers r1, r2, and r3 from the filename
            return map(int, match.groups())
    return None


#write or append to the history log file
class LogFile(object):
    def __init__(self, log_name_main, log_name_opt):
        self.log_name_main = log_name_main
        self.log_name_opt = log_name_opt
    def write(self, text):
        with open(self.log_name_main, 'a+') as file:
            file.write(text)
    def write_opt(self, text):
        with open(self.log_name_opt, 'a+') as file:
            file.write(text)
    def clean(self):
        with open(self.log_name_main, 'w') as file:
            file.write("step,return\n")
        with open(self.log_name_opt, 'w') as file:
            file.write("ep,return,q_std,q_ema,scale\n")


numbers = extract_r1_r2_r3()

#derive random numbers from history file or generate new random seeds
r1, r2, r3 = numbers if numbers != None else seed_reset()
print(r1, ", ", r2, ", ", r3)

log_name_main = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".csv"
log_name_opt = "episodes_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".csv"
log_file = LogFile(log_name_main, log_name_opt)


def save(algo, episode_return, episode_steps, total_steps):

    average_return = round(np.mean(episode_return[-300:]), 2)
    average_steps = int(np.mean(episode_steps[-300:]))

    torch.save(algo.nets.online.state_dict(), 'nets_online_model.pt')
    torch.save(algo.nets.target.state_dict(), 'nets_target_model.pt')
    torch.save(algo.nets.optimizer.state_dict(), 'nets_optimizer.pt')
    torch.save(algo.nets.replay_buffer.state_dict(), 'nets_replay_buffer.pt')
    print("saving... the buffer length = ", algo.nets.replay_buffer.length.item(), " avg return = ", average_return, " avg steps = ", average_steps, end="")
    with open('data', 'wb') as file:
        pickle.dump({'episode_return': episode_return, 'episode_steps': episode_steps, 'total_steps' : total_steps}, file)
    print(" > done")


def load(algo, stage):

    episode_return, episode_steps, total_steps = [], [], 0

    try:
        print("loading models...")
        algo.nets.online.load_state_dict(torch.load('nets_online_model.pt', weights_only=True))
        algo.nets.target.load_state_dict(torch.load('nets_target_model.pt', weights_only=True))
        algo.nets.optimizer.load_state_dict(torch.load('nets_optimizer.pt', weights_only=True))
        print('models loaded')
        if pre_valid:
            stage = {"exploration": False, "training": False, "testing": True}
            sim_loop(env_valid, 100, stage, algo, [], [], total_steps==0, limit_steps=limit_test)
    except:
        print("problem during loading models")


    try:
        print("loading buffer...")
        algo.nets.replay_buffer.load_state_dict(torch.load('nets_replay_buffer.pt', weights_only=True))
        with open('data', 'rb') as file:
            dict = pickle.load(file)
            episode_return = dict['episode_return']
            episode_steps = dict['episode_steps']
            total_steps = dict['total_steps']
            if algo.nets.replay_buffer.length>=explore_time and not stage["training"]: 
                stage = {"exploration": False, "training": True, "testing": False}
        
        print('buffer loaded, Q_ema', round(algo.nets.target.q_ema.item(), 2), ', average_reward = ', round(np.mean(episode_return[-300:]), 2))
        
    except:
        print("problem during loading buffer")

    return stage, episode_return, episode_steps, total_steps








# Loop for episodes:[ State -> Loop for one episode: [ Action, Next State, Reward, Done, State = Next State ] ]
def sim_loop(env, episodes, stage, algo, episode_return, episode_steps, total_steps, limit_steps):

    

    start_episode = len(episode_return) + 1
    average_steps = np.mean(episode_steps[-300:])


    for episode in range(start_episode, episodes+1):

        if total_steps>=3000000: break
            
        Return = 0.0     
        state = env.reset()[0]
        
        for steps in range(1,limit_steps+1):

            total_steps += 1

            

            # Activate training if explore time is reached and if it is not testing mode:
            if not stage["testing"]:
                if algo.nets.replay_buffer.length>=explore_time and not stage["training"]:
                    stage = {"exploration": False, "training": True, "testing": False}
                    algo.nets.replay_buffer.norm_fill(times)
                    print("exploration end")
            
           

            # if total steps is divisible to 2500 save models, stop training and do testing, return to training:
            if stage["training"] and total_steps%10000==0:
                save(algo, episode_return, episode_steps, total_steps)
                
                print("start testing")
                stage_test = {"exploration": False, "training": False, "testing": True}
                test_return = sim_loop(env_test, 25, stage_test, algo, [], [], total_steps=0, limit_steps=limit_test)
                log_file.write(str(total_steps) + "," + str(round(test_return, 2)) + "\n")
                print("end of testing")


            # if steps is close to episode limit (e.g. 900) we shut down actions to get Terminal Transition:
            active = True if stage["testing"] else (steps<(limit_steps-100))
            action = algo.select_action(state,  active=active, test=stage["testing"]) 
            next_state, reward, done, truncated, info = env.step(action)

            if not stage["testing"]: algo.nets.replay_buffer.add(state, action, reward, next_state, done)
            Return += reward
            
            # actual training
            if stage["training"]: algo.train()
            if done: break
            state = next_state
        
        episode_steps.append(steps)
        average_steps = np.mean(episode_steps[-300:])
        episode_return.append(Return)
        average_reward = np.mean(episode_return[-300:])


        if stage["training"]:
            action, scale, beta, q_ema, q_std = algo.info()
            print(f"Ep {episode}: Rtrn = {Return:.2f}, Avg300 = {average_reward:.2f}| q_ema = {q_ema:.2f}| q_std = {q_std:.4f} | scale = {scale:.4f} | beta = {beta:.4f} |  ep steps = {steps} | total_steps = {total_steps}") 
            log_file.write_opt(str(episode) + "," + str(round(Return, 2)) + "," + str(round(q_std, 4)) + "," + str(round(q_ema, 4)) + "," + str(round(scale, 4)) + "\n")
        else:
            print(f"Ep {episode}: Rtrn = {Return:.2f}, Avg300 = {average_reward:.2f}| ep steps = {steps} | total_steps = {total_steps}") 

    return np.mean(episode_return).item()




# Loading existing models
stage, episode_return, episode_steps, total_steps = load(algo, stage)
if not stage["training"]: log_file.clean(); algo.nets.replay_buffer.init()

# Training
sim_loop(env, num_episodes, stage, algo, episode_return, episode_steps, total_steps, limit_step)
