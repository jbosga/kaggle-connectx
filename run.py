import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from src.dqn import DQN
from src.env import ConnectX
from src.replay_memory import ReplayMemory
from src.helpers import preprocess, select_action, optimize_model


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
HIDDEN_UNITS = 100

env = ConnectX()
policy_net = DQN(env.observation_space.n+1, HIDDEN_UNITS, env.action_space.n).to(device)
target_net = DQN(env.observation_space.n+1, HIDDEN_UNITS, env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)





num_episodes = 50
# pbar = tqdm(range(num_episodes))
all_total_rewards = np.empty(num_episodes)
all_avg_rewards = np.empty(num_episodes) # Last 100 steps
all_epsilons = np.empty(num_episodes)


for i_episode in range(num_episodes):
    

    state = env.reset()
    state = preprocess(state)
    rewards = 0
    for t in count():
        
        # Select and perform an action
        eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * t*i_episode / EPS_DECAY)

        action = select_action(state, eps, policy_net, env)
        next_state, reward, done, _ = env.step(action.item())
        next_state = preprocess(next_state)
        # Observe new state
        if not done:
            reward = -0.05 # Try to prevent the agent from taking a long move
        else:
            next_state = None
            # Apply new rules
            if reward == 1: # Won
                reward = 20
            elif reward == 0: # Lost
                reward = -20
            else:  # Draw
                reward = 10
             
        reward = torch.tensor([reward], device=device).float()
        rewards += reward
            
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        
        # Move to the next state
        state = next_state
        
        # Perform one optimization step on the target network
        optimize_model(optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory, policy_net=policy_net, target_net=target_net, gamma=GAMMA)
        
        if done:
            break
            
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    all_total_rewards[i_episode] = rewards
    avg_reward = all_total_rewards[max(0, i_episode - 100):(i_episode + 1)].mean()
    all_avg_rewards[i_episode] = avg_reward
    # all_epsilons[i_episode] = epsilon
            
    # pbar.set_postfix({
    #     'episode reward': total_reward,
    #     'avg (100 last) reward': avg_reward,
    #     'epsilon': epsilon
    # })
print(all_total_rewards)
print(all_avg_rewards)
