 ##########################################################################################
 # Reference for DQN implementation taken from official pytorch example doc               #
 # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm #
 ##########################################################################################

import gym, gym_minigrid
from gym_minigrid.plot import Plotter
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import os
import yaml

env = gym.make("MiniGrid-Empty-8x8-v0").unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

screen_width = 600
resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])

plotter = Plotter()

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def get_screen(obs):
    screen = {}
    for key, value in obs.items():
        screen[key] = env.get_obs_render(obs, mode='rgb_array')[key].transpose((2, 0, 1))
        screen[key] = np.ascontiguousarray(screen[key], dtype=np.float32) / 255
        screen[key] = torch.from_numpy(screen[key])
        screen[key] = resize(screen[key]).unsqueeze(0).to(device)
    return screen

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    #def __init__(self):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

obs = env.reset()
init_screen = get_screen(obs)
_,_,screen_height, screen_width= init_screen[0].shape
n_actions = env.action_space.n
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    action = {}
    for key in state:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                #action[key] = policy_net(state[key]).max(1)[1].view(1, 1)[0]
                action[key] = policy_net(state[key]).max(1)[1].view(1, 1)
        else:
            rand_agents_a = np.random.randint(4, size=1)
            #action[key] = torch.tensor(rand_agents_a.tolist(), device=device, dtype=torch.long)
            action[key] = torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
    steps_done += 1
    #print(action)
    return action

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
                                        #  batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #print(state_batch.shape, action_batch.shape, reward_batch.shape)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    #state_action_values = policy_net(state_batch).gather(1, action_batch.resize(len(action_batch), 1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = cfg['ql']['episodes']
steps_to_complete = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = env.reset()
    last_screen = get_screen(obs)
    current_screen = get_screen(obs)
    state = {}
    for key in current_screen:
        state[key] = current_screen[key] - last_screen[key]
    for t in count():
        # Select and perform an action
        ####### Select action should be to pool all the actions of every agent (dictioonary not an int)
        action = select_action(state)
        ac_to_step = [None] * len(action)
        for key in action:
            ac_to_step[key] = action[key].tolist()[0]
        obs, reward, done, agents, info = env.step(ac_to_step)

        #reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(obs)
        next_state = {}
        if not done:
            for key in current_screen:
                next_state[key] = current_screen[key] - last_screen[key]
        else:
            for key in current_screen:
                next_state[key] = None

        # Store the transition in memory
        for key in state:
            memory.push(state[key], action[key], next_state[key], torch.tensor([reward[key]], device=device))
            #memory.push(state[key], action[key], next_state[key], torch.tensor([[reward[key]]], device=device))
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if(cfg['rnd']['grid_render']):
            env.render()
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            steps_to_complete.append(t)
            print(steps_to_complete)
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plotter.plot_steps(steps_to_complete)
env.render()
env.close()
plt.ioff()
plt.show()
