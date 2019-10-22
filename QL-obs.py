#!/usr/bin/env python3

import gym
import gym_minigrid
from gym_minigrid.plot import Plotter
import numpy as np
import time
from optparse import OptionParser
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import hashlib
import csv
import pickle
from functools import partial

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-8x8-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        env.seed()
        env.reset()

    def sha1(s):
        return hashlib.sha1(s).hexdigest()

    # Convert action from numeric value to environmental directional actions
    def get_action(temp_action):
        act = None

        if temp_action == 0:
            act = env.actions.left
        elif temp_action == 1:
            act = env.actions.up
        elif temp_action == 2:
            act = env.actions.right
        elif temp_action == 3:
            act = env.actions.down
        else:
            print("unknown key")
            return

        return act

    # Assign state values from 2d array(positions on grid are mapped out in 2d(rows/columns)) to 1d for states
    # e.g. 8x8 grid(2d array) mapped to equivalent states
    # def table_conversion():
    #     width = env.width - 2
    #     pos_loc = []
    #
    #     for i in range(width):
    #         pos_loc.append(np.arange(width*i, width*(i+1)))
    #     return pos_loc

    plotter = Plotter()

    # Initialize environment
    resetEnv()

    # parameters, can be adjusted
    episodes = 500
    epsilon = 0.8
    decay = 0.99
    alpha = 0.1
    gamma = 0.6

    # metrics
    steps_to_complete = []

    # Initalize q-table [observation space x action space]
    # q_table = defaultdict(lambda: np.random.uniform(size=(env.action_space.n,)))
    q_table = defaultdict(lambda: np.zeros(shape=(env.action_space.n,)))
    # table_locator = table_conversion()

    for e in range(episodes):
        # Calculate new epsilon-decay value -- decays with each new episode
        epsilon = epsilon*decay

        # Initial agents
        init_obs = env.reset()
        # print(agents)
        states = {}
        for agent_id in init_obs:
            # Convert state(grid position) to a 1d state value
            # states[agent_id] = agents['image'][agent_id]
            states[agent_id] = sha1(np.array(init_obs[agent_id]))
        # print(states)

        while True:
            renderer = env.render('human')

            # time.sleep(0.05)

            # Determine whether to explore or exploit for all agents during current step
            if random.uniform(0, 1) < epsilon:
                exploit = False #explore
            else:
                exploit = True  #exploit

            # Determine action for each agent
            actions = {}
            for agent_id in init_obs:
                if exploit is False:
                    temp_action = env.action_space.sample() #explore
                else:
                    temp_action = np.argmax(q_table[states[agent_id]]) #exploit

                # Convert action from numeric to environment-accepted directional action
                actions[agent_id] = get_action(temp_action)

            # Take step
            obs, reward, done, agents, info = env.step(actions)
            # print('agents ', agents)
            # print('reward=%.2f' % (reward))
            # print(obs['image'])
            # print('q table1 ', q_table)
            # Calculate q-table values for each agent
            for agent_id in obs:
                # Using the agents new position returned from the environment, convert from grid coordinates to table based state for next state
                next_state = sha1(np.array(obs[agent_id]))
                old_val = q_table[states[agent_id]][actions[agent_id]]
                # print('old val ', old_val)
                # print('next state ', next_state)
                # New possible max at the next state for q table calculations
                next_max = np.max(q_table[next_state])
                # Calculate new q value
                new_q_val = (1-alpha) * old_val + alpha * (reward[agent_id] + gamma + next_max)
                print(str(agent_id) + ':' + 'step=%s,reward=%.2f, new_q_val=%.2f, state=%s, action=%s' % (env.step_count, reward[agent_id], new_q_val, states[agent_id], actions[agent_id]))
                # print(obs[agent_id])
                q_table[states[agent_id]][actions[agent_id]] = new_q_val

                states[agent_id] = next_state

            # time.sleep(10000)
            # time.sleep(1.5)

            if done:
                # plot steps by episode
                steps_to_complete.append(env.step_count)
                plotter.plot_steps(steps_to_complete, '-lr')

                print('done!')
                print(q_table)
                break


    print("Training finished.\n")
    #csv store
    w = csv.writer(open("qt_output.csv", "w"))
    for key, val in q_table.items():
        w.writerow([key, val])
    #pkl
    f = open("qt.pkl","wb")
    pickle.dump(dict(q_table), f)
    f.close()
    #
    # while True:
    #     env.render('human')
    #     time.sleep(0.01)

        # # If the window was closed
        # if renderer.window == None:
        #     break

if __name__ == "__main__":
    main()
