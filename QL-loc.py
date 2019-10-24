#!/usr/bin/env python3

import gym
import gym_minigrid
from gym_minigrid.plot import Plotter
import numpy as np
import time
from optparse import OptionParser
import random
import matplotlib.pyplot as plt

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
    def table_conversion():
        width = env.width - 2
        pos_loc = []

        for i in range(width):
            pos_loc.append(np.arange(width*i, width*(i+1)))
        return pos_loc

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
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    table_locator = table_conversion()

    for e in range(episodes):
        # Calculate new epsilon-decay value -- decays with each new episode
        epsilon = epsilon*decay

        # Initial agents
        agents = env.reset()

        states = {}
        for agent_id, agent_pos in agents.agent_pos.items():
            # Convert state(grid position) to a 1d state value
            states[agent_id] = table_locator[agent_pos[0]-1][agent_pos[1]-1]

        while True:
            renderer = env.render('human')

            time.sleep(5)

            # Determine whether to explore or exploit for all agents during current step
            if random.uniform(0, 1) < epsilon:
                exploit = False #explore
            else:
                exploit = True  #exploit

            # Determine action for each agent
            actions = {}
            for agent_id, agent_pos in agents.agent_pos.items():
                if exploit is False:
                    temp_action = env.action_space.sample() #explore
                else:
                    temp_action = np.argmax(q_table[states[agent_id]]) #exploit

                # Convert action from numeric to environment-accepted directional action
                actions[agent_id] = get_action(temp_action)

            # Take step
            obs, reward, done, agents, info = env.step(actions)
            # print('reward=%.2f' % (reward))
            print(obs['image'][0])

            # Calculate q-table values for each agent
            for agent_id, agent_pos in agents.agent_pos.items():
                # Using the agents new position returned from the environment, convert from grid coordinates to table based state for next state
                next_state = table_locator[agent_pos[0]-1][agent_pos[1]-1]
                old_val = q_table[states[agent_id], actions[agent_id]]

                # New possible max at the next state for q table calculations
                next_max = np.max(q_table[next_state])

                # Calculate new q value
                new_q_val = (1-alpha) * old_val + alpha * (reward[agent_id] + gamma + next_max)
                print(str(agent_id) + ':' + 'step=%s,reward=%.2f, new_q_val=%.2f, state=%i, action=%s' % (env.step_count, reward[agent_id], new_q_val, states[agent_id], actions[agent_id]))
                # print(obs[agent_id])
                q_table[states[agent_id], actions[agent_id]] = new_q_val

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
    #
    # while True:
    #     env.render('human')
    #     time.sleep(0.01)

        # # If the window was closed
        # if renderer.window == None:
        #     break

if __name__ == "__main__":
    main()
