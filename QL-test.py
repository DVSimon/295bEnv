#!/usr/bin/env python3
'''
Run simulation based on q-table output in qt.pkl
'''

import gym, gym_minigrid
from gym_minigrid.plot import Plotter
import numpy as np
import time
from optparse import OptionParser
from collections import defaultdict
import csv
import pickle
import yaml
# import hashlib

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

    # def resetEnv():
    #     env.seed()
    #     env.reset()

    # def sha1(s):
    #     return hashlib.sha1(s).hexdigest()

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

    with open("config.yml", 'r') as ymlfile:
        cfg= yaml.load(ymlfile)
        ql = cfg['ql']
        rnd = cfg['rnd']

    # render boolean
    grid_render = rnd['grid_render']
    obs_render = rnd['obs_render']
    gray = rnd['grayscale']
    sleep = rnd['sleep']

    q_table = pickle.load(open('qt.pkl', 'rb'))
    for key in q_table:
        print(key)
    # print(len(q_table))

    obs = env.reset()
    states = {}
    for agent_id in obs:
        # Convert state(grid position) to a 1d state value
        # states[agent_id] = sha1(np.array(init_obs[agent_id]))
        temp_obs = ''
        for list in obs[agent_id]:
            temp = ','.join(map(str, list))
            temp_obs += ',' + temp
        states[agent_id]  = temp_obs

    knp = 0
    while True:
        if obs_render:
            env.get_obs_render(obs, grayscale=gray)
        if grid_render:
            env.render('human', highlight=True, grayscale=gray, info="Step: %s" % (str(env.step_count)))
        #test2

        time.sleep(sleep)

        if np.random.uniform(0, 1) < 0.1:
            exploit = False #explore
        else:
            exploit = True  #exploit

        # Determine action for each agent
        actions = {}
        for agent_id in obs:
            if states[agent_id] not in q_table:
                print('key not present:', knp)
                knp += 1
                print(states[agent_id])
                exploit = False
            if exploit is False:
                temp_action = env.action_space.sample() #explore
            else:
                temp_action = np.argmax(q_table[states[agent_id]]) #exploit

            # Convert action from numeric to environment-accepted directional action
            actions[agent_id] = get_action(temp_action)

        # Take step
        obs, reward, done, agents, info = env.step(actions)

        if done:
            print('done!')
            break

        # Calculate q-table values for each agent
        for agent_id in obs:
            # Using the agents new position returned from the environment, convert from grid coordinates to table based state for next state
            # next_state = sha1(np.array(obs[agent_id]))
            next_state = ''
            for list in obs[agent_id]:
                temp = ','.join(map(str, list))
                next_state += ',' + temp

            states[agent_id] = next_state

if __name__ == "__main__":
    main()