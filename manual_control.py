#!/usr/bin/env python3
'''
Manual simulation, with user input
'''

from __future__ import division, print_function

import sys
import numpy as np
import gym
import time
from optparse import OptionParser

import gym_minigrid

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

    obs = env.reset()

    n_agents = env.agents.n_agents

    # Create a window to render into
    renderer = env.render('human', highlight=True, grayscale=False)

    def keyDownCb(keyName):
        keyDownCb.num += 1
        if keyDownCb.num == n_agents:
            keyDownCb.num = 0

        if keyName == 'BACKSPACE':
            obs = env.reset()
            keyDownCb.num = -1
            return

        if keyName == 'ESCAPE':
            keyDownCb.num = -1
            sys.exit(0)

        if keyName == 'LEFT':
            action[keyDownCb.num] = env.actions.left
        elif keyName == 'RIGHT':
            action[keyDownCb.num] = env.actions.right
        elif keyName == 'UP':
            action[keyDownCb.num] = env.actions.up
        elif keyName == 'DOWN':
            action[keyDownCb.num] = env.actions.down
        else:
            print("unknown key %s" % keyName)
            keyDownCb.num = -1
            return

        if keyDownCb.num == n_agents-1:
            obs, reward, done, agents, info = env.step(action)
            print('step,reward=',env.step_count, reward)
            
            # env.get_obs_render(obs)

            if done:
                print('done!')
                obs = env.reset()

    keyDownCb.num = -1
    action = [None] * n_agents
    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human', highlight=True, grayscale=False)

        # time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
