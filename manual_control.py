#!/usr/bin/env python3

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

    def resetEnv():
        env.seed()
        env.reset()

    resetEnv()
    n_agents = env.agents.n_agents
    
    # Create a window to render into
    renderer = env.render('human')

    def table_conversion():
        width = env.width - 2
        pos_loc = []
       
        for i in range(width):
            pos_loc.append(np.arange(width*i, width*(i+1)))
        return pos_loc

    def keyDownCb(keyName):
        keyDownCb.num += 1
        if keyDownCb.num == n_agents:
            keyDownCb.num = 0

        if keyName == 'BACKSPACE':
            resetEnv()
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
        elif keyName == 'SPACE':
            action[keyDownCb.num] = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action[keyDownCb.num] = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action[keyDownCb.num] = env.actions.drop

        elif keyName == 'RETURN':
            action[keyDownCb.num] = env.actions.done

        else:
            print("unknown key %s" % keyName)
            keyDownCb.num = -1
            return
        if keyDownCb.num == n_agents-1:
            obs, reward, done, agents, info = env.step(action)
            # print('step,reward=',env.step_count, reward)

        # print('step=%s, reward=%.2f, position=%s' % (env.step_count, reward, agent_position))
        # print('obs_space=%s, action_space=%s, grid_size=%s' % (env.observation_space, env.action_space, grid_size))
        # obs, reward, done, grid_str, info = env.step(action)
        #
        # print('step=%s, reward=%.2f, string=%s' % (env.step_count, reward, grid_str))
            # print(obs)
            if done:
                print('done!')
                resetEnv()

    keyDownCb.num = -1
    action = [None] * n_agents
    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
