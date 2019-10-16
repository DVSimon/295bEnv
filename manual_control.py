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
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        env.seed()
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def table_conversion():
        width = env.width - 2
        pos_loc = []
        # i = 0
        # j = 0
        for i in range(width):
            pos_loc.append(np.arange(width*i, width*(i+1)))
        return pos_loc

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.up
        elif keyName == 'DOWN':
            action = env.actions.down
        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        table_locator = table_conversion()
        obs, reward, done, agent_position, grid_size, info = env.step(action)
        print(table_locator[agent_position[0]-1][agent_position[1]-1])

        print('step=%s, reward=%.2f, position=%s' % (env.step_count, reward, agent_position))
        # print('obs_space=%s, action_space=%s, grid_size=%s' % (env.observation_space, env.action_space, grid_size))
        # obs, reward, done, grid_str, info = env.step(action)
        print(obs)
        #
        # print('step=%s, reward=%.2f, string=%s' % (env.step_count, reward, grid_str))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
