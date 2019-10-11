#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
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
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()
    n_agents = env.getagents()
    #print('agents are: ',n_agents)
    
    # Create a window to render into
    renderer = env.render('human')

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

        # if keyDownCb.num == 0:
        #     action = [None] * n_agents

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
            obs, reward, done, info = env.step(action)
            print('step,reward=',env.step_count, reward)

        # obs, reward, done, grid_str, info = env.step(action)
        #
        # print('step=%s, reward=%.2f, string=%s' % (env.step_count, reward, grid_str))

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
