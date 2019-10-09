import gym
import gym_minigrid
import numpy as np
import time
from optparse import OptionParser
import random


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
            print("unknown key %s" % keyName)
            return

        return act

    def table_conversion():
        width = env.width - 2
        pos_loc = []
        # i = 0
        # j = 0
        for i in range(width):
            pos_loc.append(np.arange(width*i, width*(i+1)))
        return pos_loc


    resetEnv()


    #parameters
    episodes = 500
    epsilon = 0.2
    alpha = 0.1
    gamma = 0.6

    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    table_locator = table_conversion()
    for i in range(episodes):
        temp_state = env.reset()
        state = table_locator[temp_state[0]-1][temp_state[1]-1]
        while True:
            # print(q_table)
            # print('state = ')
            # print(state)
            renderer = env.render('human')
            time.sleep(0.1)
            if random.uniform(0, 1) < epsilon:
                temp_action = env.action_space.sample() #explore
            else:
                # print(q_table[state])
                # print(np.argmax(q_table[state]))
                temp_action = np.argmax(q_table[state]) #greedy
            # print(temp_action)
            # print(type(temp_action))
            action = get_action(temp_action)
            # print(action)
            obs, reward, done, agent_position, grid_size, info = env.step(action)
            # print('reward=%.2f' % (reward))
            next_state = table_locator[agent_position[0]-1][agent_position[1]-1]
            old_val = q_table[state, action]
            # print(old_val)
            next_max = np.max(q_table[next_state])
            new_q_val = (1-alpha) * old_val + alpha * (reward + gamma + next_max)
            print('step=%s,reward=%.2f, new_q_val=%.2f, state=%i, action=%s' % (env.step_count, reward, new_q_val, state, action))
            q_table[state, temp_action] = new_q_val

            state = next_state
            # time.sleep(1.5)

            if done:
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
