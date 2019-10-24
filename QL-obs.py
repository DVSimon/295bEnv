#!/usr/bin/env python3

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

    plotter = Plotter()

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)['ql']

    # parameters, can be adjusted in config.yml    
    episodes = cfg['episodes']
    epsilon = cfg['epsilon']
    decay = cfg['decay']
    alpha = cfg['alpha']
    gamma = cfg['gamma']

    # render boolean
    render = cfg['render']

    # metrics
    steps_to_complete = []

    # Initalize q-table [observation space x action space]
    # q_table = defaultdict(lambda: np.random.uniform(size=(env.action_space.n,)))
    q_table = defaultdict(lambda: np.zeros(shape=(len(env.actions),)))

    for e in range(episodes):
        # Calculate new epsilon-decay value -- decays with each new episode
        epsilon = epsilon*decay

        # Initial agents
        init_obs = env.reset()

        states = {}
        for agent_id in init_obs:
            # Convert state(grid position) to a 1d state value
            # states[agent_id] = sha1(np.array(init_obs[agent_id]))
            temp_obs = ''
            for list in init_obs[agent_id]:
                temp = ','.join(map(str, list))
                temp_obs += ',' + temp
            states[agent_id]  = temp_obs

        while True:
            if render:
                env.render('human', info="Episode: %s \tStep: %s" % (str(e),str(env.step_count)))

            # time.sleep(3)

            # Determine whether to explore or exploit for all agents during current step
            if np.random.uniform(0, 1) < epsilon:
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
            
            # Calculate q-table values for each agent
            for agent_id in obs:
                # Using the agents new position returned from the environment, convert from grid coordinates to table based state for next state
                # next_state = sha1(np.array(obs[agent_id]))
                next_state = ''
                for list in obs[agent_id]:
                    temp = ','.join(map(str, list))
                    next_state += ',' + temp
                old_val = q_table[states[agent_id]][actions[agent_id]]
                
                # New possible max at the next state for q table calculations
                next_max = np.max(q_table[next_state])

                # Calculate new q value
                new_q_val = (1-alpha) * old_val + alpha * (reward[agent_id] + gamma * next_max)
                print(str(agent_id) + ':' + 'step=%s,reward=%.2f, new_q_val=%.2f, state=%s, action=%s' % (env.step_count, reward[agent_id], new_q_val, states[agent_id], actions[agent_id]))
                
                q_table[states[agent_id]][actions[agent_id]] = new_q_val

                states[agent_id] = next_state
           
            if done:
                print('done!')
                # print("q table:",q_table)
                # print("times_visited:",env.grid_visited)

                # plot steps by episode
                steps_to_complete.append(env.step_count)
                if e % 100 == 0:
                    plotter.plot_steps(steps_to_complete)
                    with open("qt_output.csv", "w") as outfile:
                        writer = csv.writer(outfile)
                        for key, val in q_table.items():
                            writer.writerow([key, *val])

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

if __name__ == "__main__":
    main()
