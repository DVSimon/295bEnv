import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def plot_steps(self, steps_list, option='-l'):
        if len(steps_list) >= 2:
            x = list(range(0,len(steps_list)))

            plt.clf()

            # if option is '-lr': # linear regression scatter plot
            fit = np.polyfit(x,steps_list,1)
            fit_fn = np.poly1d(fit) 

            plt.plot(x,steps_list, 'yo', x, fit_fn(x), '--k')
            # else: # line plot
            #     plt.plot(steps_list)

            plt.xlabel('Episode')
            plt.ylabel('Steps to completion')
            
            plt.ion()
            plt.show()