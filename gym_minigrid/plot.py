import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import yaml

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

class Plotter:
    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)['plt']
        
        self.regression_type = cfg['regression_type']

    def plot_steps(self, steps_list):
        if len(steps_list) >= 2:
            x = np.arange(0, len(steps_list)) 

            plt.clf()

            if self.regression_type == 'lin':
                fit = np.polyfit(x,steps_list,1) # linear regression
                fit_fn = np.poly1d(fit)
                plt.plot(x,steps_list, 'yo', x, fit_fn(x), '--k')
            elif self.regression_type == 'quad':
                fit = np.polyfit(x,steps_list,2) # quadratic regression
                fit_fn = np.poly1d(fit)
                plt.plot(x,steps_list, 'yo', x, fit_fn(x), '--k')
            elif self.regression_type == 'exp':
                popt, pcov = curve_fit(exp_func, x, steps_list) # exponential regression
                fit_fn = exp_func(x, *popt)
                plt.plot(x,steps_list, 'yo', x, exp_func(x, *popt), '--k')
            else:
                assert False, "invalid regression type"

            plt.xlabel('Episode')
            plt.ylabel('Steps to completion')
            
            plt.ion()
            plt.ioff()
            plt.show()
            # plt.savefig('steps_graph.png')
