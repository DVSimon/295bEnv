#!/usr/bin/env python3
'''
Generate rendering of coverage map

./render_coverage.py -i 12x12_o3_a3_r1_t1 -s 12 -e 999
'''
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0,200]),
    'green' : np.array([0, 153, 0,200]),
    'blue'  : np.array([51, 153, 255,200]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0,200]),
    'grey'  : np.array([100, 100, 100]),
    'black' : np.array([0, 0, 0]),
    'white' : np.array([255, 255, 255]),
    'electricblue' : np.array([72, 151, 216]),
    'banana' : np.array([255, 219, 92]),
    'watermelon' : np.array([250, 110, 89]),
    'canteloupe' : np.array([248, 160, 85]),
    'purple-ish' : np.array([180, 106, 226])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'banana' : 0,
    'canteloupe' : 1,
    'purple-ish' : 2,
    'electricblue' : 3,
    'watermelon' : 4,
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

def render_cov(coverage_maps, episode, N, file_params):
    # select map by episode, and reshape into 2D grid
    last_episode = coverage_maps[episode]
    last_episode_grid = np.reshape(last_episode, (N,N))

    # colormap of grid
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    my_cmap = matplotlib.colors.ListedColormap([COLORS['grey']/255,
                                                COLORS['electricblue']/255,
                                                COLORS['banana']/255,
                                                COLORS['watermelon']/255,
                                                COLORS['canteloupe']/255,
                                                COLORS['purple-ish']/255])
    
    # draw gridlines
    for x in range(N + 1):
        ax.axhline(x, lw=1, color=COLORS['grey']/255, zorder=5)
        ax.axvline(x, lw=1, color=COLORS['grey']/255, zorder=5)

    ax.imshow(last_episode_grid, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)

    ax.axis('off')
    plt.savefig("coverage_"+file_params+".png")
    print('coverage map saved')
    plt.show()

def render_traj(trajectories, episode, N, coverage_maps, file_params):
    # select trajectory by episode
    trajectory = trajectories[episode]

    #select coverage by episode, and reshape into 2D grid
    last_cov = coverage_maps[episode]
    last_cov_grid = np.reshape(last_cov, (N,N))

    # colormap of grid
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    my_cmap = matplotlib.colors.ListedColormap([COLORS['grey']/255,
                                                COLORS['white']/255,
                                                COLORS['white']/255,
                                                COLORS['white']/255,
                                                COLORS['white']/255,
                                                COLORS['white']/255,])
    
    # draw gridlines
    for x in range(N + 1):
        ax.axhline(x, lw=1, color=COLORS['grey']/255, zorder=5)
        ax.axvline(x, lw=1, color=COLORS['grey']/255, zorder=5)

    # offset to center of cells
    offset = (0.5, 0.5)
    
    for key in trajectory:
        color = COLORS[IDX_TO_COLOR[key]]/255

        pos = trajectory[key]

        # draw point at starting position of agents
        ax.plot(np.add(pos[0][0],offset), N-(np.add(pos[0][1],offset)), marker='o', color=color)

        # draw trajectory step by step
        for i in range(len(trajectory[key])-1):
            ax.plot((np.add(pos[i][0],offset), np.add(pos[i+1][0],offset)), 
                    (N-np.add(pos[i][1],offset), N-np.add(pos[i+1][1],offset)), color=color)

    ax.imshow(last_cov_grid, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)    

    ax.axis('off')
    plt.savefig("trajectory_"+file_params+".png")
    print('trajectory map saved')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Input csv file of coverage map')
    parser.add_argument('-i','--input', help='Input file parameters', required=True)
    parser.add_argument('-s','--size', help='Grid size', required=True)
    parser.add_argument('-e','--episode', help='Episode', required=True)
    args = parser.parse_args()

    reader = csv.reader(open("coverage_"+args.input+".csv"), delimiter=",")
    x = list(reader)
    coverage_maps = np.array(x).astype("int")

    trajectories = pickle.load(open("trajectory_"+args.input+".pkl", 'rb'))

    # render coverage map
    render_cov(coverage_maps, int(args.episode), int(args.size), args.input)
    # render trajectory map
    render_traj(trajectories, int(args.episode), int(args.size), coverage_maps, args.input)

if __name__ == "__main__":
    main()
