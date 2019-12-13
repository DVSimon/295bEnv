#!/usr/bin/env python3
'''
Generate rendering of coverage map

./render_coverage.py -i coverage_12x12_o3_a3_r1_t1.csv
'''

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 153, 0]),
    'blue'  : np.array([51, 153, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'black' : np.array([0, 0, 0]),
    'white' : np.array([255, 255, 255]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'black' : 6,
    'white' : 7,
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

def render(color_map, N=12):
    last_episode = color_map[-1]
    last_episode_grid = np.reshape(last_episode, (N,N))
    #print(last_episode_grid)
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    my_cmap = matplotlib.colors.ListedColormap([COLORS['grey']/255, COLORS['blue']/255, COLORS['yellow']/255, COLORS['green']/255])
    for x in range(N + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
        ax.axvline(x, lw=2, color='k', zorder=5)
    ax.imshow(last_episode_grid, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)

    #plt.matshow(last_episode_grid)

    ax.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Input csv file of coverage map')
    parser.add_argument('-i','--input', help='Input file name', required=True)
    args = parser.parse_args()

    reader = csv.reader(open(args.input), delimiter=",")
    x = list(reader)
    coverage_maps = np.array(x).astype("int")

    print(coverage_maps)
    img = render(coverage_maps, 12)

    #TODO: render coverage map for select episodes
    # ensure map is consistent with gym environment

if __name__ == "__main__":
    main()
