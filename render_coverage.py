#!/usr/bin/env python3
'''
Generate rendering of coverage map

./render_coverage.py -i coverage_12x12_o3_a3_r1_t1.csv
'''

import argparse
import csv
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(
        description='Input csv file of coverage map')
    parser.add_argument('-i','--input', help='Input file name', required=True)
    args = parser.parse_args()

    reader = csv.reader(open(args.input), delimiter=",")
    x = list(reader)
    coverage_maps = np.array(x).astype("int")

    print(coverage_maps)

    #TODO: render coverage map for select episodes
    # ensure map is consistent with gym environment

if __name__ == "__main__":
    main()