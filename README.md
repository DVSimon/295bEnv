TODO: Update README

# Multi-Agent Coverage Control with RL
Adapted from [Minimalistic Gridworld Environment (MiniGrid)](https://github.com/maximecb/gym-minigrid)

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- PyQT 5
- Matplotlib
- Pickle


## Configuration
- Navigate to config.yml file

### Environment Variables
- grid_size determines the height and width of the grid environment including the outer walls.
- obstacles decides the # of obstacles to be placed into the environment(randomized).
- agents decides the # of agents within environment.
- obs_radius decides the agents surrounding visibility of grid.
- reward_type of 0 decides to use generic +1, -1 reward formula.
- reward_type of 1 decides to use custom reward formula.
- seed decides the seed for generation for reproducability.

### Q-Learning Parameters
- These parameters only affect the Q-learning algorithm implementation itself, not DQN.

### To turn on overall grid rendering
- Change grid_render to True

### To turn on individual agent perception rendering 
- Change grid_obs_render to True
- Change obs_render to True

### To add sleep timer between agent actions
- Change sleep parameter to desired time(seconds) between actions.

### To change plot type of number of moves taken
- Change regression_type to null/lin/quad/exp

## How to run
```
./QL-obs.py
```
