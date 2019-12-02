# Multi-Agent Coverage Control utilizing Reinforcement Learning
Adapted from [Minimalistic Gridworld Environment (MiniGrid)](https://github.com/maximecb/gym-minigrid)


## Installation and Setup
Install Python >= 3.5 and clone this project:
```
$ git clone https://github.com/DVSimon/295bEnv.git
$ cd 295bEnv
```
Set up the virtual environment:
```
$ python3 -m venv env
$ source env/bin/activate
```
Install project dependencies:
```
$ pip3 install -r requirements.txt
```


## Basic Usage
Observation-based Q-Learning simulation and training
```
./QL-obs.py
```

Location-based DQN training
```
./DQN-loc.py
```

Observation-based DQN training
```
./DQN-obs.py
```

Manually control agents with keyboard input
```
./manual_control.py
```


## Configuration
- Navigate to config.yml file

### Environment Variables
- grid_size determines the height and width of the grid environment including the outer walls
- obstacles decides the # of obstacles to be placed into the environment (randomized)
- agents decides the # of agents within environment
- obs_radius decides the agents surrounding visibility of grid
- reward_type of 0 decides to use generic +1, -1 reward formula
- reward_type of 1 decides to use custom reward formula based on times visited
- seed decides the seed for generation for reproducability

### Q-Learning Parameters
- These parameters only affect the Q-learning algorithm implementation itself, not DQN

### To turn on environment grid rendering
- Set grid_render to True

### To turn on agent observation rendering within the environment grid
- Set grid_obs_render to True

### To turn on isolated agent observation rendering
- Set obs_render to True

### To add sleep timer between steps
- Set sleep to desired time (in seconds)

### To change plot regression type of number of steps taken
- Set regression_type to null/lin/quad/exp


## Deep Q Network (DQN) approach:
- DQN_loc.py script is location based DQN implementation.

  Each agent takes action one by one.

  Entire Image of Environment is fed to Neural Network(NN).
  
  NN outputs single value of optimum action for that each agents one by one.
  
- DQN_obs.py script is observation based DQN implementation
  
  Each agent's observation space is fed to NN.
  
  NN outputs action value for each agents individually & simultaneously. 
  
