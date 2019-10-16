from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        obstacles=3,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.obstacles = obstacles

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Set Seed
        self.np_random, _ = seeding.np_random(self.seed_val)

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate obstacles
        for _ in range(self.obstacles):
            while True:
                x = self.np_random.randint(1,width-1)
                y = self.np_random.randint(1,height-1)

                if (x,y) != self.agent_start_pos and self.grid.get(x,y) is None:
                    break
            
            self.grid.set(x,y,Wall())
            
        
        #Set all non-agent squares as uncovered
        self.grid.setAll(Uncovered())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.grid.set(*self.agent_pos,None)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # self.mission = "Explore every grid space."

class EmptyEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)
