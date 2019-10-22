import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

#from gym_minigrid.action_space import MultiAgentActionSpace
#from gym_minigrid.observation_space import MultiAgentObservationSpace

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 153, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'uncovered'     : 8,
    'agent'         : 10,

}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
# we aren't using directions, need to change this to be just u, d , l , r w/o orientation?
# Map of agent direction indices to vectors
# DIR_TO_VEC = [
#     # Pointing right (positive X)
#     np.array((1, 0)),
#     # Down (positive Y)
#     np.array((0, 1)),
#     # Pointing left (negative X)
#     np.array((-1, 0)),
#     # Up (negative Y)
#     np.array((0, -1)),
# ]

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def _set_color(self, r):
        """Set the color of this object as the active drawing color"""
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

class Covered(WorldObj):
    def __init__(self):
        super().__init__('empty', None)

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Uncovered(WorldObj):
    def __init__(self):
        super().__init__('uncovered', 'green')

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c/2)
        r.drawPolygon([
            (1          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           1),
            (1          ,           1)
        ])

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Agents():
    """
    Agent Class for holding all the agent specific information
    """

    def __init__(self, n_agents=2):
        super().__init__()
        self.n_agents = n_agents
        self.agent_pos = {_: None for _ in range(self.n_agents)}

    def can_overlap(self):
        return False

    def get_n_agents(self):
        return self.n_agents

    def reset_agent_pos(self):
        self.agent_pos = {_: None for _ in range(self.n_agents)}

    def set_agent_pos(self, new_pos):
        self.agent_pos = {_: new_pos[_] for _ in range(self.n_agents)}

class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def setAll(self, v):
        for i in range(1, self.width-1):
            for j in range(1, self.height-1):
                if self.grid[j* self.width + i] is None:
                    self.grid[j* self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, Wall())

    def vert_wall(self, x, y, length=None):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, Wall())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)
        #print('slice:',len(topX))
        for a in range(len(topX)):
            for j in range(0, height):
                for i in range(0, width):
                    x = topX[a] + i
                    y = topY[a] + j

                    if x >= 0 and x < self.width and \
                       y >= 0 and y < self.height:
                        v = self.get(x, y)
                    else:
                        v = Wall()

                    grid.set(i, j, v)

        return grid

    def render(self, r, tile_size):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        assert r.width == self.width * tile_size
        assert r.height == self.height * tile_size

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

        # Draw the background of the in-world cells black
        r.fillRect(
            0,
            0,
            widthPx,
            heightPx,
            0, 0, 0
        )

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(0, self.height):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, widthPx, y)
        for colIdx in range(0, self.width):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, heightPx)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                if cell == None:
                    continue
                r.push()
                r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                cell.render(r)
                r.pop()

        r.pop()

    def encode(self, vis_mask=None):
        print('encode called')
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')
        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    print(i,j)
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                    else:
                        # State, 0: open, 1: closed, 2: locked
                        state = 0
                        if hasattr(v, 'is_open') and not v.is_open:
                            state = 1
                        if hasattr(v, 'is_locked') and v.is_locked:
                            state = 2

                        array[i, j, 0] = OBJECT_TO_IDX[v.type]
                        array[i, j, 1] = COLOR_TO_IDX[v.color]
                        array[i, j, 2] = state
        '''
        print('===encoded array:width,height is: ',self.width,self.height)
        for i in range(self.width):
            for j in range(self.height):
                for k in range(3):
                    print(array[i,j,k], ' -',end = '')
                print('')
        '''
        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3
        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                typeIdx, colorIdx, state = array[i, j]

                if typeIdx == OBJECT_TO_IDX['unseen'] or \
                        typeIdx == OBJECT_TO_IDX['empty']:
                    continue

                objType = IDX_TO_OBJECT[typeIdx]
                color = IDX_TO_COLOR[colorIdx]
                # State, 0: open, 1: closed, 2: locked
                is_open = state == 0
                is_locked = state == 2

                if objType == 'wall':
                    v = Wall(color)
                elif objType == 'floor':
                    v = Floor(color)
                elif objType == 'uncovered':
                    v = Uncovered()
                else:
                    assert False, "unknown obj type in decode '%s'" % objType

                grid.set(i, j, v)

        return grid

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

    

class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        #go left, up, right down
        left = 0
        right = 1
        up = 2
        down = 3

        # # Pick up an object
        # pickup = 3
        # # Drop an object
        # drop = 4
        # # Toggle/activate an object
        # toggle = 5

        # Done completing task
        done = 4

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        n_agents=1,
        max_steps=100,
        see_through_walls=False,
        seed=23,
        agent_view_radius=1,
    ):
        print("---init called")
        self.agents = Agents(n_agents)
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # total number of agents
        #self.n_agents = n_agents

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        self.width = width

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions)-1)

        # Number of cells (width and height) in the agent view
        self.agent_view_size = 2*agent_view_radius+1

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.agent_view_size, self.agent_view_size, 3),
        #     dtype='uint8'
        # )
        # self.observation_space = spaces.Dict({
        #     'image': self.observation_space
        # })
        self.observation_space = spaces.Discrete((self.width-2)**2)
        # Change to (-1,1) ? neeed negative reward
        # Range of possible rewards
        self.reward_range = (-1, 1)

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        # self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # (ma.)Current position and direction of the agent
        #self.agent_pos = None
        #self.agent_pos = {_: None for _ in range(self.n_agents)}
        #print('init: agent pos type: ',type(self.agent_pos))
        '''
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        print("agent position is:")
        for i in range(self.n_agents):
            print(self.agent_pos)
        '''
        # self.agent_dir = None

        # Initialize the RNG
        self.seed_val = seed
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        #print("reset called")
        # (ma.) Current position and direction of the agent
        #self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agents.reset_agent_pos()
        #for i in range(len(self.agent_pos)):
        #print('reset1:pos:',self.agent_pos)
        #self.agent_pos = None
        #print('reset:',type(self.agent_pos))
        # self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)
        #print('reset2:pos:',self.agent_pos)
        # (ma.) These fields should be defined by _gen_grid
        #assert self.agent_pos is not None
        #for i in range(len(self.agent_pos)):
        for i in range(self.agents.n_agents):
            #print("reset asser:",type(self.agent_pos))
            assert self.agents.agent_pos[i] is not None
        # assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        #for i in range(len(self.agent_pos)):
        for i in range(self.agents.n_agents):
            #x = *self.agent_pos[i]
            start_cell = self.grid.get(*self.agents.agent_pos[i])
            assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        #print('reset3:pos:',self.agents.agent_pos)
        # Return first observation
        obs = self.gen_obs()
        # return self.agents
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'uncovered'     : 'U',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        #no need for directions
        # AGENT_DIR_TO_STR = {
        #     0: '>',
        #     1: 'V',
        #     2: '<',
        #     3: '^'
        # }

        str = ''
        for x in range(self.agents.n_agents):
            for j in range(self.grid.height):
                #i don't know what this does, how do i replace the agent_dir string portion here?
                for i in range(self.grid.width):
                    #(ma)convert agent_pos in list of all agents
                    if i == self.agents.agent_pos[x][0] and j == self.agents.agent_pos[x][1]:
                        #print('agent pos['+i+'][0]:',self.agent_pos[i][0],'pos['+i+'][1]:',self.agent_pos[i][1])
                        str += '  '
                        continue

                    c = self.grid.get(i, j)
                    #print(c)

                    if c == None:
                        str += '  '
                        continue
                    #
                    # if c.type == 'door':
                    #     if c.is_open:
                    #         str += '__'
                    #     elif c.is_locked:
                    #         str += 'L' + c.color[0].upper()
                    #     else:
                    #         str += 'D' + c.color[0].upper()
                    #     continue

                    str += OBJECT_TO_STR[c.type] + c.color[0].upper()

                if j < self.grid.height - 1:
                    str += '\n'
                    #print('str is:')
        '''
        for i in range(self.n_agents):
            for j in range(2):
                print(self.agent_pos[i][j])
        '''

        #print(self.agent_pos[1])
        #print('***grid str is:',str)
        return str

    def _gen_grid(self, width, height):
        print('minigrid:gen')
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given at given step
        """
        reward = [None] * self.agents.n_agents
        #need to change reward from giving out reward at end goal(removed) to giving rewards at every step?
        for i in range(self.agents.n_agents):
            cell = self.grid.get(self.agents.agent_pos[i][0],self.agents.agent_pos[i][1])
            if cell is None:
                reward[i] = 0
            elif cell.type == 'uncovered':
                reward[i] = 100
            else:
                reward[i] = 0
        return reward

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

#    def getagents(self):
#        return self.n_agents

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            for i in range(len(self.agents.agent_pos)):
                if np.array_equal(pos, self.agents.agent_pos[i]):
                    continue
            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """
        for i in range(self.agents.n_agents):
            self.agents.agent_pos[i] = None
            pos = self.place_obj(None, top, size, max_tries=max_tries)
            self.agents.agent_pos[i] = pos

        # if rand_dir:
        #     self.agent_dir = self._rand_int(0, 4)

        return pos
    # Not sure if i need to keep this since removing direction
    # @property
    # def dir_vec(self):
    #     """
    #     Get the direction vector for the agent, pointing in the direction
    #     of forward movement.
    #     """

    #     assert self.agent_dir >= 0 and self.agent_dir < 4
    #     print(DIR_TO_VEC[self.agent_dir])
    #     return DIR_TO_VEC[self.agent_dir]

    # @property
    # def right_vec(self):
    #     """
    #     Get the vector pointing to the right of the agent.
    #     """

    #     # dx, dy = self.dir_vec
    #     return np.array((0, 1))

    # @property
    # def front_pos(self):
    #     """
    #     Get the position of the cell that is right in front of the agent
    #     """
    #
    #     return self.agent_pos + self.dir_vec

    @property
    def right_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        x = [None] * self.agents.n_agents
        for i in range(self.agents.n_agents):
            x[i] = self.agents.agent_pos[i] + np.array((1, 0))
        return x

    @property
    def down_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        x = [None] * self.agents.n_agents
        for i in range(self.agents.n_agents):
            x[i] = self.agents.agent_pos[i] + np.array((0, 1))
        return x

    @property
    def left_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        #print('pos[i],',self.agent_pos[0])
        x = [None] * self.agents.n_agents
        for i in range(self.agents.n_agents):
            x[i] = self.agents.agent_pos[i] + np.array((-1,0))
        return x
        #return self.agent_pos + np.array((-1, 0))

    # @property
    # def get_agent_pos(self):
    #     return self.agent_pos

    @property
    def up_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        x = [None] * self.agents.n_agents
        for i in range(self.agents.n_agents):
            x[i] = self.agents.agent_pos[i] + np.array((0,-1))
        return x

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agents.agent_pos
        dx, dy = np.array((1, 0))
        rx, ry = np.array((0, 1))

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self):
        #print('get_view_exts:0',self.agent_pos)

        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        topX = [None] * self.agents.n_agents
        topY = [None] * self.agents.n_agents
        botX = [None] * self.agents.n_agents
        botY = [None] * self.agents.n_agents


        for i in range(self.agents.n_agents):
            # Facing right
            if self.agent_dir == 0:
            #print('topx: type is ',type(self.agent_pos[0][0]))
                topX[i] = self.agents.agent_pos[i][0]
                #print('i is:',i,'type:',type(topX))
                topY[i] = self.agents.agent_pos[i][1] - self.agent_view_size // 2
            # Facing down
            elif self.agent_dir == 1:
                topX[i] = self.agents.agent_pos[i][0] - self.agent_view_size // 2
                topY[i] = self.agents.agent_pos[i][1]
            # Facing left
            elif self.agent_dir == 2:
                topX[i] = self.agents.agent_pos[i][0] - self.agent_view_size + 1
                topY[i] = self.agents.agent_pos[i][1] - self.agent_view_size // 2
            # Facing up
            elif self.agent_dir == 3:
                topX[i] = self.agents.agent_pos[i][0] - self.agent_view_size // 2
                topY[i] = self.agents.agent_pos[i][1] - self.agent_view_size + 1
            else:
                assert False, "invalid agent direction"

            botX[i] = topX[i] + self.agent_view_size
            botY[i] = topY[i] + self.agent_view_size

        #print('get_view_exts:1',topX,topY,botX,botY)
        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, action):
        # print(action)
        self.step_count += 1
        #add reward here to check if square moved to is in unchecked squares?
        reward = [None] * self.agents.n_agents
        done = False

        if len(action) != self.agents.n_agents:
            print('len of actions and # of agents is not same')
            #TODO: check o/p with return
            return
        # # Get the position in front of the agent
        # fwd_pos = self.front_pos
        #
        # # Get the contents of the cell in front of the agent
        # fwd_cell = self.grid.get(*fwd_pos)

        #initializing lists for multi-agent

        left_pos = [None] * len(action)
        left_cell = [None] * len(action)
        right_pos = [None] * len(action)
        right_cell = [None] * len(action)
        up_pos = [None] * len(action)
        up_cell = [None] * len(action)
        down_pos = [None] * len(action)
        down_cell = [None] * len(action)

        #cell direction contents
        #(m.a- TODO: make self.left_pos and all list instead of single value)
        left_pos = self.left_pos
        right_pos = self.right_pos
        up_pos = self.up_pos
        down_pos = self.down_pos

        for i in range(len(action)):
            left_cell[i] = self.grid.get(left_pos[i][0],left_pos[i][1])
            right_cell[i] = self.grid.get(right_pos[i][0],right_pos[i][1])
            up_cell[i] = self.grid.get(up_pos[i][0],up_pos[i][1])
            down_cell[i] = self.grid.get(down_pos[i][0],down_pos[i][1])


        #move left
        #for i in range(len(action)):
            #print("position actions: ",i,left_pos[i], right_pos[i], up_pos[i], down_pos[i])
            #print("agents ===>", self.agents.agent_pos)
            # print("before=====>",i, self.agents.agent_pos.values())
            if action[i] == self.actions.left:
                if (left_cell[i] == None or left_cell[i].can_overlap()) and tuple(left_pos[i]) not in self.agents.agent_pos.values():
                    self.agents.agent_pos[i] = tuple(left_pos[i])
            elif action[i] == self.actions.right:
                if (right_cell[i] == None or right_cell[i].can_overlap()) and tuple(right_pos[i]) not in self.agents.agent_pos.values():
                    self.agents.agent_pos[i] = tuple(right_pos[i])
            elif action[i] == self.actions.up:
                if (up_cell[i] == None or up_cell[i].can_overlap()) and tuple(up_pos[i]) not in self.agents.agent_pos.values():
                    self.agents.agent_pos[i] = tuple(up_pos[i])
            elif action[i] == self.actions.down:
                if (down_cell[i] == None or down_cell[i].can_overlap()) and tuple(down_pos[i]) not in self.agents.agent_pos.values():
                    self.agents.agent_pos[i] = tuple(down_pos[i])
            # Done action (not used by default)
            elif action[i] == self.actions.done:
                pass
            else:
                assert False, "unknown action"
            # print("after=====>",i, self.agents.agent_pos.values())

        #determine reward
        reward = self._reward()

        #grid string here
        grid_str = self.__str__()
        if 'U' not in grid_str:
            done = True

        #agent position
        # agent_position = self.agents.agent_pos
        # print(agent_position)
        # agent_cell = self.grid.get(*agent_position)
        # grid_size = self.width - 2
        #set cell as covered
        for i in range(self.agents.n_agents):
            self.grid.set(self.agents.agent_pos[i][0],self.agents.agent_pos[i][1],None)

        # if self.step_count >= self.max_steps:
        #     done = True

        obs = self.gen_obs()

        return obs, reward, done, self.agents, {}

        # return obs, reward, done, grid_str, {}

    def gen_obs_grid(self):
        #print('called get_obs_grid')
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """
        #print("genobsgrid: ",self.agent_pos)
        topX, topY, botX, botY = self.get_view_exts()
        new_topX = [x-1 for x in topX]
        grid = self.grid.slice(new_topX, topY, self.agent_view_size, self.agent_view_size)

        # Process occluders and visibility
        # Note that this incurs some performance cost
        #print('===genobsgrid 3:',self.agent_pos)
        # if not self.see_through_walls:
        #     vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        # else:
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
            0 - out of bounds
            1 - covered
            2 - wall
            8 - uncovered
            10- agent
        """
        obs = {}

        for key,val in self.agents.agent_pos.items():
            obs[key] = np.zeros((self.agent_view_size, self.agent_view_size), dtype='uint8')

            for j in range(0,self.agent_view_size):
                for i in range(0,self.agent_view_size):
                    #calculate actual position on env grid
                    actual_pos = (val[0]-self.agent_view_size//2+i, val[1]-self.agent_view_size//2+j)

                    if i==self.agent_view_size//2 and j ==self.agent_view_size//2:                  #agent location
                        obs[key][j,i] = 10
                    elif ~(0<=actual_pos[0]<self.grid.width and 0<=actual_pos[1]<self.grid.height): #outside env grid
                        obs[key][j,i] = 0
                    elif actual_pos in self.agents.agent_pos.values():                              #other agent
                        obs[key][j,i] = 10
                    elif self.grid.get(actual_pos[0], actual_pos[1]) is None:                       #covered cell
                        obs[key][j,i] = 1
                    else:                                                                           #uncovered/wall
                        obs[key][j,i] = OBJECT_TO_IDX[self.grid.get(actual_pos[0], actual_pos[1]).type]

        return obs

    def get_obs_render(self, obs, tile_size=CELL_PIXELS//2, mode='pixmap'):
        """
        Render an agent observation for visualization
        """

        if self.obs_render == None:
            from gym_minigrid.rendering import Renderer
            self.obs_render = Renderer(
                self.agent_view_size * tile_size,
                self.agent_view_size * tile_size
            )

        r = self.obs_render

        r.beginFrame()

        grid = Grid.decode(obs)

        # Render the whole grid
        grid.render(r, tile_size)

        # Draw the agent
        ratio = tile_size / CELL_PIXELS
        r.push()
        r.scale(ratio, ratio)
        r.translate(
            CELL_PIXELS * (0.5 + self.agent_view_size // 2),
            CELL_PIXELS * (self.agent_view_size - 0.5)
        )
        r.rotate(3 * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            ( 12,  0),
            (-12, -10)
        ])
        r.pop()

        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()
        return r

    def render(self, mode='human', close=False, highlight=True, tile_size=CELL_PIXELS):
        """
        Render the whole-grid human view
        """
        #print('render called, mode=',mode)
        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None or self.grid_render.window is None or (self.grid_render.width != self.width * tile_size):
            from gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.width * tile_size,
                self.height * tile_size,
                True if mode == 'human' else False
            )

        r = self.grid_render

        # if r.window:
        #     r.window.setText(self.mission)

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, tile_size)

        # Draw the agent
        for i in range(self.agents.n_agents):
            ratio = tile_size / CELL_PIXELS
            r.push()
            r.scale(ratio, ratio)
            from pprint import pprint
            #print("here\n\n")
            #pprint(self.agents.agent_pos)
            r.translate(
                CELL_PIXELS * (self.agents.agent_pos[i][0] + 0.5),
                CELL_PIXELS * (self.agents.agent_pos[i][1] + 0.5)
            )
            r.rotate(self.agent_dir * 90)
            r.setLineColor(0, 0, 0)
            r.setColor(51, 153, 255)
            # r.drawPolygon([
            #     (-12, 10),
            #     ( 12,  0),
            #     (-12, -10)
            # ])
            r.drawCircle(0,0,10)
            r.pop()

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the absolute coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = np.array((1, 0))
        r_vec = np.array((0, 1))
        top_left = [None] * self.agents.n_agents
        for i in range(self.agents.n_agents):
            top_left[i] = self.agents.agent_pos[i] + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2) - [self.agent_view_size // 2, 0]

            # For each cell in the visibility mask
            if highlight:
                for vis_j in range(0, self.agent_view_size):
                    for vis_i in range(0, self.agent_view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_mask[vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left[i] - (f_vec * vis_j) + (r_vec * vis_i)

                        # Highlight the cell
                        r.fillRect(
                            abs_i * tile_size,
                            abs_j * tile_size,
                            tile_size,
                            tile_size,
                            255, 255, 255, 75
                        )

        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()
        return r
