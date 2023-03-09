from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, ROOM_NAMES, \
    ROOM_NAMES_TO_COLORS, ROOM_OBJS, OBJFUNC, OBJNAMES, OBJFUNCLIST, DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Cup, Apple, Tshirt, Wall
from minigrid.minigrid_env import MiniGridEnv
import random
import numpy as np

class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False
        self.name = None

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)
    


class GridHouseEnv(MiniGridEnv):

    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Drop an object            |
    | 5   | toggle       | Open the door             |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered posigurations

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(self, size=15, max_steps: int | None = None, **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 10 * size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[sorted(OBJFUNCLIST)],
        )
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )
        self.target_object = None
        self.planner = AStar()


    @staticmethod
    def _gen_mission(obj_func: str):
        return (
            f"get something to {obj_func}."
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # Room splitting walls
        for n in range(0, 2):
            j = n * (height // 2)
            for i in range(0, lWallIdx):
                self.grid.set(i, j, Wall())
            for i in range(rWallIdx, width):
                self.grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 2 + 1
            self.rooms.append(LockedRoom((0, j), (roomW, roomH), (lWallIdx, j + 3)))
            self.rooms.append(
                LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 4))
            )

        # Choose one random room to be locked

        # Assign the door colors
        room_names = sorted(ROOM_NAMES)
        idx = 0
        for room in self.rooms:
            room_name = room_names[idx]
            room.name = room_name
            room.color = ROOM_NAMES_TO_COLORS[room_name]
            self.grid.set(*room.doorPos, Door(ROOM_NAMES_TO_COLORS[room_name]))
            idx += 1 
        # Select a random room to contain the key
        obj_list = ["apple", "t-shirt", "cup"]
        obj_rooms = {}

        for obj_name in ["apple", "t-shirt", "cup"]:
            for room in self.rooms:
                rn = random.uniform(0, 1)
                if rn > 0.2:
                    if obj_name in ROOM_OBJS[room.name] and obj_name in obj_list:
                        obj_rooms[obj_name] = room
                        obj_list.remove(obj_name)
                        continue
                elif obj_name in obj_list:
                    obj_rooms[obj_name] = self._rand_elem(self.rooms)
                    obj_list.remove(obj_name)
                    continue
        assert len(obj_rooms.keys()) == 3
        # print(obj_rooms.keys())
        for obj_name in list(obj_rooms.keys()):
            ObjPos = obj_rooms[obj_name].rand_pos(self)
            if obj_name == "apple":
                self.grid.set(*ObjPos, Apple("red"))
            elif obj_name == "t-shirt":
                self.grid.set(*ObjPos, Tshirt("green"))
            else:
                self.grid.set(*ObjPos, Cup("yellow"))
        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(lWallIdx, 0), size=(rWallIdx - lWallIdx, height)
        )

        tar_obj = random.choice(["apple", "t-shirt", "cup"])
        self.mission = (
            "get something to %s."
        ) % (OBJFUNC[tar_obj])
        self.target_object = tar_obj

       
    def gen_traj(self):
        room = self._rand_elem(self.rooms)
        while True:
            pos = room.rand_pos(self)
            goal_grid = self.grid.get(*pos)
            if goal_grid is None or goal_grid.can_overlap():
                break
        action = self.planner(self.agent_pos, self.agent_dir, pos, self.grid)
        return action

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs


    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    if fwd_cell.type == self.target_object:
                        terminated = True
                        reward = self._reward()
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


class Node:
    def __init__(self, came_from, node_type=None, pos=None, cost=None, 
                 direc=None, action_from=None,
                 ):
        if isinstance(action_from , int):
            action_from = [action_from]
        self.type = node_type
        self.pos = pos
        self.direc = direc
        self.cost = cost
        self.came_from = came_from
        self.action_from = action_from
        self.neighbor =None
    def get_neighbor(self, grid: Grid):
        direc=self.direc
        fwd_pos = self.front_pos
        """
        ## Action Space

        | Num | Name         | Action                    |
        |-----|--------------|---------------------------|
        | 0   | left         | Turn left                 |
        | 1   | right        | Turn right                |
        | 2   | forward      | Move forward              |
        | 3   | pickup       | Pick up an object         |
        | 4   | drop         | Drop an object            |
        | 5   | toggle       | Open the door             |
        | 6   | done         | Unused                    |
        """
        neighbor = []
        fwd_cell = grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            neighbor.append(Node(pos = fwd_pos, 
                                 direc=direc, action_from=[2], came_from=self))
        elif fwd_cell.type=='door':
             neighbor.append(Node(pos =fwd_pos, 
                             direc=direc, came_from=self, action_from=[2, 5],))
        right_cell = grid.get(*self.get_front_pos(self.pos, (direc+1)%4))
        if right_cell is None or right_cell.can_overlap():
            neighbor.append(Node(node_type = self.type, pos = self.get_front_pos(self.pos, (direc+1)%4), 
                             direc=(direc+1)%4, came_from=self, action_from=[2, 1]))
        elif right_cell.type == "door":
            neighbor.append(Node(node_type = self.type, pos = self.get_front_pos(self.pos, (direc+1)%4), 
                             direc=(direc+1)%4, came_from=self, action_from=[2, 5, 1],))
        left_cell = grid.get(*self.get_front_pos(self.pos, (direc-1)%4))
        if left_cell is None or left_cell.can_overlap():
            neighbor.append(Node(node_type = self.type, pos = self.get_front_pos(self.pos, (direc-1)%4), 
                             direc=(direc-1)%4, came_from=self, action_from=[2, 0,]))
        elif left_cell.type == "door":
            neighbor.append(Node(node_type = self.type, pos = self.get_front_pos(self.pos, (direc-1)%4), 
                             direc=(direc-1)%4, came_from=self, action_from=[2, 5, 0],))
        back_cell = grid.get(*self.get_front_pos(self.pos, (direc-2)%4))
        if back_cell is None or back_cell.can_overlap():
            neighbor.append(Node(node_type = self.type, pos = self.get_front_pos(self.pos, (direc-2)%4), 
                             direc=(direc-2)%4, came_from=self, action_from=random.choice([[2, 0, 0], [2, 1, 1]])))
        elif back_cell.type == "door":
            neighbor.append(Node(node_type = self.type, pos = self.get_front_pos(self.pos, (direc-2)%4), 
                             direc=(direc-2)%4, came_from=self, action_from=random.choice([[2, 5, 0, 0], [2, 5, 1, 1,]])))
        return neighbor
    @property
    def front_pos(self):
        return self.get_front_pos(self.pos, self.direc)
        # return tuple((self.pos + DIR_TO_VEC[self.direc]).tolist())
    @staticmethod
    def get_front_pos(pos, direc):
        return tuple((pos + DIR_TO_VEC[direc]).tolist())

class AStar:
    def __init__(self):
        pass

    @staticmethod
    def H(start: Node, goal: Node):
        # Heuristic function to estimate the cost from start to goal
        s = np.array(start.pos)
        g = np.array(goal.pos)
        return np.sum(np.abs(s-g))

    def _search(self, start: Node, goal: Node, grid: Grid):
        start.cost = self.H(start, goal)
        openset = [start]
        gscore = {start.pos: 0}
        fscore = {start.pos: start.cost}

        while len(openset) != 0:
            current = min(set(openset), key=lambda x: fscore[x.pos])
            if current.pos[0] == goal.pos[0] and current.pos[1] == goal.pos[1]:
                print("Trajectory found by A-Star!")
                # If we reach the goal node, reconstruct the path and return it
                path = []
                while current.pos[0] != start.pos[0] or current.pos[1] != start.pos[1]\
                    or current.direc != start.direc:
                    path = path + current.action_from
                    current = current.came_from
                path.reverse()
                return path

            openset.remove(current)
            # closedset.append(current)
            current.neighbor = current.get_neighbor(grid)
            for neighbor in current.neighbor:
                neighbor.cost = self.H(neighbor, goal)
                tentative_gscore = gscore[current.pos] + self.H(neighbor, current)
                if neighbor.pos not in gscore or tentative_gscore < gscore[neighbor.pos]:
                    # This path is the best until now. Record it!
                    gscore[neighbor.pos] = tentative_gscore
                    fscore[neighbor.pos] = gscore[neighbor.pos] + self.H(neighbor, goal)
                    if neighbor not in openset:
                        openset.append(neighbor)

        return None

    def __call__(self, start_pos, start_direc, goal_pos, grid: Grid):
        start = Node(pos=start_pos, direc=start_direc, came_from=None)
        goal = Node(pos=goal_pos, direc=start_direc, came_from=None)
        return self._search(start, goal, grid=grid)