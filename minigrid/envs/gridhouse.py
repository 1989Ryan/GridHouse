from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, ROOM_NAMES, \
    ROOM_NAMES_TO_COLORS, ROOM_OBJS, OBJFUNC, OBJNAMES
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
    | 5   | toggle       | Unused                    |
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

    ## Registered Configurations

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(self, size=19, max_steps: int | None = None, **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 10 * size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[OBJNAMES],
        )
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )
        self.target_object = None


    @staticmethod
    def _gen_mission(tar_obj: str):
        return (
            f"get something to {OBJFUNC[tar_obj]}."
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
            self.rooms.append(LockedRoom((0, j), (roomW, roomH), (lWallIdx, j + 2)))
            self.rooms.append(
                LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 2))
            )

        # Choose one random room to be locked
        lockedRoom = self._rand_elem(self.rooms)
        # lockedRoom.locked = True


        # Assign the door colors
        room_names = set(ROOM_NAMES)
        for room in self.rooms:
            room_name = self._rand_elem(sorted(room_names))
            room_names.remove(room_name)
            room.name = room_name
            room.color = ROOM_NAMES_TO_COLORS[room_name]
            self.grid.set(*room.doorPos, Door(ROOM_NAMES_TO_COLORS[room_name]))

        # Select a random room to contain the key
        rn = random.uniform(0, 1)
        obj_list = ["apple", "t-shirt", "cup"]
        obj_rooms = {}
        objposlist = {
            "apple": None,
            "t-shirt": None,
            "cup": None,
        }

        for obj_name in ["apple", "t-shirt", "cup"]:
            for room in self.rooms:
                if rn > 0.2:
                    if obj_name in ROOM_OBJS['room'] and obj_name in obj_list:
                        obj_rooms[obj_name] = room
                elif obj_name in obj_list:
                    obj_rooms[obj_name] = self._rand_elem(self.rooms)
                obj_list.remove(obj_name)
        for obj_name in list(obj_rooms.keys()):
            ObjPos = obj_rooms[obj_name].rand_pos(self)
            if obj_name == "apple":
                self.grid.set(*ObjPos, Apple(lockedRoom.color))
            if obj_name == "t-shirt":
                self.grid.set(*ObjPos, Tshirt(lockedRoom.color))
            if obj_name == "cup":
                self.grid.set(*ObjPos, Cup(lockedRoom.color))
            objposlist[obj_name] = ObjPos
        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(lWallIdx, 0), size=(rWallIdx - lWallIdx, height)
        )
        # goalPos = lockedRoom.rand_pos(self)
        # goal_obj = random.choice(["apple", "t-shirt", "cup"])
        # self.grid.set(*objposlist[goal_obj], Goal())
        # Generate the mission string
        tar_obj = random.choice(["apple", "t-shirt", "cup"])
        self.mission = (
            "give me something to %s"
        ) % (OBJFUNC[tar_obj])
        self.target_object = tar_obj

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