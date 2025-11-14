import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.logger import Logger

import numpy as np
import random 
import copy
import pygame

# ----- Game constants -----
GRAIN_SIZE = 5
GRID_WIDTH = 85
GRID_HEIGHT = 150
BLOCK_GRAIN = 5

COLORS = [
    (200, 100, 100),
    (100, 100, 200),
    (100, 200, 100),
    (200, 200, 100)
]

SHAPES = [
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # I
    [(0, 0), (1, 0), (0, 1), (1, 1)],  # O
    [(0, 0), (1, 0), (2, 0), (1, 1)],  # T
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # S
    [(1, 0), (2, 0), (0, 1), (1, 1)],  # Z
    [(0, 0), (0, 1), (1, 1), (2, 1)],  # J
    [(2, 0), (0, 1), (1, 1), (2, 1)],  # L
]

def make_world():
    return [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]



# ----- Piece class -----
class Piece:
    def __init__(self, world):
        self.world = world
        self.shape = random.choice(SHAPES)
        self.color = random.choice(COLORS)
        self.x = GRID_WIDTH // 2 - BLOCK_GRAIN
        self.y = 0
        self.grains = []
        self.update_grains()

    def update_grains(self):
        self.grains = []
        for bx, by in self.shape:
            for gx in range(BLOCK_GRAIN):
                for gy in range(BLOCK_GRAIN):
                    nx = self.x + bx*BLOCK_GRAIN + gx
                    ny = self.y + by*BLOCK_GRAIN + gy
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        self.grains.append([nx, ny])

    def move(self, dx, dy):
        if all(0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT and self.world[y+dy][x+dx] is None for x,y in self.grains):
            self.x += dx
            self.y += dy
            self.update_grains()
            return True
        return False

    def lock(self):
        for x, y in self.grains:
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                self.world[y][x] = self.color


# ----- Sandtris Environment -----
class SandtrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.world = make_world()
        self.piece = Piece(self.world)
        self.done = False

        self.action_space = spaces.Discrete(3)  # noop, left, right
        # rotation and down are removed for simplicity

        self.observation_space = spaces.Box(0, 1.0,
                                            shape=(3, GRID_HEIGHT, GRID_WIDTH),  # RGB channels for CnnPolicy
                                            dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((GRID_WIDTH*5, GRID_HEIGHT*5))
            self.clock = pygame.time.Clock()
            pass
            
    def reset(self, seed=None, options=None): # gym compatible reset
        super().reset(seed=seed)

        self.world = make_world()
        self.piece = Piece(self.world)
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
    
        reward = 0.0

        # Apply action
        if action == 1: 
            self.piece.move(-1, 0)
            reward = 0.01
        elif action == 2: 
            self.piece.move(1, 0)
            reward = 0.01
        
        # Penalty for noop
        elif action == 0:
            reward = -0.1 
            
        # Gravity
        locked = False
        if not self.piece.move(0, 1):
            locked = True
            self.piece.lock()
            self.step_sand()
            # Applies sand step before reward is calculated
            # Allows for rewarding bridges that form after sand step
            reward += self._clear_reward()
            self.piece = Piece(self.world)
        
        # Step sand if piece doesn't lock
        if not locked:
            self.step_sand()

        # Penalty for max height
        if not locked and self.piece.grains:
            max_y = max(y for x,y in self.piece.grains)
            reward -= (max_y / GRID_HEIGHT) * 0.5

        # Game over check if new piece immediately overlaps
        for x, y in self.piece.grains:
            if self.world[y][x] is not None:
                self.done = True
                reward -= 10.0

        return self._get_obs(), reward, self.done, False, {}
    
    def step_sand(self):
        world = self.world
        changed = True
        passes = 0
        max_passes = 5  # Grains can fall 5 rows per step
        
        while changed and passes < max_passes:
            changed = False
            for y in range(GRID_HEIGHT - 2, -1, -1): # Bottom up
                for x in range(GRID_WIDTH):
                    if world[y][x] is not None:
                        if world[y + 1][x] is None:
                            world[y + 1][x] = world[y][x]
                            world[y][x] = None
                            changed = True
                        else:
                            dirs = [-1, 1]
                            random.shuffle(dirs)
                            for d in dirs:
                                nx = x + d
                                if 0 <= nx < GRID_WIDTH and world[y + 1][nx] is None:
                                    world[y + 1][nx] = world[y][x]
                                    world[y][x] = None
                                    changed = True
                                    break
            passes += 1

        ''' 
        old step_sand with one pass

        world = self.world

        for y in range(GRID_HEIGHT - 2, -1, -1):
            for x in range(GRID_WIDTH):
                c = world[y][x]
                if c is not None and world[y + 1][x] is None:
                    world[y + 1][x] = c
                    world[y][x] = None
                elif c is not None:
                    dirs = [-1, 1]
                    random.shuffle(dirs)
                    for d in dirs:
                        nx = x + d
                        if 0 <= nx < GRID_WIDTH and world[y + 1][nx] is None:
                            world[y + 1][nx] = c
                            world[y][x] = None
                            break
        '''

    def _get_obs(self):
        # build the grid with ints
        obs = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8) 
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.world[y][x]:
                    obs[y][x] = COLORS.index(self.world[y][x]) + 1
        for x, y in self.piece.grains:
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                obs[y][x] = COLORS.index(self.piece.color) + 1
        
        # Convert to RGB format for CnnPolicy (3 channels) and normalize to 0-1 range
        rgb_obs = np.zeros((3, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        # Normalize the observation values to 0-1 range
        normalized_obs = obs.astype(np.float32) / (len(COLORS) + 1)
        rgb_obs[0] = normalized_obs  # Red channel
        rgb_obs[1] = normalized_obs  # Green channel  
        rgb_obs[2] = normalized_obs  # Blue channel
        return rgb_obs


    def _clear_reward(self):
        visited = [[False]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
        cleared_count = 0
        stepping_stone_reward = 0.0
        left_wall = 0
        right_wall = 0

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.world[y][x] and not visited[y][x]:
                    color = self.world[y][x]
                    stack = [(x, y)]
                    comp = [] # components/grains of the pile
                    left, right = GRID_WIDTH, -1
                    while stack:
                        cx, cy = stack.pop()
                        if not (0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT):
                            continue
                        if visited[cy][cx] or self.world[cy][cx] != color:
                            continue
                        visited[cy][cx] = True
                        comp.append((cx, cy))
                        left = min(left, cx)
                        right = max(right, cx)
                        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                            stack.append((cx+dx, cy+dy))
                    
        #-*-*-*-*-*-Reward-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

                    component_size = len(comp)
                    touches_left = (left == 0)
                    touches_right = (right == GRID_WIDTH - 1)
                    is_bridge = touches_left and touches_right

                    if is_bridge:
                        for cx, cy in comp:
                            self.world[cy][cx] = None
                        cleared_count += component_size
                    
                    elif (touches_left or touches_right) and component_size > 30:
                        stepping_stone_reward += (component_size ** 1.1) * 0.1

                        if touches_left:
                            left_wall += sum(1 for cx,cy in comp if cx == 0)
                        if touches_right:
                            right_wall += sum(1 for cx,cy in comp if cx == GRID_WIDTH-1)
                    
                    elif component_size > 50:
                        stepping_stone_reward += component_size * 0.05

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        wall_reward = (left_wall + right_wall) * 0.05

        filled_rows = [any(self.world[y][x] for x in range(GRID_WIDTH)) for y in range(GRID_HEIGHT)]
        maxheight = GRID_HEIGHT - np.argmax(filled_rows) if any(filled_rows) else 0

        # gets a list of how high each column is
        heights = [GRID_HEIGHT - next((y for y in range(GRID_HEIGHT) if self.world[y][x]), GRID_HEIGHT) for x in range(GRID_WIDTH)]
        heightstd = np.std(heights) # measures smoothness/flatness
        roughness = heightstd/GRID_HEIGHT # normalised

        #-*-*-*-*-*-Reward-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        reward = 0.0
        reward += (cleared_count**1.5) * 0.5
        reward += stepping_stone_reward
        reward += wall_reward
        
        reward -= roughness * 1.0
        reward -= ((maxheight/GRID_HEIGHT)**2) * 3.0

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # logging for tensorboard
        self.debug_info = {
            "cleared": cleared_count,
            "stepping_stone_reward": stepping_stone_reward,
            "wall_reward": wall_reward,
            "height": maxheight,
            "roughness": roughness,
            "reward": reward
        }

        if hasattr(self, "logger"):
            self.logger.record("reward/cleared", cleared_count)
            self.logger.record("reward/stepping_stone", stepping_stone_reward)
            self.logger.record("reward/roughness", roughness)
            self.logger.record("reward/maxheight", maxheight)
            self.logger.record("reward/total", reward)

        return reward

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((0,0,0))

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color = self.world[y][x]
                if color:
                    pygame.draw.rect(self.screen, color, (x*5, y*5, 5, 5))

        for x, y in self.piece.grains:
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                pygame.draw.rect(self.screen, self.piece.color, (x*5, y*5, 5, 5))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()