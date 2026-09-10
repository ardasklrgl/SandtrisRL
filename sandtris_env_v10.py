import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from scipy.ndimage import label
import ctypes
import os

_c_core = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandtris_c_core.so'))
_c_core.c_step_sand.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS')]

# ----- Original Sandtris Game Constants -----
GRAIN_SIZE = 5
GRID_WIDTH = 85      # Original 85 columns
GRID_HEIGHT = 150    # Original 150 rows
BLOCK_GRAIN = 5     # Original 5x5 grains per block (100 grains per piece!)

COLORS = [
    (200, 100, 100),  # Red
    (100, 100, 200),  # Blue
    (100, 200, 100),  # Green
    (200, 200, 100)   # Yellow
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

def rotate_shape(shape, rotation):
    """Rotate block coordinates clockwise N times and normalize min coords to (0,0)."""
    curr = list(shape)
    for _ in range(rotation % 4):
        curr = [(-y, x) for x, y in curr]
    min_x = min(x for x, y in curr)
    min_y = min(y for x, y in curr)
    return [(x - min_x, y - min_y) for x, y in curr]

def make_world():
    return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)

class Piece:
    """Original Sandtris Piece: 4 blocks x (5x5 grains) = 100 fine sand grains per piece."""
    def __init__(self, world, rng):
        self.world = world
        self.rng = rng
        shape_idx = int(self.rng.integers(0, len(SHAPES)))
        self.base_shape = SHAPES[shape_idx]
        self.shape = list(self.base_shape)
        self.color_idx = int(self.rng.integers(1, len(COLORS) + 1))
        
        self.x = 0
        self.y = 0
        self.grains = []
        self.update_grains()

    def set_rotation(self, rotation):
        self.shape = rotate_shape(self.base_shape, rotation)
        self.update_grains()

    def update_grains(self):
        self.grains = []
        for bx, by in self.shape:
            base_x = self.x + bx * BLOCK_GRAIN
            base_y = self.y + by * BLOCK_GRAIN
            for gx in range(BLOCK_GRAIN):
                for gy in range(BLOCK_GRAIN):
                    nx = base_x + gx
                    ny = base_y + gy
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        self.grains.append((nx, ny))

    def can_move(self, dx, dy):
        for x, y in self.grains:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                return False
            if self.world[ny, nx] != 0:
                return False
        return True

    def move(self, dx, dy):
        if self.can_move(dx, dy):
            self.x += dx
            self.y += dy
            self.update_grains()
            return True
        return False

    def lock(self):
        for x, y in self.grains:
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                self.world[y, x] = self.color_idx

    def clone(self, new_world):
        p = Piece.__new__(Piece)
        p.world = new_world
        p.rng = self.rng
        p.base_shape = list(self.base_shape)
        p.shape = list(self.shape)
        p.color_idx = self.color_idx
        p.x = self.x
        p.y = self.y
        p.grains = list(self.grains)
        return p


class SandtrisEnv(gym.Env):
    """
    Sandtris Gym Environment v10 (Original Full 150x85 Game + Exponential Survival Rewards)
    
    Action Space: Discrete(68) -> 17 block columns x 4 rotations.
      action = rot_idx * 17 + block_col (snaps cleanly to 5px grain grid)
    Observation Space: Box(0.0, 1.0, shape=(3, 150, 85), dtype=np.float32)
      Channel 0: World Terrain (85x150 fine sand)
      Channel 1: Active Piece (100 grains)
      Channel 2: Occupation Mask
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.world = make_world()
        self._rng = np.random.default_rng()
        self.piece = Piece(self.world, self._rng)
        self.done = False
        self.step_count = 0

        # Clean 68 discrete actions (17 block columns x 4 rotations)
        self.num_rotations = 4
        self.num_block_cols = GRID_WIDTH // BLOCK_GRAIN  # 85 // 5 = 17
        self.action_space = spaces.Discrete(self.num_block_cols * self.num_rotations)

        # Full Original 3-channel visual observation (3, 150, 85)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self.world = make_world()
        self.piece = Piece(self.world, self._rng)
        self.done = False
        self.step_count = 0
        return self._get_obs(), {}

    def clone(self):
        new_env = SandtrisEnv(render_mode=self.render_mode)
        new_env.world = self.world.copy()
        new_env._rng = self._rng
        new_env.piece = self.piece.clone(new_env.world)
        new_env.done = self.done
        new_env.step_count = self.step_count
        return new_env

    def _get_obs(self):
        """
        Observation tensor for an RL agent, e.g. a CNN.
        Creates the 3 channels. Normalised between 0.0-1.0
        """
        denom = float(len(COLORS) + 1) # 5.0

        # Channel 0, fallen sand
        world_layer = self.world.astype(np.float32) / denom

        player_layer = np.zeros_like(world_layer)
        for x, y in self.piece.grains:
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                # Channel 1, active piece
                player_layer[y, x] = float(self.piece.color_idx) / denom

        # Channel 2, the occupied grains, active or not, ignoring color, just 0.0 or 1.0.
        mask_layer = ((world_layer > 0.0) | (player_layer > 0.0)).astype(np.float32)

        return np.stack([world_layer, player_layer, mask_layer], axis=0).astype(np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        self.step_count += 1

        # 1. Decode Action (Rotation & Target Block Column)
        rot_idx = action // self.num_block_cols
        block_col = action % self.num_block_cols

        self.piece.set_rotation(rot_idx)

        # Calculate bounding width of piece in pixels
        max_bx = max(bx for bx, _ in self.piece.shape)
        piece_pixel_width = (max_bx + 1) * BLOCK_GRAIN
        max_safe_x = GRID_WIDTH - piece_pixel_width

        target_pixel_x = block_col * BLOCK_GRAIN
        safe_x = max(0, min(target_pixel_x, max_safe_x))
        safe_x = (safe_x // BLOCK_GRAIN) * BLOCK_GRAIN

        self.piece.x = safe_x
        self.piece.y = 0
        self.piece.update_grains()

        # Check immediate game over (Spawn kill / blockage)
        if not self.piece.can_move(0, 0):
            self.done = True
            return self._get_obs(), -20.0, True, False, {}

        # 2. Hard Drop (Move straight down)
        while self.piece.move(0, 1):
            pass

        """
        Rewards Section for an RL agent.
        This part can be and should be modified.
        It is not used in the CMA-ES training.
        """
        # 3. Calculate Rewards
        reward = 0.0

        # A. EXPONENTIAL / PROGRESSIVE SURVIVAL REWARD (Rewards staying alive!)
        survival_reward = 0.05 * ((1.0 + self.step_count / 80.0) ** 1.4)
        reward += survival_reward

        # B. Color Matching Touch Reward
        reward += self._calculate_touch_reward()

        # Lock piece into world and run sand cellular automata
        self.piece.lock()
        self.step_sand()

        # C. Original Sandtris Fine-Grain Line Clearing Reward
        clear_bonus, cleared_grains = self._clear_reward()
        reward += clear_bonus

        # Check Game Over condition (top row filled)
        if np.any(self.world[0, :] != 0):
            self.done = True
            reward -= 20.0

        # Spawn new piece for next step
        self.piece = Piece(self.world, self._rng)

        return self._get_obs(), float(reward), self.done, False, {'cleared_grains': cleared_grains} 
        """Cleared Grains is the only fitness/reward metric the CMA-ES solution uses, 
        along with the steps (no. of pieces) survived. """

    def _calculate_touch_reward(self):
        """Dense reward when piece grains touch existing sand of matching color."""
        matches = 0
        total_grains = len(self.piece.grains)
        if total_grains == 0:
            return 0.0

        for x, y in self.piece.grains:
            check_y = y + 1
            if check_y < GRID_HEIGHT:
                if self.world[check_y, x] == self.piece.color_idx:
                    matches += 1

        return (matches / total_grains) * 0.4

    def step_sand(self):
        """Vectorized Cellular Automata for original fine sand particles."""
        # Fast C-core simulation
        c_world = np.ascontiguousarray(self.world, dtype=np.uint8)
        _c_core.c_step_sand(c_world)
        self.world[:] = c_world

    def _clear_reward(self):
        """Original Sandtris Line Clear: Scipy component label spanning left (x=0) to right (x=84)."""
        cleared_grains = 0
        H, W = self.world.shape
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]])

        for color_idx in range(1, len(COLORS) + 1):
            mask = (self.world == color_idx)
            if not np.any(mask):
                continue
            labeled, num_features = label(mask, structure=structure)
            for feat in range(1, num_features + 1):
                comp_mask = (labeled == feat)
                cols_with_comp = np.where(comp_mask)[1]
                if 0 in cols_with_comp and (W - 1) in cols_with_comp:
                    self.world[comp_mask] = 0
                    cleared_grains += np.sum(comp_mask)

        reward = 0.0
        if cleared_grains > 0:
            # Non-linear clear bonus (e.g. 85 grains cleared = 1 full line = +25.0 reward)
            reward = float((cleared_grains ** 1.3) * 0.1 + 15.0)

        return reward, cleared_grains

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "rgb_array":
            img = np.zeros((GRID_HEIGHT * GRAIN_SIZE, GRID_WIDTH * GRAIN_SIZE, 3), dtype=np.uint8)
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    idx = int(self.world[y, x])
                    if idx != 0:
                        c = COLORS[idx - 1]
                        img[y*GRAIN_SIZE:(y+1)*GRAIN_SIZE, x*GRAIN_SIZE:(x+1)*GRAIN_SIZE] = c
            return img

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((GRID_WIDTH * GRAIN_SIZE, GRID_HEIGHT * GRAIN_SIZE))
                self.clock = pygame.time.Clock()

            self.screen.fill((0, 0, 0))
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    idx = int(self.world[y, x])
                    if idx != 0:
                        pygame.draw.rect(
                            self.screen, COLORS[idx - 1],
                            (x * GRAIN_SIZE, y * GRAIN_SIZE, GRAIN_SIZE, GRAIN_SIZE)
                        )

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None and self.render_mode == "human":
            pygame.quit()
            self.screen = None
