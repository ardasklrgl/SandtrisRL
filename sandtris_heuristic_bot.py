import time
import copy
import numpy as np
import pygame
from scipy.ndimage import label

from sandtris_env_v10 import SandtrisEnv, GRID_WIDTH, GRID_HEIGHT, BLOCK_GRAIN, COLORS, Piece


def evaluate_board_state(world, active_color):
    """
    Heuristic Evaluator for Sandtris:
    Calculates the 'closeness to clearing' by finding color components and measuring
    how many grains away they are from spanning wall-to-wall (x=0 to x=GRID_WIDTH-1).
    """
    H, W = world.shape
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
    
    total_gap_distance = 0
    active_color_gap = W  # Best gap for the color of the current piece
    cleared_grains = 0

    for color_idx in range(1, len(COLORS) + 1):
        mask = (world == color_idx) # boolean numpy 2d array where value is True if world[y, x] == color_idx
        
        if not np.any(mask): # if no sand of current color, skip
            continue
            
        labeled, num_features = label(mask, structure=structure) # finding clusters in sand
        # labeled is a 2d int array same size as world. 0 is background, 1 is all pixels of the first cluster, 2 is next cluster...
        # num_features is the number of clusters.
        # structure is the connectivity parameter of clusters, anything touching orthogonally

        for feat in range(1, num_features + 1):
            comp_mask = (labeled == feat) # get mask of a single cluster
            cols = np.where(comp_mask)[1] # find all 2d coordinates (row, col) where comp_mask is true.
            # Tuple of 1d np arrays is returned, and we get the col array.

            if len(cols) == 0:
                continue

            x_min = np.min(cols)
            x_max = np.max(cols)

            # Check wall touching status
            left_touch = (x_min == 0)
            right_touch = (x_max == W - 1)

            if left_touch and right_touch:
                # Component spans completely wall-to-wall!
                cleared_grains += np.sum(comp_mask)
            elif left_touch:
                # Distance remaining to right wall
                gap = (W - 1) - x_max
                total_gap_distance += gap #**************# 
                if color_idx == active_color:
                    active_color_gap = min(active_color_gap, gap) 
            elif right_touch:
                # Distance remaining to left wall
                gap = x_min
                total_gap_distance += gap #**************# 
                if color_idx == active_color:
                    active_color_gap = min(active_color_gap, gap)
            else:
                # Floating island component in middle
                gap = (W - 1 - x_max) + x_min
                total_gap_distance += gap * 1.5 # heuristic weight
                # can be removed???

    # Surface height penalty
    filled_rows = [np.any(world[y, :] != 0) for y in range(H)]
    max_height = (H - np.argmax(filled_rows)) if any(filled_rows) else 0

    # Final Cost Function (Lower is better!)
    # Its not a pure certain value based cost, there are weight. Keep in mind....
    """
    cost = (
        - 50.0 * cleared_grains          # Huge reward for completing a bridge
        + 10.0 * active_color_gap        # Priority: Get active piece color closer to clearing
        + 2.0 * total_gap_distance       # Overall closeness to clearing across all colors
        + 0.5 * max_height               # Keep stack low
    )
    return cost

    """
    return total_gap_distance
    

    

def find_best_placement(env):
    """
    Scans all valid horizontal placements (17 block columns), simulates the placement
    and sand flow on a cloned copy of the environment, and selects the minimum cost action.
    """
    best_action = 0
    best_cost = float("inf")
    num_block_cols = GRID_WIDTH // BLOCK_GRAIN

    # Search through target columns (assuming 0 rotation for now)
    rot_idx = 0
    for block_col in range(num_block_cols):
        action = rot_idx * num_block_cols + block_col

        # Clone current world & piece
        sim_world = env.world.copy()
        sim_piece = Piece(sim_world, env._rng)
        sim_piece.shape = list(env.piece.shape)
        sim_piece.color_idx = env.piece.color_idx

        # Calculate safe horizontal position
        max_bx = max(bx for bx, _ in sim_piece.shape)
        piece_pixel_width = (max_bx + 1) * BLOCK_GRAIN
        max_safe_x = GRID_WIDTH - piece_pixel_width
        safe_x = max(0, min(block_col * BLOCK_GRAIN, max_safe_x))
        safe_x = (safe_x // BLOCK_GRAIN) * BLOCK_GRAIN

        sim_piece.x = safe_x
        sim_piece.y = 0
        sim_piece.update_grains()

        # If spawn is blocked, skip
        if not sim_piece.can_move(0, 0):
            continue

        # Hard drop
        while sim_piece.move(0, 1):
            pass

        # Lock & run sand flow
        sim_piece.lock()
        
        # Simple sand physics simulation for evaluation
        for _ in range(5):
            empty_below = (sim_world[1:, :] == 0) & (sim_world[:-1, :] != 0)
            if np.any(empty_below):
                rows, cols = np.where(empty_below)
                sim_world[rows + 1, cols] = sim_world[rows, cols]
                sim_world[rows, cols] = 0

        # Evaluate candidate placement
        cost = evaluate_board_state(sim_world, sim_piece.color_idx)

        if cost < best_cost:
            best_cost = cost
            best_action = action

    return best_action


def run_heuristic_bot(episodes=3, render=True):
    """Run the Heuristic Search Sandtris Bot live."""
    env = SandtrisEnv(render_mode="human" if render else None)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        print(f"\n🎮 Starting Episode {ep}...")

        while not done:
            # Choose best placement using Closeness-to-Clearing Heuristic
            action = find_best_placement(env)

            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.05)

        print(f"✅ Episode {ep} Finished | Steps Survived: {steps} | Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    run_heuristic_bot(episodes=3, render=True)
