import os
import numpy as np
import time
import pygame
import argparse
from sandtris_env_v10 import SandtrisEnv
from sandtris_parameterized_bot import find_best_placement_parameterized

def watch_bot():
    parser = argparse.ArgumentParser(description="Watch Sandtris RL Bot")
    parser.add_argument('--mode', type=str, choices=['clipped', 'linear', 'engineered', 'binned'], default='clipped', help="Feature extraction mode")
    parser.add_argument('--run_id', type=int, default=1, help="Run ID to load weights from")
    parser.add_argument('--lookahead', action='store_true', help="Use Beam Search Lookahead")
    args = parser.parse_args()
    if args.mode == "binned":
        num_params = 47
    elif args.mode == "engineered":
        num_params = 27
    else:
        num_params = 24
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    weights_path = os.path.join(script_dir, f"best_{args.mode}_weights_run{args.run_id}.npy")
    fallback_path = os.path.join(script_dir, "best_heuristic_weights_final.npy")
    
    if os.path.exists(weights_path):
        learned_weights = np.load(weights_path)
        print(f"Loaded {args.mode} mode weights from {weights_path}")
    elif os.path.exists(fallback_path):
        learned_weights = np.load(fallback_path)
        print(f"Loaded fallback weights from {fallback_path}")
    else:
        print(f"Error: Could not find weights at {weights_path} or {fallback_path}")
        return
        
    if len(learned_weights) == 25 and args.mode != "engineered":
        learned_weights = np.delete(learned_weights, 5)
        print("Auto-sliced 25-param weights down to 24-param vector for compatibility.")
        
    print(f"Loaded learned weights: {learned_weights}")
    
    env = SandtrisEnv(render_mode="human")
    seed = int(time.time())
    print(f"Playing on random seed: {seed}")
    
    obs, info = env.reset(seed=seed)
    env.render()
    
    done = False
    steps = 0
    clears = 0
    
    while not done:
        # Event pump to keep Pygame window responsive and allow closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
                
        if done:
            break
            
        t0 = time.time()
        if args.lookahead:
            from sandtris_lookahead_bot import find_best_placement_lookahead
            action = find_best_placement_lookahead(env, learned_weights, extraction_mode=args.mode)
        else:
            action = find_best_placement_parameterized(env, learned_weights, extraction_mode=args.mode)
        t1 = time.time()
        
        obs, r, done, _, _ = env.step(action)
        env.render()
        
        steps += 1
        if r >= 100.0:
            clears += 1
            
        print(f"Step: {steps:3d} | Clears: {clears} | Think Time: {t1-t0:.3f}s")
        
        # Add a small delay so you can actually see it play
        time.sleep(0.001)
        
    print(f"\\nGame Over! Survived {steps} steps with {clears} clears.")
    pygame.quit()

if __name__ == "__main__":
    watch_bot()
