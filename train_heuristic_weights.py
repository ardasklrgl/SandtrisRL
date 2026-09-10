import cma
import numpy as np
import multiprocessing as mp
import os
import argparse
from tqdm import tqdm

from sandtris_env_v10 import SandtrisEnv
from sandtris_parameterized_bot import find_best_placement_parameterized

def evaluate_candidate(args):
    """
    Evaluates a single weight vector over the given seeds.
    Returns negative fitness (since CMA-ES minimizes).
    """
    weights, seeds, mode = args 
    
    total_fitness = 0.0
    sum_steps = 0.0
    sum_clears = 0.0
    
    for seed in seeds:
        env = SandtrisEnv()
        env.reset(seed=seed)
        
        done = False
        steps = 0
        clears = 0
        total_grains_cleared = 0
        
        while not done and steps < 1000:
            action = find_best_placement_parameterized(env, weights, extraction_mode=mode) # mode??? 
            obs, r, done, _, info = env.step(action)
            steps += 1
            
            grains_cleared_this_step = info.get('cleared_grains', 0)
            if grains_cleared_this_step > 0:
                clears += 1
                total_grains_cleared += grains_cleared_this_step
                
        fitness = steps + total_grains_cleared
        total_fitness += fitness
        sum_steps += steps
        sum_clears += clears
        
    avg_fitness = total_fitness / len(seeds)
    avg_steps = sum_steps / len(seeds)
    avg_clears = sum_clears / len(seeds)
    
    return (-avg_fitness, avg_steps, avg_clears)

def run_training(args):
    generations = args.generations
    popsize = 20
    seeds_per_gen = 3
    
    if args.mode == "binned":
        num_params = 47
        initial_std = 3.0
    elif args.mode == "engineered":
        num_params = 27
        initial_std = 3.0
    elif args.mode == "linear":
        num_params = 24
        initial_std = 3.0
    else:
        num_params = 24
        initial_std = 1.0
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_weights = np.zeros(num_params)
    
    if args.resume_from is not None:
        resume_path = os.path.join(script_dir, f"best_{args.mode}_weights_run{args.resume_from}.npy")
        if os.path.exists(resume_path):
            initial_weights = np.load(resume_path)
            print(f"Resuming {args.mode} training from {resume_path}!")
            initial_std = initial_std * 0.33 # Reduce std for fine-tuning
        else:
            print(f"Warning: {resume_path} not found. Starting from zeros.")
    
    es = cma.CMAEvolutionStrategy(initial_weights, initial_std, {'popsize': popsize})
    
    print(f"Starting CMA-ES Training for {generations} generations in {args.mode} mode.")
    
    rng = np.random.RandomState(42)
    pool = mp.Pool(processes=mp.cpu_count())
    fitness_history = []
    
    pbar = tqdm(range(generations), desc="Training CMA-ES")
    
    for gen in pbar:
        solutions = es.ask()
        gen_seeds = rng.randint(0, 1000000, size=seeds_per_gen).tolist()
        args_list = [(sol, gen_seeds, args.mode) for sol in solutions]
        
        results = list(pool.imap(evaluate_candidate, args_list))
        
        fitnesses = [res[0] for res in results]
        steps = [res[1] for res in results]
        clears = [res[2] for res in results]
        
        es.tell(solutions, fitnesses)
        
        best_idx = np.argmin(fitnesses)
        best_fit = -fitnesses[best_idx]
        best_steps = steps[best_idx]
        best_clears = clears[best_idx]
        avg_fit = -np.mean(fitnesses)
        
        fitness_history.append(best_fit)
        
        pbar.set_postfix({
            "Best Fit": f"{best_fit:.1f}", 
            "Avg Fit": f"{avg_fit:.1f}",
            "Max Steps": f"{best_steps:.1f}",
            "Max Clears": f"{best_clears:.1f}"
        })
        
        best_weights = es.result.xbest
            
    # Save final best weights
    final_path = os.path.join(script_dir, f"best_{args.mode}_weights_run{args.run_id}.npy")
    np.save(final_path, best_weights)
    
    # Save learning curve
    curve_path = os.path.join(script_dir, f"learning_curve_{args.mode}_run{args.run_id}.npy")
    np.save(curve_path, np.array(fitness_history))
    
    print(f"\nFinal Training Complete! Saved to {final_path}")
    print("Best Weights Array:")
    print(repr(best_weights))
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Sandtris RL Bot")
    parser.add_argument('--mode', type=str, choices=['clipped', 'linear', 'engineered', 'binned'], default='engineered', help="Feature extraction mode")
    parser.add_argument('--run_id', type=int, default=1, help="Run ID for multi-seed evaluation")
    parser.add_argument('--resume_from', type=int, default=None, help="Run ID to load initial weights from")
    parser.add_argument('--generations', type=int, default=20, help="Number of generations")
    args = parser.parse_args()
    
    # Set start method to 'spawn' for safe multiprocessing with ctypes on mac/linux
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    run_training(args)
