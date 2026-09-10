import numpy as np
from sandtris_parameterized_bot import extract_features
from sandtris_env_v10 import SHAPES

def find_best_placement_lookahead(env, weights, k=3, extraction_mode="clipped"):
    """
    K-Beam Search Lookahead.
    1. Evaluates all 68 current moves.
    2. Takes the top K highest scoring valid moves.
    3. For each of those K moves, clones the env, takes the move, and evaluates the expected value of the NEXT piece.
    4. Returns the move that maximizes this expected future value.
    """
    active_color = env.piece.color_idx
    
    # 1. Base Evaluation
    valid_actions, features_matrix = extract_features(env, active_color, extraction_mode)
    if len(valid_actions) == 0:
        return 0 # Default fallback if game is inevitably over
        
    base_scores = np.dot(features_matrix, weights)
    
    # 2. Pick top K actions
    k = min(k, len(valid_actions))
    # Get indices of top k scores
    top_k_indices = np.argsort(base_scores)[-k:][::-1]
    
    best_overall_action = None
    best_overall_score = -np.inf
    
    # 3. Simulate future for top K actions
    for idx in top_k_indices:
        action = valid_actions[idx]
        base_score = base_scores[idx]
        
        # Clone env and step
        future_env = env.clone()
        obs, reward, done, _, _ = future_env.step(action)
        
        if done:
            # If this action leads to immediate death, it's a terrible move.
            expected_future_score = -np.inf
        else:
            # 4. Expected Value over 7 next shapes
            total_future_score = 0.0
            shapes_evaluated = 0
            
            # Color doesn't matter much for physical geometry lookahead, just use the active color
            next_color = active_color 
            
            for shape in SHAPES:
                # To be completely isolated from other shape simulations, 
                # we don't even need to clone again because extract_features 
                # internally handles copying the state before dropping.
                future_env.piece.base_shape = list(shape)
                future_env.piece.shape = list(shape)
                future_env.piece.color_idx = next_color
                
                next_valid_actions, next_features = extract_features(future_env, next_color, extraction_mode)
                if len(next_valid_actions) > 0:
                    next_scores = np.dot(next_features, weights)
                    max_next_score = np.max(next_scores)
                    total_future_score += max_next_score
                else:
                    # Next piece causes unavoidable death
                    total_future_score -= 1000000.0 # Huge penalty
                shapes_evaluated += 1
                
            expected_future_score = total_future_score / shapes_evaluated
            
        # Combine base score and expected future score
        # Base score helps value immediate clears and immediate state quality
        combined_score = base_score + 0.5 * expected_future_score
        
        if combined_score > best_overall_score:
            best_overall_score = combined_score
            best_overall_action = action
            
    if best_overall_action is None:
        return valid_actions[0]
        
    return best_overall_action
