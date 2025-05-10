import pandas as pd
import numpy as np
from collections import defaultdict
import multiprocessing
from functools import lru_cache
from tqdm import tqdm
from joblib import Parallel, delayed

# Load data
u_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
users = pd.read_csv("./data/ml-100k/u.user", sep="|", header=None, names=u_cols, encoding="latin-1")

# Load ratings
r_cols = ["user_id", "item_id", "rating", "unix_timestamp"]
ratings_base = pd.read_csv("./data/ml-100k/ua.base", sep="\t", header=None, names=r_cols, encoding="latin-1")
ratings_test = pd.read_csv("./data/ml-100k/ua.test", sep="\t", header=None, names=r_cols, encoding="latin-1")

# Precompute user means and create user-item matrices for faster lookups
user_mean_ratings = ratings_base.groupby('user_id')['rating'].mean().to_dict()
global_mean = ratings_base['rating'].mean()

# Create user-item matrices for fast lookups
user_item_matrix = ratings_base.pivot_table(index='user_id', columns='item_id', values='rating')
user_item_dict = {}
for user_id, group in ratings_base.groupby('user_id'):
    user_item_dict[user_id] = dict(zip(group['item_id'], group['rating']))

# Find all unique users and items for faster iteration
all_users = ratings_base['user_id'].unique()
all_items = ratings_base['item_id'].unique()

# Precompute user biases (deviation from mean)
user_biases = {}
for user_id in all_users:
    if user_id in user_mean_ratings:
        user_ratings = user_item_dict.get(user_id, {})
        if user_ratings:
            user_biases[user_id] = {item_id: rating - user_mean_ratings[user_id] 
                                    for item_id, rating in user_ratings.items()}


class OptimizedSimilarityMatrix:
    def __init__(self):
        self.similarity_cache = {}
        self.top_k_cache = {}
        
    def pearson_correlation_vectorized(self, user1, user2):
        """Calculate Pearson correlation using vectorized operations"""
        if user1 == user2:
            return 1.0
            
        # Use the precomputed user-item dict
        items_user1 = user_item_dict.get(user1, {})
        items_user2 = user_item_dict.get(user2, {})
        
        # Find common items
        common_items = set(items_user1.keys()) & set(items_user2.keys())
        
        # Need at least 3 items for meaningful correlation
        if len(common_items) < 3:
            return 0
            
        # Get ratings for common items
        ratings_user1 = np.array([items_user1[item] for item in common_items])
        ratings_user2 = np.array([items_user2[item] for item in common_items])
        
        # Get mean ratings
        mean_rating_user1 = user_mean_ratings.get(user1, global_mean)
        mean_rating_user2 = user_mean_ratings.get(user2, global_mean)
        
        # Calculate numerator and denominators using numpy for speed
        diff1 = ratings_user1 - mean_rating_user1
        diff2 = ratings_user2 - mean_rating_user2
        
        numerator = np.sum(diff1 * diff2)
        
        denominator_user1 = np.sqrt(np.sum(diff1**2))
        denominator_user2 = np.sqrt(np.sum(diff2**2))
        
        # Check for division by zero
        if denominator_user1 == 0 or denominator_user2 == 0:
            return 0
            
        correlation = numerator / (denominator_user1 * denominator_user2)
        
        # Apply significance weighting
        if len(common_items) < 50:
            correlation = correlation * (len(common_items) / 50)
            
        return correlation
    
    def get_similarity(self, user1, user2):
        """Get similarity between two users with caching"""
        # Use a consistent order for the key
        cache_key = tuple(sorted([user1, user2]))
        
        if cache_key not in self.similarity_cache:
            sim = self.pearson_correlation_vectorized(user1, user2)
            self.similarity_cache[cache_key] = sim
            
        return self.similarity_cache[cache_key]
    
    def get_top_k_similar_users(self, user_id, k=20, exclude_negative=True):
        """Find k users most similar to the given user with caching"""
        cache_key = (user_id, k, exclude_negative)
        
        if cache_key in self.top_k_cache:
            return self.top_k_cache[cache_key]
        
        similarities = []
        
        for other_user_id in all_users:
            if other_user_id != user_id:
                sim = self.get_similarity(user_id, other_user_id)
                if not exclude_negative or sim > 0:
                    similarities.append((other_user_id, sim))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        result = similarities[:k]
        
        self.top_k_cache[cache_key] = result
        return result


# Create similarity matrix
similarity_matrix = OptimizedSimilarityMatrix()

# Cache for rating predictions to avoid redundant computations
prediction_cache = {}


def predict_rating(user_id, item_id, k=20, depth=1, visited=None, rating_range=(1, 5)):
    """
    Predict rating with optimizations:
    1. Reduced recursion depth
    2. Enhanced caching
    3. Vectorized operations
    """
    # Check cache first
    cache_key = (user_id, item_id, k, depth)
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]
    
    # Get mean rating for current user
    mean_rating_user = user_mean_ratings.get(user_id, global_mean)
    
    # Check if user has already rated this item
    if user_id in user_item_dict and item_id in user_item_dict[user_id]:
        return user_item_dict[user_id][item_id]
    
    # Initialize visited set for cycle detection
    if visited is None:
        visited = set()
    
    # Prevent cycles and limit recursion depth
    if (user_id, item_id) in visited or depth == 0:
        return mean_rating_user
    
    visited.add((user_id, item_id))
    
    # Get k most similar users
    similar_users = similarity_matrix.get_top_k_similar_users(user_id, k)
    
    if not similar_users:
        return mean_rating_user
    
    numerator = 0
    denominator = 0
    
    for neighbor_id, sim in similar_users:
        # Check if neighbor has rated the item directly
        if neighbor_id in user_item_dict and item_id in user_item_dict[neighbor_id]:
            # Use precomputed biases when available
            if neighbor_id in user_biases and item_id in user_biases[neighbor_id]:
                neighbor_diff = user_biases[neighbor_id][item_id]
            else:
                neighbor_mean = user_mean_ratings.get(neighbor_id, global_mean)
                neighbor_diff = user_item_dict[neighbor_id][item_id] - neighbor_mean
        else:
            # Only recurse if necessary and within depth limit
            if depth > 0:
                pred_rating = predict_rating(neighbor_id, item_id, k, depth-1, visited)
                neighbor_mean = user_mean_ratings.get(neighbor_id, global_mean)
                neighbor_diff = pred_rating - neighbor_mean
            else:
                continue
        
        numerator += sim * neighbor_diff
        denominator += abs(sim)
    
    if denominator == 0:
        predicted = mean_rating_user
    else:
        predicted = mean_rating_user + (numerator / denominator)
    
    # Ensure prediction is within valid range
    predicted = max(min(predicted, rating_range[1]), rating_range[0])
    
    # Cache the result
    prediction_cache[cache_key] = predicted
    
    return predicted


def compute_error_for_pair(user_id, item_id, k, depth=1):
    """Compute error for a single user-item pair for parallel processing"""
    actual = ratings_test[(ratings_test['user_id'] == user_id) & 
                          (ratings_test['item_id'] == item_id)]['rating'].iloc[0]
    predicted = predict_rating(user_id, item_id, k=k, depth=depth)
    return (predicted - actual) ** 2


def evaluate_model_parallel(k=20, metric='rmse', depth=1, n_jobs=-1):
    """Evaluate the model using parallel processing"""
    # Create a list of all user-item pairs in the test set
    test_pairs = list(zip(ratings_test['user_id'], ratings_test['item_id']))
    
    # Use parallel processing to compute errors
    with Parallel(n_jobs=n_jobs) as parallel:
        squared_errors = parallel(
            delayed(compute_error_for_pair)(user_id, item_id, k, depth) 
            for user_id, item_id in tqdm(test_pairs, desc=f"Evaluating k={k}")
        )
    
    mse = np.mean(squared_errors)
    
    if metric.lower() == 'rmse':
        return np.sqrt(mse)
    elif metric.lower() == 'mse':
        return mse
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def find_optimal_k_parallel(k_values, depth=1, metric='rmse', n_jobs=-1):
    """Find optimal k value using parallel processing"""
    results = []
    
    for k in tqdm(k_values, desc="Finding optimal k"):
        error = evaluate_model_parallel(k=k, metric=metric, depth=depth, n_jobs=n_jobs)
        results.append((k, error))
        print(f"k={k}, {metric.upper()}={error:.4f}")
    
    # Find the best k
    optimal_k, optimal_error = min(results, key=lambda x: x[1])
    import matplotlib.pyplot as plt
    k_vals, errors = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, errors, marker='o', linestyle='-', color='b')
    plt.xlabel('k values')
    plt.ylabel(f'{metric.upper()} error')
    plt.title(f'{metric.upper()} vs k values')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig("test1.png")
    plt.close()
    return optimal_k, optimal_error


# Example usage:
# For testing with a single k value
# error = evaluate_model_parallel(k=20, metric='rmse')
# print(f"RMSE with k=20: {error:.4f}")

# For finding optimal k
# Note: Set n_jobs to number of cores you want to use (or -1 for all available)
# optimal_k, optimal_rmse = find_optimal_k_parallel(
#     k_values=[5, 10, 20, 30, 50, 100],
#     depth=1,
#     n_jobs=4  # Adjust based on your CPU cores
# )
# print(f"Optimal k: {optimal_k} with RMSE: {optimal_rmse:.4f}")

# For the full evaluation:
if __name__ == "__main__":
    # Use 4 cores (adjust as needed)
    n_jobs = 4
    print(f"Running with {n_jobs} parallel jobs")
    
    # Test a single k value first
    error = evaluate_model_parallel(k=20, metric='rmse', n_jobs=n_jobs)
    print(f"RMSE with k=20: {error:.4f}")
    
    # Find optimal k with smaller subset first
    print("Finding optimal k...")
    k_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    optimal_k, optimal_rmse = find_optimal_k_parallel(
        k_values=k_values, 
        depth=3, 
        n_jobs=n_jobs
    )
    print(f"Optimal k: {optimal_k} with RMSE: {optimal_rmse:.4f}")