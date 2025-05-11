import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

class RecursiveCF:
    def __init__(self, users, items, ratings_base, ratings_test):
        self.users = users
        self.items = items
        self.ratings_base = ratings_base
        self.ratings_test = ratings_test
        self.user_mean_ratings = self.ratings_base.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean = self.ratings_base['rating'].mean()
        self.user_item_matrix = self.ratings_base.pivot_table(index='user_id', columns='item_id', values='rating')
        self.user_item_dict = {}
        for user_id, group in self.ratings_base.groupby('user_id'):
            self.user_item_dict[user_id] = dict(zip(group['item_id'], group['rating']))

        # Find all unique users and items for faster iteration
        self.all_users = ratings_base['user_id'].unique()
        self.all_items = ratings_base['item_id'].unique()

        
        self.user_biases = {}
        for user_id in self.all_users:
            if user_id in self.user_mean_ratings:
                user_ratings = self.user_item_dict.get(user_id, {})
                if user_ratings:
                    self.user_biases[user_id] = {item_id: rating - self.user_mean_ratings[user_id] 
                                            for item_id, rating in user_ratings.items()}
        
        self.similarity_matrix = self.SimilarityMatrix(self)  
        
        self.prediction_cache = {}      
        
    class SimilarityMatrix:
        def __init__(self, outer_instance):
            self.outer = outer_instance
            self.similarity_cache = {}
            self.top_k_cache = {}
        
        def pearson_correlation_vectorized(self, user1, user2):
            """Calculate Pearson correlation using vectorized operations"""
            if user1 == user2:
                return 1.0
                
            # Use the precomputed user-item dict
            items_user1 = self.outer.user_item_dict.get(user1, {})
            items_user2 = self.outer.user_item_dict.get(user2, {})
            
            # Find common items
            common_items = set(items_user1.keys()) & set(items_user2.keys())
            
            # Need at least 3 items for meaningful correlation
            if len(common_items) < 3:
                return 0
                
            # Get ratings for common items
            ratings_user1 = np.array([items_user1[item] for item in common_items])
            ratings_user2 = np.array([items_user2[item] for item in common_items])
            
            # Get mean ratings
            mean_rating_user1 = self.outer.user_mean_ratings.get(user1, self.outer.global_mean)
            mean_rating_user2 = self.outer.user_mean_ratings.get(user2, self.outer.global_mean)
            
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
            
            for other_user_id in self.outer.all_users:
                if other_user_id != user_id:
                    sim = self.get_similarity(user_id, other_user_id)
                    if not exclude_negative or sim > 0:
                        similarities.append((other_user_id, sim))
            
            # Sort by similarity (descending) and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            result = similarities[:k]
            
            self.top_k_cache[cache_key] = result
            return result
    
    def predict_rating(self, user_id, item_id, k=20, depth=1, visited=None, rating_range=(1, 5)):
        """
        Predict rating with optimizations:
        1. Reduced recursion depth
        2. Enhanced caching
        3. Vectorized operations
        """
        # Check cache first
        cache_key = (user_id, item_id, k, depth)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Get mean rating for current user
        mean_rating_user = self.user_mean_ratings.get(user_id, self.global_mean)
        
        # Check if user has already rated this item
        if user_id in self.user_item_dict and item_id in self.user_item_dict[user_id]:
            return self.user_item_dict[user_id][item_id]
        
        # Initialize visited set for cycle detection
        if visited is None:
            visited = set()
        
        # Prevent cycles and limit recursion depth
        if (user_id, item_id) in visited or depth == 0:
            return mean_rating_user
        
        visited.add((user_id, item_id))
        
        # Get k most similar users
        similar_users = self.similarity_matrix.get_top_k_similar_users(user_id, k)
        
        if not similar_users:
            return mean_rating_user
        
        numerator = 0
        denominator = 0
        
        for neighbor_id, sim in similar_users:
            # Check if neighbor has rated the item directly
            if neighbor_id in self.user_item_dict and item_id in self.user_item_dict[neighbor_id]:
                # Use precomputed biases when available
                if neighbor_id in self.user_biases and item_id in self.user_biases[neighbor_id]:
                    neighbor_diff = self.user_biases[neighbor_id][item_id]
                else:
                    neighbor_mean = self.user_mean_ratings.get(neighbor_id, self.global_mean)
                    neighbor_diff = self.user_item_dict[neighbor_id][item_id] - neighbor_mean
            else:
                # Only recurse if necessary and within depth limit
                if depth > 0:
                    pred_rating = self.predict_rating(neighbor_id, item_id, k, depth-1, visited)
                    neighbor_mean = self.user_mean_ratings.get(neighbor_id, self.global_mean)
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
        self.prediction_cache[cache_key] = predicted
        
        return predicted

    def compute_error_for_pair(self, user_id, item_id, k, depth=1):
        """Compute error for a single user-item pair for parallel processing"""
        try:
            actual = self.ratings_test[(self.ratings_test['user_id'] == user_id) & 
                                       (self.ratings_test['item_id'] == item_id)]['rating'].iloc[0]
            predicted = self.predict_rating(user_id, item_id, k=k, depth=depth)
            return (predicted - actual) ** 2
        except IndexError:
            return 0  # Handle case where the user-item pair is not found

    def evaluate_model_parallel(self, train_set, val_set, k, depth=1, n_jobs=-1):
        """Evaluate the model using parallel processing"""
        test_pairs = list(zip(val_set['user_id'], val_set['item_id']))

        with Parallel(n_jobs=n_jobs) as parallel:
            squared_errors = parallel(
                delayed(self.compute_error_for_pair)(user_id, item_id, k, depth) 
                for user_id, item_id in test_pairs
            )

        mse = np.mean(squared_errors)
        return np.sqrt(mse)

    def find_optimal_k(self, k_values=None, depth=1, n_folds=5, save_plot=False, plot_name=None, metric='rmse', n_jobs=-1):
        if k_values is None:
            k_values = [5, 10, 20, 30, 40, 50]

        print("Finding optimal k using cross-validation with parallel processing...")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []

        for k in tqdm(k_values, desc="Finding optimal k"):
            fold_errors = []
            for train_index, val_index in kf.split(self.ratings_base):
                train_set = self.ratings_base.iloc[train_index]
                val_set = self.ratings_base.iloc[val_index]
                fold_rmse = self.evaluate_model_parallel(train_set, val_set, k, depth=depth, n_jobs=n_jobs)
                fold_errors.append(fold_rmse)
            avg_rmse = np.mean(fold_errors)
            results.append((k, avg_rmse))
            print(f"k={k}, Average RMSE: {avg_rmse:.4f}")

        if save_plot:
            plt.figure(figsize=(10, 6))
            k_vals, errors = zip(*results)
            plt.plot(k_vals, errors, marker='o')
            plt.xlabel('k (Number of neighbors)')
            plt.ylabel('Average RMSE')
            plt.title(f'RMSE vs. k ({n_folds}-fold Cross-Validation)')
            plt.grid()
            if plot_name:
                plt.savefig(f'{plot_name}.png')
            plt.show()

        optimal_k, optimal_rmse = min(results, key=lambda x: x[1])
        print(f"Optimal k: {optimal_k} with average RMSE: {optimal_rmse:.4f}")
        return optimal_k, optimal_rmse

    
if __name__ == "__main__":
    u_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
    users = pd.read_csv("./data/ml-100k/u.user", sep="|", header=None, names=u_cols, encoding="latin-1")
    print("Number of users:", users.shape[0])

    #rates
    r_cols = ["user_id", "item_id", "rating", "unix_timestamp"]
    ratings_base = pd.read_csv("./data/ml-100k/ua.base", sep="\t", header=None, names=r_cols, encoding="latin-1")
    ratings_test = pd.read_csv("./data/ml-100k/ua.test", sep="\t", header=None, names=r_cols, encoding="latin-1")
    print("Number of ratings in base set:", ratings_base.shape[0])
    print("Number of ratings in test set:", ratings_test.shape[0])

    #items
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    print("Number of items:", items.shape[0])
    
    rcf = RecursiveCF(users, items, ratings_base, ratings_test)
    # predict = rcf.predict_rating(2, 281, k=20, depth=3)
    # print("Predicted rating for user 2 on item 281:", predict)
    # rmse = rcf.find_optimal_k(n_jobs=4)
    