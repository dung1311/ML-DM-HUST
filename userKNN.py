import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

class UserColaborativeFiltering:
    def __init__(self, users, items, ratings_base, ratings_test, similarity_type: Literal['pearson', 'cosine'] = 'pearson'):
        self.users = users
        self.items = items
        self.ratings_base = ratings_base
        self.ratings_test = ratings_test
        self.similarity_matrix = self.LazySimilarityMatrix(self)
        self.user_mean_ratings = self.ratings_base.groupby('user_id')['rating'].mean()
        self.similarity_type = similarity_type
    
    def pearson_correlation(self, user1, user2):
        """
        Calculate Pearson correlation between ratings of two users
        
        Parameters:
        user1, user2: user IDs to compare
        
        Returns:
        float: Pearson correlation coefficient between the two users
        """
        # Get items rated by each user
        items_user1 = self.ratings_base[self.ratings_base["user_id"] == user1][["item_id", "rating"]]
        items_user2 = self.ratings_base[self.ratings_base["user_id"] == user2][["item_id", "rating"]]
        
        # Find common items
        common_items = pd.merge(items_user1, items_user2, on="item_id", suffixes=('_user1', '_user2'))

        # If less than 3 common items, return 0 (not enough data for reliable correlation)
        if len(common_items) < 3:
            return 0
        
        # Get mean ratings for each user
        mean_rating_user1 = items_user1["rating"].mean()
        mean_rating_user2 = items_user2["rating"].mean()
        
        # Calculate numerator and denominators for Pearson formula
        numerator = sum((common_items["rating_user1"] - mean_rating_user1) * 
                        (common_items["rating_user2"] - mean_rating_user2))
        
        denominator_user1 = sum((common_items["rating_user1"] - mean_rating_user1)**2)**0.5
        denominator_user2 = sum((common_items["rating_user2"] - mean_rating_user2)**2)**0.5
        
        # Check for division by zero
        if denominator_user1 == 0 or denominator_user2 == 0:
            return 0
        
        # Calculate and return correlation
        correlation = numerator / (denominator_user1 * denominator_user2)
        
        # Apply significance weighting (optional but recommended)
        # This gives more weight to correlations based on more common items
        if len(common_items) < 50:
            correlation = correlation * (len(common_items) / 50)
        
        return correlation
    
    def cosine_similarity(self, user1, user2):
        """
        Calculate Cosine Similarity between two users based on their ratings.

        Parameters:
        user1, user2: user IDs to compare
        ratings_base: DataFrame containing user-item ratings with columns ['user_id', 'item_id', 'rating']

        Returns:
        float: Cosine similarity between the two users
        """
        # Get items rated by each user
        items_user1 = self.ratings_base[self.ratings_base["user_id"] == user1][["item_id", "rating"]]
        items_user2 = self.ratings_base[self.ratings_base["user_id"] == user2][["item_id", "rating"]]

        # Find common items
        common_items = pd.merge(items_user1, items_user2, on="item_id", suffixes=('_user1', '_user2'))

        # If no common items, return 0
        if common_items.empty:
            return 0

        # Extract ratings for common items
        ratings_user1 = common_items["rating_user1"].values
        ratings_user2 = common_items["rating_user2"].values

        # Calculate cosine similarity
        similarity = 1 - cosine(ratings_user1, ratings_user2)

        # Handle cases where cosine returns NaN (e.g., zero vectors)
        if np.isnan(similarity):
            return 0

        return similarity
    
    class LazySimilarityMatrix:
        def __init__(self, outer_instance):
            self.similarity_cache = {}
            self.outer = outer_instance
            
        def get_similarity(self, user1, user2):
            """Get similarity between two users with caching"""
            
            if user1 == user2:
                return 1.0
            
            # Use a consistent order for the key to avoid duplicates
            cache_key = tuple(sorted([user1, user2]))
            
            if cache_key not in self.similarity_cache and self.outer.similarity_type == 'pearson':
                sim = self.outer.pearson_correlation(user1, user2)
                self.similarity_cache[cache_key] = sim
            
            elif cache_key not in self.similarity_cache and self.outer.similarity_type == 'cosine':
                sim = self.outer.cosine_similarity(user1, user2)
                self.similarity_cache[cache_key] = sim    
                
            return self.similarity_cache[cache_key]
        
        def get_top_k_similar_users(self, user_id, k=20, exclude_negative=True):
            """Find k users most similar to the given user"""
            user_ids = self.outer.ratings_base["user_id"].unique()
            similarities = []
            
            for other_user_id in user_ids:
                if other_user_id != user_id:
                    sim = self.get_similarity(user_id, other_user_id)
                    # Option to exclude users with negative correlation
                    if not exclude_negative or sim > 0:
                        similarities.append((other_user_id, sim))
            
            # Sort by similarity (descending) and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
    
    def predict_rating(self, user_id, item_id, k=20, rating_range=(1, 5)):
        """
        Predict rating for a given user-item pair using user-based collaborative filtering
        
        Parameters:
        user_id: ID of the user
        item_id: ID of the item
        k: Number of similar users to consider
        rating_range: Valid range for ratings (min, max)
        
        Returns:
        float: Predicted rating
        """
        # Check if the user has already rated this item
        is_rated = self.ratings_base[(self.ratings_base['user_id'] == user_id) & (self.ratings_base['item_id'] == item_id)]
        if not is_rated.empty:
            return is_rated['rating'].iloc[0]
        
        # Get k most similar users
        similar_users = self.similarity_matrix.get_top_k_similar_users(user_id, k)
        
        # Get mean rating for the current user
        if user_id in self.user_mean_ratings:
            mean_rating_user = self.user_mean_ratings[user_id]
        else:
            mean_rating_user = self.ratings_base['rating'].mean()  # Global mean as fallback
        
        # Get all ratings for this item from users in our similar users list
        similar_user_ids = [uid for uid, _ in similar_users]
        item_ratings = self.ratings_base[
            (self.ratings_base['item_id'] == item_id) & 
            (self.ratings_base['user_id'].isin(similar_user_ids))
        ]
        
        # Convert to dictionary for faster lookup
        similar_user_ratings = {row['user_id']: row['rating'] for _, row in item_ratings.iterrows()}
        
        numerator = 0
        denominator = 0
        
        for similar_user_id, similarity_score in similar_users:
            if similar_user_id in similar_user_ratings:
                # Get the rating this similar user gave to the item
                similar_user_rating = similar_user_ratings[similar_user_id]
                
                # Get the mean rating for this similar user
                similar_user_mean = self.user_mean_ratings.get(similar_user_id, 
                                                self.ratings_base['rating'].mean())
                
                # Use normalized rating in the calculation
                numerator += similarity_score * (similar_user_rating - similar_user_mean)
                denominator += abs(similarity_score)
        
        # If no similar users rated this item, return user's mean rating
        if denominator == 0:
            return mean_rating_user
        
        # Calculate predicted rating using weighted average of normalized ratings
        predicted_rating = mean_rating_user + (numerator / denominator)
        
        min_rating, max_rating = rating_range
        return max(min_rating, min(max_rating, predicted_rating))
    
    
    def evaluate_model(self, eval_set, k=20, metric='rmse'):
        """
        Evaluate the recommender model on the test set
        
        Parameters:
        k: Number of similar users to consider
        metric: Evaluation metric ('rmse' or 'mae')
        
        Returns:
        float: Error metric value (lower is better)
        """
        errors = []
        total = len(eval_set)
        print(f"Evaluating model with k={k} on {total} test samples...")
        
        # Optional: show progress for long evaluations
        progress_step = max(1, total // 10)
        
        with tqdm(total=total, desc="Evaluating", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            for _, row in eval_set.iterrows():
                user_id, item_id, actual_rating = row['user_id'], row['item_id'], row['rating']
                predicted_rating = self.predict_rating(user_id, item_id, k)

                if metric == 'rmse':
                    errors.append((predicted_rating - actual_rating) ** 2)
                else:  # mae
                    errors.append(abs(predicted_rating - actual_rating))
                
                pbar.update(1)
                
        if metric == 'rmse':
            return np.sqrt(np.mean(errors))
        else:  # mae
            return np.mean(errors)
    
    def find_optimal_k(self, k_values=None, save_plot=False, plot_name=None):
        """
        Find the optimal k value using a simple evaluation method

        Parameters:
        k_values: List of k values to evaluate
        save_plot: Whether to save the plot
        plot_name: Name of the plot file

        Returns:
        tuple: (optimal_k, rmse_score)
        """
        if k_values is None:
            k_values = [5, 10, 20, 30, 40, 50]

        print("Finding optimal k...")

        results = []
        
        valid_set = train_test_split(self.ratings_base, test_size=0.2, random_state=42)[1]
        
        for k in k_values:
            print(f"Evaluating k={k}...")
            rmse = self.evaluate_model(valid_set, k=k, metric='rmse')
            results.append((k, rmse))

        if save_plot:
            try:
                k_list, rmse_list = zip(*results)
                plt.figure(figsize=(10, 6))
                plt.plot(k_list, rmse_list, marker='o')
                plt.xlabel('k (Number of neighbors)')
                plt.ylabel('RMSE')
                plt.title('RMSE vs. k')
                plt.grid(True)
                plt.xticks(k_list)
                if plot_name:
                    plt.savefig(f'{plot_name}.png')
                plt.show()
            except Exception as e:
                print(f"Error while plotting: {e}")

        optimal_k, optimal_rmse = min(results, key=lambda x: x[1])
        print(f"Optimal k: {optimal_k} with RMSE: {optimal_rmse:.4f}")
        return optimal_k, optimal_rmse
    
    
    def find_optimal_k_with_cv(self, k_values=None, n_folds=5, save_plot=False, plot_name=None):
        """
        Find the optimal k value using cross-validation

        Parameters:
        k_values: List of k values to evaluate
        n_folds: Number of folds for cross-validation
        save_plot: Whether to save the plot
        plot_name: Name of the plot file

        Returns:
        tuple: (optimal_k, rmse_score)
        """
        if k_values is None:
            k_values = [5, 10, 20, 30, 40, 50]

        print("Finding optimal k using cross-validation...")

        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []

        for k in k_values:
            print(f"Evaluating k={k}...")
            fold_rmse = []

            for train_idx, val_idx in tqdm(
                kf.split(self.ratings_base),
                total=n_folds,
                desc=f"Cross-validation for k={k}",
                leave=False
            ):
                train_set = self.ratings_base.iloc[train_idx]
                val_set = self.ratings_base.iloc[val_idx]

                if train_set.empty or val_set.empty:
                    print(f"Warning: Empty train or validation set in fold. Skipping...")
                    continue

                temp_cf = UserColaborativeFiltering(
                    self.users, self.items, train_set, val_set,
                    similarity_type=self.similarity_type
                )

                rmse = temp_cf.evaluate_model(val_set, k=k, metric='rmse')
                fold_rmse.append(rmse)

            avg_rmse = np.mean(fold_rmse)
            results.append((k, avg_rmse))

        if save_plot:
            try:
                k_list, rmse_list = zip(*results)
                plt.figure(figsize=(10, 6))
                plt.plot(k_list, rmse_list, marker='o')
                plt.xlabel('k (Number of neighbors)')
                plt.ylabel('Average RMSE')
                plt.title(f'RMSE vs. k ({n_folds}-fold CV)')
                plt.grid(True)
                plt.xticks(k_list)
                if plot_name:
                    plt.savefig(f'{plot_name}.png')
                # plt.show()
            except Exception as e:
                print(f"Error while plotting: {e}")

        if not results:
            raise ValueError("No valid results obtained from cross-validation. Check data or folds.")
        
        optimal_k, optimal_rmse = min(results, key=lambda x: x[1])
        print(f"Optimal k: {optimal_k} with average RMSE: {optimal_rmse:.4f}")
        return optimal_k, optimal_rmse
    
    def recommend_items(self, user_id, n=10, exclude_rated=True):
        """
        Recommend top N items for a given user
        
        Parameters:
        user_id: ID of the user
        n: Number of recommendations to make
        exclude_rated: Whether to exclude items the user has already rated
        
        Returns:
        list: Tuples of (item_id, predicted_rating)
        """
        # Get all items
        all_items = self.items['movie id'].unique()
        
        # Get items already rated by the user if we want to exclude them
        if exclude_rated:
            rated_items = self.ratings_base[self.ratings_base['user_id'] == user_id]['item_id'].unique()
            candidate_items = [item for item in all_items if item not in rated_items]
        else:
            candidate_items = all_items
        
        # Predict ratings for candidate items
        predictions = []
        for item_id in candidate_items:
            pred_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating (descending) and take top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

if __name__ == "__main__":
    # Comment out ml-100k code
    """
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
    
    userCF = UserColaborativeFiltering(users, items, ratings_base, ratings_test, similarity_type='pearson')
    userCF2 = UserColaborativeFiltering(users, items, ratings_base, ratings_test, similarity_type='cosine')
    # print("Predicting rating for user 2 and item 281:", userCF.predict_rating(2, 281, k=20))
    # print("Predicting rating for user 2 and item 281 with cosine:", userCF2.predict_rating(2, 281, k=20))
    # print("pearson", userCF2.evaluate_model(ratings_test, k=20, metric='rmse'))
    # print("cosine", userCF2.evaluate_model(ratings_test, k=20, metric='rmse'))
    # userCF.find_optimal_k_with_cv(k_values=list(range(10, 50, 5)), n_folds=5, save_plot=True, plot_name='userKNN_pearson_100k')
    # userCF2.find_optimal_k_with_cv(k_values=list(range(10, 50, 5)), save_plot=True, plot_name='userKNN_cosine_100k', n_folds=5)
    rmse = userCF.evaluate_model(ratings_test, k=45, metric='rmse')
    print("RMSE for k=45:", rmse)
    """

    # Load ml-1m dataset
    print("Loading ml-1m dataset...")
    
    # Load users
    users = pd.read_csv("./data/ml-1m/users.dat", sep="::", header=None, 
                       names=["user_id", "gender", "age", "occupation", "zip_code"],
                       engine='python', encoding='latin-1')
    print("Number of users:", users.shape[0])

    # Load ratings
    ratings = pd.read_csv("./data/ml-1m/ratings.dat", sep="::", header=None,
                         names=["user_id", "item_id", "rating", "timestamp"],
                         engine='python', encoding='latin-1')
    print("Total number of ratings:", ratings.shape[0])

    # Load movies
    items = pd.read_csv("./data/ml-1m/movies.dat", sep="::", header=None,
                       names=["movie id", "movie title", "genres"],
                       engine='python', encoding='latin-1')
    print("Number of items:", items.shape[0])

    # Split ratings into train and test (80-20)
    from sklearn.model_selection import train_test_split
    ratings_base, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
    print("Number of ratings in base set:", ratings_base.shape[0])
    print("Number of ratings in test set:", ratings_test.shape[0])

    # Create and evaluate model
    userCF = UserColaborativeFiltering(users, items, ratings_base, ratings_test, similarity_type='pearson')
    rmse = userCF.evaluate_model(ratings_test, k=45, metric='rmse')
    print("RMSE for k=45:", rmse)
    