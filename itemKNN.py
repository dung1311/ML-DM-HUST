import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class ItemColaborativeFiltering:
    def __init__(self, users, items, ratings_base, ratings_test, similarity_type: Literal['pearson', 'cosine'] = 'pearson'):
        self.users = users
        self.items = items
        self.ratings_base = ratings_base
        self.ratings_test = ratings_test
        self.similarity_matrix = self.LazySimilarityMatrix(self)
        self.user_mean_ratings = self.ratings_base.groupby('user_id')['rating'].mean()
        self.similarity_type = similarity_type
    
    def pearson_correlation(self, item1, item2):
        """
        Calculate the Pearson correlation coefficient between two items.
        """
        # Get the ratings for both items
        ratings1 = self.ratings_base[self.ratings_base['item_id'] == item1]
        ratings2 = self.ratings_base[self.ratings_base['item_id'] == item2]

        # Merge the two dataframes on user_id
        merged_ratings = pd.merge(ratings1, ratings2, on='user_id', suffixes=('_item1', '_item2'))

        # if there are less than 3 common users, return 0
        if len(merged_ratings) < 3:
            return 0
        
        # Calculate the mean ratings for both items
        mean_item1 = merged_ratings['rating_item1'].mean()
        mean_item2 = merged_ratings['rating_item2'].mean()

        # Calculate the numerator and denominator for the Pearson correlation
        numerator = np.sum((merged_ratings['rating_item1'] - mean_item1) * (merged_ratings['rating_item2'] - mean_item2))
        denominator = np.sqrt(np.sum((merged_ratings['rating_item1'] - mean_item1) ** 2) * np.sum((merged_ratings['rating_item2'] - mean_item2) ** 2))

        if denominator == 0:
            return 0

        correlation = numerator / denominator
        
        if len(merged_ratings) < 50:
            correlation = correlation * (len(merged_ratings)/50)
        
        return correlation
    
    def cosine_similarity(self, item1, item2):
        """
        Calculate the cosine similarity between two items.
        """
        # Get the ratings for both items
        ratings1 = self.ratings_base[self.ratings_base['item_id'] == item1]
        ratings2 = self.ratings_base[self.ratings_base['item_id'] == item2]

        # Merge the two dataframes on user_id
        merged_ratings = pd.merge(ratings1, ratings2, on='user_id', suffixes=('_item1', '_item2'))

        # if there are less than 3 common users, return 0
        if len(merged_ratings) < 3:
            return 0
        
        # Calculate the cosine similarity
        vector1 = merged_ratings['rating_item1'].values
        vector2 = merged_ratings['rating_item2'].values

        similarity = 1 - cosine(vector1, vector2)
        
        if np.isnan(similarity):
            return 0
        
        return similarity
    
    class LazySimilarityMatrix:
        def __init__(self, outer_instance):
            self.similarity_cache = {}
            self.outer = outer_instance
            
        def get_similarity(self, item1, item2):
            """Get similarity between two items with caching"""
            if item1 == item2:
                return 1.0
            
            # Use a consistent order for the key to avoid duplicates
            cache_key = tuple(sorted([item1, item2]))
            
            if cache_key not in self.similarity_cache and self.outer.similarity_type == 'pearson':
                sim = self.outer.pearson_correlation(item1, item2)
                self.similarity_cache[cache_key] = sim
            
            elif cache_key not in self.similarity_cache and self.outer.similarity_type == 'cosine':
                sim = self.outer.cosine_similarity(item1, item2)
                self.similarity_cache[cache_key] = sim
                
            return self.similarity_cache[cache_key]
        
        def get_top_k_similar_items(self, item_id, k=20, exclude_negative=True):
            """Find k items most similar to the given item"""
            item_ids = self.outer.ratings_base["user_id"].unique()
            similarities = []
            
            for other_item_id in item_ids:
                if other_item_id != item_id:
                    sim = self.get_similarity(item_id, other_item_id)
                    # Option to exclude users with negative correlation
                    if not exclude_negative or sim > 0:
                        similarities.append((other_item_id, sim))
            
            # Sort by similarity (descending) and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
    
    def predict_rating(self, user_id, item_id, k=20, rating_range=(1, 5)):
        # Check if the user has already rated this item
        is_rated = self.ratings_base[(self.ratings_base['user_id'] == user_id) & (self.ratings_base['item_id'] == item_id)]
        if not is_rated.empty:
            return is_rated['rating'].iloc[0]
        
        items_rated_by_user = self.ratings_base[self.ratings_base['user_id'] == user_id]
        
        similar_items = self.similarity_matrix.get_top_k_similar_items(item_id, k)
        
        numerator = 0
        denominator = 0
        
        for similar_item_id, sim in similar_items:
            # Check if the user has rated the similar item
            user_rating = items_rated_by_user[items_rated_by_user['item_id'] == similar_item_id]
            
            if not user_rating.empty:
                rating = user_rating['rating'].iloc[0]
                numerator += sim * rating
                denominator += abs(sim)
        
        if denominator == 0:
            if user_id in self.user_mean_ratings:
                return self.user_mean_ratings[user_id]
            else:
                return self.ratings_base['rating'].mean()
        
        predicted_rating = numerator / denominator
        
        predicted_rating = max(rating_range[0], min(predicted_rating, rating_range[1]))
        
        return predicted_rating

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
    
    def find_optimal_k(self, k_values=None, save_plot=False, plot_name='itemKNN'):
        """Find the optimal k value for the KNN algorithm"""
        print("Finding optimal k...")
        if k_values is None:
            k_values = [5, 10, 20, 30, 40, 50]
        
        results = []
        
        for k in k_values:
            print(f"Evaluating k={k}...")
            # Use a small sample for efficiency
            sample_size = len(self.ratings_test)
            sample = self.ratings_test.sample(sample_size)
            
            errors = []
            for _, row in sample.iterrows():
                user_id, item_id, actual_rating = row['user_id'], row['item_id'], row['rating']
                predicted_rating = self.predict_rating(user_id, item_id, k)
                errors.append((predicted_rating - actual_rating) ** 2)
            
            rmse = np.sqrt(np.mean(errors))
            results.append((k, rmse))
            print(f"k={k}, RMSE={rmse:.4f}")
        
        import matplotlib.pyplot as plt
        
        if save_plot:
            plt.figure(figsize=(10, 6))
            
            k_list = [result[0] for result in results]
            rmse_list = [result[1] for result in results]
            
            plt.plot(k_list, rmse_list, marker='o')
            plt.xlabel('k (Number of neighbors)')
            plt.ylabel('RMSE')
            plt.title('RMSE vs. k')
            plt.grid(True)
            plt.savefig(f'{plot_name}.png')
            plt.show()
        
        
        # Find k with lowest error
        optimal_k = min(results, key=lambda x: x[1])
        optimal_rmse = optimal_k[1]
        print(f"Optimal k: {optimal_k[0]} with RMSE: {optimal_rmse:.4f}")
        
        return optimal_k
    
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

                temp_cf = ItemColaborativeFiltering(
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
                plt.show()
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
    
    # itemCF = ItemColaborativeFiltering(users, items, ratings_base, ratings_test, similarity_type='pearson')
    itemCF2 = ItemColaborativeFiltering(users, items, ratings_base, ratings_test, similarity_type='cosine')
    # print("Predicting rating for user 2 and item 281:", itemCF.predict_rating(2, 281, k=20))
    # print("Predicting rating for user 2 and item 281 with cosine:", itemCF2.predict_rating(2, 281, k=20))
    # print("pearson", itemCF.evaluate_model(ratings_test, k=20, metric='rmse'))
    # print("cosine", itemCF2.evaluate_model(ratings_test, k=20, metric='rmse'))
    # itemCF.find_optimal_k_with_cv(k_values=list(range(10, 50, 5)), n_folds=5, save_plot=True, plot_name='itemKNN_pearson_100k')
    itemCF2.find_optimal_k_with_cv(k_values=list(range(10, 55, 5)), save_plot=True, plot_name='itemKNN_cosine_100k')
    