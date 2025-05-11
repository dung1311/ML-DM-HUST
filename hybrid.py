import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from userKNN import UserColaborativeFiltering
from itemKNN import ItemColaborativeFiltering

class HybridRecommender:
    def __init__(self, users, items, ratings_base, ratings_test):
        self.user_knn = UserColaborativeFiltering(users, items, ratings_base, ratings_test)
        self.item_knn = ItemColaborativeFiltering(users, items, ratings_base, ratings_test)
        self.users = users
        self.items = items
        self.ratings_base = ratings_base
        self.ratings_test = ratings_test

    def predict_rating(self, user_id, item_id, alpha=0.5):
        # Get predictions from both user-based and item-based collaborative filtering
        user_prediction = self.user_knn.predict_rating(user_id, item_id, k=45)
        item_prediction = self.item_knn.predict_rating(user_id, item_id, k=45)

        # Combine the predictions (simple average in this case)
        hybrid_prediction = alpha*user_prediction + (1-alpha)*item_prediction

        return hybrid_prediction

    def evaluate_model(self, eval_set, alpha=0.5, metric='rmse'):
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
        print(f"Evaluating model with alpha = {alpha} on {total} test samples...")
        
        with tqdm(total=total, desc="Evaluating", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            for _, row in eval_set.iterrows():
                user_id, item_id, actual_rating = row['user_id'], row['item_id'], row['rating']
                predicted_rating = self.predict_rating(user_id, item_id, alpha=alpha)

                if metric == 'rmse':
                    errors.append((predicted_rating - actual_rating) ** 2)
                else:  # mae
                    errors.append(abs(predicted_rating - actual_rating))
                
                pbar.update(1)
                
        if metric == 'rmse':
            return np.sqrt(np.mean(errors))
        else:  # mae
            return np.mean(errors)
        
    def find_optimal_alpha_with_cv(self, alpha_values=None, n_folds=5, save_plot=False, plot_name=None):
        """
        Find the optimal alpha value using cross-validation

        Parameters:
        alpha_values: List of alpha values to evaluate
        n_folds: Number of folds for cross-validation
        save_plot: Whether to save the plot
        plot_name: Name of the plot file

        Returns:
        tuple: (optimal_alpha, rmse_score)
        """
        if alpha_values is None:
            alpha_values = list(np.arange(0.1, 1.0, 0.1))

        print("Finding optimal alpha using cross-validation...")

        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []

        for alpha in alpha_values:
            print(f"Evaluating alpha={alpha}...")
            fold_rmse = []

            for train_idx, val_idx in tqdm(
                kf.split(self.ratings_base),
                total=n_folds,
                desc=f"Cross-validation for alpha={alpha}",
                leave=False
            ):
                train_set = self.ratings_base.iloc[train_idx]
                val_set = self.ratings_base.iloc[val_idx]

                if train_set.empty or val_set.empty:
                    print(f"Warning: Empty train or validation set in fold. Skipping...")
                    continue

                temp_cf = HybridRecommender(
                    self.users, self.items, train_set, val_set,
                )

                rmse = temp_cf.evaluate_model(val_set, alpha=alpha, metric='rmse')
                fold_rmse.append(rmse)

            avg_rmse = np.mean(fold_rmse)
            results.append((alpha, avg_rmse))

        if save_plot:
            try:
                alpha_list, rmse_list = zip(*results)
                plt.figure(figsize=(10, 6))
                plt.plot(alpha_list, rmse_list, marker='o')
                plt.xlabel('alpha (weight)')
                plt.ylabel('Average RMSE')
                plt.title(f'RMSE vs. alpha ({n_folds}-fold CV)')
                plt.grid(True)
                plt.xticks(alpha_list)
                if plot_name:
                    plt.savefig(f'{plot_name}.png')
                plt.show()
            except Exception as e:
                print(f"Error while plotting: {e}")

        if not results:
            raise ValueError("No valid results obtained from cross-validation. Check data or folds.")
        
        optimal_alpha, optimal_rmse = min(results, key=lambda x: x[1])
        print(f"Optimal alpha: {optimal_alpha} with average RMSE: {optimal_rmse:.4f}")
        return optimal_alpha, optimal_rmse
    
    
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
    
    hybrid = HybridRecommender(users, items, ratings_base, ratings_test)
    # print("Hybrid prediction for user 1 and item 1:", hybrid.predict_rating(2, 281))
    # print(hybrid.evaluate_model(ratings_test, alpha=0.5, metric='rmse'))
    # alpha, rmse = hybrid.find_optimal_alpha_with_cv(alpha_values=None, n_folds=5, save_plot=True, plot_name="alpha_vs_rmse")
    # print(f"Optimal alpha: {alpha} with RMSE: {rmse:.4f}")
    print(hybrid.evaluate_model(ratings_test, alpha=0.7))