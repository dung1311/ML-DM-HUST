import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from typing import Literal
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

class ContentFiltering:
    def __init__(self, users, items, ratings_base, ratings_test):
        self.users = users
        self.items = items
        self.ratings_base = ratings_base
        self.original_ratings_base = ratings_base.copy()
        self.ratings_test = ratings_test
        self.all_users = users['user_id'].unique()
        self.all_items = items['item_id'].unique()
        self.transformer = TfidfTransformer(smooth_idf=True, norm='l2')
        self.genres = self.items.iloc[:, -19:].values
        self.tfidf = self.transformer.fit_transform(self.genres).toarray()
        self.models = {}

    def train(self, algo: Literal['ridge', 'lasso'] = 'ridge', alpha=1.0):
        for uid in self.all_users:
            user_ratings = self.ratings_base[self.ratings_base['user_id'] == uid]
            item_ids = user_ratings['item_id'].values
            ratings = user_ratings['rating'].values
            X = self.tfidf[item_ids - 1]

            if algo == 'ridge':
                model = Ridge(alpha=alpha)
            elif algo == 'lasso':
                model = Lasso(alpha=alpha, max_iter=5000, tol=1)
            else:
                raise ValueError("Invalid algorithm. Choose 'ridge' or 'lasso'.")

            if len(ratings) > 0:
                model.fit(X, ratings)
                self.models[uid] = model

        # Construct the prediction matrix
        predict_matrix = np.zeros((len(self.all_users), len(self.all_items)))
        for user_index, uid in enumerate(self.all_users):
            if uid in self.models:
                model = self.models[uid]
                predict_matrix[user_index] = model.predict(self.tfidf)

        return predict_matrix

    def validate(self, n_folds=5, algo: Literal['ridge', 'lasso'] = 'ridge', alpha_range=(0.1, 1.0, 10)):
        kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
        alphas = np.linspace(*alpha_range)
        mean_errors = []

        for alpha in alphas:
            fold_errors = []
            for train_index, val_index in kf.split(self.original_ratings_base):
                train_data = self.original_ratings_base.iloc[train_index]
                val_data = self.original_ratings_base.iloc[val_index]

                self.ratings_base = train_data
                predict_matrix = self.train(algo=algo, alpha=alpha)

                val_users = val_data['user_id'].values
                val_items = val_data['item_id'].values
                val_ratings = val_data['rating'].values

                preds = [predict_matrix[self.all_users.tolist().index(uid)][item_id - 1] for uid, item_id in zip(val_users, val_items)]
                fold_errors.append(np.mean((val_ratings - preds) ** 2))

            mean_errors.append(np.mean(fold_errors))

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mean_errors, marker='o', linestyle='-', label=f'{algo} validation error')
        plt.xlabel('Alpha')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Validation Error vs Alpha for {algo} Regression')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'validation_{algo}.png')
        plt.show()

        # Save the results
        np.save(f'validation_{algo}_results.npy', mean_errors)
        return alphas[np.argmin(mean_errors)], min(mean_errors)

    def predict_rating(self, user_id, item_id):
        if user_id in self.models:
            model = self.models[user_id]
            item_vector = self.tfidf[item_id - 1].reshape(1, -1)
            return model.predict(item_vector)[0]
        else:
            return np.nan

    def evaluate(self):
        y_true = self.ratings_test['rating'].values
        y_pred = [self.predict_rating(uid, iid) for uid, iid in zip(self.ratings_test['user_id'].values, self.ratings_test['item_id'].values)]
        y_pred = np.array(y_pred)
        rmse = root_mean_squared_error(y_true, y_pred, squared=False)
        print(f"RMSE: {rmse}")
        return rmse

if __name__ == '__main__':
    u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

    r_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('./data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('./data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

    i_cols = ['item_id', 'item title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

    cf = ContentFiltering(users, items, ratings_base, ratings_test)
    # optimal_alpha, min_error = cf.validate(algo='ridge', alpha_range=(0.1, 10, 20))
    # print(f'Optimal alpha for Ridge: {optimal_alpha} with MSE: {min_error}')

    optimal_alpha, min_error = cf.validate(algo='lasso', alpha_range=(0.1, 10, 20))
    print(f'Optimal alpha for Lasso: {optimal_alpha} with MSE: {min_error}')
    # predicted_matrix = cf.train()
    # # print(cf.predict_rating(2, 281))
    # print(predicted_matrix.shape)
    # print(predicted_matrix[2][281])