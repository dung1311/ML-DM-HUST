import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import pandas as pd
import NeuMF_model
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import tqdm

# Hàm vẽ biểu đồ RMSE
def plot_grid_search_results(results):
    results_df = pd.DataFrame(results, columns=['RMSE', 'Params'])
    params_df = pd.json_normalize(results_df['Params'])
    results_df = pd.concat([results_df.drop('Params', axis=1), params_df], axis=1)
    results_df['nums_hiddens'] = results_df['nums_hiddens'].apply(lambda x: str(x))

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    param_names = ['num_factors', 'lr', 'weight_decay', 'nums_hiddens', 'epoch','batch_size', 'dropout']

    for ax, param in zip(axes.flat, param_names):
        avg = results_df.groupby(param)['RMSE'].mean().reset_index()
        sns.lineplot(data=avg, x=param, y='RMSE', marker='o', ax=ax)
        ax.set_title(f"{param} vs RMSE")

    plt.tight_layout()
    plt.show()

# Grid Search
def grid_search_neumf(param_grid, df, num_users, num_items, devices):
    best_rmse = float('inf')
    best_params = None
    results = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for factors, lr, wd, hiddens, epochs, batch_size, drop in product(param_grid['num_factors'], 
                                                    param_grid['lr'],
                                                    param_grid['weight_decay'], 
                                                    param_grid['nums_hiddens'],
                                                    param_grid['epoch'],
                                                    param_grid['batch_size'],
                                                    param_grid['dropout']):
        
        fold_rmses = []

        for fold, (train_index, val_index) in enumerate(kf.split(df)):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]

            users_train, items_train, ratings_train = NeuMF_model.load_data(train_df, num_users, num_items)
            users_val, items_val, ratings_val = NeuMF_model.load_data(val_df, num_users, num_items)

            train_ds = TensorDataset(users_train, items_train, ratings_train)
            val_ds = TensorDataset(users_val, items_val, ratings_val)

            train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_iter = DataLoader(val_ds, batch_size=batch_size)

            model = NeuMF_model.NeuMF(factors, num_users, num_items, hiddens,drop).to(devices)
            for param in model.parameters():
                nn.init.normal_(param, 0, 0.01)

            loss_fn = NeuMF_model.RMSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            for epoch in range(epochs):
                model.train()
                for user, item, rating in tqdm.tqdm(train_iter, desc=f"Epoch {epoch+1}", leave=False):
                    user, item, rating = user.to(devices), item.to(devices), rating.float().to(devices)
                    optimizer.zero_grad()
                    pred = model(user, item).squeeze()
                    loss = loss_fn(pred, rating)
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for user, item, rating in val_iter:
                    user, item = user.to(devices), item.to(devices)
                    pred = model(user, item).squeeze().cpu()
                    val_preds.extend(pred.numpy())
                    val_labels.extend(rating.numpy())
            val_preds = np.clip(val_preds, 1, 5)
            rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_labels)) ** 2))
            fold_rmses.append(rmse)

            avg_rmse = np.mean(fold_rmses)
            print(f"Average RMSE over {kf.n_splits} folds: {avg_rmse:.4f}")
            print(f"Tested params: factors={factors}, lr={lr}, wd={wd}, hiddens={hiddens}, epoch={epochs}, batch_size={batch_size}, drop_out={drop} → RMSE: {rmse:.4f}")

            results.append((avg_rmse, {
                'num_factors': factors,
                'lr': lr,
                'weight_decay': wd,
                'nums_hiddens': hiddens,
                'epoch': epochs,
                'batch_size': batch_size,
                'dropout': drop
            }))

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = results[-1][1]

    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"{k} = {v}")
    print(f"Best RMSE = {best_rmse:.4f}")
    return best_params, results

# Thiết bị
devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tải dữ liệu
df, num_users, num_items = NeuMF_model.read_data()

# Grid search
param_grid = {
    'num_factors': [8, 16, 32, 64],
    'lr': [	1e-4, 5e-4, 1e-3, 5e-3],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'nums_hiddens': [
            [64, 32],
            [128, 64],
            [64, 32, 16],
            [128, 64, 32],
            [256, 128, 64, 32]],
    'epoch': [20, 50, 100],
    'batch_size': [ 128, 512,1024],
    'dropout': [0.0, 0.2, 0.4]
}

best_params, all_results = grid_search_neumf(param_grid, df, num_users, num_items, devices)

# Vẽ biểu đồ
plot_grid_search_results(all_results)
