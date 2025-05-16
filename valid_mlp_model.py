import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load MovieLens 100K
def load_data():
    df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df['label'] = df['rating'].astype(float)
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    return df, num_users, num_items

data, num_users, num_items = load_data()
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# 2. Dataset class
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# 3. MLP model
class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_units, dropout):
        super().__init__()
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embed = nn.Embedding(num_items + 1, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze()

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# 4. Grid Search with K-Fold
device = torch.device('cuda' )
results = []
best_loss = float('inf')
best_params = None
#kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'embedding_dim': [4, 8, 16, 32],
    'hidden_units': [
            [64, 32],
            [128, 64],
            [64, 32, 16],
            [128, 64, 32]],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [64, 128, 256],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'dropout': [0.0, 0.2, 0.4],
    'epochs': [20, 50, 100]
}

for emb_dim in param_grid['embedding_dim']:
    for hidden in param_grid['hidden_units']:
        for lr in param_grid['learning_rate']:
            for bs in param_grid['batch_size']:
                for wd in param_grid['weight_decay']:
                    for drop in param_grid['dropout']:
                        for epoch in param_grid['epochs']:
                            #fold_losses = []

                            '''for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                                train_df_fold = data.iloc[train_idx]
                                val_df_fold = data.iloc[val_idx]'''

                            train_loader = DataLoader(MovieLensDataset(train_df), batch_size=bs, shuffle=True)
                            val_loader = DataLoader(MovieLensDataset(val_df), batch_size=bs)

                            model = MLP(num_users, num_items, emb_dim, hidden, drop).to(device)
                            criterion = RMSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                            for epoch in range(epoch):
                                model.train()
                                for users, items, labels in train_loader:
                                    users, items, labels = users.to(device), items.to(device), labels.to(device)
                                    optimizer.zero_grad()
                                    outputs = model(users, items)
                                    loss = criterion(outputs, labels)
                                    loss.backward()
                                    optimizer.step()

                            model.eval()
                            val_loss = 0
                            with torch.no_grad():
                                for users, items, labels in val_loader:
                                    users, items, labels = users.to(device), items.to(device), labels.to(device)
                                    outputs = model(users, items)
                                    val_loss += criterion(outputs, labels).item()
                            val_loss /= len(val_loader)
                            #fold_losses.append(val_loss)

                            #avg_val_loss = sum(fold_losses) / len(fold_losses)
                            print(f" Embedding={emb_dim}, Hidden={hidden}, LR={lr}, BS={bs}, WD={wd}, Dropout={drop},epoch={epoch} → Val Loss={val_loss:.4f}")

                            results.append({
                                'embedding_dim': emb_dim,
                                'hidden_units': hidden,
                                'learning_rate': lr,
                                'batch_size': bs,
                                'weight_decay': wd,
                                'dropout': drop,
                                'epoch': epoch,
                                'val_loss': val_loss
                            })

                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_params = {
                                    'embedding_dim': emb_dim,
                                    'hidden_units': hidden,
                                    'learning_rate': lr,
                                    'batch_size': bs,
                                    'weight_decay': wd,
                                    'dropout': drop,
                                    'epoch': epoch
                                }

# 5. Best result
print("\nBest Hyperparameters:")
print(best_params)
print(f"→ Best Validation RMSE: {best_loss:.4f}")

# 6. Plot
df_results = pd.DataFrame(results)
sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
param_names = ['embedding_dim', 'hidden_units', 'learning_rate', 'batch_size', 'weight_decay', 'dropout', 'epoch']

for ax, param in zip(axes.flat, param_names):
    sns.lineplot(data=df_results, x=param, y='val_loss', marker='o', ax=ax)
    ax.set_title(f"{param} vs Val Loss")

for ax in axes.flat[len(param_names):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
