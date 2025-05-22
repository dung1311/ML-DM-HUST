import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from tqdm import tqdm

# ----- Set random seed -----
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----- GMF Model -----
class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        x = user_embed * item_embed
        out = self.output(x)
        return out.view(-1)

# ----- Dataset -----
class MovieLensDataset(Dataset):
    def __init__(self, user_item_pairs, labels):
        self.user_item_pairs = user_item_pairs
        self.labels = labels

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# ----- RMSE Loss -----
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# ----- Load data -----
def load_data():
    df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    df['label'] = df['rating'].astype(float)
    user_map = {u: i for i, u in enumerate(df['user'].unique())}
    item_map = {i: j for j, i in enumerate(df['item'].unique())}
    df['user'] = df['user'].map(user_map)
    df['item'] = df['item'].map(item_map)
    return df, len(user_map), len(item_map)

# ----- Training with K-Fold -----
data, num_users, num_items = load_data()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []
best_loss = float('inf')
best_params = None

param_grid = {
    'embedding_dim': [4, 8, 16, 32],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [32, 64, 128],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'epochs': [20, 30, 50, 100]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for emb_dim in param_grid['embedding_dim']:
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for wd in param_grid['weight_decay']:
                for ep in param_grid['epochs']:
                    fold_losses = []
                    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                        train_df = data.iloc[train_idx]
                        val_df = data.iloc[val_idx]

                        train_dataset = MovieLensDataset(list(zip(train_df['user'], train_df['item'])), train_df['label'].tolist())
                        val_dataset = MovieLensDataset(list(zip(val_df['user'], val_df['item'])), val_df['label'].tolist())
                        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=bs)

                        model = GMF(num_users, num_items, emb_dim).to(device)
                        criterion = RMSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                        # Training loop
                        for epoch in range(ep):
                            model.train()
                            for users, items, labels in train_loader:
                                users, items, labels = users.to(device), items.to(device), labels.to(device)
                                optimizer.zero_grad()
                                outputs = model(users, items)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()

                        # Evaluation
                        model.eval()
                        all_preds, all_labels = [], []
                        with torch.no_grad():
                            for users, items, labels in val_loader:
                                users, items, labels = users.to(device), items.to(device), labels.to(device)
                                outputs = model(users, items)
                                outputs = torch.clamp(outputs, 1.0, 5.0)
                                all_preds.append(outputs.cpu())
                                all_labels.append(labels.cpu())

                        preds = torch.cat(all_preds)
                        labels = torch.cat(all_labels)
                        rmse = torch.sqrt(torch.mean((preds - labels) ** 2)).item()
                        fold_losses.append(rmse)

                    avg_rmse = np.mean(fold_losses)
                    print(f"[Params] emb={emb_dim}, lr={lr}, bs={bs}, wd={wd}, epochs={ep} → RMSE: {avg_rmse:.4f}")

                    results.append({
                        'embedding_dim': emb_dim,
                        'learning_rate': lr,
                        'batch_size': bs,
                        'weight_decay': wd,
                        'epochs': ep,
                        'val_loss': avg_rmse
                    })

                    if avg_rmse < best_loss:
                        best_loss = avg_rmse
                        best_params = {
                            'embedding_dim': emb_dim,
                            'learning_rate': lr,
                            'batch_size': bs,
                            'weight_decay': wd,
                            'epochs': ep
                        }

# ----- Best Result -----
print("\nBest Parameters:")
print(best_params)
print(f"→ Best RMSE: {best_loss:.4f}")

# ----- Plotting -----
df_results = pd.DataFrame(results)
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
param_names = ['embedding_dim', 'learning_rate', 'batch_size', 'weight_decay', 'epochs']

for ax, param in zip(axes.flat, param_names):
    sns.lineplot(data=df_results, x=param, y='val_loss', marker='o', ax=ax)
    ax.set_title(f"{param} vs Validation RMSE")

plt.tight_layout()
plt.show()
