import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import time

# -------------------------------
# Hàm vẽ biểu đồ RMSE và các tham số
# -------------------------------
def plot_grid_search_results(results):
    results_df = pd.DataFrame(results, columns=['RMSE', 'Params'])
    params_df = pd.json_normalize(results_df['Params'])
    results_df = pd.concat([results_df.drop('Params', axis=1), params_df], axis=1)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    param_names = ['num_factors', 'L', 'd', 'd_prime', 'drop_ratio', 'lr', 'batch_size', 'weight_decay', 'epoch']

    for ax, param in zip(axes.flat, param_names):
        avg = results_df.groupby(param)['RMSE'].mean().reset_index()
        sns.lineplot(data=avg, x=param, y='RMSE', marker='o', ax=ax)
        ax.set_title(f"{param} vs RMSE")

    plt.tight_layout()
    plt.show()

# -------------------------------
# Data processing
# -------------------------------
def read_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', header=None, names=names)
    num_users = data.user_id.nunique()
    num_items = data.item_id.nunique()
    return data, num_users, num_items

def load_data(data, feedback='explicit'):
    users, items, scores = [], [], []
    inter = {}
    for line in data.itertuples():
        user_index, item_index = line[1] - 1, line[2] - 1
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        inter.setdefault(user_index, []).append((item_index, score))
    return users, items, scores, inter

class SeqDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings, L, num_users, num_items, candidates):
        self.L = L
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        ratings = np.array(ratings)

        sort_idx = np.lexsort((item_ids, user_ids))
        u_ids = user_ids[sort_idx]
        i_ids = item_ids[sort_idx]
        r_vals = ratings[sort_idx]

        self.data = []
        user_hist = {}

        for u, i, r in zip(u_ids, i_ids, r_vals):
            user_hist.setdefault(u, []).append((i, r))

        self.test_seq = np.zeros((num_users, L))

        for u in user_hist:
            items_ratings = user_hist[u]
            items = [x[0] for x in items_ratings]
            ratings = [x[1] for x in items_ratings]
            if len(items) < L + 1:
                continue
            for i in range(L, len(items)):
                self.data.append((u, items[i - L:i], ratings[i]))
            self.test_seq[u] = items[-L:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, seq, rating = self.data[idx]
        return torch.LongTensor([user]), torch.LongTensor(seq), torch.FloatTensor([rating])

class Caser(nn.Module):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16, d_prime=4, drop_ratio=0.05):
        super(Caser, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)

        self.conv_v = nn.Conv2d(1, d_prime, (L, 1))
        self.convs_h = nn.ModuleList([
            nn.Conv2d(1, d, (i, num_factors)) for i in range(1, L + 1)
        ])

        self.dropout = nn.Dropout(drop_ratio)
        self.fc = nn.Linear(d_prime * num_factors + d * L + num_factors, 1)

    def forward(self, user_id, seq):
        item_embs = self.Q(seq).unsqueeze(1)
        out_v = self.conv_v(item_embs).squeeze(2).view(seq.size(0), -1)
        out_hs = []
        for conv in self.convs_h:
            h = torch.relu(conv(item_embs)).squeeze(3)
            pool = nn.functional.max_pool1d(h, h.shape[2]).squeeze(2)
            out_hs.append(pool)
        out_h = torch.cat(out_hs, dim=1)

        user_emb = self.P(user_id.squeeze(1))
        z = self.fc(self.dropout(torch.cat([out_v, out_h, user_emb], dim=1)))
        return z.squeeze()

def evaluate(model, data_loader, device, epoch=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = nn.MSELoss()
    eval_start = time.time()
    desc = f"Evaluating" if epoch is None else f"Epoch {epoch+1} (val)"
    with torch.no_grad():
        for user, seq, rating in tqdm(data_loader, desc=desc, leave=False):
            user, seq, rating = user.to(device), seq.to(device), rating.to(device)
            pred = model(user, seq)
            loss = criterion(pred, rating.squeeze())
            total_loss += loss.item() * rating.size(0)
            total_samples += rating.size(0)
    rmse = np.sqrt(total_loss / total_samples)
    eval_end = time.time()
    print(f"Validation RMSE: {rmse:.4f} | Evaluation time: {eval_end - eval_start:.2f}s")
    return rmse

def train(model, train_loader, optimizer, device, epochs=5, val_loader=None):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0
        for user, seq, rating in tqdm(train_loader, desc=f"Epoch {epoch+1} (train)", leave=False):
            user, seq, rating = user.to(device), seq.to(device), rating.to(device)
            pred = model(user, seq)
            loss = criterion(pred, rating.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rating.size(0)
        epoch_end = time.time()
        print(f"Epoch {epoch+1} training loss: {running_loss/len(train_loader.dataset):.4f} | Time: {epoch_end - epoch_start:.2f}s")
        # Nếu muốn đánh giá từng epoch: 
        # if val_loader is not None:
        #     evaluate(model, val_loader, device, epoch)

# -------------------------------
# Grid search
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, num_users, num_items = read_data()
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    u_train, i_train, r_train, candidates_train = load_data(train_df, "explicit")
    u_val, i_val, r_val, candidates_val = load_data(val_df, "explicit")

    param_grid = {
        "num_factors": [8, 16, 32, 64],
        'L': [3, 5, 7],
        'd': [8, 16, 32],
        'd_prime': [4, 8, 16],
        "drop_ratio": [0.1, 0.2, 0.3, 0.5],
        "lr": [0.001, 0.005, 0.01],
        'batch_size': [164, 128, 256],
        'weight_decay': [0, 1e-5, 1e-4, 1e-3],
        'epoch': [20, 50, 100]
    }

    all_combinations = list(itertools.product(
        param_grid["num_factors"],
        param_grid["L"],
        param_grid["d"],
        param_grid["d_prime"],
        param_grid["drop_ratio"],
        param_grid["lr"],
        param_grid["batch_size"],
        param_grid["weight_decay"],
        param_grid["epoch"],
    ))

    print(f"Running grid search over {len(all_combinations)} combinations...\n")

    best_rmse = float("inf")
    best_params = None
    results = []

    grid_start = time.time()

    for comb in tqdm(all_combinations, desc="Grid Search Progress"):
        num_factors, L, d, d_prime, drop_ratio, lr, batch_size, wd, epoch = comb
        print(f"\nTrying: num_factors={num_factors}, L={L}, d={d}, d_prime={d_prime}, drop_ratio={drop_ratio}, lr={lr}, batch_size={batch_size}, weight_decay={wd}, epoch={epoch}")

        # Tạo datasets và dataloaders 
        train_dataset = SeqDataset(u_train, i_train, r_train, L, num_users, num_items, candidates_train)
        val_dataset = SeqDataset(u_val, i_val, r_val, L, num_users, num_items, candidates_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = Caser(num_factors=num_factors, num_users=num_users, num_items=num_items,
                    L=L, d=d, d_prime=d_prime, drop_ratio=drop_ratio).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train(model, train_loader, optimizer, device, epochs=epoch)
        rmse = evaluate(model, val_loader, device)

        results.append((rmse, {
            "num_factors": num_factors,
            "L": L,
            "d": d,
            "d_prime": d_prime,
            "drop_ratio": drop_ratio,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": wd,
            "epoch": epoch
        }))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = results[-1][1]

    grid_end = time.time()
    print(f"\nTotal grid search time: {grid_end - grid_start:.2f} seconds")

    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"{k} = {v}")
    print(f"Best RMSE = {best_rmse:.4f}")

    plot_grid_search_results(results)
