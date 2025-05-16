import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# 1. Read and prepare data
def read_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', header=None, names=names)
    num_users = data.user_id.nunique()
    num_items = data.item_id.nunique()
    return data, num_users, num_items

def split_data(data, num_users, num_items, split_mode='seq-aware', test_ratio=0.1):
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = np.random.rand(len(data)) < (1 - test_ratio)
        train_data = data[mask]
        test_data = data[~mask]
    return train_data, test_data

def load_data(data, num_users, num_items, feedback='explicit'):
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

# 2. Dataset for RMSE training
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
            for i in range(L, len(items)):
                self.data.append((u, items[i - L:i], ratings[i]))
            self.test_seq[u] = items[-L:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, seq, rating = self.data[idx]
        return torch.LongTensor([user]), torch.LongTensor(seq), torch.FloatTensor([rating])

# 3. Caser Model (adapted for rating prediction)
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
        item_embs = self.Q(seq).unsqueeze(1)  # (batch, 1, L, emb)

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

# 4. Training function for rating prediction
def train_rating(model, train_loader, loss_fn, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        for user, seq, rating in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            user, seq, rating = user.to(device), seq.to(device), rating.to(device)

            pred = model(user, seq)
            loss = loss_fn(pred, rating.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * rating.size(0)
            total_samples += rating.size(0)

        rmse = np.sqrt(total_loss / total_samples)
        print(f"Epoch {epoch+1}, RMSE: {rmse:.4f}")

# 5. Main
if __name__ == "__main__":
    L = 5
    batch_size = 512
    num_factors = 10
    num_epochs = 20
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, num_users, num_items = read_data()
    train_data, test_data = split_data(df, num_users, num_items, split_mode='seq-aware')
    users_train, items_train, ratings_train, candidates = load_data(train_data, num_users, num_items, feedback="explicit")

    train_dataset = SeqDataset(users_train, items_train, ratings_train, L, num_users, num_items, candidates)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Caser(num_factors, num_users, num_items, L).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_rating(model, train_loader, loss_fn, optimizer, device, num_epochs)
