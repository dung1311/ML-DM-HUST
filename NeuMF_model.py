import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import random

# === Timer và Accumulator ===
class Timer:
    def __init__(self):
        self.start_time = time.time()
    def sum(self):
        return time.time() - self.start_time

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

# === Đọc dữ liệu MovieLens 100K ===
def read_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data_path = "data/ml-100k/u.data"
    data = pd.read_csv(data_path, sep='\t', names=names, engine='python')
    num_users = data['user_id'].nunique()
    num_items = data['item_id'].nunique()
    return data, num_users, num_items

# === Tách tập train/test ===
def split_data(data, num_users, num_items, split_mode='random', test_ratio=0.1):
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
        mask = np.random.uniform(0, 1, len(data)) < 1 - test_ratio
        train_data = data[mask]
        test_data = data[~mask]
    return train_data, test_data

# === Tiền xử lý dữ liệu ===
def load_data(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
    return torch.tensor(users), torch.tensor(items), torch.tensor(scores)

# === Mô hình NeuMF cho hồi quy ===
class NeuMF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens):
        super(NeuMF, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for i, num_hidden in enumerate(nums_hiddens):
            in_features = num_factors * 2 if i == 0 else nums_hiddens[i-1]
            self.mlp.add_module(f'dense_{i}', nn.Linear(in_features, num_hidden))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
        self.prediction_layer = nn.Linear(nums_hiddens[-1] + num_factors, 1)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat([p_mlp, q_mlp], dim=1))
        con_res = torch.cat([gmf, mlp], dim=1)
        return self.prediction_layer(con_res).squeeze(1)

# === Hàm mất mát RMSE ===
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

# === Đánh giá RMSE trên tập test ===
def evaluate_regression(net, test_iter, device):
    net.eval()
    total_loss, total_num = 0.0, 0
    rmse_loss = RMSELoss()
    with torch.no_grad():
        for user, item, rating in test_iter:
            user, item, rating = user.to(device), item.to(device), rating.float().to(device)
            pred = net(user, item)
            loss = rmse_loss(pred, rating)
            total_loss += loss.item() * user.size(0)
            total_num += user.size(0)
    return total_loss / total_num

# === Hàm huấn luyện ===
def train(net, train_iter, test_iter, loss_fn, optimizer,
                  num_users, num_items, num_epochs, devices, evaluator):
    timer = Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        net.train()
        for user, item, rating in train_iter:
            user, item, rating = user.to(devices), item.to(devices), rating.float().to(devices)
            optimizer.zero_grad()
            pred = net(user, item)
            loss = loss_fn(pred, rating)
            loss.backward()
            optimizer.step()
            metric.add(loss.item() * user.shape[0], user.shape[0])
        test_rmse = evaluator(net, test_iter, devices)
        print(f"Epoch {epoch+1}: train RMSE {(metric[0] / metric[1]):.3f}, test RMSE {test_rmse:.3f}")
    print(f'Training speed: {metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

# === Main ===
if __name__ == "__main__":
    # Siêu tham số
    batch_size = 1024
    lr, num_epochs, wd = 0.0001, 50, 1e-3
    nums_hiddens = [128, 64]

    # Thiết bị
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Đọc và xử lý dữ liệu
    df, num_users, num_items = read_data()
    train_data, test_data = split_data(df, num_users, num_items, 'seq-aware')
    users_train, items_train, ratings_train = load_data(train_data, num_users, num_items)
    users_test, items_test, ratings_test = load_data(test_data, num_users, num_items)

    train_dataset = TensorDataset(users_train, items_train, ratings_train)
    test_dataset = TensorDataset(users_test, items_test, ratings_test)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size)

    # Khởi tạo mô hình
    net = NeuMF(8, num_users, num_items, nums_hiddens)
    net.to(devices)
    for param in net.parameters():
        init.normal_(param, 0, 0.01)

    # Khởi tạo loss và optimizer
    loss_fn = RMSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # Huấn luyện
    train(net, train_iter, test_iter, loss_fn, optimizer,
                  num_users, num_items, num_epochs, devices, evaluate_regression)

    # Lưu mô hình
    torch.save(net.state_dict(), 'neumf_model.pth')
