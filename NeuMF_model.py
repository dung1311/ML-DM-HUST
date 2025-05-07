import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import random
import sys
import pandas as pd
import time

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
    
def read_data_ml100k():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data_path = "data/ml-100k/u.data"
    data = pd.read_csv(data_path, sep='\t', names=names, engine='python')

    
    num_users = data['user_id'].nunique()
    num_items = data['item_id'].nunique()
    
    return data, num_users, num_items


def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
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
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    
    # Chuyển dữ liệu sang tensor PyTorch
    users = torch.tensor(users)
    items = torch.tensor(items)
    scores = torch.tensor(scores)
    
    if feedback == 'explicit':
        inter = torch.tensor(inter)
    
    return users, items, scores, inter
def split_and_load_ml100k_pytorch(split_mode='seq-aware', feedback='explicit',
                                  test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    
    # Chuyển dữ liệu sang tensor PyTorch
    train_u, train_i, train_r = torch.tensor(train_u), torch.tensor(train_i), torch.tensor(train_r)
    test_u, test_i, test_r = torch.tensor(test_u), torch.tensor(test_i), torch.tensor(test_r)
    
    # Tạo bộ dữ liệu PyTorch
    train_set = TensorDataset(train_u, train_i, train_r)
    test_set = TensorDataset(test_u, test_i, test_r)
    
    # Tạo DataLoader
    train_iter = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_iter = DataLoader(test_set, batch_size=batch_size)
    
    return num_users, num_items, train_iter, test_iter
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc


test_input = {}  
candidates = {} 

def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_items)])  # Sửa lại thành num_items
    
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, scores = [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        
        # Tạo bộ dữ liệu
        dataset = TensorDataset(torch.tensor(user_ids), torch.tensor(item_ids))
        test_data_iter = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        for batch in test_data_iter:
            user_ids_batch, item_ids_batch = batch
            user_ids_batch, item_ids_batch = user_ids_batch.to(devices), item_ids_batch.to(devices)
            
            # Tính điểm số
            scores_batch = net(user_ids_batch, item_ids_batch).detach().cpu().numpy()
            scores.extend(scores_batch)
        
        # Xếp hạng mặt hàng
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        
        # Tính hit rate và auc
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))

class NeuMF (nn.Module):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens):
        super(NeuMF, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        
        self.mlp = nn.Sequential()
        for i, num_hidden in enumerate(nums_hiddens):
            if i == 0:
                self.mlp.add_module(f'dense_{i}', nn.Linear(num_factors * 2, num_hidden))
            else:
                self.mlp.add_module(f'dense_{i}', nn.Linear(nums_hiddens[i-1], num_hidden))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
        
        self.prediction_layer = nn.Linear(nums_hiddens[-1] + num_factors, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat([p_mlp, q_mlp], dim=1))
        con_res = torch.cat([gmf, mlp], dim=1)
        return self.sigmoid(self.prediction_layer(con_res))
    

class PRDataset(Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.candidates = candidates
        self.all_items = set(range(num_items))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all_items - set(self.candidates[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx],self.items[idx], neg_items[indices]

def train_ranking(net, train_iter, test_iter, loss_fn, optimizer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer = Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(v.to(devices))
            optimizer.zero_grad()
            p_pos = net(*input_data[:-1])
            p_neg = net(*input_data[:-2], input_data[-1])
            ls = [loss_fn(p, n) for p, n in zip(p_pos, p_neg)]
            loss = torch.mean(torch.stack(ls))
            loss.backward()
            optimizer.step()
            metric.add(loss.item(), len(values[0]), len(values[0]))
        net.eval()
        with torch.no_grad():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
# Định nghĩa hàm mất mát
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_pred, neg_pred):
        return -torch.log(torch.sigmoid(pos_pred - neg_pred)).mean()

loss_fn = BPRLoss()

batch_size = 1024
df, num_users, num_items = read_data_ml100k()
train_data, test_data = split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')

# Load dữ liệu
users_train, items_train, ratings_train, candidates = load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")


train_dataset = PRDataset(users_train, items_train, candidates, num_items)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Chọn thiết bị (GPU hoặc CPU)
devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình NeuMF
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])

# Chuyển mô hình sang thiết bị
net.to(devices)

# Khởi tạo tham số mô hình

for param in net.parameters():
    init.normal_(param, 0, 0.01)


# Khởi tạo bộ tối ưu hóa
lr, num_epochs, wd, optimizer_name = 0.01, 10, 1e-5, 'adam'
if optimizer_name == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

if __name__ == "__main__":
    train_ranking(net, train_iter, test_iter, loss_fn, optimizer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)

torch.save(net.state_dict(), 'neumf_model.pth')