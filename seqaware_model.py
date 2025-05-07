import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import urllib.request
import os

# 1. Lớp Caser 
class Caser(nn.Module):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16, d_prime=4, drop_ratio=0.05):
        super(Caser, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        
        self.conv_v = nn.Conv2d(1, d_prime, kernel_size=(L,1))
        
        h = [i+1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, d, kernel_size=(i, num_factors)) for i in h])
        self.max_pool = nn.ModuleList([nn.MaxPool1d(kernel_size=L-i+1) for i in h])
        
        self.fc1_dim_v = d_prime * num_factors
        self.fc1_dim_h = d * len(h)
        self.fc = nn.Linear(self.fc1_dim_v + self.fc1_dim_h, num_factors)
        
        self.Q_prime = nn.Embedding(num_items, num_factors*2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)
        self.relu = nn.ReLU()

    def forward(self, user_id, seq, item_id):
        item_embs = self.Q(seq).unsqueeze(1)
        user_emb = self.P(user_id)
        
        out_v, out_h = None, None
        
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(out_v.size(0), -1)
            
        if self.d:
            out_hs = []
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = conv(item_embs).squeeze(3)
                conv_out = self.relu(conv_out)
                pool_out = maxp(conv_out).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, dim=1)
            
        out = torch.cat([out_v, out_h], dim=1)
        z = self.fc(self.dropout(out))
        
        x = torch.cat([z, user_emb], dim=1)
        q_prime_i = self.Q_prime(item_id).squeeze(1)
        b = self.b(item_id).squeeze(1)
        res = (x * q_prime_i).sum(dim=1) + b
        return res

# 2. Lớp Dataset 
class SeqDataset(Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items, candidates):
        self.L = L
        self.candidates = candidates
        self.all_items = set(range(num_items))
        
        # Sắp xếp dữ liệu
        sorted_idx = np.argsort(user_ids)
        user_ids = user_ids[sorted_idx]
        item_ids = item_ids[sorted_idx]
        
        # Tạo sequences
        self.sequences = {}
        current_user = None
        current_seq = []
        
        for uid, iid in zip(user_ids, item_ids):
            if uid != current_user:
                if current_user is not None:
                    self.sequences[current_user] = current_seq
                current_user = uid
                current_seq = []
            current_seq.append(iid)
        if current_user is not None:
            self.sequences[current_user] = current_seq
        
        # Tạo samples
        self.samples = []
        self.test_seq = {}
        
        for user, seq in self.sequences.items():
            if len(seq) > L:
                for i in range(L, len(seq)):
                    self.samples.append((user, seq[i-L:i], seq[i]))
                self.test_seq[user] = seq[-L:]
            else:
                self.test_seq[user] = seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, seq, target = self.samples[idx]
        neg = random.choice(list(self.all_items - self.candidates[user]))
        return (
            torch.LongTensor([user]),
            torch.LongTensor(seq),
            torch.LongTensor([target]),
            torch.LongTensor([neg])
        )
    
    def get_test_seq(self):
        return {u: torch.LongTensor(s) for u, s in self.test_seq.items()}

# 3. Hàm xử lý dữ liệu 
def read_data_ml100k():
    df = pd.read_csv(
        'data/ml-100k/u.data', 
        sep='\t', 
        header=None, 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    return df, num_users, num_items

def split_data_ml100k(df, mode='seq-aware'):
    train_data = []
    test_data = []
    
    grouped = df.groupby('user_id')
    for user, group in grouped:
        group = group.sort_values('timestamp')
        if mode == 'seq-aware' and len(group) > 1:
            train_data.append(group.iloc[:-1])
            test_data.append(group.iloc[-1:])
        else:
            train_data.append(group)
    
    return pd.concat(train_data), pd.concat(test_data)

def load_data_ml100k(df):
    candidates = {}
    for u, i in zip(df['user_id'], df['item_id']):
        if u not in candidates:
            candidates[u] = set()
        candidates[u].add(i)
    return df['user_id'].values, df['item_id'].values, candidates

# 4. Pipeline huấn luyện
def main():
    # Đọc và xử lý dữ liệu
    df, num_users, num_items = read_data_ml100k()
    train_df, test_df = split_data_ml100k(df)
    
    users_train, items_train, candidates = load_data_ml100k(train_df)
    
    # Tạo dataset
    L = 5
    batch_size = 4096
    dataset = SeqDataset(
        users_train, items_train, L, num_users, num_items, candidates
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Khởi tạo model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Caser(10, num_users, num_items, L).to(device)
    
    # Hàm loss và optimizer
    def bpr_loss(pos_preds, neg_preds):
        return -torch.log(torch.sigmoid(pos_preds - neg_preds)).mean()
    
    optimizer = optim.Adam(model.parameters(), lr=0.04, weight_decay=1e-5)
    
    # Huấn luyện
    for epoch in range(8):
        model.train()
        total_loss = 0
        for batch in train_loader:
            users, seqs, targets, negs = [x.to(device).squeeze() for x in batch]
            
            optimizer.zero_grad()
            pos_preds = model(users, seqs, targets)
            neg_preds = model(users, seqs, negs)
            loss = bpr_loss(pos_preds, neg_preds)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

if __name__ == "__main__":
    main()
