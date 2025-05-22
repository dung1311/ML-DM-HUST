import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. Chuẩn bị dữ liệu 
def load_data():
    df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df['label'] = df['rating'].astype(float)
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    return df, num_users, num_items


data, num_users, num_items = load_data()


# 2. Tạo dataset 
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
# Split data
train, test = train_test_split(data, test_size=0.2)

train_loader = DataLoader(MovieLensDataset(train), batch_size=64, shuffle=True)
val_loader = DataLoader(MovieLensDataset(test), batch_size=64)

# 3. Định nghĩa MLP Model 
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
    
# 4. Khởi tạo model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(num_users+1, num_items+1, 4, [128,64],0.2).to(device)
criterion = RMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# 5. Training loop 
batch_size = 64
train_loader = DataLoader(MovieLensDataset(train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(MovieLensDataset(test), batch_size=batch_size)

for epoch in range(100):
    model.train()
    train_loss = 0
    for users, items, labels in train_loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(users, items).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for users, items, labels in test_loader:
            outputs = model(users.to(device), items.to(device)).squeeze()
            test_loss += criterion(outputs, labels.to(device)).item()
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss/len(train_loader):.4f} | Test Loss: {test_loss/len(test_loader):.4f}')

