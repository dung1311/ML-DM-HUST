import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. Chuẩn bị dữ liệu 
def load_data():
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', 
                       names=['user_id','item_id','rating','timestamp'])
    num_users = data['user_id'].nunique()
    num_items = data['item_id'].nunique()
    return data, num_users, num_items


data, num_users, num_items = load_data()
data['interaction'] = 1

# 2. Tạo dataset 
class MovieLensDataset(Dataset):
    def __init__(self, users, items, interactions):
        self.users = users
        self.items = items
        self.interactions = interactions
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.interactions[idx], dtype=torch.float)
        )

# Split data
train, test = train_test_split(data, test_size=0.2)
train_dataset = MovieLensDataset(train['user_id'], train['item_id'], train['interaction'])
test_dataset = MovieLensDataset(test['user_id'], test['item_id'], test['interaction'])

# 3. Định nghĩa MLP Model 
class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128,64,32]):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(2*embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
    def forward(self, user, item):
        user_vec = self.user_embed(user)
        item_vec = self.item_embed(item)
        concat = torch.cat([user_vec, item_vec], dim=-1)
        return self.layers(concat)

# 4. Khởi tạo model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(num_users+1, num_items+1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop 
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

for epoch in range(20):
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

