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

# Chuyển đổi sang implicit feedback (rating > 0 → 1)
data, num_users, num_items = load_data()
data['label'] = (data['rating'] > 0).astype(float)

# 2. Tạo Dataset 
class MovieLensDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = users
        self.items = items
        self.labels = labels
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

# Split data 
train, test = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = MovieLensDataset(
    train['user_id'].reset_index(drop=True),
    train['item_id'].reset_index(drop=True),
    train['label'].reset_index(drop=True)
)
test_dataset = MovieLensDataset(
    test['user_id'].reset_index(drop=True),
    test['item_id'].reset_index(drop=True),
    test['label'].reset_index(drop=True)
)

# 3. Định nghĩa GMF Model
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




# 4. Khởi tạo model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GMF(num_users+1, num_items+1, embedding_dim=64).to(device)
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
        outputs = model(users, items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for users, items, labels in test_loader:
            outputs = model(users.to(device), items.to(device))
            test_loss += criterion(outputs, labels.to(device)).item()
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss/len(train_loader):.4f} | Test Loss: {test_loss/len(test_loader):.4f}')

# 6. Dự đoán 
def recommend(user_id, k=10):
    user = torch.tensor([user_id]).to(device)
    all_items = torch.arange(num_items+1).to(device)
    with torch.no_grad():
        scores = model(user.repeat(num_items+1), all_items)
    top_items = scores.argsort(descending=True)[:k]
    return top_items.cpu().numpy()
