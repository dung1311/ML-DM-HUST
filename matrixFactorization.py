import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load user, ratings, and item data
u_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
users = pd.read_csv("./data/ml-100k/u.user", sep="|", header=None, names=u_cols, encoding="latin-1")
print("Number of users:", users.shape[0])

# Ratings
r_cols = ["user_id", "item_id", "rating", "unix_timestamp"]
ratings_base = pd.read_csv("./data/ml-100k/ua.base", sep="\t", header=None, names=r_cols, encoding="latin-1")
ratings_test = pd.read_csv("./data/ml-100k/ua.test", sep="\t", header=None, names=r_cols, encoding="latin-1")
print("Number of ratings in base set:", ratings_base.shape[0])
print("Number of ratings in test set:", ratings_test.shape[0])

# Items
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
print("Number of items:", items.shape[0])

# mkdir to save plots
os.makedirs("plots", exist_ok=True)

# Custom Dataset for MovieLens
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings
        self.users = ratings['user_id'].unique()
        self.items = ratings['item_id'].unique()
        self.user_to_index = {user: idx for idx, user in enumerate(self.users)}
        self.item_to_index = {item: idx for idx, item in enumerate(self.items)}

        self.data = [
            (self.user_to_index[row['user_id']], self.item_to_index[row['item_id']], row['rating'])
            for _, row in ratings.iterrows()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, rating = self.data[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(rating, dtype=torch.float32)

# Matrix Factorization Model
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, n_factors=10):
        super(MatrixFactorization, self).__init__()
        self.user_factors = torch.nn.Embedding(num_users, n_factors)
        self.item_factors = torch.nn.Embedding(num_items, n_factors)
        self.user_bias = torch.nn.Embedding(num_users, 1)
        self.item_bias = torch.nn.Embedding(num_items, 1)
        
        # Initialize weights
        torch.nn.init.normal_(self.user_factors.weight, 0, 0.01)
        torch.nn.init.normal_(self.item_factors.weight, 0, 0.01)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)
        
        # Global bias
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        
        # Move model to GPU if available
        self.to(device)

    def forward(self, user, item):
        # Move inputs to device
        user = user.to(device)
        item = item.to(device)
        
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        user_bias = self.user_bias(user).squeeze(-1)  # Use -1 to safely squeeze last dimension
        item_bias = self.item_bias(item).squeeze(-1)  # Use -1 to safely squeeze last dimension

        # Handle both batch and single item cases
        if user_embedding.dim() > 1:
            # For batches: sum along embedding dimension
            rating = torch.sum(user_embedding * item_embedding, dim=1) + user_bias + item_bias + self.global_bias
        else:
            # For single items: sum the entire tensor
            rating = torch.sum(user_embedding * item_embedding) + user_bias + item_bias + self.global_bias
            
        return rating

    def train_model(self, train_loader, val_loader=None, n_epochs=10, lr=0.01, weight_decay=0.001, early_stopping=5):
        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Training phase
            self.train()
            total_train_loss = 0
            train_samples = 0
            
            for user, item, rating in train_loader:
                # Move data to device
                user, item, rating = user.to(device), item.to(device), rating.to(device)
                
                optimizer.zero_grad()
                prediction = self(user, item)
                loss = criterion(prediction, rating)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item() * user.size(0)
                train_samples += user.size(0)
            
            avg_train_loss = total_train_loss / train_samples
            train_losses.append(avg_train_loss)
            
            # Validation phase (if validation data is provided)
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion)
                val_losses.append(val_loss)
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {key: value.cpu() for key, value in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Restore best model state
                        if best_model_state is not None:
                            self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        break
            else:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Return losses for plotting
        if val_loader:
            return best_val_loss, train_losses, val_losses
        else:
            return avg_train_loss, train_losses, None

    def evaluate(self, data_loader, criterion=None):
        if criterion is None:
            criterion = torch.nn.MSELoss()
            
        self.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for user, item, rating in data_loader:
                # Move data to device
                user, item, rating = user.to(device), item.to(device), rating.to(device)
                
                prediction = self(user, item)
                loss = criterion(prediction, rating)
                total_loss += loss.item() * user.size(0)
                total_samples += user.size(0)
        
        return total_loss / total_samples  # Return MSE, not RMSE here

def predict_ratings(model, user_indices, item_indices):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor(user_indices, dtype=torch.long).to(device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)
        predictions = model(user_tensor, item_tensor)
    return predictions.cpu().numpy()

def k_fold_cv_parameter_search(dataset, n_factors_list, batch_size=64, n_folds=5, n_epochs=10, weight_decay=0.001):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    
    # For plotting
    nfactors_rmse = []
    
    for n_factors in n_factors_list:
        start_time = time.time()
        fold_losses = []
        
        print(f"\n=== Testing parameter: n_factors={n_factors} ===")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
            fold_start = time.time()
            print(f"Fold {fold+1}/{n_folds}")
            
            # Create train and validation subsets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            # Create and train model
            model = MatrixFactorization(len(dataset.users), len(dataset.items), n_factors=n_factors)
            val_loss, _, _ = model.train_model(
                train_loader, val_loader, 
                n_epochs=n_epochs,
                weight_decay=weight_decay
            )
            
            fold_losses.append(val_loss)
            fold_time = time.time() - fold_start
            print(f"Fold {fold+1} completed in {fold_time:.2f}s with validation loss: {val_loss:.4f}")
        
        # Calculate average validation loss across folds
        avg_val_loss = np.mean(fold_losses)
        avg_rmse = np.sqrt(avg_val_loss)
        total_time = time.time() - start_time
        print(f"Average validation loss: {avg_val_loss:.4f}, RMSE: {avg_rmse:.4f}")
        print(f"Total time for parameter: {total_time:.2f}s")
        
        # Store results for plotting
        nfactors_rmse.append((n_factors, avg_rmse))
        
        results.append({
            'n_factors': n_factors,
            'avg_val_loss': avg_val_loss,
            'rmse': avg_rmse
        })
    
    # Plot n_factors vs RMSE
    plt.figure(figsize=(10, 6))
    factors, rmses = zip(*nfactors_rmse)
    plt.plot(factors, rmses, marker='o', linestyle='-', color='blue')
    plt.title('n_factors vs RMSE')
    plt.xlabel('Number of Factors')
    plt.ylabel('RMSE')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for x, y in zip(factors, rmses):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Highlight the best point
    best_idx = rmses.index(min(rmses))
    best_factor = factors[best_idx]
    best_rmse = rmses[best_idx]
    plt.scatter(best_factor, best_rmse, color='red', s=100, zorder=5)
    plt.annotate(f"Best: {best_factor}, {best_rmse:.4f}", 
                 (best_factor, best_rmse), 
                 textcoords="offset points", 
                 xytext=(0, -20), 
                 ha='center',
                 fontweight='bold')
    
    # Save the plot
    plt.savefig("plots/n_factors_vs_rmse.png", dpi=300, bbox_inches='tight')
    print("Plot saved as plots/n_factors_vs_rmse.png")
    plt.close()
    
    # Find best parameters
    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['avg_val_loss'].idxmin()]
    
    print("\n=== Parameter Search Results ===")
    print(results_df.sort_values('avg_val_loss'))
    
    return best_params

def train_and_evaluate_final_model(train_dataset, test_dataset, best_params, batch_size=64):
    n_factors = int(best_params['n_factors'])
    
    print(f"\n=== Training final model with best parameter: n_factors={n_factors} ===")
    
    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Create and train model
    start_time = time.time()
    final_model = MatrixFactorization(len(train_dataset.users), len(train_dataset.items), n_factors=n_factors)
    _, train_losses, _ = final_model.train_model(train_loader, n_epochs=50)  # Train longer for final model
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss = final_model.evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', markersize=3)
    plt.title(f'Training Loss for Final Model (n_factors={n_factors})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("plots/final_model_training_loss.png", dpi=300, bbox_inches='tight')
    print("Training loss plot saved as plots/final_model_training_loss.png")
    plt.close()
    
    print(f"Training completed in {training_time:.2f}s")
    print(f"Final test loss (MSE): {test_loss:.4f}")
    print(f"Final test loss (RMSE): {test_rmse:.4f}")
    
    return final_model, test_loss, test_rmse

def calculate_top_n_recommendations(model, dataset, user_id, n=10):
    model.eval()
    
    if user_id not in dataset.user_to_index:
        raise ValueError(f"User ID {user_id} not found in dataset.")
    
    user_idx = dataset.user_to_index[user_id]
    
    rated_items = set(dataset.ratings[dataset.ratings['user_id'] == user_id]['item_id'])
    all_items = set(dataset.items)
    unrated_items = list(all_items - rated_items)
    
    unrated_indices = [dataset.item_to_index[item] for item in unrated_items if item in dataset.item_to_index]
    
    if not unrated_indices:
        return []
    
    user_tensor = torch.tensor([user_idx] * len(unrated_indices), dtype=torch.long).to(device)
    item_tensor = torch.tensor(unrated_indices, dtype=torch.long).to(device)
    
    with torch.no_grad():
        predicted_ratings = model(user_tensor, item_tensor).cpu().numpy()
    
    predictions = list(zip(unrated_items, predicted_ratings))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:n]

if __name__ == "__main__":
    total_start_time = time.time()
    
    base_dataset = MovieLensDataset(ratings_base)
    test_dataset = MovieLensDataset(ratings_test)
    
    n_factors_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_size = 64  
    
    # Perform K-fold cross-validation to find best parameters
    print("\nStarting K-fold cross-validation parameter search...")
    best_params = k_fold_cv_parameter_search(
        base_dataset, 
        n_factors_list=n_factors_list,
        batch_size=batch_size,
        n_folds=3,  # Using 3 folds for faster execution
        n_epochs=5  # Fewer epochs during parameter search
    )
    
    print("\n=== Best Parameters ===")
    print(f"n_factors: {best_params['n_factors']}")
    print(f"validation loss (MSE): {best_params['avg_val_loss']:.4f}")
    print(f"validation loss (RMSE): {best_params['rmse']:.4f}")
    
    # Train final model with best parameters and evaluate on test set
    final_model, test_loss, test_rmse = train_and_evaluate_final_model(
        base_dataset, test_dataset, best_params, batch_size=batch_size
    )
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    
    print("\n=== Final Results ===")
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # print("\n=== Sample Recommendations ===")
    # sample_users = [1, 100, 200]
    # for user_id in sample_users:
    #     try:
    #         top_recommendations = calculate_top_n_recommendations(final_model, base_dataset, user_id, n=5)
    #         print(f"\nTop 5 recommendations for User {user_id}:")
    #         for item_id, pred_rating in top_recommendations:
    #             movie_title = items[items['movie id'] == item_id]['movie title'].values[0]
    #             print(f"Movie: {movie_title} (ID: {item_id}), Predicted Rating: {pred_rating:.2f}")
    #     except Exception as e:
    #         print(f"Could not generate recommendations for User {user_id}: {str(e)}")