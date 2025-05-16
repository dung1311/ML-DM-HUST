import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
import os

# Check and use GPU if available
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

# Create plots directory if it doesn't exist
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

# Matrix Factorization Model with SVD Initialization
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, n_factors=10, dataset=None):
        super(MatrixFactorization, self).__init__()
        self.user_factors = torch.nn.Embedding(num_users, n_factors)
        self.item_factors = torch.nn.Embedding(num_items, n_factors)
        self.user_bias = torch.nn.Embedding(num_users, 1)
        self.item_bias = torch.nn.Embedding(num_items, 1)
        
        # Define global_bias before potentially calling initialize_with_svd
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        
        if dataset is not None:
            self.initialize_with_svd(dataset, n_factors)
        else:
            # Initialize weights randomly
            torch.nn.init.normal_(self.user_factors.weight, 0, 0.01)
            torch.nn.init.normal_(self.item_factors.weight, 0, 0.01)
            torch.nn.init.zeros_(self.user_bias.weight)
            torch.nn.init.zeros_(self.item_bias.weight)
        
        # Move model to GPU if available
        self.to(device)

    def initialize_with_svd(self, dataset, k):
        print("Initializing model with SVD")
        ratings = dataset.ratings
        μ = ratings['rating'].mean()
        
        # Compute user biases
        user_means = ratings.groupby('user_id')['rating'].mean()
        b_u = user_means - μ
        
        # Compute item biases
        ratings['b_u'] = ratings['user_id'].map(b_u)
        ratings['residual'] = ratings['rating'] - μ - ratings['b_u']
        item_means = ratings.groupby('item_id')['residual'].mean()
        b_i = item_means
        
        # Construct residual matrix R_prime
        num_users = len(dataset.users)
        num_items = len(dataset.items)
        R_prime = np.zeros((num_users, num_items))
        
        user_indices = ratings['user_id'].map(dataset.user_to_index).values
        item_indices = ratings['item_id'].map(dataset.item_to_index).values
        bu_array = ratings['b_u'].values
        bi_array = ratings['item_id'].map(b_i).values
        residuals = ratings['rating'].values - μ - bu_array - bi_array
        R_prime[user_indices, item_indices] = residuals
        
        # Compute SVD using numpy
        U, S, Vh = np.linalg.svd(R_prime, full_matrices=False)
        S_k = S[:k]
        sqrt_S_k = np.sqrt(S_k)
        P = U[:, :k] @ np.diag(sqrt_S_k)
        Q = Vh.T[:, :k] @ np.diag(sqrt_S_k)
        
        # Set factors
        self.user_factors.weight.data = torch.from_numpy(P).float().to(device)
        self.item_factors.weight.data = torch.from_numpy(Q).float().to(device)
        
        # Set biases
        b_u_values = [b_u[user] for user in dataset.users]
        b_i_values = [b_i[item] for item in dataset.items]
        self.user_bias.weight.data = torch.tensor(b_u_values).view(-1, 1).float().to(device)
        self.item_bias.weight.data = torch.tensor(b_i_values).view(-1, 1).float().to(device)
        self.global_bias.data = torch.tensor([μ]).float().to(device)

    def forward(self, user, item):
        user = user.to(device)
        item = item.to(device)
        
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        user_bias = self.user_bias(user).squeeze(-1)
        item_bias = self.item_bias(item).squeeze(-1)

        if user_embedding.dim() > 1:
            rating = torch.sum(user_embedding * item_embedding, dim=1) + user_bias + item_bias + self.global_bias
        else:
            rating = torch.sum(user_embedding * item_embedding) + user_bias + item_bias + self.global_bias
            
        return rating

    def train_model(self, train_loader, val_loader=None, n_epochs=10, lr=0.01, weight_decay=0.001, early_stopping=5):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            self.train()
            total_train_loss = 0
            train_samples = 0
            
            for user, item, rating in train_loader:
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
            
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion)
                val_losses.append(val_loss)
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {key: value.cpu() for key, value in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        if best_model_state is not None:
                            self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        break
            else:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
        
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
                user, item, rating = user.to(device), item.to(device), rating.to(device)
                prediction = self(user, item)
                loss = criterion(prediction, rating)
                total_loss += loss.item() * user.size(0)
                total_samples += user.size(0)
        
        return total_loss / total_samples

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
    
    nfactors_rmse = []
    
    for n_factors in n_factors_list:
        start_time = time.time()
        fold_losses = []
        
        print(f"\n=== Testing parameter: n_factors={n_factors} ===")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
            fold_start = time.time()
            print(f"Fold {fold+1}/{n_folds}")
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            model = MatrixFactorization(len(dataset.users), len(dataset.items), n_factors=n_factors, dataset=dataset)
            val_loss, _, _ = model.train_model(
                train_loader, val_loader, 
                n_epochs=n_epochs,
                weight_decay=weight_decay
            )
            
            fold_losses.append(val_loss)
            fold_time = time.time() - fold_start
            print(f"Fold {fold+1} completed in {fold_time:.2f}s with validation loss: {val_loss:.4f}")
        
        avg_val_loss = np.mean(fold_losses)
        avg_rmse = np.sqrt(avg_val_loss)
        total_time = time.time() - start_time
        print(f"Average validation loss: {avg_val_loss:.4f}, RMSE: {avg_rmse:.4f}")
        print(f"Total time for parameter: {total_time:.2f}s")
        
        nfactors_rmse.append((n_factors, avg_rmse))
        
        results.append({
            'n_factors': n_factors,
            'avg_val_loss': avg_val_loss,
            'rmse': avg_rmse
        })
    
    plt.figure(figsize=(10, 6))
    factors, rmses = zip(*nfactors_rmse)
    plt.plot(factors, rmses, marker='o', linestyle='-', color='blue')
    plt.title('n_factors vs RMSE')
    plt.xlabel('Number of Factors')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for x, y in zip(factors, rmses):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
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
    
    plt.savefig("plots/n_factors_vs_rmse.png", dpi=300, bbox_inches='tight')
    print("Plot saved as plots/n_factors_vs_rmse.png")
    plt.close()
    
    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['avg_val_loss'].idxmin()]
    
    print("\n=== Parameter Search Results ===")
    print(results_df.sort_values('avg_val_loss'))
    
    return best_params

def train_and_evaluate_final_model(train_dataset, test_dataset, best_params, batch_size=64):
    n_factors = int(best_params['n_factors'])
    
    print(f"\n=== Training final model with best parameter: n_factors={n_factors} ===")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    start_time = time.time()
    final_model = MatrixFactorization(len(train_dataset.users), len(train_dataset.items), n_factors=n_factors, dataset=train_dataset)
    _, train_losses, _ = final_model.train_model(train_loader, n_epochs=50)
    training_time = time.time() - start_time
    
    test_loss = final_model.evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)
    
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

def recommend_movies(model, dataset, items, user_id, n=10):
    """
    Recommend movies for a given user ID
    
    Args:
        model: Trained MatrixFactorization model
        dataset: MovieLensDataset instance
        items: DataFrame containing movie information
        user_id: ID of the user to get recommendations for
        n: Number of recommendations to return (default: 10)
    
    Returns:
        List of tuples (movie_title, predicted_rating)
    """
    try:
        # Get recommendations
        recommendations = calculate_top_n_recommendations(model, dataset, user_id, n=n)
        
        # Format recommendations with movie titles
        formatted_recommendations = []
        for item_id, pred_rating in recommendations:
            movie_title = items[items['movie id'] == item_id]['movie title'].values[0]
            formatted_recommendations.append((movie_title, pred_rating))
            
        return formatted_recommendations
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []

if __name__ == "__main__":
    total_start_time = time.time()
    
    # Comment out ml-100k code
    """
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
    """

    # Load ml-1m dataset
    print("Loading ml-1m dataset...")
    
    # Load users
    users = pd.read_csv("./data/ml-1m/users.dat", sep="::", header=None, 
                       names=["user_id", "gender", "age", "occupation", "zip_code"],
                       engine='python', encoding='latin-1')
    print("Number of users:", users.shape[0])

    # Load ratings
    ratings = pd.read_csv("./data/ml-1m/ratings.dat", sep="::", header=None,
                         names=["user_id", "item_id", "rating", "timestamp"],
                         engine='python', encoding='latin-1')
    print("Total number of ratings:", ratings.shape[0])

    # Load movies
    items = pd.read_csv("./data/ml-1m/movies.dat", sep="::", header=None,
                       names=["movie id", "movie title", "genres"],
                       engine='python', encoding='latin-1')
    print("Number of items:", items.shape[0])

    # Split ratings into train and test (80-20)
    from sklearn.model_selection import train_test_split
    ratings_base, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
    print("Number of ratings in base set:", ratings_base.shape[0])
    print("Number of ratings in test set:", ratings_test.shape[0])

    # Create datasets
    base_dataset = MovieLensDataset(ratings_base)
    test_dataset = MovieLensDataset(ratings_test)
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)
    train_loader = DataLoader(base_dataset, batch_size=64, shuffle=True, pin_memory=True)
    
    # Initialize and train model with n_factors=10
    print("\nInitializing and training model with n_factors=10...")
    mf = MatrixFactorization(len(base_dataset.users), len(base_dataset.items), n_factors=10, dataset=base_dataset)
    
    # Train model
    print("\nTraining model...")
    mf.train_model(train_loader=train_loader, n_epochs=100)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    test_loss = mf.evaluate(test_loader)
    test_rmse = np.sqrt(test_loss)
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Generate recommendations for some sample users
    print("\nGenerating movie recommendations for sample users...")
    sample_users = [1, 100, 200]  # Example user IDs
    for user_id in sample_users:
        try:
            top_recommendations = calculate_top_n_recommendations(mf, base_dataset, user_id, n=5)
            print(f"\nTop 5 recommendations for User {user_id}:")
            for item_id, pred_rating in top_recommendations:
                movie_title = items[items['movie id'] == item_id]['movie title'].values[0]
                print(f"Movie: {movie_title} (ID: {item_id}), Predicted Rating: {pred_rating:.2f}")
        except Exception as e:
            print(f"Could not generate recommendations for User {user_id}: {str(e)}")
    
    # After training and evaluating the model, add interactive recommendation
    print("\n=== Interactive Movie Recommendations ===")
    while True:
        try:
            user_input = input("\nEnter user ID to get recommendations (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
                
            user_id = int(user_input)
            recommendations = recommend_movies(mf, base_dataset, items, user_id, n=5)
            
            if recommendations:
                print(f"\nTop 5 movie recommendations for User {user_id}:")
                for i, (movie_title, pred_rating) in enumerate(recommendations, 1):
                    print(f"{i}. {movie_title} (Predicted Rating: {pred_rating:.2f})")
            else:
                print("No recommendations available for this user.")
                
        except ValueError:
            print("Please enter a valid user ID (number) or 'q' to quit.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    print("\nThank you for using the movie recommendation system!")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")