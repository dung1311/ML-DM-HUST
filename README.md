# Movie Recommendation System

This project implements various recommendation algorithms using both MovieLens datasets (ml-100k and ml-1m). The system includes collaborative filtering (user-based and item-based), matrix factorization, SVD, hybrid approaches, and deep learning models.

## Datasets

### MovieLens 100K Dataset
The MovieLens 100K dataset contains:
- 100,000 ratings (1-5)
- 943 users
- 1,682 movies
- User demographics (age, gender, occupation, zip code)
- Movie genres (19 categories)
- Timestamps for ratings

File structure:
```
data/ml-100k/
├── u.user     # User information
├── u.item     # Movie information
├── ua.base    # Training set
└── ua.test    # Test set
```

### MovieLens 1M Dataset
The MovieLens 1M dataset contains:
- 1,000,000 ratings (1-5)
- 6,000 users
- 4,000 movies
- User demographics (age, gender, occupation, zip code)
- Movie genres
- Timestamps for ratings

File structure:
```
data/ml-1m/
├── users.dat     # User information
├── ratings.dat   # Ratings data
└── movies.dat    # Movie information
```

## Requirements

```bash
pip install -r requirements.txt
```

## Files and Usage

### 1. userKNN.py - User-based Collaborative Filtering

This file implements user-based collaborative filtering using KNN.

```python
# Run the file
python userKNN.py

# The file will:
# 1. Load the ml-1m dataset (or ml-100k if specified)
# 2. Split data into train/test (80/20)
# 3. Train the model with k=45
# 4. Calculate RMSE on test set
# 5. Display results

# To use ml-100k dataset:
# Modify the data loading section to use ml-100k paths and separators
```

### 2. itemKNN.py - Item-based Collaborative Filtering

This file implements item-based collaborative filtering using KNN.

```python
# Run the file
python itemKNN.py

# The file will:
# 1. Load the ml-1m dataset (or ml-100k if specified)
# 2. Split data into train/test (80/20)
# 3. Train the model with k=45
# 4. Calculate RMSE on test set
# 5. Display results

# To use ml-100k dataset:
# Modify the data loading section to use ml-100k paths and separators
```

### 3. matrixFactorization.py - Matrix Factorization

This file implements matrix factorization for collaborative filtering.

```python
# Run the file
python matrixFactorization.py

# The file will:
# 1. Load the ml-1m dataset (or ml-100k if specified)
# 2. Split data into train/test (80/20)
# 3. Train model with n_factors=5
# 4. Calculate RMSE on test set
# 5. Generate recommendations for sample users

# Key functions:
- MatrixFactorization.train_model(): Train the model
- MatrixFactorization.evaluate(): Evaluate model performance
- calculate_top_n_recommendations(): Get top N recommendations for a user

# To use ml-100k dataset:
# Modify the data loading section to use ml-100k paths and separators
```

### 4. svd.py - Singular Value Decomposition

This file implements SVD-based matrix factorization.

```python
# Run the file
python svd.py

# The file will:
# 1. Load the ml-1m dataset (or ml-100k if specified)
# 2. Split data into train/test (80/20)
# 3. Train model with n_factors=10
# 4. Calculate RMSE on test set
# 5. Provide interactive movie recommendations

# Key functions:
- MatrixFactorization.initialize_with_svd(): Initialize model weights using SVD
- MatrixFactorization.train_model(): Train the model
- MatrixFactorization.evaluate(): Evaluate model performance
- recommend_movies(): Get movie recommendations for a user

# Interactive Usage:
1. Run the program
2. After training, enter a user ID to get recommendations
3. Enter 'q' to quit

# To use ml-100k dataset:
# Modify the data loading section to use ml-100k paths and separators
```

### 5. hybrid.py - Hybrid Recommender

This file implements a hybrid recommendation system combining user-based and item-based approaches.

```python
# Run the file
python hybrid.py

# The file will:
# 1. Load the ml-1m dataset (or ml-100k if specified)
# 2. Split data into train/test (80/20)
# 3. Train model with alpha=0.7
# 4. Calculate RMSE on test set
# 5. Display results

# Key functions:
- HybridRecommender.train(): Train the hybrid model
- HybridRecommender.evaluate(): Evaluate model performance
- HybridRecommender.recommend(): Get recommendations for a user

# To use ml-100k dataset:
# Modify the data loading section to use ml-100k paths and separators
```

### 6. Generalized Matrix Factorization (GMF_model.py)

This implementation uses a generalized matrix factorization model for collaborative filtering.

#### Running the Model
```python
python GMF_model.py
```

#### Model Parameters
- `embed_dim`: Dimension of user and item embeddings (default=16)
- `learning_rate`: Learning rate for optimization (default=0.001)
- `weight_decay`: L2 regularization parameter (default=1e-4)
- `batch_size`: Batch size for training (default=128)
- `epochs`: Number of training epochs (default=100)

#### Evaluation
The model is evaluated using:
- RMSE (Root Mean Square Error)
- Training and validation loss curves

#### Validation
- Uses k-fold cross-validation to find optimal embedding dimension
- Implements early stopping to prevent overfitting
- Saves training/validation loss plots in 'plots' directory

### 7. Multi-Layer Perceptron (MLP_model.py)

This implementation uses a deep neural network for collaborative filtering.

#### Running the Model
```python
python MLP_model.py
```

#### Model Parameters
- `embedding_dim`: Dimension of user and item embeddings (default=64)
- `hidden_dims`: List of hidden layer dimensions (default=[128,64,32])
- `learning_rate`: Learning rate for optimization (default=0.001)
- `batch_size`: Batch size for training (default=1024)
- `epochs`: Number of training epochs (default=20)

#### Evaluation
The model is evaluated using:
- Binary Cross Entropy Loss
- Training and validation loss curves

#### Validation
- Uses k-fold cross-validation to find optimal architecture
- Implements early stopping to prevent overfitting
- Saves training/validation loss plots in 'plots' directory

### 8. Neural Collaborative Filtering (NeuMF_model.py)

This implementation combines GMF and MLP approaches using a neural network architecture.

#### Running the Model
```python
python NeuMF_model.py
```

#### Model Parameters
- `num_factors`: Number of latent factors (default=8)
- `nums_hiddens`: List of hidden layer dimensions (default=[128,64])
- `learning_rate`: Learning rate for optimization (default=0.0001)
- `weight_decay`: L2 regularization parameter (default=1e-3)
- `batch_size`: Batch size for training (default=1024)
- `epochs`: Number of training epochs (default=50)

#### Evaluation
The model is evaluated using:
- RMSE (Root Mean Square Error)
- Training and validation loss curves

#### Validation
- Uses k-fold cross-validation to find optimal architecture
- Implements early stopping to prevent overfitting
- Saves training/validation loss plots in 'plots' directory

### 9. Convolutional Sequence Model (caser_rating_rmse.py)

This implementation uses a convolutional neural network to capture sequential patterns in user-item interactions.

#### Running the Model
```python
python caser_rating_rmse.py
```

#### Model Parameters
- `num_factors`: Number of latent factors (default=10)
- `L`: Sequence length (default=5)
- `d`: Number of horizontal filters (default=16)
- `d_prime`: Number of vertical filters (default=4)
- `drop_ratio`: Dropout ratio (default=0.05)
- `learning_rate`: Learning rate for optimization (default=0.001)
- `batch_size`: Batch size for training (default=512)
- `epochs`: Number of training epochs (default=20)

#### Evaluation
The model is evaluated using:
- RMSE (Root Mean Square Error)
- Training and validation loss curves

#### Validation
- Uses k-fold cross-validation to find optimal architecture
- Implements early stopping to prevent overfitting
- Saves training/validation loss plots in 'plots' directory

## Common Functions Across Files

### Data Loading for ml-1m
```python
# Load users
users = pd.read_csv("./data/ml-1m/users.dat", sep="::", header=None, 
                   names=["user_id", "gender", "age", "occupation", "zip_code"],
                   engine='python', encoding='latin-1')

# Load ratings
ratings = pd.read_csv("./data/ml-1m/ratings.dat", sep="::", header=None,
                     names=["user_id", "item_id", "rating", "timestamp"],
                     engine='python', encoding='latin-1')

# Load movies
items = pd.read_csv("./data/ml-1m/movies.dat", sep="::", header=None,
                   names=["movie id", "movie title", "genres"],
                   engine='python', encoding='latin-1')
```

### Data Loading for ml-100k
```python
# Load users
users = pd.read_csv("./data/ml-100k/u.user", sep="|", header=None,
                   names=["user_id", "age", "gender", "occupation", "zip_code"],
                   encoding='latin-1')

# Load ratings
ratings_base = pd.read_csv("./data/ml-100k/ua.base", sep="\t", header=None,
                          names=["user_id", "item_id", "rating", "timestamp"],
                          encoding='latin-1')
ratings_test = pd.read_csv("./data/ml-100k/ua.test", sep="\t", header=None,
                          names=["user_id", "item_id", "rating", "timestamp"],
                          encoding='latin-1')

# Load movies
items = pd.read_csv("./data/ml-100k/u.item", sep="|", header=None,
                   names=['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
                         'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                   encoding='latin-1')
```

### Data Splitting
```python
from sklearn.model_selection import train_test_split
ratings_base, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
```

### Model Evaluation
```python
# Calculate RMSE
test_loss = model.evaluate(test_loader)
test_rmse = np.sqrt(test_loss)
print(f"Test RMSE: {test_rmse:.4f}")
```

### Getting Recommendations
```python
# Get top N recommendations for a user
recommendations = model.recommend(user_id, n=5)
for movie_title, pred_rating in recommendations:
    print(f"Movie: {movie_title}, Predicted Rating: {pred_rating:.2f}")
```

## Notes

1. Models can use either ml-100k or ml-1m dataset
2. GPU acceleration is used if available
3. Results are saved in the 'plots' directory
4. Each model can be tuned by adjusting its parameters:
   - userKNN.py: k (number of neighbors)
   - itemKNN.py: k (number of neighbors)
   - matrixFactorization.py: n_factors
   - svd.py: n_factors
   - hybrid.py: alpha (weight between user and item-based approaches)

## Directory Structure

```
.
├── data/
│   ├── ml-100k/
│   │   ├── u.user
│   │   ├── u.item
│   │   ├── ua.base
│   │   └── ua.test
│   └── ml-1m/
│       ├── users.dat
│       ├── ratings.dat
│       └── movies.dat
├── plots/
│   ├── userKNN_rmse.png
│   ├── itemKNN_rmse.png
│   ├── matrixFactorization_loss.png
│   ├── svd_loss.png
│   ├── hybrid_rmse.png
│   ├── gmf_validation.png
│   ├── mlp_validation.png
│   ├── neumf_validation.png
│   └── caser_validation.png
├── userKNN.py
├── itemKNN.py
├── matrixFactorization.py
├── svd.py
├── hybrid.py
├── GMF_model.py
├── valid_gmf_model.py
├── MLP_model.py
├── valid_mlp_model.py
├── NeuMF_model.py
├── valid_NeuMF_model.py
├── caser_rating_rmse.py
├── caser_hyperparam_search.py
├── requirements.txt
└── README.md
```

## Model Validation and Evaluation

### 1. userKNN.py Validation
```python
# Cross-validation with k-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Validate different k values
k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
for k in k_values:
    model = UserColaborativeFiltering(ratings_base, k=k)
    rmse = model.evaluate_model(ratings_test)
    print(f"k={k}, RMSE={rmse:.4f}")

# Plot validation results
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o')
plt.title('RMSE vs k for User-based CF')
plt.xlabel('k (number of neighbors)')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('plots/user_knn_validation.png')
```

### 2. itemKNN.py Validation
```python
# Cross-validation with k-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Validate different k values
k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
for k in k_values:
    model = ItemColaborativeFiltering(ratings_base, k=k)
    rmse = model.evaluate_model(ratings_test)
    print(f"k={k}, RMSE={rmse:.4f}")

# Plot validation results
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o')
plt.title('RMSE vs k for Item-based CF')
plt.xlabel('k (number of neighbors)')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('plots/item_knn_validation.png')
```

### 3. matrixFactorization.py Validation
```python
# Cross-validation with different n_factors
n_factors_list = [5, 10, 15, 20, 25, 30, 40, 50]
best_params = k_fold_cv_parameter_search(
    base_dataset, 
    n_factors_list=n_factors_list,
    batch_size=64,
    n_folds=5,
    n_epochs=10
)

# Plot validation results
plt.figure(figsize=(10, 6))
plt.plot(n_factors_list, rmse_values, marker='o')
plt.title('RMSE vs n_factors for Matrix Factorization')
plt.xlabel('Number of Factors')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('plots/mf_validation.png')

# Final model evaluation
final_model, test_loss, test_rmse = train_and_evaluate_final_model(
    base_dataset, test_dataset, best_params
)
```

### 4. svd.py Validation
```python
# Cross-validation with different n_factors
n_factors_list = [5, 10, 15, 20, 25, 30, 40, 50]
best_params = k_fold_cv_parameter_search(
    base_dataset, 
    n_factors_list=n_factors_list,
    batch_size=64,
    n_folds=5,
    n_epochs=10
)

# Plot validation results
plt.figure(figsize=(10, 6))
plt.plot(n_factors_list, rmse_values, marker='o')
plt.title('RMSE vs n_factors for SVD')
plt.xlabel('Number of Factors')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('plots/svd_validation.png')

# Final model evaluation with SVD initialization
final_model = MatrixFactorization(
    len(base_dataset.users), 
    len(base_dataset.items), 
    n_factors=best_params['n_factors'],
    dataset=base_dataset
)
```

### 5. hybrid.py Validation
```python
# Cross-validation with different alpha values
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for alpha in alpha_values:
    model = HybridRecommender(ratings_base, alpha=alpha)
    rmse = model.evaluate_model(ratings_test)
    print(f"alpha={alpha}, RMSE={rmse:.4f}")

# Plot validation results
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, rmse_values, marker='o')
plt.title('RMSE vs alpha for Hybrid Recommender')
plt.xlabel('Alpha (weight)')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('plots/hybrid_validation.png')
```

## Common Validation Metrics

### RMSE (Root Mean Square Error)
```python
def calculate_rmse(predictions, actual):
    return np.sqrt(np.mean((predictions - actual) ** 2))
```

### MAE (Mean Absolute Error)
```python
def calculate_mae(predictions, actual):
    return np.mean(np.abs(predictions - actual))
```

## Validation Best Practices

1. Always use cross-validation to avoid overfitting
2. Use multiple metrics (RMSE, MAE)
3. Plot validation results to visualize model performance
4. Save validation plots for comparison
5. Use early stopping when training deep learning models
6. Compare results across different parameter values
7. Use the same random seed for reproducibility

## Model Validation and Tuning

### Cross-Validation
All models use k-fold cross-validation (default k=5) to:
1. Find optimal hyperparameters
2. Prevent overfitting
3. Get more reliable performance estimates

### Hyperparameter Tuning
Each model has specific hyperparameters that can be tuned:

1. User-based CF:
   - `k`: Number of neighbors (try values: 5, 10, 20, 30, 40, 50)
   - `similarity_type`: 'pearson' or 'cosine'

2. Item-based CF:
   - `k`: Number of neighbors (try values: 5, 10, 20, 30, 40, 50)
   - `similarity_type`: 'pearson' or 'cosine'

3. Matrix Factorization:
   - `n_factors`: Number of latent factors (try values: 5, 10, 20, 30, 40)
   - `learning_rate`: Try values: 0.001, 0.01, 0.1
   - `weight_decay`: Try values: 0.0001, 0.001, 0.01

4. SVD:
   - `n_factors`: Number of latent factors (try values: 5, 10, 20, 30, 40)
   - `learning_rate`: Try values: 0.001, 0.01, 0.1
   - `weight_decay`: Try values: 0.0001, 0.001, 0.01

5. Hybrid:
   - `alpha`: Weight for user-based predictions (try values: 0.1 to 0.9 in steps of 0.1)
   - `k`: Number of neighbors (try values: 5, 10, 20, 30, 40, 50)

6. GMF:
   - `embedding_dim`: Try values: 4, 8, 16, 32
   - `learning_rate`: Try values: 1e-4, 5e-4, 1e-3, 5e-3
   - `batch_size`: Try values: 32, 64, 128
   - `weight_decay`: Try values: 0, 1e-5, 1e-4, 1e-3
   - `epochs`: Try values: 20, 30, 50

7. MLP:
   - `embedding_dim`: Try values: 4, 8, 16, 32
   - `hidden_units`: Try combinations: [64,32], [128,64], [64,32,16], [128,64,32]
   - `learning_rate`: Try values: 1e-4, 5e-4, 1e-3, 5e-3
   - `batch_size`: Try values: 64, 128, 256
   - `weight_decay`: Try values: 0, 1e-5, 1e-4, 1e-3
   - `dropout`: Try values: 0.0, 0.2, 0.4
   - `epochs`: Try values: 20, 50, 100

8. NeuMF:
   - `num_factors`: Try values: 8, 16, 32, 64
   - `nums_hiddens`: Try combinations: [64,32], [128,64], [64,32,16], [128,64,32], [256,128,64,32]
   - `learning_rate`: Try values: 1e-4, 5e-4, 1e-3, 5e-3
   - `weight_decay`: Try values: 0, 1e-5, 1e-4, 1e-3
   - `epochs`: Try values: 20, 50, 100

9. Caser:
   - `num_factors`: Try values: 8, 16, 32, 64
   - `L`: Try values: 3, 5, 7
   - `d`: Try values: 8, 16, 32
   - `d_prime`: Try values: 4, 8, 16
   - `drop_ratio`: Try values: 0.1, 0.2, 0.3, 0.5
   - `learning_rate`: Try values: 0.001, 0.005, 0.01
   - `batch_size`: Try values: 164, 128, 256
   - `weight_decay`: Try values: 0, 1e-5, 1e-4, 1e-3
   - `epochs`: Try values: 20, 50, 100

### Early Stopping
All deep learning models implement early stopping to prevent overfitting:
- Monitors validation loss
- Stops training if validation loss doesn't improve for specified number of epochs
- Saves best model state

### Performance Visualization
All models generate plots to visualize performance:
- RMSE vs hyperparameter values
- Training and validation loss curves
- Plots are saved in the 'results' directory

## Notes

1. Models can use either ml-100k or ml-1m dataset
2. GPU acceleration is used if available (PyTorch models)
3. Results are saved in the 'results' directory
4. Each model implements caching for similarity calculations to improve performance
5. The hybrid model combines the strengths of user-based and item-based approaches
6. Deep learning models (GMF, MLP, NeuMF, Caser) require PyTorch
7. All models support both explicit and implicit feedback

## Directory Structure

```
.
├── data/
│   ├── ml-100k/
│   │   ├── u.user
│   │   ├── u.item
│   │   ├── ua.base
│   │   └── ua.test
│   └── ml-1m/
│       ├── users.dat
│       ├── ratings.dat
│       └── movies.dat
├── plots/
│   ├── userKNN_rmse.png
│   ├── itemKNN_rmse.png
│   ├── matrixFactorization_loss.png
│   ├── svd_loss.png
│   ├── hybrid_rmse.png
│   ├── gmf_validation.png
│   ├── mlp_validation.png
│   ├── neumf_validation.png
│   └── caser_validation.png
├── userKNN.py
├── itemKNN.py
├── matrixFactorization.py
├── svd.py
├── hybrid.py
├── GMF_model.py
├── valid_gmf_model.py
├── MLP_model.py
├── valid_mlp_model.py
├── NeuMF_model.py
├── valid_NeuMF_model.py
├── caser_rating_rmse.py
├── caser_hyperparam_search.py
├── requirements.txt
└── README.md
```
