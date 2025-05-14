# Movie Recommendation System

This project implements various recommendation algorithms using both MovieLens datasets (ml-100k and ml-1m). The system includes collaborative filtering (user-based and item-based), matrix factorization, SVD, and hybrid approaches.

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
├── userKNN.py
├── itemKNN.py
├── matrixFactorization.py
├── svd.py
├── hybrid.py
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