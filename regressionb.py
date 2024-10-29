import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Box plot comparing model errors
def plot_model_comparison(results_df):
    plt.figure(figsize=(10, 6))
    error_data = pd.melt(results_df[:-2],  # Exclude Mean and Std rows
                        value_vars=['ridge_error', 'ann_error', 'baseline_error'],
                        var_name='Model', value_name='Error')
    error_data['Model'] = error_data['Model'].map({
        'ridge_error': 'Ridge',
        'ann_error': 'Neural Network',
        'baseline_error': 'Baseline'
    })
    sns.boxplot(x='Model', y='Error', data=error_data)
    plt.title('Model Performance Comparison')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# Line plot showing errors across folds
def plot_error_trends(results_df):
    plt.figure(figsize=(12, 6))
    folds = results_df['fold'][:-2]  # Exclude Mean and Std rows
    plt.plot(folds, results_df['ridge_error'][:-2], 'o-', label='Ridge')
    plt.plot(folds, results_df['ann_error'][:-2], 's-', label='Neural Network')
    plt.plot(folds, results_df['baseline_error'][:-2], '^-', label='Baseline')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Trends Across Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# Hyperparameter visualization
def plot_hyperparameters(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    folds = results_df['fold'][:-2]
    # Ridge lambda values
    ax1.plot(folds, results_df['ridge_lambda'][:-2], 'o-')
    ax1.set_ylabel('Lambda Value')
    ax1.set_xlabel('Fold')
    ax1.set_title('Ridge Regression Lambda Values')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # ANN hidden units
    ax2.plot(folds, results_df['ann_hidden_units'][:-2], 'o-')
    ax2.set_ylabel('Number of Hidden Units')
    ax2.set_xlabel('Fold')
    ax2.set_title('Neural Network Hidden Units')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

# Distribution of errors
def plot_error_distributions(results_df):
    plt.figure(figsize=(10, 6))
    errors = results_df[['ridge_error', 'ann_error', 'baseline_error']].iloc[:-2]
    
    sns.kdeplot(data=errors, fill=True, alpha=0.5)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Density')
    plt.title('Distribution of Errors Across Folds')
    plt.legend(labels=['Ridge', 'Neural Network', 'Baseline'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


warnings.filterwarnings('ignore')

def generatePCA(X,Y):
    #X_np = X.to_numpy()
    #y_np = Y.to_numpy()
    N, M = X.shape
    pca = PCA(n_components=M)
    X_pca = pca.fit_transform(X)
    #n_components_range = range(2, M+1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance > 0.85)[0][0] + 1
    X_pca_reduced = X_pca[:, :n_components]
    return X_pca_reduced, n_components, cumulative_variance[n_components]



class MeanPredictor:
    def __init__(self):
        self.mean = None
        
    def fit(self, X, y):
        self.mean = np.mean(y)
        return self
        
    def predict(self, X):
        return np.full(len(X), self.mean)

def calculate_per_observation_error(y_true, y_pred):
    """Calculate squared loss per observation."""
    return np.mean((y_true - y_pred) ** 2)

def evaluate_models_with_shared_splits(X, y, K1=10, K2=10):
    """
    Evaluate all models using shared train/test splits.
    Returns errors and optimal parameters for each fold.
    """
    # Parameters for models
    lambda_values = np.logspace(-4, 4, 20)
    hidden_units = [1, 2, 4, 8, 12, 32, 64, 128, 256]
    
    # Initialize storage for results
    results = {
        'fold': [],
        'ridge_lambda': [],
        'ann_hidden_units': [],
        'ridge_error': [],
        'ann_error': [],
        'baseline_error': []
    }
    
    # Create outer fold splits
    outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    # Iterate through outer folds
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_train_outer, X_test = X[train_idx], X[test_idx]
        y_train_outer, y_test = y[train_idx], y[test_idx]
        print(f"outer fold {fold_idx}")
        # Store best parameters and errors for each model
        best_params = {
            'ridge': {'error': float('inf'), 'lambda': None},
            'ann': {'error': float('inf'), 'hidden_units': None}
        }
        
        # Inner cross-validation for parameter selection
        for model_type in ['ridge', 'ann']:
            param_errors = {}
            
            # Try each parameter value
            param_values = lambda_values if model_type == 'ridge' else hidden_units
            
            for param in param_values:
                fold_errors = []
                
                # Inner cross-validation
                for train_inner_idx, val_idx in inner_cv.split(X_train_outer):
                    X_train = X_train_outer[train_inner_idx]
                    X_val = X_train_outer[val_idx]
                    y_train = y_train_outer[train_inner_idx]
                    y_val = y_train_outer[val_idx]
                    
                    if model_type == 'ridge':
                        model = Ridge(alpha=param)
                    else:  # ANN
                        model = MLPRegressor(
                            hidden_layer_sizes=(param,),
                            activation='relu',
                            solver='adam',
                            learning_rate='adaptive',
                            learning_rate_init=0.01,
                            max_iter=2000,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=10,
                            random_state=42,
                            tol=1e-4
                        )
                    
                    model.fit(X_train, y_train)
                    y_pred_val = model.predict(X_val)
                    error = calculate_per_observation_error(y_val, y_pred_val)
                    fold_errors.append(error)
                
                # Average error across inner folds
                mean_error = np.mean(fold_errors)
                param_errors[param] = mean_error
                
                # Update best parameters if better
                if mean_error < best_params[model_type]['error']:
                    best_params[model_type]['error'] = mean_error
                    if model_type == 'ridge':
                        best_params[model_type]['lambda'] = param
                    else:
                        best_params[model_type]['hidden_units'] = param
        
        # Train final models with best parameters on full training set
        ridge_final = Ridge(alpha=best_params['ridge']['lambda'])
        ann_final = MLPRegressor(
            hidden_layer_sizes=(best_params['ann']['hidden_units'],),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.01,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            tol=1e-4
        )
        baseline_final = MeanPredictor()
        
        # Fit and evaluate all models
        ridge_final.fit(X_train_outer, y_train_outer)
        ann_final.fit(X_train_outer, y_train_outer)
        baseline_final.fit(X_train_outer, y_train_outer)
        
        # Calculate test errors
        ridge_pred = ridge_final.predict(X_test)
        ann_pred = ann_final.predict(X_test)
        baseline_pred = baseline_final.predict(X_test)
        
        ridge_error = calculate_per_observation_error(y_test, ridge_pred)
        ann_error = calculate_per_observation_error(y_test, ann_pred)
        baseline_error = calculate_per_observation_error(y_test, baseline_pred)
        
        # Store results
        results['fold'].append(fold_idx)
        results['ridge_lambda'].append(best_params['ridge']['lambda'])
        results['ann_hidden_units'].append(best_params['ann']['hidden_units'])
        results['ridge_error'].append(ridge_error)
        results['ann_error'].append(ann_error)
        results['baseline_error'].append(baseline_error)
    
    # Create results table
    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    
    # Add mean and std at the bottom
    means = results_df.mean().round(4)
    stds = results_df.std().round(4)
    
    means['fold'] = 'Mean'
    stds['fold'] = 'Std'
    
    results_df = pd.concat([results_df, 
                           pd.DataFrame([means]), 
                           pd.DataFrame([stds])])
    
    return results_df

# Load and prepare data
file_path = 'heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(file_path)

# Standardize features
scaler = StandardScaler()
df_centered = df - df.mean()
df_standardized = pd.DataFrame(scaler.fit_transform(df_centered), 
                             columns=df_centered.columns)
#df_standardized = df
# Prepare features and target
X = df_standardized.drop(['DEATH_EVENT', 'platelets'], axis=1).to_numpy()
y = df_standardized['platelets'].to_numpy()


#Calculate PCA
num_components = 0
X, num_components, explain = generatePCA(X,y)

# Generate results table
results_table = evaluate_models_with_shared_splits(X, y)

# Print formatted table
print("\nTwo-Level Cross-Validation Results:")
print("\nTable 1: Model Comparison across 10 Folds")
print("=" * 80)
print(results_table.to_string(index=False))
print("=" * 80)
print(f"Amount of PCA components: {num_components}, explaining {explain*100}% of the dataset")



plot_model_comparison(results_table)
plot_error_trends(results_table)
plot_hyperparameters(results_table)
plot_error_distributions(results_table)
plt.show()