import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import scipy.stats as stats
import warnings
from statsmodels.stats.contingency_tables import mcnemar
warnings.filterwarnings("ignore")


# Dataset preparation
file_path = 'heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(file_path)

# Standardization
scaler = StandardScaler()
df_centered = df - df.mean()
df_standardized = pd.DataFrame(scaler.fit_transform(df_centered), columns=df_centered.columns)

# Prepare data for classification
X = df_standardized.drop(['DEATH_EVENT'], axis=1).to_numpy()
y = df['DEATH_EVENT'].to_numpy()  # Binary target

# Apply PCA and reduce dimensionality
pca = PCA(n_components=11)
X_pca = pca.fit_transform(X)

# Determine the number of principal components to retain (85% variance)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cumulative_variance > 0.85)[0][0] + 1
X_pca = X_pca[:, :n_components]

# Define parameters for cross-validation
C_values = np.logspace(-4, 4, 20)  # Complexity parameter for Logistic Regression
hidden_units = [1, 2, 4, 8, 16, 32, 64,128,256]  # Complexity parameter for ANN

# Two-level cross-validation results storage
results = {
    'fold': [],
    'logistic_C': [],
    'ann_hidden_units': [],
    'logistic_error': [],
    'ann_error': [],
    'baseline_error': []
}

# Outer K-Fold for final model evaluation
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_pca), 1):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Outer fold {fold_idx}")
    # Baseline model: predicts the majority class
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_error = 1 - accuracy_score(y_test, baseline_model.predict(X_test))  # Error rate

    # Logistic Regression with hyperparameter tuning
    best_logistic_score = 0
    best_C = None
    for C in C_values:
        logistic = LogisticRegression(C=C, max_iter=1000)
        cv_scores = cross_val_score(logistic, X_train, y_train, cv=inner_cv, scoring='accuracy')
        mean_score = cv_scores.mean()
        if mean_score > best_logistic_score:
            best_logistic_score = mean_score
            best_C = C

    # Train final Logistic Regression with best C
    logistic_final = LogisticRegression(C=best_C, max_iter=1000)
    logistic_final.fit(X_train, y_train)
    logistic_error = 1 - accuracy_score(y_test, logistic_final.predict(X_test))  # Error rate

    # ANN with hyperparameter tuning (number of hidden units)
    best_ann_score = 0
    best_hidden_units = None
    for hidden in hidden_units:
        ann = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=2000)
        cv_scores = cross_val_score(ann, X_train, y_train, cv=inner_cv, scoring='accuracy')
        mean_score = cv_scores.mean()
        if mean_score > best_ann_score:
            best_ann_score = mean_score
            best_hidden_units = hidden

    # Train final ANN with best hidden units
    ann_final = MLPClassifier(hidden_layer_sizes=(best_hidden_units,), activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.01,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            tol=1e-4)
    ann_final.fit(X_train, y_train)
    ann_error = 1 - accuracy_score(y_test, ann_final.predict(X_test))  # Error rate

    # Store results
    results['fold'].append(fold_idx)
    results['logistic_C'].append(best_C)
    results['ann_hidden_units'].append(best_hidden_units)
    results['logistic_error'].append(logistic_error)
    results['ann_error'].append(ann_error)
    results['baseline_error'].append(baseline_error)

# Create results DataFrame
results_df = pd.DataFrame(results)
mean_row = results_df.mean().to_dict()
mean_row['fold'] = 'Mean'
std_row = results_df.std().to_dict()
std_row['fold'] = 'Std'
#results_df = results_df.append([mean_row, std_row], ignore_index=True)
results_df = pd.concat([results_df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)
# Display table of results
print("\nTwo-Level Cross-Validation Results (Classification)")
print(results_df)

# Parameter choices
print("\nChosen Parameters:")
print(f"Logistic Regression: Complexity parameter (C) range = {C_values}")
print(f"ANN: Complexity parameter (hidden units) range = {hidden_units}")

# Plot error rates across folds
def plot_error_comparison(results_df):
    plt.figure(figsize=(10, 6))
    error_data = pd.melt(results_df[:-2], 
                         value_vars=['logistic_error', 'ann_error', 'baseline_error'],
                         var_name='Model', value_name='Error Rate')
    sns.boxplot(x='Model', y='Error Rate', data=error_data)
    plt.title('Error Rate Comparison across Models')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

plot_error_comparison(results_df)
plt.show()


# Plot the feature importance
#plot_logistic_regression_feature_importance(df_standardized.drop(['DEATH_EVENT'], axis=1), y_train, logistic_model)


