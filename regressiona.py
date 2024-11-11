import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# Dataset preparation
file_path = 'heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(file_path)

# Standardization
df_centered = df - df.mean()
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_centered), columns=df_centered.columns)

# Prepare data for regression
X = df_standardized.drop(['DEATH_EVENT', 'platelets'], axis=1)
Y = df_standardized['platelets']

# Apply PCA and reduce dimensionality
X_np = X.to_numpy()
X_std = scaler.fit_transform(X_np)
pca = PCA(n_components=11)
X_pca = pca.fit_transform(X_std)

# Determine the number of principal components to retain
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cumulative_variance > 0.85)[0][0] + 1
X_pca = X_pca[:, :n_components]

# Find the best Ridge regression model using the reduced PCA components
lambda_values = np.logspace(-4, 4, 40)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mean_errors = []

for alpha in lambda_values:
    lin_reg = Ridge(alpha=alpha)
    mse_scores = -cross_val_score(lin_reg, X_pca, Y, cv=kf, scoring='neg_mean_squared_error')
    mean_errors.append(mse_scores.mean())

# Find the best lambda and the corresponding best model
best_lambda_idx = np.argmin(mean_errors)
best_lambda = lambda_values[best_lambda_idx]
best_ridge_model = Ridge(alpha=best_lambda).fit(X_pca, Y)

# Get the coefficients of the best Ridge regression model
ridge_coefs = best_ridge_model.coef_

# Create a DataFrame to store the feature importances
feature_importances = pd.DataFrame({
    'Feature': [f'PC{i+1}' for i in range(X_pca.shape[1])],
    'Coefficient': ridge_coefs
})

# Sort the DataFrame by the absolute value of the coefficients
feature_importances = feature_importances.sort_values(by='Coefficient', key=abs, ascending=False)

# Display the feature importance table
print(feature_importances)

# Visualize the feature importances
fig, ax = plt.subplots(figsize=(12, 6))
feature_importances.sort_values(by='Coefficient', key=abs).plot(kind='barh', x='Feature', y='Coefficient', ax=ax)
ax.set_title('Feature Importances in Ridge Regression')
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Feature')
plt.show()


# Get loadings matrix for only the reduced number of components
n_components = X_pca.shape[1]  # This is the reduced number of components we kept
loadings = pd.DataFrame(
    pca.components_[:n_components].T,  # Only use the components we kept
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=X.columns
)

# Show what features contribute most to the important PCs
print("\nFeature contributions to top PCs:")
print("\nPC2 (strongest positive effect on platelets, coefficient=0.003099):")
print(loadings['PC2'].sort_values(ascending=False))
print("\nPC1 (strong negative effect on platelets, coefficient=-0.002589):")
print(loadings['PC1'].sort_values(ascending=False))

# Calculate total effect of each original feature using only the reduced components
feature_effects = pd.DataFrame({
    'Feature': X.columns,
    'Total_Effect': np.dot(loadings, ridge_coefs[:n_components])  # Use only the coefficients for components we kept
})

# Sort by absolute value of effect
feature_effects = feature_effects.sort_values(by='Total_Effect', key=abs, ascending=False)

print("\nTotal effect of original features on platelets prediction:")
print(feature_effects)

# Visualize
plt.figure(figsize=(10, 6))
feature_effects.plot(kind='barh', x='Feature', y='Total_Effect')
plt.title('Total Effect of Original Features on Platelets Count')
plt.xlabel('Effect Size')
plt.tight_layout()
plt.show()


# Plot MSE vs lambda values
plt.figure(figsize=(10, 6))
plt.semilogx(lambda_values, mean_errors, 'b-', lw=2)
plt.plot(lambda_values[best_lambda_idx], mean_errors[best_lambda_idx], 'r*', markersize=15, 
         label=f'Best λ = {best_lambda:.4f}')
plt.xlabel('λ (Alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: MSE vs Regularization Parameter (λ)')
plt.grid(True)
plt.legend()
plt.show()

# Print the best lambda and its corresponding MSE
print(f"\nBest λ value: {best_lambda:.4f}")
print(f"Corresponding MSE: {mean_errors[best_lambda_idx]:.4f}")