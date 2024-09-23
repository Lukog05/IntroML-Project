import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Replace 'file_path.csv' with the actual path to your CSV file
file_path = 'heart_failure_clinical_records_dataset.csv'

# Read the CSV file into a Pandas XFrame
df = pd.read_csv(file_path)

# Display the first few rows of the XFrame
#print(df.head())

corr = df.corr()
#sb.heatmap(corr,annot=True)
X = df.to_numpy()
y = X[:, -1]
X = X[:, 0:-1]
N = X.shape[0]

#center data
Y = X - np.ones((N,1)) * X.mean(axis=0)

# 1. Standardize the data (optional, but recommended)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)  # Standardized X

# 2. Apply PCA
pca = PCA(n_components=3)  # Number of components to keep (e.g., 2 for 2D)
X_pca = pca.fit_transform(X_std)

# 3. Access explained variance and components
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Principal components:\n", pca.components_)

# 3. Plot the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color='g')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Explained Variance Ratio by Principal Components')
plt.show()

# 4. Plot PCA1 vs PCA2, colored by class label
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.colorbar(scatter)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA1 vs PCA2 with Original Class Labels')
plt.grid(True)
plt.show()

# 4. 3D Plot PCA1 vs PCA2 vs PCA3
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with PCA1, PCA2, PCA3 and color by class labels
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.8)

# Set labels and title
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('3D PCA Plot (PCA1 vs PCA2 vs PCA3)')

# Add color bar
plt.colorbar(scatter)
plt.show()












corr = np.corrcoef(Y.T)
sb.heatmap(corr,annot=True)
fig, ax = plt.subplots(2,2)

bins, value_counts = np.unique(X[:,-1], return_counts=True)
ax[0][0].bar(bins,value_counts)
ax[0][0].set_xlabel(r'Died or not during follow-up period')
ax[0][0].set_ylabel(r'Count')


color = ['red' if d == 0 else 'blue' for d in X[:,5]]
platelets = X[:,6] / X[:,6].max()
ax[0][1].scatter(X[:,0],X[:,4], c=color,  s =platelets*100, alpha=0.5)
ax[0][1].set_xlabel(r'Age')
ax[0][1].set_ylabel(r'Ejection Fraction')
ax[0][1].set_title('Size = platelets, color = High blood pressure')


color2 = ['red' if d == 0 else 'blue' for d in X[:,9]]

ax[1][0].scatter(X[:,0], X[:,-3], c=color2, s=platelets*100)
ax[1][0].set_xlabel(r'Age')
ax[1][0].set_ylabel(r'Smokes')
ax[1][0].set_title('Size = platelets, color = Sex')



float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


plt.show()
