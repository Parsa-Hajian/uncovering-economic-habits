import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the dataset
df = sns.load_dataset('tips')

# Data Preprocessing
# Convert categorical variables to numeric
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
df['smoker'] = df['smoker'].map({'No': 0, 'Yes': 1})
df['day'] = df['day'].map({'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3})
df['time'] = df['time'].map({'Lunch': 0, 'Dinner': 1})

# Select numeric columns for PCA
features = ['total_bill', 'tip', 'size', 'sex', 'smoker', 'day', 'time']
data = df[features]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA()
pca_components = pca.fit_transform(scaled_data)
explained_variance_ratio = pca.explained_variance_ratio_

# Visualize Cumulative Explained Variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), 
         explained_variance_ratio.cumsum(), 
         marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid()
plt.show()

# PCA Loadings
loadings = pd.DataFrame(
    pca.components_, columns=features, index=[f'PC{i+1}' for i in range(len(features))]
)
plt.figure(figsize=(12, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings (Contribution of Features)')
plt.show()

# 2D Visualization of PCA
pca_2d = pd.DataFrame(pca_components[:, :2], columns=['PC1', 'PC2'])
pca_2d['size'] = df['size']  # Size of the dining party
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_2d, x='PC1', y='PC2', hue='size', palette='viridis', s=100)
plt.title('PCA 2D Projection of Dining Habits')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# 3D Visualization of PCA
pca_3d = pd.DataFrame(pca_components[:, :3], columns=['PC1', 'PC2', 'PC3'])
pca_3d['size'] = df['size']
fig = px.scatter_3d(
    pca_3d, x='PC1', y='PC2', z='PC3', color=pca_3d['size'].astype(str),
    title='3D PCA Visualization of Dining Habits',
    labels={'size': 'Party Size'}
)
fig.show()
