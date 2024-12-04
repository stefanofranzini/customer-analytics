import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from umap import umap_
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import lightgbm as lgb
import shap


# Load Data
df = pd.read_csv('data/raw/new_retail_data.csv')

# --- CLEANING ---

# Select Data
df = df.dropna(subset=['Transaction_ID', 'Customer_ID'])
df['Transaction_ID'] = df['Transaction_ID'].astype(int)
df['Customer_ID'] = df['Customer_ID'].astype(int)

columns = [
    'Transaction_ID', 'Customer_ID', 'State', 'Country', 'Age', 'Gender', 'Income',
    'Amount', 'Total_Purchases', 'Total_Amount', 'Product_Category', 'Product_Brand',
    'Product_Type', 'Shipping_Method', 'Payment_Method', 'Ratings'
]

df_clean = df.copy()[columns]
df_clean = df_clean.dropna()
print(f"Lost {len(df)-len(df_clean)} rows. That is {(len(df)-len(df_clean))/len(df)*100} %")

# Map the Variables
df_clean['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

to_replace = {'Low': 1, 'Medium': 2, 'High': 3}
df_clean['Income'] = df['Income'].map(to_replace)

# Dummify
customer_columns = ['Customer_ID', 'Age', 'Gender', 'Income', 'State', 'Country']
transaction_columns = ['Transaction_ID', 'Customer_ID', 'Amount', 'Total_Purchases', 'Total_Amount', 'Ratings', 'Product_Category', 'Product_Brand', 'Product_Type', 'Shipping_Method', 'Payment_Method']

df_customer = df_clean[customer_columns].drop_duplicates(subset='Customer_ID')
df_transaction = df_clean[transaction_columns]

df_customer = pd.get_dummies(df_customer, columns=['State', 'Country'], drop_first=True, dtype=int)
df_transaction = pd.get_dummies(df_transaction, columns=['Product_Category', 'Product_Brand', 'Product_Type', 'Shipping_Method', 'Payment_Method'], drop_first=True, dtype=int)

# Aggregate Transactions
avg_columns = ['Amount', 'Total_Purchases', 'Total_Amount', 'Ratings']
sum_columns = [col for col in df_transaction.columns if col not in avg_columns + ['Customer_ID', 'Transaction_ID']]

agg_dict = {col: 'mean' for col in avg_columns}
agg_dict.update({col: 'sum' for col in sum_columns})

df_aggregated = df_transaction.groupby('Customer_ID').agg(agg_dict).reset_index()
df_aggregated = df_aggregated.merge(df_customer, on='Customer_ID')

feature_columns = list(df_aggregated.columns)
df_sampled = df_aggregated.sample(n=10000, replace=False, random_state=42)
features = df_sampled.select_dtypes(include=['number']).drop(['Customer_ID'], axis=1)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)

# --- DIMENSIONAL REDUCTION ---

# UMAP Dimensionality Reduction
umap_reducer = umap_.UMAP(n_components=3, random_state=42)
umap_data = umap_reducer.fit_transform(normalized_data)

df_sampled['UMAP1'] = umap_data[:, 0]
df_sampled['UMAP2'] = umap_data[:, 1]
df_sampled['UMAP3'] = umap_data[:, 2]

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = df_sampled['UMAP1']
y = df_sampled['UMAP2']
z = df_sampled['UMAP3']

scatter = ax.scatter(x, y, z, s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/dimensional_reduction.png')
plt.show()

# Remove Density Outliers
radius = 0.5
features = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']].values
nbrs = NearestNeighbors(radius=radius).fit(features)
neighbors_count = nbrs.radius_neighbors(features, return_distance=False)

local_density = np.array([len(neighbors) for neighbors in neighbors_count])
df_sampled['Local_Density'] = local_density
df_sampled['Q_Local_Density'] = pd.qcut(local_density, q=4, labels=[1, 2, 3, 4]).astype(int)

df_sampled = df_sampled.loc[df_sampled['Q_Local_Density'] > 1].copy()
features = df_sampled.select_dtypes(include=['number']).drop(['Customer_ID'], axis=1)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)

# 3D Scatter Plot after Density Removal
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = df_sampled['UMAP1']
y = df_sampled['UMAP2']
z = df_sampled['UMAP3']

scatter = ax.scatter(x, y, z, s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/dimensional_reduction.png')
plt.show()

# --- CLUSTERING (no dimensional reduction) ---

# K-means Clustering
kmeans_labels = []
scores = []

for k in range(2, 31, 3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels.append(kmeans.fit_predict(normalized_data))
    labels = kmeans_labels[-1]
    df_sampled['label'] = labels

    kmeans_silhouette = silhouette_score(normalized_data, df_sampled['label'])
    print(f"Silhouette Score for KMeans Clustering with {k} clusters: {kmeans_silhouette:.2f}")
    scores.append(kmeans_silhouette)

optimal = np.argmax(scores) + 2
k = list(range(2, 31, 3))[optimal - 2]
print(f"the optimal number of clusters is {optimal}, with a score of {scores[optimal - 2]:.2f}")

df_sampled['KMeans_Cluster'] = kmeans_labels[optimal - 2]

# 3D Scatter Plot with KMeans Clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = df_sampled['UMAP1']
y = df_sampled['UMAP2']
z = df_sampled['UMAP3']
labels = df_sampled['KMeans_Cluster']

scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/clustering_KMEANS_noDimensionalReduction.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df_sampled['UMAP1'], df_sampled['UMAP2'], c=df_sampled['KMeans_Cluster'], cmap='tab20', alpha=0.7)
plt.title('2D Scatter Plot of Clusters')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()

# --- CLUSTERING (*with* dimensional reduction) ---

# K-means Clustering
scores = []
kmeans_labels = []

for k in range(2, 101, 5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels.append(kmeans.fit_predict(df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]))
    labels = kmeans_labels[-1]
    df_sampled['label'] = labels

    kmeans_silhouette = silhouette_score(df_sampled[['UMAP1', 'UMAP2', 'UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for KMeans Clustering with {k} clusters: {kmeans_silhouette:.2f}")
    scores.append(kmeans_silhouette)

optimal = np.argmax(scores) + 2
k = list(range(2, 101, 5))[optimal - 2]
print(f"the optimal number of clusters is {k}, with a score of {scores[optimal - 2]:.2f}")

df_sampled['KMeans_Cluster_UMAP'] = kmeans_labels[optimal - 2]

# 3D Scatter Plot with KMeans Clusters (Dimensional Reduction)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = df_sampled['UMAP1']
y = df_sampled['UMAP2']
z = df_sampled['UMAP3']
labels = df_sampled['KMeans_Cluster_UMAP']

scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/clustering_KMEANS_withDimensionalReduction.png')
plt.show()

# ---
# HIERARCHICAL
# ---

hierarchical_labels = []
scores = []
data_to_cluster = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]

for k in range(2, 101, 5):
    hierarchical = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    hierarchical_labels += [hierarchical.fit_predict(data_to_cluster)]
    labels = hierarchical_labels[-1]
    df_sampled['label'] = labels
    silhouette = silhouette_score(df_sampled[['UMAP1','UMAP2','UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for Hierarchical Clustering with {k} clusters: {silhouette:.2f}")
    scores += [silhouette]

optimal = np.argmax(scores) + 2
k = list(range(2, 101, 5))[optimal - 2]
print(f"Optimal number of clusters is {k}, with a score of {scores[optimal - 2]:.2f}")

df_sampled['Hierarchical_Cluster_UMAP'] = hierarchical_labels[optimal - 2]

# 3D Scatter Plot for Hierarchical Clustering
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x, y, z = df_sampled['UMAP1'], df_sampled['UMAP2'], df_sampled['UMAP3']
labels = df_sampled['Hierarchical_Cluster_UMAP']
scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/clustering_hierarchical_dimensionalReduction.png')
plt.show()

# 2D Scatter Plot for Hierarchical Clustering
plt.figure(figsize=(8, 6))
plt.scatter(df_sampled['UMAP1'], df_sampled['UMAP2'], c=df_sampled['Hierarchical_Cluster_UMAP'], cmap='tab20', alpha=0.7)
plt.title('2D Scatter Plot of Clusters')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()

# ---
# DBSCAN
# ---

dbscan_labels = []
scores = []
data_to_cluster = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]

for scale in np.linspace(0.1, 5, 20)[::-1]:
    dbscan = DBSCAN(eps=scale, min_samples=10)
    dbscan_labels += [dbscan.fit_predict(data_to_cluster)]
    labels = dbscan_labels[-1]
    df_sampled['label'] = labels
    k = len(df_sampled['label'].unique())
    silhouette = silhouette_score(df_sampled[['UMAP1','UMAP2','UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for DBSCAN Clustering with {k} clusters (eps = {scale:.2f}): {silhouette:.2f}")
    scores += [silhouette]

optimal = np.argmax(scores)
df_sampled['DBSCAN_Cluster_UMAP'] = dbscan_labels[optimal]
k = len(df_sampled['DBSCAN_Cluster_UMAP'].unique())
print(f"Optimal number of clusters is {k}, with a score of {scores[optimal]:.2f}")

# 3D Scatter Plot for DBSCAN
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x, y, z = df_sampled['UMAP1'], df_sampled['UMAP2'], df_sampled['UMAP3']
labels = df_sampled['DBSCAN_Cluster_UMAP']
scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/clustering_DBSCAN_dimensionalReduction.png')
plt.show()

# 2D Scatter Plot for DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(df_sampled['UMAP1'], df_sampled['UMAP2'], c=df_sampled['DBSCAN_Cluster_UMAP'], cmap='tab20', alpha=0.7)
plt.title('2D Scatter Plot of Clusters')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()

# ---
# GAUSSIAN MIXTURES
# ---

gaussian_labels = []
scores = []
data_to_cluster = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]

for k in range(2, 101, 5):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gaussian_labels += [gmm.fit_predict(data_to_cluster)]
    labels = gaussian_labels[-1]
    df_sampled['label'] = labels
    silhouette = silhouette_score(df_sampled[['UMAP1','UMAP2','UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for Gaussian Mixtures with {k} gaussians: {silhouette:.2f}")
    scores += [silhouette]

optimal = np.argmax(scores) + 2
k = list(range(2, 101, 5))[optimal - 2]
print(f"Optimal number of clusters is {k}, with a score of {scores[optimal - 2]:.2f}")

df_sampled['Gaussian_Cluster_UMAP'] = gaussian_labels[optimal - 2]

# 3D Scatter Plot for Gaussian Mixtures
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x, y, z = df_sampled['UMAP1'], df_sampled['UMAP2'], df_sampled['UMAP3']
labels = df_sampled['Gaussian_Cluster_UMAP']
scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/clustering_gaussian_dimensionalReduction.png')
plt.show()

# 2D Scatter Plot for Gaussian Mixtures
plt.figure(figsize=(8, 6))
plt.scatter(df_sampled['UMAP1'], df_sampled['UMAP2'], c=df_sampled['Gaussian_Cluster_UMAP'], cmap='tab20', alpha=0.7)
plt.title('2D Scatter Plot of Clusters')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()

# ---
# CUSTOMER SEGMENT
# ---

df_sampled = df_sampled.merge(df[['Customer_ID', 'Customer_Segment']].drop_duplicates(subset=['Customer_ID']), on='Customer_ID', how='left')

to_replace = {'Regular': 2, 'Premium': 3, 'New': 1}
df_sampled['Customer_Segment_encoded'] = df_sampled['Customer_Segment'].map(to_replace)
df_sampled['Customer_Segment_encoded'] = df_sampled['Customer_Segment_encoded'].fillna(0)

# 3D Scatter Plot for Customer Segment
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x, y, z = df_sampled['UMAP1'], df_sampled['UMAP2'], df_sampled['UMAP3']
labels = df_sampled['Customer_Segment_encoded']
scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.savefig('artifacts/imgs/clustering_customerSegments.png')
plt.show()

# ---
# SAVE OUTPUTS
# ---

df_customer = df_clean[customer_columns].drop_duplicates(subset='Customer_ID')
df_transaction = df_clean[transaction_columns]
df_transaction = pd.get_dummies(df_transaction, columns=['Product_Category', 'Product_Brand', 'Product_Type', 'Shipping_Method', 'Payment_Method'], drop_first=False, dtype=int)

df_aggregated_ = df_transaction.groupby('Customer_ID').agg(agg_dict).reset_index()
df_aggregated_ = df_aggregated_.merge(df_customer, on='Customer_ID')

df_sampled['Cluster'] = df_sampled['Hierarchical_Cluster_UMAP']
df_aggregated_ = df_aggregated_.merge(df_sampled[['Customer_ID', 'Cluster', 'UMAP1', 'UMAP2', 'UMAP3']], on='Customer_ID')

df_aggregated_.to_csv('data/processed/new_retail_customer_clustering.csv', index=False)

# ---
# Explain Clusters
# ---

df_sampled_ = df_sampled[feature_columns + ['Cluster']]
X = df_sampled_.drop(['Customer_ID', 'Cluster'], axis=1)
y = df_sampled

# --- 
# LGBM Classifier Training
clf_km = lgb.LGBMClassifier(colsample_by_tree=0.8)
clf_km.fit(X=X, y=y)

# --- 
# SHAP values
explainer_km = shap.TreeExplainer(clf_km)
shap_values_km = explainer_km(X)

# --- 
# Plot SHAP values
class_names = clf_km.classes_
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
axes = axes.flatten()

for k, ax in enumerate(axes):
    if k < len(class_names):
        shap.plots.bar(shap_values_km[:, :, k], show=False, ax=ax)
        ax.set_title(f"Class: {class_names[k]}")
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel("mean(|SHAP value|)", fontsize=10)
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# --- 
# Cluster Names Mapping
cluster_names = {
    '0': 'English Customers',
    '1': 'Mitsubhisi Lovers',
    '2': 'Berliners Customers',
    '3': 'New Mexican Customers',
    '4': 'Ontario Customers',
    '5': 'Whirlpool Lovers',
    '6': 'Bluestar Lovers',
    '7': 'Kansas Customers',
    '8': 'Maine Customers',
    '9': 'Connetticut Customers',
    '10': 'New South Wales Customers',
    '11': 'Georgia Customers',
}

df_sampled['Cluster_Name'] = df_sampled['Cluster'].astype(str).replace(cluster_names)

# --- 
# 3D Scatter Plot with Cluster Names
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = df_sampled['UMAP1']
y = df_sampled['UMAP2']
z = df_sampled['UMAP3']
labels = df_sampled['Cluster']
cluster_names = df_sampled['Cluster_Name'].unique()

scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=5, alpha=0.5)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')

for i, cluster_name in enumerate(cluster_names):
    ax.scatter([], [], [], c=[plt.cm.tab20((i+1)/len(cluster_names))], label=cluster_name, s=50)

ax.legend(title="Cluster Names", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('artifacts/imgs/clustering_customerSegments.png', bbox_inches='tight')
plt.show()

# --- 
# 3D Scatter Plot with Cluster Names (Alternate Color Mapping)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = df_sampled['UMAP1']
y = df_sampled['UMAP2']
z = df_sampled['UMAP3']
labels = df_sampled['Cluster']
cluster_names = df_sampled['Cluster_Name'].unique()

scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=5, alpha=0.5)
ax.set_title('3D Scatter Plot of Clusters')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')

for i, cluster_name in enumerate(cluster_names):
    ax.scatter([], [], [], c=[plt.cm.tab20(i/len(cluster_names))], label=cluster_name, s=50)

ax.legend(title="Cluster Names", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('artifacts/imgs/clustering_customerSegments.png', bbox_inches='tight')
plt.show()