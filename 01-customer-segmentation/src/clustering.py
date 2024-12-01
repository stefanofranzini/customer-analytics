import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from umap import umap_
from sklearn.metrics import silhouette_score
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

def plot_3D(df,label_column=None):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x = df['UMAP1'] 
    y = df['UMAP2'] 
    z = df['UMAP3']

    if label_column is not None:
        labels = df[label_column]

        # Create the scatter plot
        scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)

    else:
        scatter = ax.scatter(x, y, z, s=20, alpha=0.8)

    # Label axes
    ax.set_title(f'3D Scatter Plot of Clusters: {label_column}')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')

    return ax

def plot_2D(df,label_column=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['UMAP1'], df['UMAP2'], c=df[label_column], cmap='tab20', alpha=0.7)
    plt.title(f'2D Scatter Plot of Clusters: {label_column}')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

df = pd.read_csv('data/raw/new_retail_data.csv')
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# CLEANING

# ------------------------------------------------------------------------------------------------------
### SELECT DATA

df = df.dropna(subset=['Transaction_ID','Customer_ID'])


df['Transaction_ID'] = df['Transaction_ID'].astype(int)
df['Customer_ID'] = df['Customer_ID'].astype(int)


columns = [
    'Transaction_ID',
    'Customer_ID',
    'State',
    'Country',
    'Age',
    'Gender',
    'Income',
    'Amount',
    'Total_Purchases',
    'Total_Amount',
    'Product_Category',
    'Product_Brand',
    'Product_Type',
    'Shipping_Method',
    'Payment_Method',
    'Ratings']


df_clean = df.copy()[columns]
df_clean = df_clean.dropna()
print(f"Lost {len(df)-len(df_clean)} rows. That is {(len(df)-len(df_clean))/len(df)*100} %")

# ------------------------------------------------------------------------------------------------------
### MAP THE VARIABLES


df_clean['Gender'] = df['Gender'].apply(lambda x: 1 if x=='Male' else 0)


to_replace = {
    'Low' : 1,
    'Medium': 2,
    'High' : 3 
}

df_clean['Income'] = df['Income'].map(to_replace)

# ------------------------------------------------------------------------------------------------------
### DUMMIFY


customer_columns = ['Customer_ID','Age','Gender','Income','State','Country']
transaction_columns = ['Transaction_ID','Customer_ID','Amount','Total_Purchases','Total_Amount','Ratings','Product_Category','Product_Brand','Product_Type','Shipping_Method','Payment_Method']


df_customer = df_clean[customer_columns].drop_duplicates(subset='Customer_ID')
df_transaction = df_clean[transaction_columns]


df_customer = pd.get_dummies(df_customer,columns=['State','Country'], drop_first=True, dtype=int)
df_transaction = pd.get_dummies(df_transaction,columns=['Product_Category','Product_Brand','Product_Type','Shipping_Method','Payment_Method'], drop_first=True, dtype=int)


# ------------------------------------------------------------------------------------------------------
### AGGREGATE TRANSACTIONS


avg_columns = ['Amount', 'Total_Purchases', 'Total_Amount', 'Ratings']
sum_columns = [col for col in df_transaction.columns if col not in avg_columns + ['Customer_ID', 'Transaction_ID']]

agg_dict = {col: 'mean' for col in avg_columns}
agg_dict.update({col: 'sum' for col in sum_columns})



df_aggregated = df_transaction.groupby('Customer_ID').agg(agg_dict).reset_index()


df_aggregated = df_aggregated.merge(df_customer,on='Customer_ID')


# ------------------------------------------------------------------------------------------------------
### SCALING


df_sampled = df_aggregated.sample(n=10000, replace=False, random_state=42)

features = df_sampled.select_dtypes(include=['number']).drop(['Customer_ID'],axis=1)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# DIMENSIONAL REDUCTION


# Perform UMAP for dimensionality reduction to 2 dimensions
umap_reducer = umap_.UMAP(n_components=3, random_state=42)
umap_data = umap_reducer.fit_transform(normalized_data)

df_sampled['UMAP1'] = umap_data[:,0]
df_sampled['UMAP2'] = umap_data[:,1]
df_sampled['UMAP3'] = umap_data[:,2]

plot_3D(df_sampled)
plt.savefig('artifacts/imgs/dimensional_reduction.png')
plt.show()

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# CLUSTERING (no dimensional reduction)


# ------------------------------------------------------------------------------------------------------
### K-means Clustering


kmeans_labels = []
scores = []

for k in range(2,31):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels += [ kmeans.fit_predict(normalized_data) ]

    labels = kmeans_labels[-1]

    df_sampled['label'] = labels

    kmeans_silhouette = silhouette_score(normalized_data, df_sampled['label'])
    print(f"Silhouette Score for KMeans Clustering with {k} clusters: {kmeans_silhouette:.2f}")

    scores += [kmeans_silhouette]

optimal = np.argmax(scores)+2

print(f"the optimal number of clusters is {optimal}, with a score of {scores[optimal-2]:.2f}")


# Assign K-means cluster labels to the DataFrame
df_sampled['KMeans_Cluster'] = kmeans_labels[optimal-2]

plot_3D(df_sampled,'KMeans_Cluster')
plt.savefig('artifacts/imgs/clustering_KMEANS_noDimensionalReduction.png')
plt.show()


plot_2D(df_sampled,'KMeans_Cluster')
plt.show()

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# CLUSTERING (*with* dimensional reduction)


# ------------------------------------------------------------------------------------------------------
### K-means Clustering


scores = []
kmeans_labels = []


for k in range(2,101):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels += [ kmeans.fit_predict(df_sampled[['UMAP1','UMAP2','UMAP3']]) ]

    labels = kmeans_labels[-1]

    df_sampled['label'] = labels

    kmeans_silhouette = silhouette_score(df_sampled[['UMAP1','UMAP2','UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for KMeans Clustering with {k} clusters: {kmeans_silhouette:.2f}")

    scores += [kmeans_silhouette]

optimal = np.argmax(scores)+2

print(f"the optimal number of clusters is {optimal}, with a score of {scores[optimal-2]:.2f}")


# Assign K-means cluster labels to the DataFrame
df_sampled['KMeans_Cluster_UMAP'] = kmeans_labels[optimal - 2]

plot_3D(df_sampled,'KMeans_Cluster_UMAP')
plt.savefig('artifacts/imgs/clustering_KMEANS_dimensionalReduction.png')
plt.show()


plot_2D(df_sampled,'KMeans_Cluster_UMAP')
plt.show()



# ------------------------------------------------------------------------------------------------------
# HIERARCHICAL


hierarchical_labels = []
scores = []

data_to_cluster = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]

for k in range(2,101):

    hierarchical = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    hierarchical_labels += [hierarchical.fit_predict(data_to_cluster)]

    labels = hierarchical_labels[-1]

    df_sampled['label'] = labels

    silhouette = silhouette_score(df_sampled[['UMAP1','UMAP2','UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for Hierarchical Clustering with {k} clusters: {silhouette:.2f}")

    scores += [silhouette]

optimal = np.argmax(scores)+2

print(f"the optimal number of clusters is {optimal}, with a score of {scores[optimal-2]:.2f}")

df_sampled['Hierarchical_Cluster_UMAP'] = hierarchical_labels[optimal - 2]

plot_3D(df_sampled,'Hierarchical_Cluster_UMAP')
plt.savefig('artifacts/imgs/clustering_hierarchical_dimensionalReduction.png')
plt.show()


plot_2D(df_sampled,'Hierarchical_Cluster_UMAP')
plt.show()


# ------------------------------------------------------------------------------------------------------
# DBSCAN


dbscan_labels = []
scores = []

data_to_cluster = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]

for scale in np.linspace(0.1,5,100)[::-1]:

    dbscan = DBSCAN(eps=scale, min_samples=10)  # Adjust eps and min_samples as needed
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

print(f"the optimal number of clusters is {k}, with a score of {scores[optimal]:.2f}")

plot_3D(df_sampled,'DBSCAN_Cluster_UMAP')
plt.savefig('artifacts/imgs/clustering_DBSCAN_dimensionalReduction.png')
plt.show()


plot_2D(df_sampled,'DBSCAN_Cluster_UMAP')
plt.show()


# ------------------------------------------------------------------------------------------------------
### GAUSSIAN MIXTURES


gaussian_labels = []
scores = []

data_to_cluster = df_sampled[['UMAP1', 'UMAP2', 'UMAP3']]


for k in range(2,101):

    gmm = GaussianMixture(n_components=k, random_state=42)  # Adjust n_components as needed
    gaussian_labels += [gmm.fit_predict(data_to_cluster)]

    labels = gaussian_labels[-1]

    df_sampled['label'] = labels

    silhouette = silhouette_score(df_sampled[['UMAP1','UMAP2','UMAP3']], df_sampled['label'])
    print(f"Silhouette Score for Gaussian Mixures with {k} gaussians: {silhouette:.2f}")

    scores += [silhouette]

optimal = np.argmax(scores)+2

print(f"the optimal number of clusters is {optimal}, with a score of {scores[optimal-2]:.2f}")


df_sampled['Gaussian_Cluster_UMAP'] = gaussian_labels[optimal - 2]

plot_3D(df_sampled,'Gaussian_Cluster_UMAP')
plt.savefig('artifacts/imgs/clustering_gaussian_dimensionalReduction.png')
plt.show()


plot_2D(df_sampled,'Gaussian_Cluster_UMAP')
plt.show()


#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# EXTEND LABELS TO THE ORIGINAL DATASET

df_aggregated = df_aggregated.merge(df_sampled[['Customer_ID','DBSCAN_Cluster_UMAP']],on='Customer_ID',how='left')
df_aggregated = df_aggregated.merge(df[['Customer_ID','Customer_Segment']].drop_duplicates(subset=['Customer_ID']),on='Customer_ID',how='left')

features = df_aggregated.select_dtypes(include=['number']).drop(['Customer_ID','DBSCAN_Cluster_UMAP'],axis=1)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)

umap_reducer = umap_.UMAP(n_components=3, random_state=42)
umap_data = umap_reducer.fit_transform(normalized_data)

df_aggregated['UMAP1'] = umap_data[:,0]
df_aggregated['UMAP2'] = umap_data[:,1]
df_aggregated['UMAP3'] = umap_data[:,2]

#-------------------------------------------------------------------------------------------------------

# KNN IMPUTATION

df_aggregated['Cluster'] = df_aggregated['DBSCAN_Cluster_UMAP']

# Identify rows with and without missing labels
train_data = df_aggregated[df_aggregated['DBSCAN_Cluster_UMAP'].notnull()]
test_data = df_aggregated[df_aggregated['DBSCAN_Cluster_UMAP'].isnull()]

# Features and labels for training
X_train = train_data[['UMAP1', 'UMAP2','UMAP3']].values
y_train = train_data['DBSCAN_Cluster_UMAP'].values

# Features for prediction
X_test = test_data[['UMAP1', 'UMAP2', 'UMAP3']].values

# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the missing labels
if len(X_test) > 0:  # Ensure there are missing labels to predict
    y_pred = knn.predict(X_test)

    # Assign predicted labels to the test data
    df_aggregated.loc[df_aggregated['DBSCAN_Cluster_UMAP'].isnull(), 'Cluster'] = y_pred

plot_3D(df_aggregated,label_column='Cluster')
plt.savefig('artifacts/imgs/clustering.png')

plt.show()

# -------------------------------------------------------------------------------------------------

### UNDUMMIFY

df_customer = df_clean[customer_columns].drop_duplicates(subset='Customer_ID')
df_transaction = df_clean[transaction_columns]

df_transaction = pd.get_dummies(df_transaction,columns=['Product_Category','Product_Brand','Product_Type','Shipping_Method','Payment_Method'], drop_first=False, dtype=int)

df_aggregated_ = df_transaction.groupby('Customer_ID').agg(agg_dict).reset_index()
df_aggregated_ = df_aggregated_.merge(df_customer,on='Customer_ID')
df_aggregated_ = df_aggregated_.merge(df_aggregated[['Customer_ID','Cluster','UMAP1','UMAP2','UMAP3']],on='Customer_ID')

# Save the DataFrame with the new column
df_aggregated_.to_csv('data/processed/new_retail_customer_clustering.csv', index=False)

print("Imputation completed and saved to 'new_retail_customer_clustering.csv'")
