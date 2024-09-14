import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Obesity Risk Clustering App')
df = pd.read_csv('ObesityDataSet.csv')

# Feature Engineering
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Adolescent', 'Adult', 'Elderly'])
df['FAVC_CAEC'] = df['FAVC'] + "_" + df['CAEC']
df['Healthy_Score'] = df['FCVC'] + df['FAF'] - df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})
df.drop(columns=['Height', 'Weight'], inplace=True)  # BMI replaces these

# Preprocessing
numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Healthy_Score']
n_features = len(numerical_features)
n_cols = 2  # Number of columns (2 plots per row)
n_rows = (n_features + 1) // n_cols  # Calculate rows needed based on the number of features
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))  # Adjust the figure size
axes = axes.flatten()  # Flatten the axes array to easily iterate
for i, feature in enumerate(numerical_features):
    sns.boxplot(x=df[feature], ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature}')
for i in range(n_features, len(axes)):
    fig.delaxes(axes[i])
numerical_features_excl_age = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Healthy_Score']
for feature in numerical_features_excl_age:
    df = df[(np.abs(stats.zscore(df[feature])) < 3)]
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
for feature in numerical_features_excl_age:
    skewness = df[feature].skew(
for feature in numerical_features_excl_age:
    skewness = df[feature].skew()
    if skewness > 1:
        df[feature] = np.log1p(df[feature])  # Apply log1p transformation to handle zero values
df_before_skewness = df.copy()
for i, feature in enumerate(numerical_features_excl_age):
    sns.boxplot(x=df_before_skewness[feature], ax=axes[0, i])
    axes[0, i].set_title(f'Before: {feature}')
for i, feature in enumerate(numerical_features_excl_age):
    sns.boxplot(x=df[feature], ax=axes[1, i])
    axes[1, i].set_title(f'After: {feature}') 
nominal_features = ['Gender', 'FAVC_CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'CAEC'] 
ordinal_features = ['NObeyesdad', 'Age_Group']  # Ordinal features
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_nominal = onehot_encoder.fit_transform(df[nominal_features])
encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=onehot_encoder.get_feature_names_out(nominal_features))
encoded_nominal_df.reset_index(drop=True, inplace=True)
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])
df['Age_Group'] = label_encoder.fit_transform(df['Age_Group'])
df_numerical_only = df.drop(columns=nominal_features).reset_index(drop=True)
df_encoded = pd.concat([df_numerical_only, encoded_nominal_df], axis=1
corr_matrix = df_encoded.corr()
target_variable = 'NObeyesdad'
correlation_with_target = df_encoded.corr()[target_variable]
threshold = 0.1
weak_corr_features = correlation_with_target[correlation_with_target.abs() < threshold].index.tolist()
df_reduced = df_encoded.drop(columns=weak_corr_features)
high_corr_features = [
    'FAVC_CAEC_yes_Always', 
    'FAVC_CAEC_no_Frequently', 
    'CALC_no', 
    'CALC_Frequently', 
    'MTRANS_Walking'
]
weak_corr_features = [
    'FAVC', 'FCVC', 'NCP', 'CH2O', 'TUE', 'Gender_Male', 
    'FAVC_CAEC_no_no', 'FAVC_CAEC_yes_no', 'SMOKE_yes', 'SCC_yes', 
    'CALC_Frequently', 'MTRANS_Bike', 'MTRANS_Motorbike', 
    'MTRANS_Public_Transportation', 'MTRANS_Walking', 'CAEC_no'
]
columns_to_drop = list(set(high_corr_features + weak_corr_features))  # Use set to avoid duplicates
clean_df = df_encoded.drop(columns=columns_to_drop)
further_columns_to_drop = [
    'FAVC_CAEC_yes_Sometimes', 
    'CAEC_Sometimes', 
    'Healthy_Score', 
    'CAEC_Frequently'
]
clean_df_final_minimized = clean_df.drop(columns=further_columns_to_drop)
corr_matrix_minimized = clean_df_final_minimized.corr()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_df_final_minimized)
scaled_df = pd.DataFrame(scaled_data, columns=clean_df_final_minimized.columns)
target = "NObeyesdad"
X = clean_df  
numerical_features = X.select_dtypes(include=['number']).columns
scaler = RobustScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(9)])

pca_df = pd.DataFrame(X_pca[:, :5], columns=[f'PC{i+1}' for i in range(5)])
scaler_reduced = StandardScaler()
scaled_pca_df = scaler_reduced.fit_transform(pca_df)
inertia = []
k_range = range(1, 11)  # Try different values of k (number of clusters)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_pca_df)
    inertia.append(kmeans.inertia_)
silhouette_scores = []
valid_k = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_pca_df)
    labels = kmeans.labels_
    if len(np.unique(labels)) > 1:
        silhouette_avg = silhouette_score(scaled_pca_df, labels)
        silhouette_scores.append(silhouette_avg)
        valid_k.append(k)
    else:
        silhouette_scores.append(np.nan)  # Use NaN to indicate invalid silhouette score
filtered_valid_k = [k for k, s in zip(k_range, silhouette_scores) if not np.isnan(s)]
filtered_silhouette_scores = [s for s in silhouette_scores if not np.isnan(s)]

# Modeling K-Means
n_clusters_range = range(1, 11)
inertia = []
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_pca_df)
    inertia.append(kmeans.inertia_)
silhouette_scores = []
for i in range(2, 11):  
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_df)
    score = silhouette_score(pca_df, cluster_labels)
    silhouette_scores.append(score)
optimal_n_clusters = 4 
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
pca_df['KMeans_Cluster'] = kmeans.fit_predict(scaled_pca_df)

# Modeling Mean Shift
bandwidth = estimate_bandwidth(scaled_pca_df, quantile=0.2, n_samples=500)
mean_shift = MeanShift(bandwidth=bandwidth)
pca_df['MeanShift_Cluster'] = mean_shift.fit_predict(scaled_pca_df)
n_clusters = len(np.unique(pca_df['MeanShift_Cluster']))

# Modeling DBSCAN
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_pca_df)
distances, indices = neighbors_fit.kneighbors(scaled_pca_df)
dbscan = DBSCAN(eps=2.0, min_samples=10)
n_clusters = len(np.unique(pca_df['DBSCAN_Cluster']))

# Modeling GMM
n_components_range = range(1, 11)
aics = []
bics = []
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(scaled_pca_df)
    aics.append(gmm.aic(scaled_pca_df))
    bics.append(gmm.bic(scaled_pca_df))
optimal_n_components = 5 
gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
pca_df['GMM_Cluster'] = gmm.fit_predict(scaled_pca_df)

# Modeling Agglomerative Hierarchical Clustering
linkage_matrix = linkage(scaled_pca_df, method='ward') 
n_clusters = 3
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
pca_df['Agg_Cluster'] = agg_clustering.fit_predict(scaled_pca_df)

# Modeling Spectral
n_clusters_range = range(2, 11)
inertia = []
for n_clusters in n_clusters_range:
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    labels = spectral.fit_predict(scaled_pca_df)
    inertia.append(np.sum([np.min(pairwise_distances_argmin_min(scaled_pca_df[labels == i], scaled_pca_df[labels])[1]) for i in range(n_clusters)]))
optimal_n_clusters = 5
spectral = SpectralClustering(n_clusters=optimal_n_clusters, affinity='nearest_neighbors', random_state=42)
pca_df['Spectral_Cluster'] = spectral.fit_predict(scaled_pca_df)

#Evaluation metrics
def apply_kmeans(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

def apply_gmm(n_components, data):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    return gmm.fit_predict(data)
def apply_dbscan(eps, min_samples, data):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)
def apply_mean_shift(bandwidth, data):
    mean_shift = MeanShift(bandwidth=bandwidth)
    return mean_shift.fit_predict(data)
def apply_agglomerative(n_clusters, data):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return agg_clustering.fit_predict(data)
def apply_spectral(n_clusters, data):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    return spectral.fit_predict(data)
pca_data = pca_df[['PC1', 'PC2']]
clustering_labels = {
    'KMeans': kmeans_labels,
    'GMM': gmm_labels,
    'Hierarchical': agglomerative_labels,
    'DBSCAN': dbscan_labels,
    'Spectral': spectral_labels,
    'Mean Shift': mean_shift_labels
}
metrics_list = []
for model_name, labels in clustering_labels.items():
    if len(set(labels)) > 1:  # Ensure that there are at least two clusters
        silhouette = silhouette_score(pca_data, labels)
        calinski_harabasz = calinski_harabasz_score(pca_data, labels)
        davies_bouldin = davies_bouldin_score(pca_data, labels)
        
        metrics_list.append({
            'Model': model_name,
            'Silhouette Score': silhouette,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin
        })
# Select clustering algorithm
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Input for number of clusters (for applicable methods)
if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

# Input parameters based on selected clustering model
if cluster_model == 'KMeans':
    init_method = st.selectbox('Initialization method', ['k-means++', 'random'])
    max_iter = st.slider('Maximum iterations', min_value=100, max_value=1000, value=300)

elif cluster_model == 'MeanShift':
    bandwidth = st.slider('Bandwidth', min_value=0.8, max_value=1.5, value=1.18, step=0.1)

elif cluster_model == 'DBSCAN':
    eps = st.slider('Epsilon', min_value=0.1, max_value=0.5, value=0.32, step=0.1)
    min_samples = st.slider('Minimum samples', min_value=3, max_value=12, value=8)

elif cluster_model == 'Gaussian Mixture':
    covariance_type = st.selectbox('Covariance type', ['full', 'tied', 'diag', 'spherical'])
    max_iter = st.slider('Maximum iterations', min_value=100, max_value=1000, value=300)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    affinity = st.selectbox('Affinity', ['euclidean'])
    linkage = st.selectbox('Linkage', ['ward', 'complete', 'average', 'single'])

elif cluster_model == 'Spectral Clustering':
    affinity = st.selectbox('Affinity', ['nearest_neighbors', 'rbf'])
    n_neighbors = st.slider('Number of neighbors', min_value=2, max_value=20, value=10)

# Applying PCA for visualization
apply_pca = st.checkbox('Display PCA Visualization')

# Perform clustering based on selected model
if cluster_model == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters, init=init_method, n_init=10, max_iter=max_iter)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'MeanShift':
    clustering = MeanShift(bandwidth=bandwidth)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'DBSCAN':
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'Gaussian Mixture':
    clustering = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, max_iter=max_iter)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'Spectral Clustering':
    clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors)
    labels = clustering.fit_predict(df_scaled)

# Display PCA visualization
if apply_pca:
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    st.write('PCA Result:')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    # Handle long legends
    if len(np.unique(labels)) > 5:  # If more than 5 clusters, adjust the legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Standard legend position
    st.pyplot(fig)

# Silhouette Score
if len(np.unique(labels)) > 1:  # Silhouette score needs at least 2 clusters
    silhouette_avg = silhouette_score(df_scaled, labels)
    st.write(f'Silhouette Score: {silhouette_avg:.2f}')

# Number of records in each cluster
st.subheader('Number of records in each cluster:')
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.write(cluster_counts)

# Mean statistics for each cluster
st.subheader('Mean statistics for each cluster:')
cluster_mean_stats = pd.DataFrame(df).groupby(labels).mean()
st.write(cluster_mean_stats)

# Median statistics for each cluster
st.subheader('Median statistics for each cluster:')
cluster_median_stats = pd.DataFrame(df).groupby(labels).median()
st.write(cluster_median_stats)

