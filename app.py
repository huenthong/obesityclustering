import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.title('Obesity Risk Clustering App')

# Load data
df = pd.read_csv('ObesityDataSet.csv')

# Feature Engineering
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Adolescent', 'Adult', 'Elderly'])
df['FAVC_CAEC'] = df['FAVC'] + "_" + df['CAEC']
df['Healthy_Score'] = df['FCVC'] + df['FAF'] - df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})
df.drop(columns=['Height', 'Weight'], inplace=True)  # BMI replaces these

# Preprocessing and Outlier Removal
numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Healthy_Score']
for feature in numerical_features:
    df = df[(np.abs(stats.zscore(df[feature])) < 3)]

df = df.drop_duplicates()

# Encoding Categorical Features
nominal_features = ['Gender', 'FAVC_CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'CAEC'] 
ordinal_features = ['NObeyesdad', 'Age_Group']

onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_nominal = onehot_encoder.fit_transform(df[nominal_features])
encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=onehot_encoder.get_feature_names_out(nominal_features))
df['NObeyesdad'] = LabelEncoder().fit_transform(df['NObeyesdad'])
df['Age_Group'] = LabelEncoder().fit_transform(df['Age_Group'])

# Combine Encoded and Numerical Data
df_numerical_only = df.drop(columns=nominal_features).reset_index(drop=True)
df_encoded = pd.concat([df_numerical_only, encoded_nominal_df], axis=1)

# PCA and Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded.drop(columns=['NObeyesdad']))
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Clustering Options in Streamlit
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Number of Clusters Input
n_clusters = None
if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

# Clustering Model Selection with Pre-Fine-tuned Parameters
if cluster_model == 'KMeans':
    kmeans_param_grid = {"n_clusters": [2, 3, 4, 5]}
    kmeans = KMeans(n_clusters=n_clusters if n_clusters else 3, random_state=42)
    pca_df['Cluster'] = kmeans.fit_predict(X_pca)

elif cluster_model == 'MeanShift':
    bandwidth_values = np.linspace(0.8, 1.5, num=10)
    bandwidth = st.slider('Bandwidth', min_value=0.5, max_value=2.0, value=1.0)
    mean_shift = MeanShift(bandwidth=bandwidth)
    pca_df['Cluster'] = mean_shift.fit_predict(X_pca)

elif cluster_model == 'DBSCAN':
    eps_values = np.linspace(0.1, 0.5, num=10)
    min_samples_values = np.arange(3, 12)
    eps = st.slider('Epsilon', min_value=0.1, max_value=2.0, value=0.5)
    min_samples = st.slider('Minimum samples', min_value=5, max_value=20, value=10)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    pca_df['Cluster'] = dbscan.fit_predict(X_pca)

elif cluster_model == 'Gaussian Mixture':
    gmm_param_grid = {"n_components": [2, 3, 4]}
    gmm = GaussianMixture(n_components=n_clusters if n_clusters else 3, random_state=42)
    pca_df['Cluster'] = gmm.fit_predict(X_pca)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    linkage_options = ['ward', 'complete', 'average', 'single']
    linkage = st.selectbox('Linkage criterion', linkage_options, index=0)
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters if n_clusters else 3, linkage=linkage)
    pca_df['Cluster'] = agg_clustering.fit_predict(X_pca)

elif cluster_model == 'Spectral Clustering':
    affinity_options = ['rbf', 'nearest_neighbors']
    affinity = st.selectbox('Affinity', affinity_options, index=0)
    spectral = SpectralClustering(n_clusters=n_clusters if n_clusters else 3, affinity=affinity, random_state=42)
    pca_df['Cluster'] = spectral.fit_predict(X_pca)

# Evaluation Metrics
silhouette = silhouette_score(X_pca, pca_df['Cluster'])
calinski_harabasz = calinski_harabasz_score(X_pca, pca_df['Cluster'])
davies_bouldin = davies_bouldin_score(X_pca, pca_df['Cluster'])

# Display Metrics
st.write(f"Silhouette Score: {silhouette}")
st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
st.write(f"Davies-Bouldin Index: {davies_bouldin}")

# Plot Clusters
fig, ax = plt.subplots()
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df['Cluster'], palette='tab10', ax=ax)
st.pyplot(fig)

