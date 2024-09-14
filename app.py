import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import estimate_bandwidth
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min

# Title of the app
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
df.drop(columns=['Height', 'Weight'], inplace=True)

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
st.pyplot(fig)

# Removing outliers based on Z-score
from scipy import stats
for feature in numerical_features:
    df = df[(np.abs(stats.zscore(df[feature])) < 3)]

# Removing duplicates
df = df.drop_duplicates()

# Handling skewness
for feature in numerical_features:
    skewness = df[feature].skew()
    if skewness > 1:
        df[feature] = np.log1p(df[feature])  # Apply log1p transformation to handle zero values

# Encoding categorical features
nominal_features = ['Gender', 'FAVC_CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'CAEC']
ordinal_features = ['NObeyesdad', 'Age_Group']
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_nominal = onehot_encoder.fit_transform(df[nominal_features])
encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=onehot_encoder.get_feature_names_out(nominal_features))
encoded_nominal_df.reset_index(drop=True, inplace=True)

label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])
df['Age_Group'] = label_encoder.fit_transform(df['Age_Group'])
df_numerical_only = df.drop(columns=nominal_features).reset_index(drop=True)
df_encoded = pd.concat([df_numerical_only, encoded_nominal_df], axis=1)

# Correlation and feature reduction
target_variable = 'NObeyesdad'
correlation_with_target = df_encoded.corr()[target_variable]
threshold = 0.1
weak_corr_features = correlation_with_target[correlation_with_target.abs() < threshold].index.tolist()
df_reduced = df_encoded.drop(columns=weak_corr_features)

# Define high correlation features and drop them
high_corr_features = ['FAVC_CAEC_yes_Always', 'FAVC_CAEC_no_Frequently', 'CALC_no', 'CALC_Frequently', 'MTRANS_Walking']
weak_corr_features = ['FAVC', 'FCVC', 'NCP', 'CH2O', 'TUE', 'Gender_Male', 'FAVC_CAEC_no_no', 'FAVC_CAEC_yes_no', 'SMOKE_yes', 'SCC_yes', 'CALC_Frequently', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking', 'CAEC_no']
columns_to_drop = list(set(high_corr_features + weak_corr_features))
clean_df = df_encoded.drop(columns=columns_to_drop)
further_columns_to_drop = ['FAVC_CAEC_yes_Sometimes', 'CAEC_Sometimes', 'Healthy_Score', 'CAEC_Frequently']
clean_df_final_minimized = clean_df.drop(columns=further_columns_to_drop)

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_df_final_minimized)
scaled_df = pd.DataFrame(scaled_data, columns=clean_df_final_minimized.columns)

# PCA Transformation
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Perform clustering and visualization
cluster_model = st.selectbox('Select a clustering model', 
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering'))

# Input parameters based on selected clustering model
if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

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

# Perform clustering based on selected model
if cluster_model == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters, init=init_method, n_init=10, max_iter=max_iter)
    labels = clustering.fit_predict(scaled_data)

elif cluster_model == 'MeanShift':
    clustering = MeanShift(bandwidth=bandwidth)
    labels = clustering.fit_predict(scaled_data)

elif cluster_model == 'DBSCAN':
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(scaled_data)

elif cluster_model == 'Gaussian Mixture':
    clustering = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, max_iter=max_iter)
    labels = clustering.fit_predict(scaled_data)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = clustering.fit_predict(scaled_data)

elif cluster_model == 'Spectral Clustering':
    clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    labels = clustering.fit_predict(scaled_data)

# Visualization
fig, ax = plt.subplots()
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=labels, cmap='viridis', marker='o')
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Cluster Visualization')
st.pyplot(fig)

# Metrics
if st.checkbox('Show metrics'):
    silhouette_avg = silhouette_score(scaled_data, labels)
    calinski_harabasz = calinski_harabasz_score(scaled_data, labels)
    davies_bouldin = davies_bouldin_score(scaled_data, labels)
    
    st.write(f'Silhouette Score: {silhouette_avg}')
    st.write(f'Calinski-Harabasz Score: {calinski_harabasz}')
    st.write(f'Davies-Bouldin Score: {davies_bouldin}')

