#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 20:05:03 2023

@author: vladasliusar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# Load the dataset
file_path = '/Users/vladasliusar/Downloads/yusuf_demo.csv'
data = pd.read_csv(file_path)

data_half = data.sample(frac=0.5, random_state=42)

# Display basic information about the dataset
print(data.info())
print(data.head())

# Handle Missing Values
# Check for missing values
print(data.isnull().sum())

# Option 1: Drop rows with missing values
data_cleaned = data_half.dropna()
data_cleaned = data_cleaned.drop(columns = ['CUSTOMER_HK', 'D_LAST_INTERACTION_REF_DT', 'APPOINTMENT_SCHEDULING_DT'])


# Convert Categorical Variables to Numerical
# Identify categorical columns
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns

# One-Hot Encoding
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns)

# Display the encoded data
print(data_encoded.info())
print(data_encoded.head())

# Scale the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Convert the scaled data back to a DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=data_encoded.columns)

# Display the scaled data
print(data_scaled_df.info())

# Determine the Optimal Number of Clusters using the Elbow Method
def plot_elbow_method(data, max_clusters=10):
    sse = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_half)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters+1), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()

plot_elbow_method(data_scaled_df)

# Determine the Optimal Number of Clusters using the Silhouette Method
def plot_silhouette_method(data, max_clusters=10):
    silhouette_avg = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg.append(silhouette_score(data, cluster_labels))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters+1), silhouette_avg, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()

plot_silhouette_method(data_scaled_df)

# Assume optimal_k is determined based on the plots (e.g., from the Elbow or Silhouette Method)
optimal_k = 5  # Replace with the number determined from the methods

# Perform K-Means Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data_scaled_df['cluster'] = kmeans.fit_predict(data_scaled_df)

# Evaluate Clustering
silhouette_avg = silhouette_score(data_scaled_df.drop('cluster', axis=1), data_scaled_df['cluster'])
print(f'Silhouette Score for k={optimal_k}: {silhouette_avg:.2f}')

# Visualize Clusters (using the first two principal components if dimensionality is high)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled_df.drop('cluster', axis=1))
data_pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
data_pca_df['cluster'] = data_scaled_df['cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='viridis', data=data_pca_df, legend='full')
plt.title('K-Means Clustering Results (PCA)')
plt.show()

loading_scores = pd.Series(pca.components_[0], index=data_encoded.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# Display top features contributing to the first principal component
top_n = 10
top_pca_features = sorted_loading_scores.head(top_n)
print("Top PCA features contributing to the first principal component:")
print(top_pca_features)


cluster_summary = data_scaled_df.groupby('cluster').agg(['mean', 'median', 'std', 'min', 'max']).reset_index()

# Analyze Feature Contribution
def calculate_feature_contributions(data, cluster_col='cluster'):
    cluster_means = data.groupby(cluster_col).mean()
    overall_mean = data.drop(columns=[cluster_col]).mean()
    
    feature_contributions = cluster_means.subtract(overall_mean).abs()
    feature_contributions = feature_contributions.sum(axis=0).sort_values(ascending=False)
    return feature_contributions

feature_contributions = calculate_feature_contributions(data_scaled_df)
print("Top features contributing to cluster formation:")
print(feature_contributions)

# Visualize Top Features
top_n = 20  # Number of top features to visualize
top_features = feature_contributions.head(top_n).index

loading_scores = pd.Series(pca.components_[0], index=data_encoded.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# Display top features contributing to the first principal component
top_n = 10
top_pca_features = sorted_loading_scores.head(top_n)
print("Top PCA features contributing to the first principal component:")
print(top_pca_features)
