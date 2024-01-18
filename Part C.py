import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import numpy as np

# Load the dataset
data = pd.read_csv('UNSW-NB15_3.csv', header=None, encoding='us-ascii')
features = pd.read_csv('NUSW-NB15_features.csv', encoding='latin-1')
data.columns = features['Name']

# Select the significant features identified in Task 2
significant_features = ['dtcpb', 'stcpb', 'Sload', 'Dload', 'dbytes']  # Replace with actual feature names if necessary

# Filter the data for the significant features
data_filtered = data[significant_features]

# Use a random sample of the data to ensure the dendrogram plots are manageable
# Adjust the sample size as needed
data_sample = data_filtered.sample(n=100, random_state=42)

# Scatter plots for each pair of significant features
feature_pairs = list(combinations(significant_features, 2))
for (feature1, feature2) in feature_pairs:
    plt.figure(figsize=(8, 6))
    plt.scatter(data_sample[feature1], data_sample[feature2])
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Scatter Plot of {feature1} vs {feature2}')
    plt.savefig(f'scatter_{feature1}_{feature2}.png')
    plt.close()

# Perform Hierarchical Agglomerative Clustering and plot dendrogram
for k in [2, 5, 10]:
    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=k)
    predictions = clustering.fit_predict(data_sample)

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    plt.title(f'Dendrogram for k={k}')
    # Use the linkage function from scipy's hierarchy module
    Z = linkage(data_sample, method='ward')
    dendrogram(Z)
    plt.savefig(f'dendrogram_k{k}.png')
    plt.close()

    # Scatter plot of the cluster assignments for the first pair
    plt.figure(figsize=(8, 6))
    plt.scatter(data_sample[feature_pairs[0][0]], data_sample[feature_pairs[0][1]], c=predictions, cmap='rainbow')
    plt.xlabel(feature_pairs[0][0])
    plt.ylabel(feature_pairs[0][1])
    plt.title(f'Cluster Assignment for k={k}')
    plt.savefig(f'clusters_k{k}.png')
    plt.close()