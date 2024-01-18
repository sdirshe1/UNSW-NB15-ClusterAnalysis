# UNSW-NB15-ClusterAnalysis
# Overview:
UNSW-NB15-ClusterAnalysis is a detailed exploration of clustering techniques using Python on the UNSW-NB15 dataset, focusing on network intrusion detection.

# Key Components:

K-Means Clustering (Part A):
Import libraries and preprocess data:
``` import pandas as pd```
```from sklearn.cluster import KMeans```

Apply K-means and visualize:

```kmeans = KMeans(n_clusters=5)```
```kmeans.fit(data)```

Feature Selection Using Correlation Matrix (Part B):

Create and visualize correlation matrix:

```import seaborn as sns```
```correlation_matrix = data.corr()```
```sns.heatmap(correlation_matrix)```

Hierarchical Agglomerative Clustering (Part C):

Perform HAC and create dendrograms:

```from scipy.cluster.hierarchy import dendrogram, linkage```
```linked = linkage(data, method='ward')```
```dendrogram(linked)```
