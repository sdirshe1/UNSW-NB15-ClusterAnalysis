import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Function to convert non-numeric columns to integers
def convert_non_numeric_to_int(df):
    label_encoder = LabelEncoder()
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    for column in non_numeric_columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Load the dataset
data = pd.read_csv('UNSW-NB15_3.csv', header=None, encoding='us-ascii')
features = pd.read_csv('NUSW-NB15_features.csv', encoding='latin-1')

# Assign column names from the features file
data.columns = features['Name']

# Convert non-numeric values to integers
data = convert_non_numeric_to_int(data)

# Impute NaN values
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Apply K-means clustering
k_values = [2, 5, 10]
overall_top_features = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_imputed)
    data_imputed[f'cluster_{k}'] = kmeans.labels_

    # Extract and display sample data for each cluster
    print(f"\nSample data for k={k}:")
    sample_data = data_imputed.groupby(f'cluster_{k}').apply(lambda x: x.head(20))
    print(sample_data)

    # Save the sample data to a CSV file
    sample_data_filename = f'sample_data_k{k}.csv'
    sample_data.to_csv(sample_data_filename, index=False)
    print(f"Sample data saved to {sample_data_filename}")

    # Plotting clusters for each value of k
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_imputed, x='dur', y='sbytes', hue=f'cluster_{k}')
    plt.title(f'Clusters for k={k} based on duration and source bytes')
    plt.xlabel('Duration')
    plt.ylabel('Source Bytes')
    plt.legend(title=f'Cluster for k={k}')
    scatter_filename = f'scatter_plot_k{k}.png'
    plt.savefig(scatter_filename)
    plt.close()

    # Calculate variance and mean for the entire dataset
    variance = data_imputed.var().sort_values(ascending=False)
    top_features = variance.head(5).index
    top_feature_values = data_imputed[top_features].mean()

    # Aggregate the top features across all clusters
    overall_top_features[k] = top_features.tolist()

    # Plot top feature names and their average values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features, y=top_feature_values)
    plt.title(f'Top 5 Features for k={k}')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    bar_filename = f'bar_graph_k{k}.png'
    plt.savefig(bar_filename)
    plt.close()

# Display the overall top features for each k
print("Overall top features for each k:")
for k, features in overall_top_features.items():
    print(f"k={k}: {features}")
