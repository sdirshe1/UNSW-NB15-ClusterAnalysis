import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load features
features = pd.read_csv('NUSW-NB15_features.csv', encoding='latin-1')
# Make sure the feature names are in the 'Name' column
column_names = features['Name'].tolist()

# Load the dataset
data = pd.read_csv('UNSW-NB15_3.csv', header=None, encoding='utf-8', names=column_names)

# Convert non-numeric values to integers
label_encoder = LabelEncoder()
non_numeric_columns = data.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str))  # Convert to string to ensure consistency

# Impute NaN values
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Create and plot the correlation matrix
correlation_matrix = data_imputed.corr(method='pearson')
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_matrix_heatmap.png')
plt.close()

# Identify the top correlated features
# This gets the top absolute correlations without duplicates
top_correlations = correlation_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
top_correlated_pairs = top_correlations[1:]  # Exclude the self-correlation pair (1.0 values)

# Print top correlated pairs
print("Top Absolute Correlated Pairs:\n", top_correlated_pairs.head(20))

# You would then extract the feature names from these pairs
# For example, if you are interested in the top 10 features based on the highest correlation values:
top_features_correlation = top_correlated_pairs.head(10).index.tolist()
print("Top features based on correlation:\n", top_features_correlation)

# Now compare these top features from the correlation analysis with those selected in Task 1
# For the comparison, you should have a list of features from Task 1, which might look something like this:
task1_top_features = ['dtcpb', 'stcpb', 'Sload', 'Dload', 'dbytes']

# Find common features
common_features = set(top_features_correlation).intersection(task1_top_features)
print("Common significant features from Task 1 and Task 2:\n", common_features)

# Based on the common features, make conclusions about their significance
if common_features:
    print("The following features are significant for machine learning as they appear in both tasks:", common_features)
else:
    print("No common features were found between the two tasks.")
