import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from featclus.model import FeatureSelection

###############################################################################
# Part 1: Data Exploration
###############################################################################

# Load dataset using Pandas
dataset = pd.read_csv('./heart.csv')

# Display:
#   First rows of dataset
print(dataset.head())
#   Dataset shape (number of rows and columns)
rows, cols = dataset.shape
print(f'Rows: {rows}, Columns: {cols}')
#   Column names and data types
print(dataset.dtypes)

# Remove target column
dataset.drop('target', axis=1, inplace=True)
print(dataset.columns.tolist())

# Compute basic statistics (mean, standard deviation, min, max)
stats = {'mean': None, 'std_dev': None, 'min': None, 'max': None}

np.set_printoptions(legacy='1.25')  # to avoid printing np.float64

for col in dataset.columns:
    stats['mean'] = round(dataset[col].mean(), 2)
    stats['std_dev'] = round(dataset[col].std(), 2)
    stats['min'] = round(dataset[col].min(), 2)
    stats['max'] = round(dataset[col].max(), 2)
    print(f'{col} = {stats}')

# Feature distributions analysis and visualisation

# Histograms
dataset.hist()
plt.suptitle('Histograms')

# Box plots
dataset.plot(kind='box')
dataset.boxplot()
plt.suptitle('Box plots')

# Scatter plots (for selected features)
dataset.plot(kind='scatter', x='age', y='chol')  # not sure which features should be selected
plt.suptitle('Scatter plots')

# Display plots
# plt.show()

# Identify missing values
missing_vals_cnt = dataset.isnull().sum()
print(f'Missing values: \n{missing_vals_cnt}')

# Drop rows with missing values
dataset.dropna(inplace=True)
# Drop rows where all values are missing
dataset.dropna(inplace=True, how='all')

# Remove irrelevant columns (patient ids)
# There aren't any irrelevant columns so I'm not sure what I'm supposed to do

# Check for duplicates and remove them
dataset.drop_duplicates(subset=None, inplace=True)

# Encode categorical variables (One hot encoding)

categorical_cols = ['fbs', 'exang']
# Pandas method
dataset = pd.get_dummies(dataset, columns=categorical_cols, dtype=int)
print(dataset.head())

# Todo: Sklearn method

# Select features to be used for clustering
# fs = FeatureSelection(dataset, shifts=[25, 50, 75, 100], n_jobs=-1)
# metrics = fs.get_metrics()
# print(metrics)
# fs.plot_results(n_features=10) # I am confused by the number of clusters that I should use
dt = dataset[['age', 'sex', 'cp', 'chol', 'fbs_0', 'fbs_1', 'restecg', 'oldpeak', 'slope', 'ca']]

# Normalise or standardise numerical features
scaler = StandardScaler()
dt_scaled = scaler.fit_transform(dt)
print(dt_scaled)


###############################################################################
# Part 2: Model Training
###############################################################################

# Split dataset into training set (80%) and a test set (20%)

# Apply Elbow method and choose an appropriate number of clusters k

# Import K-Means algorithm from sklearn and train it using the training dataset

# Obtain cluster labels for both training and test data

# Display the cluster centers

###############################################################################
# Part 3: Evaluation
###############################################################################

# Evaluate the clustering using appropriate metrics: inertia, silhouette score,
# Davies-Boulding Index

# Compare metric values for different choices of k

# Interpret what these metrics say about cluster quality

# Visualise the clustering results using matplotlib plots, color data points
# according to their assigned cluster, and visualise the cluster centroids
