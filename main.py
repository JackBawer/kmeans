import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import numpy as np

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
plt.show()

# Identify missing values and remove them if found

# Check for duplicated and remove them

# Encode categorical variables (One hot encoding)

# Select features to be used for clustering

# Normalise or standardise numerical features

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
