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
dataset.hist(bins=20, figsize=(12, 10))
plt.suptitle('Histograms')
plt.tight_layout()

# Box plots
dataset.plot(kind='box', figsize=(12, 6))
plt.title('Box plots')
plt.tight_layout()

# Scatter plots (for selected features)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(dataset['age'], dataset['chol'])
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Cholesterol')
axes[0].set_title('Age vs Cholesterol')

axes[1].scatter(dataset['age'], dataset['thalach'])
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Max Heart Rate')
axes[1].set_title('Age vs Max Heart Rate')

plt.suptitle('Scatter plots')
plt.tight_layout()

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

categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
# Pandas method
dataset = pd.get_dummies(dataset, columns=categorical_cols, dtype=int)
print(dataset.head())

# Todo: Sklearn method

# Select features to be used for clustering
# fs = FeatureSelection(dataset, shifts=[25, 50, 75, 100], n_jobs=-1)
# metrics = fs.get_metrics()
# print(metrics)
# fs.plot_results(n_features=10)
selected_features = ['age', 'sex', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'trestbps',
                     'chol', 'thalach', 'exang', 'oldpeak']
dt = dataset[selected_features]

# Normalise or standardise numerical features
dt_scaled = dt.copy()
num_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
scaler = StandardScaler()
dt_scaled[num_cols] = scaler.fit_transform(dt_scaled[num_cols])
print(dt_scaled.head())


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
