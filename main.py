import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
from featclus.model import FeatureSelection
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

###############################################################################
# Part 1: Data Exploration
###############################################################################


# Q1: Load dataset using Pandas

# Then load the dataset
DATA_DIR = Path(__file__).parent
HEART_CSV = DATA_DIR / 'heart.csv'

dataset = pd.read_csv(HEART_CSV)

print('\n--- Dataset loaded ---')

dt = dataset.copy()

# Q2: Display:
print('\n--- Display ---')
#   First rows of dataset
print(dt.head())
#   Dataset shape (number of rows and columns)
rows, cols = dt.shape
print(f'Rows: {rows}, Columns: {cols}')
#   Column names and data types
print(dt.dtypes)

# Q3: Remove target column
dt.drop('target', axis=1, inplace=True)
print('\n--- Target column removed ---')
print(dt.columns.tolist())

# Q4: Compute basic statistics (mean, standard deviation, min, max)
np.set_printoptions(legacy='1.25')  # to avoid printing np.float64

print('\n--- Basic statics computed ---')
stats = {}
for col in dt.columns:
    stats[col] = {
            'mean': round(dt[col].mean(), 2),
            'std_dev': round(dt[col].std(), 2),
            'min': round(dt[col].min(), 2),
            'max': round(dt[col].max(), 2)
    }

# Q5: Feature distributions analysis and visualisation

print('\n--- Feature distribution visualisation ---')
# Histograms
dt.hist(bins=20, figsize=(12, 10))
plt.suptitle('Histograms')
plt.tight_layout()
# plt.show()

# Box plots
dt.plot(kind='box', figsize=(12, 6))
plt.title('Box plots')
plt.tight_layout()
# plt.show()

# Scatter plots (for selected features)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(dt['age'], dt['chol'])
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Cholesterol')
axes[0].set_title('Age vs Cholesterol')

axes[1].scatter(dt['age'], dt['thalach'])
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Max Heart Rate')
axes[1].set_title('Age vs Max Heart Rate')

plt.suptitle('Scatter plots')
plt.tight_layout()
# plt.show()

# Q6: Identify missing values
print('\n--- Missing values identification and removal ---')
missing_vals_cnt = dt.isnull().sum()
print(f'Missing values: \n{missing_vals_cnt}')

# Drop rows with missing values
dt.dropna(inplace=True)

# Q7: Remove irrelevant columns (e.g. patient ids)
# There aren't any irrelevant columns
print('\n--- Irrelavant columns removed (none) ---')

# Q8: Check for duplicates and remove them
dt.drop_duplicates(subset=None, inplace=True)
print('\n--- Duplicates removal ---')

# Q9: Encode categorical variables (One hot encoding)

categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
# Pandas method
dt = pd.get_dummies(dt, columns=categorical_cols, dtype=int)
print('\n--- One hot encoding perfomed on categorial columns ---')
print(dt.head())

# Q10: Select features to be used for clustering
fs = FeatureSelection(dt, shifts=[25, 50, 75, 100], n_jobs=-1)
metrics = fs.get_metrics()
print(metrics)
fs.plot_results(n_features=10)
selected_features = ['age', 'sex', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'trestbps',
                     'chol', 'thalach', 'exang', 'oldpeak']
dt = dt[selected_features]
print('\n--- Feature selection ---')

# Q11: Normalise or standardise numerical features
dt_scaled = dt.copy()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
dt_scaled[numerical_cols] = scaler.fit_transform(dt_scaled[numerical_cols])
print('\n--- Numerical features scaled ---')
print(dt_scaled.head())

###############################################################################
# Part 2: Model Training
###############################################################################

# Q1: Split dataset into training set (80%) and a test set (20%)
X_train, X_test = train_test_split(dt_scaled, test_size=0.2, random_state=42)
print('\n--- Dataset split ---')
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Q2: Apply Elbow method and choose an appropriate number of clusters k
#   Initialise kmeans parameters
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
}

#   Array to hold SSE for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_train)
    sse.append(kmeans.inertia_)

#   Visualise results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
# plt.show()
print('\n--- Elblow method applied ---')


# Q3: Import K-Means algorithm from sklearn and train it using the training dataset
kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=1)
kmeans.fit(X_train)
print('\n--- Dataset trained ---')


# Q4: Obtain cluster labels for both training and test data
print('\n--- Cluster labes ---')
train_labels = kmeans.labels_
test_labels = kmeans.predict(X_test)
print(train_labels)
print(test_labels)

print("\nTraining set - Cluster distribution:")
unique, counts = np.unique(train_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples")

print("\nTest set - Cluster distribution:")
unique, counts = np.unique(test_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples")

# Q5: Display the cluster centres
print('\n--- Cluster centres ---')
print(kmeans.cluster_centers_)

###############################################################################
# Part 3: Evaluation
###############################################################################


# Q1: Clustering evaluation
print("\n--- Check the metrics of our model with k=3 ---")

a = kmeans.inertia_
b = silhouette_score(X_train, train_labels)
c = davies_bouldin_score(X_train, train_labels)

print("inertia result : ", a)
print("silhouette score : ", b)
print("davies-bouldin index : ", c)

# Analysis of the metrics
print("\n the results for the chosen k (k=3) :")
print("\n- the inertia shows that patients are grouped around 3 centres but the groups are a little wide")
print("- the silhouette score is near 0.17 showing clusters overlap")
print("- the davies-bouldin index is above 1, it means the groups are not that separated")

# Q2: Comparison of metric values for different choies of k
print("\n--- Test for other k values ---")

for k_test in [2, 4, 5]:
    m_test = KMeans(n_clusters=k_test)
    l_test = m_test.fit_predict(X_train)
    sil_test = silhouette_score(X_train, l_test)
    db_test = davies_bouldin_score(X_train, l_test)
    print("k =", k_test, ": silhouette =", sil_test, "/ db =", db_test)

# Q3: Interpretation of of these metrics about cluster quality
print("\n--- Final conclusion on the best k ---")
print("From our comparison, k=2 seems to be the best choice according to metrics")
print("it gives the highest silhouette score and the lowest davies-bouldin index")
print("this means k=2 creates the most stable groups, even if k=3 gives more medical details")

# Q4: Visualisation and coloring according data points according to their assigned centre
print("\n--- Visualising the results ---")
plt.figure()

# Plotting Age vs Cholesterol
# iloc is to select the columns from our scaled data
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 7], c=train_labels, cmap='Set1')

# Adding the centroids
cen = kmeans.cluster_centers_
plt.scatter(cen[:, 0], cen[:, 7], s=180, c='black', marker='X', label='centroids')

plt.title("visual analysis of our patient groups")
plt.xlabel("age")
plt.ylabel("cholesterol")
plt.legend()
# plt.show()

plt.show()  # Display all plots
