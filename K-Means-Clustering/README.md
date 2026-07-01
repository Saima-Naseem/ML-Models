# K-Means Clustering for Customer Segmentation

## Overview

This project demonstrates how to build a K-Means Clustering model to segment customers based on their Instagram Visit Score and Spending Rank. K-Means is an unsupervised machine learning algorithm that automatically groups similar observations without requiring target labels. The complete workflow includes data loading, feature selection, data visualization, optimal cluster selection, model training, cluster visualization, model evaluation, and interpretation.

## Dataset

The dataset contains the following columns:

| Column | Description |
|---------|-------------|
| User ID | Unique identifier for each customer |
| Instagram Visit Score | Numerical score representing Instagram activity |
| Spending Rank | Numerical score (0–100) representing customer spending |

User ID was excluded from model training because identifiers do not contain useful information for clustering.

## Algorithm

Algorithm: K-Means Clustering

Learning Type: Unsupervised Learning

Objective: Segment customers into meaningful groups based on similarities in Instagram activity and spending behaviour.

## Step 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
```

## Step 2: Load the Dataset

```python
df = pd.read_csv("dataset.csv")
```

## Step 3: Explore the Dataset

The following checks were performed before model training.

```python
df.head()

df.shape

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()
```

Purpose:

Verify that the dataset is clean and ready for clustering.

## Step 4: Select Features

```python
X = df[['Instagram visit score','Spending_rank']]
```

Only numerical features were selected.

User ID was intentionally excluded because it is only an identifier.

## Step 5: Visualize the Dataset

```python
plt.figure(figsize=(8,6))
plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1]
)
plt.xlabel("Instagram Visit Score")
plt.ylabel("Spending Rank")
plt.title("Customer Distribution")
plt.show()
```

Purpose:

Visualize the relationship between the two features before clustering.

## Step 6: Determine the Optimal Number of Clusters (Elbow Method)

```python
wcss = []

for k in range(1,11):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(X)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11),wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(range(1,11))
plt.grid(True)
plt.show()
```

Observation:

The elbow occurred at K = 3, indicating that three clusters provide the best balance between model simplicity and clustering quality.

## Step 7: Train the K-Means Model

```python
kmeans = KMeans(
    n_clusters=3,
    n_init=20,
    random_state=42
)

y_kmeans = kmeans.fit_predict(X)
```

Why n_init=20?

K-Means starts with randomly selected centroids. Increasing n_init allows the algorithm to try multiple starting positions and keep the best solution, resulting in a more stable model.

## Step 8: Visualize the Clusters

```python
plt.figure(figsize=(8,6))

plt.scatter(X.iloc[y_kmeans==0,0],X.iloc[y_kmeans==0,1],label="Cluster 1")
plt.scatter(X.iloc[y_kmeans==1,0],X.iloc[y_kmeans==1,1],label="Cluster 2")
plt.scatter(X.iloc[y_kmeans==2,0],X.iloc[y_kmeans==2,1],label="Cluster 3")

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    c="black",
    marker="X",
    label="Centroids"
)

plt.xlabel("Instagram Visit Score")
plt.ylabel("Spending Rank")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()
```

The visualization confirms that the customers were successfully grouped into three distinct clusters.

## Step 9: Evaluate the Model

Unlike supervised learning algorithms, K-Means has no target labels, so traditional metrics such as Accuracy, Precision, Recall, F1 Score, RMSE, MAE, and R² cannot be calculated.

Specialized clustering evaluation metrics were used instead.

### Silhouette Score

```python
score = silhouette_score(X,y_kmeans)
print(score)
```

Result:

**0.5967**

Interpretation:

Higher values indicate better-separated clusters.

The obtained score indicates good clustering performance.

### Davies-Bouldin Index

```python
dbi = davies_bouldin_score(X,y_kmeans)
print(dbi)
```

Result:

**0.5948**

Interpretation:

Lower values indicate better clustering.

The obtained value suggests compact and well-separated clusters.

### Calinski-Harabasz Index

```python
chi = calinski_harabasz_score(X,y_kmeans)
print(chi)
```

Result:

**4811.54**

Interpretation:

Higher values indicate stronger clustering quality.

The obtained value demonstrates good overall cluster separation.

## Final Model Performance

| Evaluation Metric | Result | Interpretation |
|-------------------|--------|----------------|
| Elbow Method | K = 3 | Optimal number of clusters |
| Silhouette Score | 0.5967 | Good cluster separation |
| Davies-Bouldin Index | 0.5948 | Good cluster compactness |
| Calinski-Harabasz Index | 4811.54 | Strong clustering quality |

## Cluster Interpretation

Cluster 1 represents customers with relatively high Instagram activity but comparatively lower spending.

Cluster 2 represents customers with moderate Instagram activity and moderate spending.

Cluster 3 represents highly engaged customers with high spending behaviour and may represent premium or loyal customers.

## Important Insights

K-Means is an unsupervised learning algorithm and does not require target labels.

User IDs should never be used as clustering features.

The Elbow Method should always be performed before training the final model.

WCSS always decreases as the number of clusters increases, so the smallest WCSS does not necessarily indicate the best model.

The optimal value of K is selected where the WCSS curve begins to flatten.

Centroids represent the average position of each cluster and are not actual observations.

Silhouette Score measures how well observations fit within their assigned cluster.

Davies-Bouldin Index evaluates cluster compactness and separation.

Calinski-Harabasz Index measures overall clustering quality.

Increasing n_init improves model stability by evaluating multiple random centroid initializations.

A higher evaluation score alone should not determine the final number of clusters. Visual inspection and business interpretation are equally important.

## Common Beginner Mistakes

Including identifier columns, such as User ID, as clustering features.

Choosing K without using the Elbow Method.

Assuming that a smaller WCSS automatically means a better model.

Trying to calculate Accuracy, Precision, Recall, F1 Score, RMSE, or R² for a clustering model.

Forgetting to retrain the model after changing K before accessing cluster_centers_.

Assuming that centroids represent actual customers.

Interpreting Cluster 1, Cluster 2, and Cluster 3 as ranked categories instead of arbitrary labels.

## Technologies Used

Python

Google Colab

Pandas

NumPy

Matplotlib

Scikit-learn

## Conclusion

A K-Means Clustering model was successfully developed to segment customers using Instagram Visit Score and Spending Rank.
The Elbow Method identified K = 3 as the optimal number of clusters, while the Silhouette Score (0.5967), Davies-Bouldin Index (0.5948),
and Calinski-Harabasz Index (4811.54) confirmed that the model produced compact, well-separated, and meaningful customer segments.
The complete workflow demonstrates a practical implementation of unsupervised learning suitable for customer segmentation, portfolio projects,
and introductory machine learning applications.
