## Supervised Learning 

# 6.1_Unsupervised_Learning_K_Means_Clustering

**Algorithm Description**
K-Means clustering is an unsupervised learning algorithm that partitions a set of unlabeled data into K clusters by iteratively refining K centroids: it begins by seeding centroids (here using the “k-means++” strategy to spread them out), then assigns each point to its nearest centroid (by Euclidean distance), and recomputes each centroid as the mean of the points assigned to it. This assignment–update cycle repeats until cluster memberships no longer change or a maximum number of iterations is reached. A key extension in this notebook is MiniBatchKMeans, which uses small, random subsets (“mini-batches”) of the data at each update to dramatically reduce computation time on large datasets, trading off only a slight loss in cluster quality. The primary hyperparameters are the number of clusters (n_clusters), initialization method (init), number of restarts (n_init), and maximum iterations (max_iter), and the overall time complexity scales roughly as O(N·K·I), where N is the number of samples, K the chosen clusters, and I the average iterations to converge.


**Dataset Summary**
We use the Fashion-MNIST dataset, preprocessed and stored under data/processed/fashion_mnist1/:

- X_fashion.npy  
  - Shape: (70000, 784)  
  - Each row is a 28×28 grayscale image of a clothing item, flattened to length 784.  
  - Values normalized to [0, 1] by dividing original [0,255] pixels by 255.  
  - For initial experiments we subsample 10000 points (random state = 42) to speed up clustering.

- y_fashion.npy 
  - Shape: (70 000,)  
  - Integer labels 0–9 for the ten classes:  
    0 = T-shirt/top, 1 = Trouser, 2 = Pullover, 3 = Dress, 4 = Coat,  
    5 = Sandal, 6 = Shirt, 7 = Sneaker, 8 = Bag, 9 = Ankle boot.

We cluster X_fashion, then compare the learned labels to the true labels via confusion matrices and silhouette scores.


**Insight**

1. Cluster compactness & separation
- The inertia falls sharply from ~8.0 at K=1 to ~5.6 at K=4, then slowly decreases after K≈6, indicating diminishing returns on adding more clusters.
- The silhouette score peaks at 0.168 for K=2 (best overall separation), drops to a minimum of 0.107 at K=6, then rebounds slightly to 0.124 at K=9, reflecting tighter grouping for very small K but more fragmented clusters as K increases.

2. Moderate cohesion for K=10
- At the chosen K=10 (to match the ten true classes), inertia is ~4.55 and the silhouette score is ~0.119, showing only modest within‐cluster cohesion when splitting into ten groups.

3. Distinct prototype clarity
- The reshaped centroids for certain clusters (e.g., boots, sneakers, trousers, bags) appear as recognizable silhouettes, demonstrating that K-Means learns meaningful “prototypes” for items with unique outlines.
- Centroids for visually similar garments (T-shirts, pullovers, coats) are blurrier and less distinct, revealing that those classes share very similar raw-pixel patterns.

4. Purity & confusion patterns
- Overall clustering purity is 0.5531, so about 55% of assignments align with ground truth.
- The confusion matrix shows strong one-to-one mappings for classes with unique shapes—trousers (label 1→cluster 2), sneakers (7→7), bags (8→8), ankle‐boots (9→9)—while garments like pullovers (2), coats (4), and shirts (6) scatter across multiple clusters.

-------------------------------------------------------------------

# 6.2_Unsupervised_Learning_DBSCAN

**Algorithm Description**
DBSCAN groups data by linking points in high-density regions: it examines each point’s ε-neighborhood and marks those with at least min_samples neighbors as “core” points, absorbs “border” points that lie within ε of a core but lack sufficient neighbors themselves, and labels all remaining points as noise. By relying solely on density rather than a preset cluster count, DBSCAN can discover arbitrarily shaped clusters and automatically detect outliers, demonstrating how tuning eps and min_samples controls cluster granularity and noise identification.


**Dataset Summary**
All experiments use the preprocessed Fashion-MNIST arrays 

- X_fashion.npy (as stated above)
  - Before clustering, we apply StandardScaler so that each pixel feature has zero mean and unit variance.

- y_fashion.npy (as stated above)
  - Used to compute silhouette scores and to color the 2D PCA plots for visualization.


**Insight**
1. No clusters found with default settings
- The log reports 0 clusters and 70000 noise points, so DBSCAN labeled every sample as noise under eps=0.5, min_samples=5.

2. PCA‐2D shows a continuous cloud
- The scatter of the 70000 PCA‐reduced points forms one large, smoothly varying shape with no clear dense cores—DBSCAN sees no region dense enough to form a cluster at the chosen eps.


-------------------------------------------------------------------

# 6.3_Unsupervised_Learning_Principal_Component_Analysis

**Algorithm Description**
Principal Component Analysis is an unsupervised linear dimensionality-reduction method that transforms a set of possibly correlated variables into a smaller number of uncorrelated variables called principal components. It identifies orthogonal directions (components) of maximal variance in the data by computing the eigenvectors of the covariance matrix; these directions capture the most significant underlying structure while discarding noise. By projecting the original high-dimensional data onto the top k components—ranked by explained variance ratio—PCA yields a low-dimensional representation that preserves as much variance as possible, facilitates visualization, and can improve downstream learning tasks by reducing overfitting.

**Dataset Summary**
Preprocessed Fashion-MNIST (X_fashion.npy & y_fashion.npy) as stated above

**Insight**
1. Limited variance captured by first two PCs
- PC1 explains 22.05% and PC2 14.40%, so together they account for only 36.45% of the total variance. This means a 2D projection retains just over a third of the data’s information content.

2. Long tail of component contributions
- The explained-variance ratio drops sharply after PC2 (PC3≈5.46%, PC4≈5.11%), and by PC10 it’s down to ~1.32%. We need 50 PCs to capture 80.07% of variance, illustrating that Fashion-MNIST’s pixel space is intrinsically high-dimensional.

3. Elbow in cumulative curve around 15–20 PCs
- The cumulative-variance plot starts flattening noticeably after about 15–20 components, suggesting diminishing returns: each additional PC beyond this adds only marginal explained variance.

4. Partial class separation in 2D
- The scatter of the 70000 PCA-projected points colored by true labels shows that some classes (e.g. trousers, sandals) form denser, more distinct regions along PC1/PC2, while most others overlap heavily. No class is perfectly isolated in 2D.


-------------------------------------------------------------------

# 6.4_Unsupervised_Learning_SVD

**Algorithm Description**
Singular Value Decomposition factorizes a matrix X∈R^n*m into three matrices UΣV^T, where columns of U and V are the left and right singular vectors and Σ is the diagonal matrix of singular values. In this implementation, we use scikit‐learn’s TruncatedSVD to perform dimensionality reduction on sparse or dense data without centering. By selecting the top k singular values, we project the original data into a lower-dimensional space that captures the directions of maximum variance. The algorithm leverages randomized techniques under the hood for efficiency on large datasets and returns the transformed features along with explained variance ratios for each component.

**Dataset Summary**
X_fashion.npy (as stated above)


**Insight**
1. Rapid error decay
- With just k=5 components, the reconstructed image is visibly blurred and yields a high MSE of 0.0898.
- By k=15, MSE plummets to 0.0018, indicating that most of the image structure is captured in the first 15 singular vectors.

2. Compression vs. fidelity trade-off
- At k=5, the compression ratio is 0.36× (36% of original), but the distortion is large.
- At k=15, you achieve near-perfect fidelity (MSE≈0.002) but the data size slightly exceeds the original (ratio ≈1.09×).
- Larger k (30, 50) give true zero reconstruction error but at the cost of 2–3.6× the original storage.

3. Visual quality thresholds
- The k=5 reconstruction loses fine details in the sweater and text.
- At k=15, major contours and even small text are clearly legible.
- Beyond k=15, additional components produce negligible visible improvements.

4. Diminishing returns
- After k≈15, increasing k yields almost zero further error reduction, yet storage grows linearly with k.






