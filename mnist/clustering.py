import numpy as np
import hdbscan as hd
import sklearn.cluster as cluster
import sklearn.mixture as mixture

# loading umap data
x_pred = np.load('./data/x_true.npy')
embedding = np.reshape(x_pred, (x_pred.shape[0], 1 * 28 * 28))
print(embedding.shape)

# HDBScan clustering
print('HDBScan Clustering')
hdbscan = hd.HDBSCAN(min_cluster_size=50, min_samples=1)
hdbscan.fit(embedding)
hdbscan_labels = np.asarray(hdbscan.labels_)
hdbscan_classes = np.sort(np.unique(hdbscan_labels))
hdbscan_classes = np.append(hdbscan_classes, hdbscan_classes[-1] + 1)
np.save('./data/hdbscan_labels.npy', hdbscan_labels)
np.save('./data/hdbscan_classes.npy', hdbscan_classes)

# kmeans clustering
print('Kmeans Clustering')
kmeans = cluster.KMeans(n_clusters=10, random_state=0).fit(embedding)
kmeans_labels = np.asarray(kmeans.labels_)
kmeans_classes = np.sort(np.unique(kmeans_labels))
kmeans_classes = np.append(kmeans_classes, kmeans_classes[-1] + 1)
np.save('./data/kmeans_labels.npy', kmeans_labels)
np.save('./data/kmeans_classes.npy', kmeans_classes)

# spectral clustering
print('Spectral Clustering')
spectral = cluster.SpectralClustering(n_clusters=10, random_state=0).fit(embedding)
spectral_labels = np.asarray(spectral.labels_)
spectral_classes = np.sort(np.unique(spectral_labels))
spectral_classes = np.append(spectral_classes, spectral_classes[-1] + 1)
np.save('./data/spectral_labels.npy', spectral_labels)
np.save('./data/spectral_classes.npy', spectral_classes)

# GM clustering
print('Gaussian Mixture Clustering')
gm = mixture.GaussianMixture(n_components=10, random_state=0).fit(embedding)
gm_labels = np.asarray(gm.predict(embedding))
gm_classes = np.sort(np.unique(gm_labels))
gm_classes = np.append(gm_classes, gm_classes[-1] + 1)
np.save('./data/gm_labels.npy', gm_labels)
np.save('./data/gm_classes.npy', gm_classes)

# birch clustering
print('Birch Clustering')
birch = cluster.Birch(n_clusters=10).fit(embedding)
birch_labels = np.asarray(birch.predict(embedding))
birch_classes = np.sort(np.unique(birch_labels))
birch_classes = np.append(birch_classes, birch_classes[-1] + 1)
np.save('./data/birch_labels.npy', birch_labels)
np.save('./data/birch_classes.npy', birch_classes)
