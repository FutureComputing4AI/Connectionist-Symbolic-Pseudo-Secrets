import numpy as np
import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import matplotlib.font_manager as f

# loading umap data
embedding = np.load('./data/embedding.npy')
true_labels = np.load('./data/y_true.npy')

# kmeans clustering
kmeans_labels = np.load('./data/kmeans_labels.npy')
kmeans_classes = np.load('./data/kmeans_classes.npy')

adjusted_rand_score = metrics.adjusted_rand_score(true_labels, kmeans_labels)
homogeneity_score = metrics.homogeneity_score(true_labels, kmeans_labels)
completeness = metrics.completeness_score(true_labels, kmeans_labels)
v_measure = metrics.v_measure_score(true_labels, kmeans_labels)

print('Kmeans Clustering:')
print('Adjusted Rand Score: {0:.2f}%'.format(adjusted_rand_score * 100))
print('V measure Score: {0:.2f}%'.format(v_measure * 100))
print('Homogeneity Score: {0:.2f}%'.format(homogeneity_score * 100))
print('Completeness Score: {0:.2f}%'.format(completeness * 100))
print('\n')

# spectral clustering
spectral_labels = np.load('./data/spectral_labels.npy')
spectral_classes = np.load('./data/spectral_classes.npy')

adjusted_rand_score = metrics.adjusted_rand_score(true_labels, spectral_labels)
homogeneity_score = metrics.homogeneity_score(true_labels, spectral_labels)
completeness = metrics.completeness_score(true_labels, spectral_labels)
v_measure = metrics.v_measure_score(true_labels, spectral_labels)

print('Spectral Clustering:')
print('Adjusted Rand Score: {0:.2f}%'.format(adjusted_rand_score * 100))
print('V measure Score: {0:.2f}%'.format(v_measure * 100))
print('Homogeneity Score: {0:.2f}%'.format(homogeneity_score * 100))
print('Completeness Score: {0:.2f}%'.format(completeness * 100))
print('\n')

# gaussian mixture clustering
gm_labels = np.load('./data/gm_labels.npy')
gm_classes = np.load('./data/gm_classes.npy')

adjusted_rand_score = metrics.adjusted_rand_score(true_labels, gm_labels)
homogeneity_score = metrics.homogeneity_score(true_labels, gm_labels)
completeness = metrics.completeness_score(true_labels, gm_labels)
v_measure = metrics.v_measure_score(true_labels, gm_labels)

print('Gaussian Mixture Clustering:')
print('Adjusted Rand Score: {0:.2f}%'.format(adjusted_rand_score * 100))
print('V measure Score: {0:.2f}%'.format(v_measure * 100))
print('Homogeneity Score: {0:.2f}%'.format(homogeneity_score * 100))
print('Completeness Score: {0:.2f}%'.format(completeness * 100))
print('\n')

# birch clustering
birch_labels = np.load('./data/birch_labels.npy')
birch_classes = np.load('./data/birch_classes.npy')

adjusted_rand_score = metrics.adjusted_rand_score(true_labels, birch_labels)
homogeneity_score = metrics.homogeneity_score(true_labels, birch_labels)
completeness = metrics.completeness_score(true_labels, birch_labels)
v_measure = metrics.v_measure_score(true_labels, birch_labels)

print('Birch Clustering:')
print('Adjusted Rand Score: {0:.2f}%'.format(adjusted_rand_score * 100))
print('V measure Score: {0:.2f}%'.format(v_measure * 100))
print('Homogeneity Score: {0:.2f}%'.format(homogeneity_score * 100))
print('Completeness Score: {0:.2f}%'.format(completeness * 100))
print('\n')

# HDBScan clustering
hdbscan_labels = np.load('./data/hdbscan_labels.npy')
hdbscan_classes = np.load('./data/hdbscan_classes.npy')

adjusted_rand_score = metrics.adjusted_rand_score(true_labels, hdbscan_labels)
homogeneity_score = metrics.homogeneity_score(true_labels, hdbscan_labels)
print(true_labels)
print(hdbscan_labels)
completeness = metrics.completeness_score(true_labels, hdbscan_labels)
v_measure = metrics.v_measure_score(true_labels, hdbscan_labels)

print('HDBScan Clustering:')
print('Adjusted Rand Score: {0:.2f}%'.format(adjusted_rand_score * 100))
print('V measure Score: {0:.2f}%'.format(v_measure * 100))
print('Homogeneity Score: {0:.2f}%'.format(homogeneity_score * 100))
print('Completeness Score: {0:.2f}%'.format(completeness * 100))
print('\n')

""" plot """
sns.set_style('darkgrid')

# loading font
font = f.FontEntry(fname='./../fonts/Lato.ttf', name='lato')
f.fontManager.ttflist.insert(0, font)

# setting text font
plt.rcParams['font.family'] = 'lato'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5

# plotting figure
alpha = 0.85
fontsize = 16
width, height = 5, 5

fig, ax1 = plt.subplots(1, 1, figsize=[width, height], constrained_layout=True)
p1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap='jet', alpha=alpha)
fig.colorbar(p1, ax=ax1, boundaries=np.arange(11) - 0.5, aspect=40).set_ticks(np.arange(10))
plt.savefig('./figs1/1_true.pdf', bbox_inches="tight", pad_inches=0)

fig, ax2 = plt.subplots(1, 1, figsize=[width, height], constrained_layout=True)
p2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, cmap='jet', alpha=alpha)
fig.colorbar(p2, ax=ax2, boundaries=kmeans_classes - 0.5, aspect=40).set_ticks(kmeans_classes[:-1])
plt.savefig('./figs1/2_kmeans.pdf', bbox_inches="tight", pad_inches=0)

fig, ax3 = plt.subplots(1, 1, figsize=[width, height], constrained_layout=True)
p3 = ax3.scatter(embedding[:, 0], embedding[:, 1], c=spectral_labels, cmap='jet', alpha=alpha)
fig.colorbar(p3, ax=ax3, boundaries=spectral_classes - 0.5, aspect=40).set_ticks(spectral_classes[:-1])
plt.savefig('./figs1/3_spectral.pdf', bbox_inches="tight", pad_inches=0)

fig, ax4 = plt.subplots(1, 1, figsize=[width, height], constrained_layout=True)
p4 = ax4.scatter(embedding[:, 0], embedding[:, 1], c=gm_labels, cmap='jet', alpha=alpha)
fig.colorbar(p4, ax=ax4, boundaries=gm_classes - 0.5, aspect=40).set_ticks(gm_classes[:-1])
plt.savefig('./figs1/4_gaussian.pdf', bbox_inches="tight", pad_inches=0)

fig, ax5 = plt.subplots(1, 1, figsize=[width, height], constrained_layout=True)
p5 = ax5.scatter(embedding[:, 0], embedding[:, 1], c=birch_labels, cmap='jet', alpha=alpha)
fig.colorbar(p5, ax=ax5, boundaries=birch_classes - 0.5, aspect=40).set_ticks(birch_classes[:-1])
plt.savefig('./figs1/5_birch.pdf', bbox_inches="tight", pad_inches=0)

fig, ax6 = plt.subplots(1, 1, figsize=[width, height], constrained_layout=True)
p6 = ax6.scatter(embedding[:, 0], embedding[:, 1], c=hdbscan_labels, cmap='jet', alpha=alpha)

if len(hdbscan_classes) < 3:
    # if cluster is only unclassified
    cmap = plt.get_cmap('jet', len(hdbscan_classes) - 1)
    norm = colors.Normalize(vmin=-2, vmax=0)
    mappable = plt.cm.ScalarMappable(cmap=cmap.reversed(), norm=norm)
    fig.colorbar(mappable, ax=ax6, aspect=40).set_ticks([-1])
else:
    fig.colorbar(p6, ax=ax6, boundaries=hdbscan_classes - 0.5, aspect=40).set_ticks(hdbscan_classes[:-1])

plt.savefig('./figs1/6_hdbscan.pdf', bbox_inches="tight", pad_inches=0)

plt.show()
