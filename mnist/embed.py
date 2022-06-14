import umap
import numpy as np
import matplotlib.pyplot as plt

images = np.load('x_pred.npy')
labels = np.load('y_true.npy')
images = images.reshape((images.shape[0], -1))

reducer = umap.umap_.UMAP()
embedding = reducer.fit_transform(images)
np.save('embedding.npy', embedding)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='jet')
plt.show()
