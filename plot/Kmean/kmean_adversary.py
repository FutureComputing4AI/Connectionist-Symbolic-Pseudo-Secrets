import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
plt.figure(figsize=[10, 8])

# loading data
mnist = np.load('kmean_mnist_adversary.npy')
svhn = np.load('kmean_svhn_adversary.npy')
cifar = np.load('kmean_cifar_adversary.npy')
cifar100 = np.load('kmean_cifar100_adversary.npy')
mini_imagenet = np.load('kmean_mini-imagenet_adversary.npy')

idx = np.arange(1, len(mnist) + 1)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Helvetica"
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 1.0

# plotting
linewidth = 2.5
markersize = 12
alpha = 0.9
params = {'linewidth': linewidth, 'markersize': markersize, 'alpha': alpha}

line1, = plt.plot(idx, mnist, color='#6c17ff', label='MNIST', marker='s', **params)
line2, = plt.plot(idx, svhn, color='#12e621', label='SVHN', marker='p', **params)
line3, = plt.plot(idx, cifar, color='#ff6c17', label='CIFAR-10', marker='^', **params)
line4, = plt.plot(idx, cifar100, color='#ff1717', label='CIFAR-100', marker='o', **params)
line5, = plt.plot(idx, mini_imagenet, color='#17b9ff', label='Mini-ImageNet', marker='d', **params)

plt.ylim([-4, 46])
legend1 = plt.legend(handles=[line1, line2, line3, line4, line5],
                     loc='upper left',
                     borderpad=0.5,
                     handlelength=1.5,
                     fancybox=True,
                     framealpha=0.5)
plt.gca().add_artist(legend1)
plt.legend(handles=[line1, line2, line3, line4, line5],
           labels=['Accuracy: {0:.2f} %'.format(mnist[-1]),
                   'Accuracy: {0:.2f} %'.format(svhn[-1]),
                   'Accuracy: {0:.2f} %'.format(cifar[-1]),
                   'Accuracy: {0:.2f} %'.format(cifar100[-1]),
                   'Accuracy: {0:.2f} %'.format(mini_imagenet[-1])],
           loc='upper right',
           borderpad=0.5,
           handlelength=1.5,
           fancybox=True,
           framealpha=0.5)

plt.xlabel('Number of repeated secrets')
plt.ylabel('Top-1 Accuracy (%)')
plt.xticks(np.arange(11), list(range(0, 11)))
plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=0, hspace=0)
# plt.savefig('./../../figure/kmean.png', bbox_inches="tight", pad_inches=0)
plt.savefig('./../../figure/kmean_adversary.pdf', bbox_inches="tight", pad_inches=0)
plt.show()
