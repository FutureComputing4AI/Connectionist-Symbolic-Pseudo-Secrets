import torch
from torchvision import datasets, transforms


def cifar100(root, batch_size=64, num_workers=0):
    train_set = datasets.CIFAR100(root=root + 'cifar100/train',
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root + 'cifar100/test',
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class CIFAR100_AUGMENTED:
    def __init__(self, root):
        self.cifar = datasets.CIFAR100
        self.directory = root + 'cifar100/train'
        self.tensor = transforms.ToTensor()

    # 1. original image
    def original(self):
        train_set_original = self.cifar(root=self.directory, train=True, download=True, transform=self.tensor)
        return train_set_original

    # 2. horizontal flip and jitter
    def horizontal_flip_jitter(self):
        flip = transforms.RandomHorizontalFlip(p=0.25)
        crop = transforms.RandomResizedCrop(size=(32, 32), scale=(.5, 1.))
        jitter = transforms.ColorJitter(brightness=(.5, 1.5), contrast=(.5, 1.5), saturation=(.5, 1.5))
        transform = transforms.Compose([self.tensor, flip, crop, jitter])
        train_set_flip_jitter = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_flip_jitter

    # 3. affine transformation
    def affine_transformation_blur(self):
        affine = transforms.RandomAffine(degrees=(-90, 90), translate=(0.0, .25), scale=(.9, 1.1))
        blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.15))
        transform = transforms.Compose([self.tensor, affine, blur])
        train_set_blur = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_blur


def cifar100_augmented(root, batch_size=64, num_workers=0):
    train_set = []
    augment = CIFAR100_AUGMENTED(root=root)

    train_set.append(augment.original())
    train_set.append(augment.horizontal_flip_jitter())
    train_set.append(augment.affine_transformation_blur())

    train_set = torch.utils.data.ConcatDataset(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root + 'cifar100/test',
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt


    def imshow(img, name, n):
        img = np.transpose(img.numpy(), (1, 2, 0))
        plt.subplot(1, 2, n)
        plt.title(name)
        plt.imshow(img)


    aug = CIFAR100_AUGMENTED(root='./data/')

    train = torch.utils.data.DataLoader(aug.original(), batch_size=128, shuffle=False, num_workers=0)
    x_train1, _ = next(iter(train))

    train = torch.utils.data.DataLoader(aug.affine_transformation_blur(), batch_size=128, shuffle=False, num_workers=0)
    x_train2, _ = next(iter(train))

    for i1, i2 in zip(x_train1, x_train2):
        plt.figure('figure')
        imshow(i1, name='original image', n=1)
        imshow(i2, name='augmented image', n=2)
        plt.show()
