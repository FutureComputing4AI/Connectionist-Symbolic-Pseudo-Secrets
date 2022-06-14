import torch
import numpy as np
from math import sqrt
from utils import evaluate
from HRR.with_pytorch import projection_2d, binding_2d


def predict(test_loader, network, device, k1, k2):
    k_mean = []
    for k in range(k1, k2 + 1):
        test_acc = []
        one_hot = torch.eye(100).to(device)

        for test_data in test_loader:
            x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)
            y_test_batch = one_hot[y_test_batch.long()]
            y_pred_mean = torch.zeros(y_test_batch.size()).to(device)

            for _ in range(k):
                secret = projection_2d(torch.normal(mean=0.,
                                                    std=1. / sqrt(32. * 32. * 3.),
                                                    size=(x_test_batch.size()[0], 3, 32, 32)).to(device))

                x_test_bind = binding_2d(x_test_batch, secret)
                _, y_pred_batch, _ = network(x_test_bind, secret)
                y_pred_mean = y_pred_mean + y_pred_batch

            y_pred_mean = y_pred_mean / k
            accuracy = evaluate(y_test_batch, y_pred_mean)
            test_acc.append(accuracy.item())

        test_accuracy = sum(test_acc) / len(test_acc) * 100
        k_mean.append(test_accuracy)
        print('k = {0} : Test Accuracy: {1:>6.2f}%'.format(k, test_accuracy))
    return k_mean


if __name__ == '__main__':
    from cifar100 import cifar100_augmented
    from network import Network

    batch_size = 64
    _, loader_test = cifar100_augmented(root='./../data/', batch_size=batch_size)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.to(dev)
    model.load_state_dict(torch.load('./../weights/cifar100.h5'))
    # 1 - 1   : 128
    # 2 - 7   : 64
    # 8 - 9   : 64
    # 10 - 10 : 64
    with torch.no_grad():
        kmean_acc = predict(loader_test, model, dev, k1=1, k2=10)

    kmean_acc = np.asarray(kmean_acc)
    print(kmean_acc)
    np.save('data/kmean_cifar100_adversary.npy', kmean_acc)
