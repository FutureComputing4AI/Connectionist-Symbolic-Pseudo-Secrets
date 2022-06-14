import torch
import numpy as np
from math import sqrt
from utils import evaluate
from HRR.with_pytorch import projection_2d, binding_2d


def test(test_loader, network, device, k):
    k_mean = []
    for k in range(1, k + 1):
        print('k =', k)
        test_acc = []
        one_hot = torch.eye(10).to(device)

        for test_data in test_loader:
            x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)
            y_test_batch = one_hot[y_test_batch.long()]
            y_pred_mean = torch.zeros(y_test_batch.size()).to(device)

            for _ in range(k):
                secret = projection_2d(torch.normal(mean=0.,
                                                    std=1. / sqrt(1 * 32 * 32),
                                                    size=(x_test_batch.size()[0], 1, 32, 32)).to(device))

                x_test_bind = binding_2d(x_test_batch, secret)
                _, y_pred_batch, _ = network(x_test_bind, secret)
                y_pred_mean = y_pred_mean + y_pred_batch

            y_pred_mean = y_pred_mean / k
            accuracy = evaluate(y_test_batch, y_pred_mean)
            test_acc.append(accuracy.item())

        test_accuracy = sum(test_acc) / len(test_acc) * 100
        k_mean.append(test_accuracy)

    return k_mean


if __name__ == '__main__':
    from dataset import svhn
    from network import Network

    loader_train, loader_test = svhn(root='./../data/', batch_size=32)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.to(dev)
    model.load_state_dict(torch.load('./svhn.h5'))

    with torch.no_grad():
        kmean_acc = test(loader_test, model, dev, k=10)

    for i, acc in enumerate(kmean_acc):
        print('k = {0} : Test Accuracy: {1:>6.2f}%'.format(i + 1, acc))

    kmean_acc = np.asarray(kmean_acc)
    np.save('data/kmean_svhn_adversary.npy', kmean_acc)
