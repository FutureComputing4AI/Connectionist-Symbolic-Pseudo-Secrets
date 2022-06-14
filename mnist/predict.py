from init import root

import torch
from math import sqrt
from dataset import mnist
from utils import evaluate
from network import Network
from HRR.with_pytorch import projection_2d, binding_2d


def predict(test_loader, network, device):
    test_acc = []
    one_hot = torch.eye(10).to(device)

    for i, test_data in enumerate(test_loader):
        x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)
        y_test_batch = one_hot[y_test_batch.long()]

        secret = projection_2d(torch.normal(mean=0.,
                                            std=1. / sqrt(1 * 28 * 28),
                                            size=(x_test_batch.size()[0], 1, 28, 28)).to(device))

        x_test_bind = binding_2d(x_test_batch, secret)

        y_pred_batch, _ = network(x_test_bind, secret)

        # evaluate
        accuracy = evaluate(y_test_batch, y_pred_batch)
        test_acc.append(accuracy.item())

    test_accuracy = sum(test_acc) / len(test_acc) * 100

    return test_accuracy


if __name__ == '__main__':
    _, loader_test = mnist(root=root(), batch_size=256)

    cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(cuda)

    model = Network()
    model.to(cuda)
    model.load_state_dict(torch.load('./../weights/mnist.h5'))

    acc = predict(loader_test, model, cuda)
    print('Test Accuracy: {0:>6.2f}%'.format(acc))
