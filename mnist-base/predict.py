import torch
from math import sqrt
from utils import evaluate
from HRR.with_pytorch import projection_2d, binding_2d


def test(test_loader, network, device):
    test_acc = []
    one_hot = torch.eye(10).to(device)

    for test_data in test_loader:
        x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)

        y_test_batch = one_hot[y_test_batch.long()]

        key = projection_2d(torch.normal(mean=0.,
                                         std=1. / sqrt(32. * 32. * 3.),
                                         size=(x_test_batch.size()[0], 3, 32, 32)).to(device))

        x_test_bind = binding_2d(x_test_batch, key)

        # forward
        y_pred_test, _, _ = network(x_test_bind, key)

        # evaluate
        acc = evaluate(y_test_batch, y_pred_test)
        test_acc.append(acc.item())

    test_accuracy = sum(test_acc) / len(test_acc) * 100
    return test_accuracy


if __name__ == '__main__':
    from dataset import cifar10
    from network import Network

    loader_train, loader_test = cifar10(root='./../data/', batch_size=64)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.to(dev)
    model.load_state_dict(torch.load('./../weights/mnist-base.h5'))

    accuracy = test(loader_test, model, dev)
    print('Test Accuracy: {0:>6.2f}%'.format(accuracy))
