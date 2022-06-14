import torch
from math import sqrt
from utils import evaluate_top
from HRR.with_pytorch import projection_2d, binding_2d


def predict(test_loader, network, device):
    test_acc = []
    one_hot = torch.eye(100).to(device)

    for i, data in enumerate(test_loader):
        x_test_batch, y_test_batch = data[0].to(device), data[1].to(device)
        y_test_batch = one_hot[y_test_batch.long()]

        secret = projection_2d(torch.normal(mean=0.,
                                            std=1. / sqrt(32. * 32. * 3.),
                                            size=(x_test_batch.size()[0], 3, 32, 32)).to(device))

        x_test_bind = binding_2d(x_test_batch, secret)

        y_pred, _ = network(x_test_bind, secret)
        # evaluate
        acc = evaluate_top(y_test_batch, y_pred, top=1)
        test_acc.append(acc.item())

    mean_accuracy = sum(test_acc) / len(test_acc) * 100
    return mean_accuracy


if __name__ == '__main__':
    from cifar100 import cifar100_augmented
    from network import Network

    _, loader_test = cifar100_augmented(root='./../data/', batch_size=32)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.to(dev).eval()
    model.load_state_dict(torch.load('./../weights/cifar100.h5'))

    accuracy = predict(loader_test, model, dev)
    print('Test Accuracy: {0:>6.2f}%'.format(accuracy))
