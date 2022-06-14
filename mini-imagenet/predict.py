from init import root

import torch
from math import sqrt
from utils import evaluate_top
from HRR.with_pytorch import projection_2d, binding_2d


def predict(test_loader, network, device):
    test_acc = []
    one_hot = torch.eye(100).to(device)

    for i, test_data in enumerate(test_loader):
        x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)
        y_test_batch = one_hot[y_test_batch.long()]

        key = projection_2d(torch.normal(mean=0.,
                                         std=1. / sqrt(84. * 84. * 3.),
                                         size=(x_test_batch.size()[0], 3, 84, 84)).to(device))

        x_test_bind = binding_2d(x_test_batch, key)

        # forward
        y_pred_test, _ = network(x_test_bind, key)

        # evaluate
        acc = evaluate_top(y_test_batch, y_pred_test, top=5)
        test_acc.append(acc.item())

    test_accuracy = sum(test_acc) / len(test_acc) * 100
    return test_accuracy


if __name__ == '__main__':
    from network import Network
    from mini_imagenet import mini_imagenet_augment_84

    batch_size = 50
    _, loader_test = mini_imagenet_augment_84(root=root(), batch_size=batch_size, num_workers=0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.to(dev).eval()
    model.load_state_dict(torch.load('./../weights/mini-imagenet.h5'))

    accuracy = predict(loader_test, model, dev)
    print('Test Accuracy: {0:>6.2f}%'.format(accuracy))
