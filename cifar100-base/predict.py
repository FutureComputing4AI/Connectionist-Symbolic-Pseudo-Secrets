import torch
from utils import evaluate_top


def predict(test_loader, network, device):
    test_acc = []
    one_hot = torch.eye(100).to(device)

    for test_data in test_loader:
        x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)
        y_test_batch = one_hot[y_test_batch.long()]

        # forward
        y_pred_test = network(x_test_batch)

        # evaluate
        acc = evaluate_top(y_test_batch, y_pred_test, top=5)
        test_acc.append(acc.item())

    test_accuracy = sum(test_acc) / len(test_acc) * 100
    return test_accuracy


if __name__ == '__main__':
    from network import Network
    from cifar100 import cifar100

    loader_train, loader_test = cifar100(root='./../data/', batch_size=32)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.to(dev).eval()
    model.load_state_dict(torch.load('./../weights/cifar100-base.h5'))

    accuracy = predict(loader_test, model, dev)
    print('Test Accuracy: {0:>6.2f}%'.format(accuracy))
