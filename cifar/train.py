from init import root

import time
import torch
from math import sqrt
from predict import predict
from network import Network
from dataset import cifar10_augmented
from utils import loss_function, evaluate
from HRR.with_pytorch import projection_2d, binding_2d


train_loader, test_loader = cifar10_augmented(root=root(), batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = Network()
network.to(device)
# network.load_state_dict(torch.load('./../weights/cifar.h5'))

epochs = 10
one_hot = torch.eye(10).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
tic = time.time()

for epoch in range(1, epochs + 1):
    train_loss = []
    train_acc = []
    for i, data in enumerate(test_loader, 0):
        x_true_batch, y_true_batch = data[0].to(device), data[1].to(device)

        y_true_batch = one_hot[y_true_batch.long()]

        secret = projection_2d(torch.normal(mean=0.,
                                            std=1. / sqrt(32. * 32. * 3.),
                                            size=(x_true_batch.size()[0], 3, 32, 32)).to(device))

        x_true_bind = binding_2d(x_true_batch, secret)

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred, y_attack = network(x_true_bind, secret)

        # y_attack = network(x_true_bind)
        loss1 = loss_function(y_true_batch, y_pred)
        loss2 = loss_function(y_true_batch, y_attack)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        # evaluate
        accuracy = evaluate(y_true_batch, y_pred)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())

    mean_loss = sum(train_loss) / len(train_loss)
    mean_accuracy = sum(train_acc) / len(train_acc) * 100
    test_accuracy = predict(test_loader, network)

    form = 'Epoch: {0:>3d}/' + str(epochs) + '   Train Accuracy: {1:>6.2f}%   Train Loss: {2:>8.6f}   '
    form += 'Test Accuracy: {3:>6.2f}%'
    print(form.format(epoch, mean_accuracy, mean_loss, test_accuracy))

toc = time.time()
print('Time: {0:.2f} seconds per epoch'.format((toc - tic) / epochs))

torch.save(network.state_dict(), './../weights/cifar.h5')
print('All Done!')
