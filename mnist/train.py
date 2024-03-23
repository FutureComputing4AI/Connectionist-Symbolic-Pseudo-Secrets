from init import root

import time
import torch
from math import sqrt
from dataset import mnist
from network import Network
from predict import predict
from HRR.with_pytorch import projection_2d, binding_2d
from utils import loss_function, evaluate

train_loader, test_loader = mnist(root=root(), batch_size=32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = Network()
network.to(device)
network.load_state_dict(torch.load('./../weights/mnist.h5'))

epochs = 10
one_hot = torch.eye(10).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
tic = time.time()

for epoch in range(1, epochs + 1):
    train_loss = []
    train_acc = []
    for i, data in enumerate(train_loader, 0):
        x_true_batch, y_true_batch = data[0].to(device), data[1].to(device)
        y_true_batch = one_hot[y_true_batch.long()]

        secret = projection_2d(torch.normal(mean=0.,
                                            std=1. / sqrt(1 * 28 * 28),
                                            size=(x_true_batch.size()[0], 1, 28, 28)).to(device))

        x_true_bind = binding_2d(x_true_batch, secret)

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred, y_attack = network(x_true_bind, secret)
        loss1 = loss_function(y_true_batch, y_pred)
        loss2 = loss_function(y_true_batch, y_attack)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        # evaluate
        accuracy = evaluate(y_true_batch, y_attack)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())

    mean_loss = sum(train_loss) / len(train_loss)
    mean_accuracy = sum(train_acc) / len(train_acc) * 100
    form = 'Epoch: {0:>3d}/' + str(epochs) + '   Train Accuracy: {1:>6.2f}%   Train Loss: {2:>8.6f}'
    print(form.format(epoch, mean_accuracy, mean_loss))

toc = time.time()
print('Time: {0:.2f} seconds per epoch'.format((toc - tic) / epochs))

test_accuracy = predict(test_loader, network, device)
print('Test Accuracy: {0:>6.2f}%'.format(test_accuracy))

torch.save(network.state_dict(), './../weights/mnist.h5')
print('All Done!')
