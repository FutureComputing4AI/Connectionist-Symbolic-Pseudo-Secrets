from init import root

import time
import torch
from predict import predict
from network import Network
from dataset import cifar10
from utils import loss_function, evaluate

train_loader, test_loader = cifar10(root=root(), batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = Network()
network.to(device)
# network.load_state_dict(torch.load('./../weights/cifar-base.h5'))

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

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred_batch = network(x_true_batch)
        loss = loss_function(y_true_batch, y_pred_batch)
        loss.backward()
        optimizer.step()

        # evaluate
        accuracy = evaluate(y_true_batch, y_pred_batch)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())

    mean_loss = sum(train_loss) / len(train_loss)
    mean_accuracy = sum(train_acc) / len(train_acc) * 100
    test_accuracy = predict(test_loader, network, device)

    form = 'Epoch: {0:>3d}/' + str(epochs) + '   Train Accuracy: {1:>6.2f}%   Train Loss: {2:>8.6f}   '
    form += 'Test Accuracy: {3:>6.2f}%'
    print(form.format(epoch, mean_accuracy, mean_loss, test_accuracy))

toc = time.time()
print('Time: {0:.2f} seconds per epoch'.format((toc - tic) / epochs))

torch.save(network.state_dict(), './../weights/cifar-base.h5')
print('All Done!')
