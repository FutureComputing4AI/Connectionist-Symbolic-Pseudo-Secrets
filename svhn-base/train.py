from init import root

import time
import torch
from dataset import svhn
from network import Network
from utils import loss_function, evaluate

train_loader, test_loader = svhn(root=root(), batch_size=64, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = Network()
network.to(device)
# network.load_state_dict(torch.load('./../weights/svhn-base.h5'))

epochs = 10
one_hot = torch.eye(10).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

if __name__ == '__main__':
    tic = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = []
        train_acc = []
        for data in train_loader:
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

        form = 'Epoch: {0:>3d}/' + str(epochs) + '   Train Accuracy: {1:>6.2f}%   Train Loss: {2:>8.6f}'
        print(form.format(epoch, mean_accuracy, mean_loss))

    toc = time.time()
    print('Time: {0:.2f} seconds per epoch'.format((toc - tic) / epochs))

    test_acc = []
    for test_data in test_loader:
        x_test_batch, y_test_batch = test_data[0].to(device), test_data[1].to(device)
        y_test_batch = one_hot[y_test_batch.long()]

        y_pred_batch = network(x_test_batch)

        # evaluate
        accuracy = evaluate(y_test_batch, y_pred_batch)
        test_acc.append(accuracy.item())

    mean_accuracy = sum(test_acc) / len(test_acc) * 100
    print('Test Accuracy: {0:>6.2f}%'.format(mean_accuracy))

    torch.save(network.state_dict(), './../weights/svhn-base.h5')
    print('All Done!')
