import torch
import torch.nn as nn
from dataset import cifar10
import torch.optim as optimizer
from utils import loss_function, evaluate


class Network(nn.Module):
    def __init__(self, input_features=3 * 32 * 32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_features, 10), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        y = self.model(x)
        return y


batch_size = 256
train_loader, test_loader = cifar10(root='./../data/', batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

network = Network()
network.to(device)
# network.load_state_dict(torch.load('./../weights/cifar-linear.h5'))

epochs = 10
one_hot = torch.eye(10).to(device)
optimizer = optimizer.Adam(network.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    train_loss = []
    train_acc = []
    for i, data in enumerate(train_loader, 0):
        x_true_batch, y_true_batch = data[0].to(device), data[1].to(device)

        x_true_batch = x_true_batch.view(x_true_batch.size()[0], 3 * 32 * 32)
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

    if epoch % 100 == 0:
        torch.save(network.state_dict(), './weights/cifar-linear_' + str(epoch) + '.h5')

torch.save(network.state_dict(), './../weights/cifar-linear.h5')

# test
test_acc = []
for i, data in enumerate(test_loader, 0):
    x_test_batch, y_test_batch = data[0].to(device), data[1].to(device)

    x_test_batch = x_test_batch.view(x_test_batch.size()[0], 3 * 32 * 32)
    y_test_batch = one_hot[y_test_batch.long()]

    # forward
    y_pred_batch = network(x_test_batch)

    # evaluate
    accuracy = evaluate(y_test_batch, y_pred_batch)

    test_acc.append(accuracy.item())

test_accuracy = sum(test_acc) / len(test_acc) * 100

print('Test Accuracy: {0:>6.2f}%'.format(test_accuracy))
