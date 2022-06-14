import torch
from unet import unet
import torch.nn as nn
from pytorch_revgrad import RevGrad
from HRR.with_pytorch import unbinding_2d


class Network(nn.Module):
    def __init__(self, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        print('cifar 100 network')

        self.F_main = nn.Sequential(
            unet.UNet2D(3, 3)
        )

        self.F_attact = nn.Sequential(
            RevGrad(),
            nn.Conv2d(3, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)), nn.BatchNorm2d(128), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)), nn.BatchNorm2d(256), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(4096, 2048), activation, nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024), activation, nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 100), nn.Softmax(dim=-1)
        )

        self.F_pred = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)), nn.BatchNorm2d(128), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)), nn.BatchNorm2d(256), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(4096, 2048), activation, nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024), activation, nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 100), nn.Softmax(dim=-1)
        )

    def forward(self, x, key):
        x_main = self.F_main(x)

        y_attack = self.F_attact(x)

        x_unbind = unbinding_2d(x_main, key)

        y_pred = self.F_pred(x_unbind)

        return y_pred, y_attack


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = torch.normal(0, 1, (32, 3, 32, 32), dtype=torch.float32).to(device)
    keys = torch.normal(0, 1, (32, 3, 32, 32), dtype=torch.float32).to(device)
    network = Network()
    network.to(device)

    x_, y_ = network(inputs, keys)
    print(x_.size())
    print(y_.size())
