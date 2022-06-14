import torch
from hilbertcurve.hilbertcurve import HilbertCurve


class Hilbert_Transform:
    def __init__(self, x, p=32, n=2, dim=3):
        self.p = p
        self.dim = dim
        self.index_array = torch.FloatTensor(list(range(0, x * x))).reshape((x, x))
        self.indexes = torch.zeros(x * x, n, dtype=torch.int)
        self.hilbert_curve = HilbertCurve(p, n)

        for i in range(x * x):
            coordinates = self.hilbert_curve.coordinates_from_distance(i)
            self.indexes[i, :] = torch.FloatTensor(coordinates)

        self.encode_vec = torch.FloatTensor([self.index_array[i, j] for i, j in self.indexes])

        if dim == 3:
            self.encode_vec = torch.cat((self.encode_vec,
                                         self.encode_vec + x * x,
                                         self.encode_vec + x * x * 2), dim=0).long()

        self.decode_dec = torch.argsort(self.encode_vec)

    def encode(self, inputs):
        inputs = inputs.view(-1, self.dim * self.p * self.p)
        return inputs[:, self.encode_vec]

    def decode(self, inputs):
        inputs = inputs.view(-1, self.dim * self.p * self.p)
        return inputs[:, self.decode_dec]


def index_sequence(batch_size, dataset_size):
    index_i = list(range(0, dataset_size, batch_size))
    index_j = list(range(batch_size, dataset_size, batch_size))
    index_j.append(dataset_size)
    indices = list(zip(index_i, index_j))
    return indices


def loss_function(y_true, y_pred, eps=1e-10):
    y_t = torch.clip(y_true, eps, 1.)
    y_p = torch.clip(y_pred, eps, 1.)
    y_t_prime = torch.clip(1. - y_true, eps, 1.)
    y_p_prime = torch.clip(1. - y_pred, eps, 1.)
    losses = - torch.xlogy(y_t, y_p) - torch.xlogy(y_t_prime, y_p_prime)
    losses = torch.mean(losses)
    return losses


def evaluate(y_true, y_pred):
    y_true = torch.argmax(y_true, dim=-1)
    y_pred = torch.argmax(y_pred, dim=-1)
    total = torch.sum((y_true == y_pred).long())
    count = y_true.size()[0]
    accuracy = total / count
    return accuracy


def evaluate_top(y_true, y_pred, top=1):
    y_true = torch.argsort(y_true, dim=-1, descending=True)[:, 0:1]
    y_pred = torch.argsort(y_pred, dim=-1, descending=True)[:, 0:top]
    equal = torch.eq(y_true, y_pred)
    count = torch.any(equal, dim=-1).float()
    accuracy = torch.mean(count)
    return accuracy
