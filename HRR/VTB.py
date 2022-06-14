import torch


class VTB:
    def __init__(self, batch_size, dim):
        self.dim = dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.zeros((batch_size, self.dim * self.dim, self.dim * self.dim)).to(self.device)
        self.ones = torch.ones((batch_size, dim, dim)).to(self.device)

        for i in range(0, self.dim * self.dim, self.dim):
            self.mask[:, i:i + self.dim, i:i + self.dim] += self.ones

    def block_diagonal(self, x, n):
        batch_size = x.size()[0]
        x = torch.tile(x, dims=[n, n])
        x = x * self.mask[0:batch_size]
        return x

    def bind_single_dim(self, x, y):
        d = torch.tensor(x.size()[1])
        d_prime = torch.sqrt(d).int()
        vy_prime = torch.pow(d, 1.0 / 4.0) * torch.reshape(y, (x.shape[0], d_prime, d_prime))
        vy = self.block_diagonal(vy_prime, d_prime)
        return torch.matmul(vy, x.unsqueeze(-1)).squeeze()

    def unbind_single_dim(self, x, y):
        d = torch.tensor(x.size()[1])
        d_prime = torch.sqrt(d).int()
        vy_prime = torch.pow(d, 1.0 / 4.0) * torch.reshape(y, (x.shape[0], d_prime, d_prime))
        vy = self.block_diagonal(vy_prime.permute(0, 2, 1), d_prime)
        return torch.matmul(vy, x.unsqueeze(-1)).squeeze()

    def binding(self, x, y):
        bind = torch.zeros(x.size()).to(self.device)
        size = (x.size()[0], self.dim, self.dim)
        bind[:, 0, :, :] = self.bind_single_dim(x[:, 0, :, :].flatten(1), y[:, 0, :, :].flatten(1)).reshape(*size)
        # bind[:, 1, :, :] = self.bind_single_dim(x[:, 1, :, :].flatten(1), y[:, 1, :, :].flatten(1)).reshape(*size)
        # bind[:, 2, :, :] = self.bind_single_dim(x[:, 2, :, :].flatten(1), y[:, 2, :, :].flatten(1)).reshape(*size)
        return bind

    def unbinding(self, x, y):
        unbind = torch.zeros(x.size()).to(self.device)
        size = (x.size()[0], self.dim, self.dim)
        unbind[:, 0, :, :] = self.unbind_single_dim(x[:, 0, :, :].flatten(1), y[:, 0, :, :].flatten(1)).reshape(*size)
        # unbind[:, 1, :, :] = self.unbind_single_dim(x[:, 1, :, :].flatten(1), y[:, 1, :, :].flatten(1)).reshape(*size)
        # unbind[:, 2, :, :] = self.unbind_single_dim(x[:, 2, :, :].flatten(1), y[:, 2, :, :].flatten(1)).reshape(*size)
        return unbind


class Orthogonal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def tensor(self, size):
        random = torch.normal(mean=self.mean, std=self.std, size=size)
        # random = torch.distributions.uniform.Uniform(0, 1).sample(size)
        q, _ = torch.linalg.qr(random)
        return q

    @staticmethod
    def is_orthogonal(x):
        dim = list(range(len(x.size())))
        dim[-2], dim[-1] = dim[-1], dim[-2]
        x = torch.matmul(x.permute(dim), x)
        x = torch.diagonal(x, dim1=-2, dim2=-1)
        x = torch.sum(x, dim=-1) / x.size()[-1]
        return x


def cosine_similarity(x, y):
    dim = list(range(-len(x.size()) // 2, 0))
    return torch.sum(x * y, dim=dim) / (torch.norm(x, dim=dim) * torch.norm(y, dim=dim))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # a = torch.tensor([[-.8, 0.08, 0.04, -.6], [-.8, 0.08, 0.04, -.6]], dtype=torch.float32).to(device)
    # b = torch.tensor([[0.97, .16, .03, -.16], [0.97, .16, .03, -.16]], dtype=torch.float32).to(device)
    # (batch_size, 256)

    m = 16
    orthogonal = Orthogonal(mean=0., std=1. / m)
    a = orthogonal.tensor(size=(2, 1, m, m)).to(device)
    b = orthogonal.tensor(size=(2, 1, m, m)).to(device)

    print(a.size())
    print(b.size())

    vtb = VTB(batch_size=2, dim=m)

    s = vtb.binding(a, b)
    a_prime = vtb.unbinding(s, b) / m

    print("a :\n\t", a[0][0])
    print("a_prime :\n\t", a_prime[0][0])
    print("similarity :\n\t", cosine_similarity(a, a_prime))
