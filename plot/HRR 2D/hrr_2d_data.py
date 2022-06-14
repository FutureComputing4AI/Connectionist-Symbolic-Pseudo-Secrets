import torch
import random
import numpy as np
from math import sqrt
from HRR import projection_2d, binding_2d, unbinding_2d


def generate_random(index, avoid=None):
    rand = random.randint(0, index)
    if rand == avoid or rand > 1023:
        return generate_random(index, avoid)
    else:
        return rand


dim = 256
proj = False
batch_size = 1024
torch.manual_seed(0)

r1 = torch.normal(mean=0., std=1. / sqrt(dim * dim), size=(batch_size, dim, dim))
r2 = torch.normal(mean=0., std=1. / sqrt(dim * dim), size=(batch_size, dim, dim))

if proj:
    r1 = projection_2d(r1)
    r2 = projection_2d(r2)

bind = binding_2d(r1, r2)
bind = torch.sum(bind, dim=0)

pos_idx = []
neg_idx = []

for i in range(2, batch_size + 2):
    print('i =', i)
    repeat = 20

    # positive cases
    corr = torch.tensor([0], dtype=torch.float32)
    for _ in range(repeat):
        idx = generate_random(i - 1)
        x = r1[idx]
        x_prime = unbinding_2d(bind, r2[idx])
        corr += torch.sum(x * x_prime)
    corr = corr / repeat
    pos_idx.append(corr.item())

    # negative cases
    corr = torch.tensor([0], dtype=torch.float32)
    for _ in range(repeat):
        idx1 = generate_random(i - 1)
        idx2 = generate_random(i - 1, avoid=idx1)
        x = r1[idx1]
        x_prime = unbinding_2d(bind, r2[idx2])
        corr += torch.sum(x * x_prime)
    corr = corr / repeat
    neg_idx.append(corr.item())

pos_idx = np.array(pos_idx)
neg_idx = np.array(neg_idx)

np.save('data/proj_pos_idx.npy' if proj else 'data/pos_idx.npy', pos_idx)
np.save('data/proj_neg_idx.npy' if proj else 'data/neg_idx.npy', neg_idx)
