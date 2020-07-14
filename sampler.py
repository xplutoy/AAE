# ref https://github.com/musyoku/adversarial-autoencoder/blob/master/aae/sampler.py
import random
from math import *

import numpy as np


def onehot_categorical(batchsize, num_labels):
    y = np.zeros((batchsize, num_labels), dtype=np.float32)
    indices = np.random.randint(0, num_labels, batchsize)
    for b in range(batchsize):
        y[b, indices[b]] = 1
    return y


def uniform(bz, ndim, minv=-1, maxv=1):
    return np.random.uniform(minv, maxv, (bz, ndim)).astype(np.float32)


def gaussian(bz, ndim, mu=0, var=1):
    return np.random.normal(mu, var, (bz, ndim)).astype(np.float32)


def gaussian_mixture(batchsize, ndim, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1),
                                                 num_labels)
    return z


def supervised_gaussian_mixture(batchsize, ndim, label_indices, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], num_labels)
    return z


def swiss_roll(batchsize, ndim, num_labels):
    def sample(label, num_labels):
        uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(random.randint(0, num_labels - 1), num_labels)
    return z


def supervised_swiss_roll(batchsize, ndim, label_indices, num_labels):
    def sample(label, num_labels):
        uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(label_indices[batch], num_labels)
    return z


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # z = gaussian_mixture(1000, 2, 10)
    # print(z.shape)
    # plt.scatter(z[:, 0], z[:, 1], marker='.')
    # plt.show()
    label = np.random.randint(10, size=1000)
    z = supervised_gaussian_mixture(1000, 2, label, 10)
    plt.scatter(z[:, 0], z[:, 1], marker='.', c=label)
    plt.show()
    z = supervised_swiss_roll(1000, 2, label, 10)
    plt.scatter(z[:, 0], z[:, 1], marker='.', c=label)
    plt.show()

