import math

import numpy as np
import torch


def reset_grads(x):
    if x.grad is not None:
        x.grad.zero_()
    x.requires_grad_()
    return x


class Potential:
    def __init__(self, function, bounds):
        self._bounds = bounds
        self._function = function

    def __call__(self, x, reset_grad=False, return_grad=True, scaled=False):
        if scaled:
            pass
            # x = self.from_scaled(x)
        if (reset_grad and return_grad) or (return_grad and x.grad is None):
            x = reset_grads(x)

        val = self._function(x)
        if return_grad:
            loss = val.sum()
            loss.backward()
            return val, x.grad
        return val

    @property
    def bounds(self):
        return self._bounds

    def from_scale_x_vector(self, x):
        (x_min, x_max), _ = self.bounds
        x = ((x + 1) / 2) * (x_max - x_min) + x_min
        return x

    def from_scale_y_vector(self, y):
        _, (y_min, y_max) = self.bounds
        y = ((y + 1) / 2) * (y_max - y_min) + y_min
        return y

    def to_scale_x_vector(self, x):
        (x_min, x_max), _ = self.bounds
        x = (((x - x_min) / (x_max - x_min)) * 2) - 1
        return x

    def to_scale_y_vector(self, y):
        _, (y_min, y_max) = self.bounds
        y = (((y - y_min) / (y_max - y_min)) * 2) - 1
        return y

    def extract_x_coordinates(self, x):
        if len(x.shape) == 2:
            x_ix = x[:, 0]
            y_ix = x[:, 1]
        elif len(x.shape) == 3:
            x_ix = x[:, :, 0]
            y_ix = x[:, :, 1]
        elif len(x.shape) == 4:
            x_ix = x[:, :, 0, 0]
            y_ix = x[:, :, 0, 1]
        else:
            raise NotImplementedError()
        return x_ix, y_ix

    def from_scaled_pos(self, x):
        x_ix, y_ix = self.extract_x_coordinates(x)
        xs = self.from_scale_x_vector(x_ix).unsqueeze(-1)
        ys = self.from_scale_y_vector(y_ix).unsqueeze(-1)
        return torch.cat([xs, ys], dim=-1)

    def to_scaled_pos(self, x):
        x_ix, y_ix = self.extract_x_coordinates(x)
        xs = self.to_scale_x_vector(x_ix).unsqueeze(-1)
        ys = self.to_scale_y_vector(y_ix).unsqueeze(-1)
        return torch.cat([xs, ys], dim=-1)


def with_grad(bounds=None):
    def inner(func):
        return Potential(function=func, bounds=bounds)

    return inner


@with_grad(bounds=((-5, 5), (-5, 5)))
def holder_table(_x):
    x, y = _x[:, 0], _x[:, 1]
    exp = torch.abs(1 - (torch.sqrt(x * x + y * y) / np.pi))
    return -torch.abs(torch.sin(x) * torch.cos(y) * torch.exp(exp))


@with_grad(bounds=((-511, 511), (-511, 511)))
def eggholder(x):
    x, y = x[:, 0], x[:, 1]
    first_root = torch.sqrt(torch.abs(x / 2.0 + (y + 47)))
    second_root = torch.sqrt(torch.abs(x - (y + 47)))
    result = -1 * (y + 47) * torch.sin(first_root) - x * torch.sin(second_root)
    return result


@with_grad(bounds=((-5, 5), (-5, 5)))
def himmelbau(x):
    x, y = x[:, 0], x[:, 1]
    return (x.pow(2) + y - 11).pow(2) + (x + y.pow(2) - 7).pow(2)


@with_grad(bounds=((-5, 5), (-5, 5)))
def sphere(x):
    return torch.sum(x**2, 1)


@with_grad(bounds=((-5, 5), (-5, 5)))
def dummy(x):
    return x[:, 1]


@with_grad(bounds=((-5.12, 5.12), (-5.12, 5.12)))
def rastrigin(x: np.ndarray) -> np.ndarray:
    dims = x.shape[1]
    A = 10
    result = A * dims + torch.sum(x**2 - A * torch.cos(2 * math.pi * x), 1)
    return result.flatten()
