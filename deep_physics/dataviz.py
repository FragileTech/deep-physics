import einops
import holoviews as hv
import numpy as np
import torch


def plot_landscape(function, bounds, n_points=100):
    (x_min, x_max), (y_min, y_max) = bounds
    xi = np.linspace(x_min, x_max, n_points)
    yi = np.linspace(y_min, y_max, n_points)
    xx, yy = np.meshgrid(xi, yi)
    ins = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
    zs, _ = function(torch.from_numpy(ins), reset_grad=True)
    zs = einops.asnumpy(zs.reshape(n_points, n_points))
    landscape = hv.QuadMesh((xx, yy, zs))
    return landscape
