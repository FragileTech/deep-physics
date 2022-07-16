import einops
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from deep_physics.physics import action, discrete_diff, generate_trajectories


def split_path(path):
    first = path[:, 0].unsqueeze(1)
    last = path[:, -1].unsqueeze(1)
    bound = torch.cat([first, last], 1)
    return bound, path[:, 1:-1]


def reconstruct_path(bound, pred):
    first, last = bound[:, 0], bound[:, -1]
    x0 = einops.rearrange(first, "b v x -> b 1 v x")
    xt = einops.rearrange(last, "b v x -> b 1 v x")
    path = torch.cat([x0, pred, xt], dim=1)
    return path


class System:
    def __init__(self, potential, path_len, step_size):
        super().__init__()
        self.potential = potential
        self.path_len = path_len
        self.step_size = step_size

    def sample_trajectories(self, samples, only_valid: bool = True):
        data = generate_trajectories(
            self.potential,
            self.potential.bounds,
            n_batch=samples,
            steps=self.path_len,
            step_size=self.step_size,
            p0=None,
        )

        pos, mom, pots, pot_grads = [x.detach().clone() for x in data]
        max_x = self.potential.bounds[0][1]  # Assumes domain is a square
        if only_valid:
            ix = (pos.abs() < abs(max_x)).all(1).all(1)
            pos = pos[ix].detach().clone()
            mom = mom[ix].detach().clone()
            pots = pots[ix].detach().clone()
            pot_grads = pot_grads[ix].detach().clone()
        return pos, mom, pots, pot_grads
