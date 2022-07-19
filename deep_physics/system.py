import einops
import numpy as np
import pandas as pd
import torch
from torch import nn

from deep_physics.physics import action, discrete_diff, forces_newton, generate_trajectories


def split_path(path, n=1):
    first = path[:, :n].unsqueeze(1)
    last = path[:, -n:].unsqueeze(1)
    bound = torch.cat([first, last], 1)
    return bound, path[:, n:-n]


def reconstruct_path(bound, pred, n=1):
    first, last = bound[:, :n], bound[:, -n:]
    x0 = einops.rearrange(first, "b v x -> b 1 v x")
    xt = einops.rearrange(last, "b v x -> b 1 v x")
    path = torch.cat([x0, pred, xt], dim=1)
    return path


class Coords:
    @staticmethod
    def split_path(path):
        first = path[:, 0].unsqueeze(1)
        last = path[:, -1].unsqueeze(1)
        bound = torch.cat([first, last], 1)
        return bound, path[:, 1:-1]

    @staticmethod
    def reconstruct_path(bound, pred):
        first, last = bound[:, 0], bound[:, -1]
        x0 = einops.rearrange(first, "b v x -> b 1 v x")
        xt = einops.rearrange(last, "b v x -> b 1 v x")
        path = torch.cat([x0, pred, xt], dim=1)
        return path

    @classmethod
    def to_polar(cls, x):
        if isinstance(x, np.ndarray):
            return cls._to_polar_numpy(x)
        return cls._to_polar_torch(x)

    @staticmethod
    def _to_polar_torch(x):
        xc = x[:, :, 0] if len(x.shape) == 3 else x[:, 0].flatten()
        yc = x[:, :, 1] if len(x.shape) == 3 else x[:, 1].flatten()
        magnitude = torch.sqrt(xc**2 + yc**2)
        angle = (np.pi / 2.0) - torch.arctan2(xc / magnitude, yc / magnitude)
        return torch.cat([magnitude, angle], dim=1)

    @staticmethod
    def _to_polar_numpy(x):
        xc = x[:, :, 0] if len(x.shape) == 3 else x[:, 0].flatten()
        yc = x[:, :, 1] if len(x.shape) == 3 else x[:, 1].flatten()
        magnitude = np.sqrt(xc**2 + yc**2)
        angle = (np.pi / 2.0) - np.arctan2(xc / magnitude, yc / magnitude)
        return np.concatenate([magnitude.reshape(-1, 1), angle.reshape(-1, 1)], axis=1)


class System:
    df_columns = ["x", "y", "dx", "dy", "v", "dv", "gvx", "gvy", "fx", "fy"]

    def __init__(self, potential, path_len, step_size):
        super().__init__()
        self.potential = potential
        self.path_len = path_len
        self.step_size = step_size
        self.coords = Coords()

    def sample_trajectories(
        self, samples, only_valid: bool = True, as_numpy: bool = True, random_p0: bool = False
    ):
        data = generate_trajectories(
            self.potential,
            self.potential.bounds,
            n_batch=samples,
            steps=self.path_len,
            step_size=self.step_size,
            p0=None,
            random_p0=random_p0,
        )

        data = [x.detach().clone() for x in data]
        if only_valid:
            max_x = self.potential.bounds[0][1]  # Assumes domain is a square
            pos, mom, pots, pot_grads = data
            ix = (pos.abs() < abs(max_x)).all(1).all(1)
            pos = pos[ix].detach().clone()
            mom = mom[ix].detach().clone()
            pots = pots[ix].detach().clone()
            pot_grads = pot_grads[ix].detach().clone()
            data = pos, mom, pots, pot_grads
        if as_numpy:
            pos, mom, pots, pot_grads = [einops.asnumpy(x) for x in data]
        return pos, mom, pots, pot_grads

    def to_df(self, pos, init_mom=None):
        pos = torch.from_numpy(pos) if isinstance(pos, np.ndarray) else pos

        pots, pot_grads = self.potential(pos, return_grad=True, reset_grad=True)
        pots = pots.unsqueeze(0).unsqueeze(-1)
        pos = pos.unsqueeze(0)
        dpos = discrete_diff(pos)
        if init_mom is not None:
            dpos[0] = init_mom

        dv = discrete_diff(pots)
        fs = discrete_diff(dpos)
        data = {
            "x": einops.asnumpy(pos[:, :, 0]).flatten(),
            "y": einops.asnumpy(pos[:, :, 1]).flatten(),
            "px": einops.asnumpy(dpos[:, :, 0]).flatten(),
            "py": einops.asnumpy(dpos[:, :, 1]).flatten(),
            "v": einops.asnumpy(pots).flatten(),
            "dv": einops.asnumpy(dv[0, :, 0]).flatten(),
            "gvx": einops.asnumpy(pot_grads[:, 0]).flatten(),
            "gvy": einops.asnumpy(pot_grads[:, 1]).flatten(),
            "fx": einops.asnumpy(fs[:, :, 0]).flatten(),
            "fy": einops.asnumpy(fs[:, :, 1]).flatten(),
        }
        df = pd.DataFrame(data)
        df.loc[:, ["fm", "fa"]] = Coords.to_polar(df[["fx", "fy"]].values)
        df.loc[:, ["pm", "pa"]] = Coords.to_polar(df[["px", "py"]].values)
        df.loc[:, ["gvm", "gva"]] = Coords.to_polar(df[["gvx", "gvy"]].values)
        df.loc[:, ["fm-", "fa-"]] = df[["fm", "fa"]].shift(1).values
        df["fvm"] = -df["gvm"]
        df = df.reset_index().rename(columns={"index": "time"})
        return df


def mse(pred, target):
    return (pred - target).pow(2).mean()


class PathOptimizerModule(nn.Module):
    def __init__(self, system, lr=0.1, n_iters=100, minimize="MSE"):
        super().__init__()
        self.system = system
        self.lr = lr
        self.n_iters = n_iters
        self.minimize = minimize

    def init_step(self, pred):
        self.path = nn.Parameter(pred.detach().clone())
        self.optim = torch.optim.SGD([self.path], lr=self.lr)
        self.optim.zero_grad()

    def get_loss(self, target, n_split):
        self.optim.zero_grad()
        if self.minimize == "Action":
            bound_true, _ = split_path(target, n=n_split)
            _, inner_pred = split_path(self.path, n=n_split)
            path = reconstruct_path(bound_true, inner_pred, n=n_split)
            action_value = action(path, self.system.potential, return_grad=False)
            loss = action_value.sum()
        elif self.minimize == "MSE":

            loss = mse(self.path, target.detach())
        else:
            loss = forces_newton(self.path, self.system.potential, self.system.step_size)
        return loss

    def forward(self, pred, target, n_split):
        for i in range(self.n_iters):
            self.init_step(pred)
            loss = self.get_loss(target, n_split)
            loss.backward(retain_graph=True)
            self.optim.step()
            pred = self.path.data.detach().clone()
        return self.path.data.detach().clone()
