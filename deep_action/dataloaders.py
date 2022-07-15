import einops
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from deep_action.physics import action, discrete_diff, generate_trajectories


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


class TrajectoryDataset(Dataset):
    def __init__(self, function, n_trajectories, path_len, step_size):
        super().__init__()
        self.function = function
        self.n_trajectories = n_trajectories
        self.path_len = path_len
        self.step_size = step_size
        self.data = None
        self.action_data = None
        self.generate_trajectories()

    def generate_trajectories(self):
        data = generate_trajectories(
            self.function,
            self.function.bounds,
            n_batch=self.n_trajectories,
            steps=self.path_len,
            step_size=self.step_size,
            p0=None,
        )

        pos_raw, _, pots, pot_grads = [x.detach().clone() for x in data]

        max_x = self.function.bounds[0][1]  # Assumes domain is a square
        ix = (pos_raw.abs() < abs(max_x)).all(1).all(1)
        pos_raw = pos_raw[ix].detach().clone()
        pots = pots[ix].detach().clone()

        pos = einops.rearrange(pos_raw, "b t x -> b x t")
        norm_pos = self.function.to_scaled(pos)
        norm_pos = einops.rearrange(norm_pos, "b x t -> b t 1 x")

        norm_mom = discrete_diff(einops.rearrange(norm_pos, "b t 1 x -> b t x"))
        norm_mom = einops.rearrange(norm_mom, "b t x -> b t 1 x")

        norm_grads = pot_grads / pot_grads.norm(dim=-1).unsqueeze(-1)
        norm_grads = einops.rearrange(norm_grads, "b t x -> b t 1 x")

        delta_pots = discrete_diff(pots).repeat(1, 1, 2)
        delta_pots = einops.rearrange(delta_pots, "b t x -> b t 1 x")
        # This tensor has dimensions (batch, time, value_type, xy_values)
        self.data = torch.cat([norm_pos, norm_mom, norm_grads, delta_pots], dim=2)
        self.action_data = action(pos_raw, self.function, return_grad=False)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        path = self.data[item]
        act = self.action_data[item]
        return path, act


class PathDataModule(pl.LightningDataModule):
    def __init__(
        self,
        function,
        path_len,
        step_size,
        train_size=32000,
        val_size=500,
        test_size=200,
        **kwargs,
    ):
        super().__init__()
        self.function = function
        self.path_len = path_len
        self.step_size = step_size
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.dl_kwargs = kwargs

    # def prepare_data(self):
    # download, split, etc...
    # only called on 1 GPU/TPU in distributed
    # def setup(self, stage):
    # make assignments here (val/train/test split)
    # called on every process in DDP
    def train_dataloader(self):
        train_split = TrajectoryDataset(
            function=self.function,
            path_len=self.path_len,
            step_size=self.step_size,
            n_trajectories=self.train_size,
        )
        return DataLoader(train_split, **self.dl_kwargs)

    def val_dataloader(self):
        val_split = TrajectoryDataset(
            function=self.function,
            path_len=self.path_len,
            step_size=self.step_size,
            n_trajectories=self.val_size,
        )
        return DataLoader(val_split, **self.dl_kwargs)

    def test_dataloader(self):
        test_split = TrajectoryDataset(
            function=self.function,
            path_len=self.path_len,
            step_size=self.step_size,
            n_trajectories=self.test_size,
        )
        return DataLoader(test_split, **self.dl_kwargs)
