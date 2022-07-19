import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

from deep_physics.physics import action, forces_newton


def split_path(path, n=1):
    first = path[:, :n]  # .unsqueeze(1)
    last = path[:, -n:]  # .unsqueeze(1)
    bound = torch.cat([first, last], 1)
    return bound, path[:, n:-n]


def reconstruct_path(bound, pred, n=1):
    first, last = bound[:, :n], bound[:, -n:]

    if len(first.shape) == 2 and len(bound.shape) == 4:  # (b, t, v, x)
        x0 = einops.rearrange(first, "b v x -> b 1 v x")
        xt = einops.rearrange(last, "b v x -> b 1 v x")
    elif len(first.shape) == 2 and len(bound.shape) == 3:  # (b, t, x)
        x0 = einops.rearrange(first, "b x -> b 1 x")
        xt = einops.rearrange(last, "b x -> b 1 x")
    elif len(first.shape) == 1 and len(bound.shape) == 2:  # (t, x)
        x0 = einops.rearrange(first, "1 x -> 1 1 x")
        xt = einops.rearrange(last, "1 x -> 1 1 x")
    else:
        x0, xt = first, last
    path = torch.cat([x0, pred, xt], dim=1)
    return path


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class PathLearner(pl.LightningModule):
    def __init__(self, model, function, step_size, lr=1e-4, warmup=50, max_iters=2000):
        super().__init__()
        self.model = model
        self.function = function
        self.mse_loss = nn.MSELoss()
        self.lr_scheduler = None
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters
        self.step_size = step_size

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        target, min_action = batch
        model_in = target.detach().clone()
        model_in[:, 3:-3] = -2.0
        pred = self.model(model_in)
        action_loss, mse_loss, entropy, _ = self.calculate_losses(pred, target, min_action)
        loss = mse_loss - 0.00001 * entropy + 0.001 * action_loss
        # loss = action_loss + newton - 0.0001 * entropy
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        # self.log("newton_loss", newton)
        self.log("action_loss", action_loss)
        self.log("mse_loss", mse_loss)
        self.log("entropy_loss", entropy)
        return loss

    def validation_step(self, batch, batch_idx):
        target, min_action = batch
        pred = self.model(target)
        action_loss, mse_loss, entropy, _ = self.calculate_losses(pred, target, min_action)
        loss = mse_loss - 0.00001 * entropy + 0.001 * action_loss
        # loss = action_loss + newton - 0.0001 * entropy
        self.log("val_train_loss", loss)
        # self.log("val_newton_loss", newton)
        self.log("val_action_loss", action_loss)
        self.log("val_mse_loss", mse_loss)
        self.log("val_entropy_loss", entropy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.lr_scheduler = CosineWarmupScheduler(
        #    optimizer, warmup=self.warmup, max_iters=self.max_iters
        # )
        return optimizer

    def calculate_losses(self, pred, target, min_action):

        mse_loss = self.mse_loss(pred, target)

        action_pred = self.calculate_action(pred, target)
        action_loss = (action_pred - min_action).mean()
        _, inner_pred = split_path(pred)
        # pred_path = self.function.from_scaled_pos(inner_pred)
        # forces_reg = forces_newton(pred_path, self.function, self.step_size).mean()
        entropy = self.model.latent_dist.dist.entropy().mean()
        return action_loss, mse_loss, entropy, 0

    def calculate_action(self, pred, target):
        bound_true, _ = split_path(target)
        _, inner_pred = split_path(pred)
        action_path_norm = reconstruct_path(bound_true, inner_pred)
        path = self.function.from_scaled_pos(action_path_norm)
        return action(path, self.function, return_grad=False)
