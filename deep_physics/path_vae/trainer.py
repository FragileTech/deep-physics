import numpy as np
import pytorch_lightning as pl
from torch import nn, optim

from deep_physics.physics import action
from deep_physics.training import reconstruct_path, split_path


class PathVAELearner(pl.LightningModule):
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        target, min_action = batch
        target = target[:, :, 0]
        model_in = target.detach().clone()
        mask = np.random.randint(low=1, high=target.shape[1] - 1)
        model_in[:, mask - 1 : mask + 1] = -2.0
        pred = self.model(model_in)
        mse_loss, entropy, action_loss = self.calculate_losses(pred, target)
        loss = mse_loss + 0.1 * action_loss - 0.00001 * entropy
        # loss = action_loss + newton - 0.0001 * entropy
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log("action_loss", action_loss)
        self.log("mse_loss", mse_loss)
        self.log("entropy_loss", entropy)
        return loss

    def validation_step(self, batch, batch_idx):
        target, min_action = batch
        target = target[:, :, 0]
        model_in = target.detach().clone()
        mask = np.random.randint(low=1, high=target.shape[1] - 1)
        model_in[:, mask - 1 : mask + 1] = -2.0
        pred = self.model(model_in)
        mse_loss, entropy, action_loss = self.calculate_losses(pred, target)
        loss = mse_loss + 0.1 * action_loss - 0.00001 * entropy
        self.log("val_train_loss", loss)
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

    def calculate_losses(self, pred, target):
        action_loss = self.calculate_action(pred, target)
        mse_loss = self.mse_loss(pred, target)
        entropy = self.model.encoder.latent.dist.entropy().mean()
        return mse_loss, entropy, action_loss

    def calculate_action(self, pred, target):
        bound_true, _ = split_path(target)
        _, inner_pred = split_path(pred)
        action_path_norm = reconstruct_path(bound_true, inner_pred)
        path = self.function.from_scaled_pos(action_path_norm)
        return action(path, self.function, return_grad=False).mean()
