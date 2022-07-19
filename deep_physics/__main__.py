import sys

import numpy as np
import pytorch_lightning as pl

from deep_physics.dataloaders import PathDataModule
from deep_physics.functions import holder_table  # rastrigin, sphere
from deep_physics.models import MomentumNet
from deep_physics.storage import METRICS_REGISTRY
from deep_physics.training import PathLearner


def main():
    # Parameters
    # PathDataModule
    path_len = 10
    dim_value = 1
    dim_x = 2
    step_size = 0.025
    function = holder_table

    # CrossEmbedLayer
    dim_in_ce = 1
    kernel_sizes = (2, 4, 8)
    dim_out_ce = 96
    stride_ce = 1
    n_channels_base = 32

    # PositionalEncoding
    d_model = (n_channels_base + dim_out_ce) * (
        (dim_value * dim_x) - 1
    )  # Cross embedding features dim

    # TransformerEncoder
    num_layers = 1
    num_heads = 1
    dim_feedforward = 64
    input_dim = d_model

    # LatentDistribution
    out_dim_latent = 121  # 256
    in_dim_latent = (path_len - 1) * d_model

    # resnet decoder
    latent_channels = 64
    # dim_out_resnet = 64
    groups = 1  # divisible by latent_channels
    norm = True
    dropout = 0.0
    dim_out_decoder = path_len * dim_value * dim_x
    latent_dim = int(np.sqrt(out_dim_latent / dim_value))
    dim_resnet_block = dim_value

    # training
    warmup = 200
    lr = 1e-5
    batch_size = 16
    max_iters = 5000
    train_size = 250000
    val_size = 1000
    test_size = 200
    track_grad_norm = -1
    max_epochs = 100000
    check_val_every_n_epoch = 1

    data_module = PathDataModule(
        function=function,
        path_len=path_len - 1,
        step_size=step_size,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_workers=64,
    )
    from pytorch_lightning.loggers import MLFlowLogger

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri=f"file:{METRICS_REGISTRY}"
    )

    momnet = MomentumNet(
        d_model=d_model,
        n_channels_base=n_channels_base,
        path_len=path_len,
        dim_in_ce=dim_in_ce,
        kernel_sizes=kernel_sizes,
        dim_out_ce=dim_out_ce,
        stride_ce=stride_ce,
        num_layers=num_layers,
        input_dim_transformer=input_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        in_dim_latent=in_dim_latent,
        out_dim_latent=out_dim_latent,
        dim_resnet=dim_resnet_block,
        latent_dim=latent_dim,
        latent_channels=latent_channels,
        groups=groups,
        dim_out_decoder=dim_out_decoder,
        dim_value=dim_value,
        dim_x=dim_x,
        norm=norm,
        dropout=dropout,
    )

    learner = PathLearner(
        momnet, function=function, warmup=warmup, lr=lr, max_iters=max_iters, step_size=step_size
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=mlf_logger,
        track_grad_norm=track_grad_norm,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    trainer.fit(model=learner, datamodule=data_module)


if __name__ == "__main__":
    sys.exit(main())
