import sys

import numpy as np
import pytorch_lightning as pl

from deep_action.dataloaders import PathDataModule
from deep_action.functions import sphere
from deep_action.models import MomentumNet, PathLearner


def main():
    # Parameters
    # PathDataModule
    path_len = 5
    dim_value = 4
    dim_x = 2
    step_size = 0.05
    function = sphere
    batch_size = 32

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
    out_dim_latent = 144  # 256
    in_dim_latent = (path_len - 1) * d_model

    # resnet decoder
    latent_channels = 32
    dim_out_resnet = 32
    groups = 2  # divisible by latent_channels
    norm = True
    dropout = 0.0
    dim_out_decoder = path_len * dim_value * dim_x
    latent_dim = int(np.sqrt(out_dim_latent / dim_value))
    dim_resnet_block = dim_value

    # training
    warmup = 50
    lr = 1e-4

    data_module = PathDataModule(
        function=function,
        path_len=path_len - 1,
        step_size=step_size,
        batch_size=batch_size,
        num_workers=64,
    )
    from pytorch_lightning.loggers import MLFlowLogger

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs")

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

    learner = PathLearner(momnet, sphere, warmup=warmup, lr=lr)
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=mlf_logger)
    trainer.fit(model=learner, datamodule=data_module)


if __name__ == "__main__":
    sys.exit(main())
