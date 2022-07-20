import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from deep_physics.dataloaders import PathDataModule
from deep_physics.functions import dummy, holder_table, rastrigin, sphere
from deep_physics.path_vae.pathnet import PathVAE
from deep_physics.path_vae.trainer import PathVAELearner
from deep_physics.storage import METRICS_REGISTRY, MODEL_ACTION_REGISTRY, MODEL_REGISTRY


def main():
    path_len = 20
    dim_x = 2
    step_size = 0.25
    function = holder_table
    kernel_sizes = (2, 4, 8)
    stride = 2
    dim_embedding = 32
    layers_transformer = 2
    n_heads_attention = 2
    dropout = 0
    dim_feedforward = 32
    use_pos_encoding = True
    dim_latent = 100
    # training
    warmup = 200
    lr = 1e-3
    batch_size = 64
    max_iters = 5000
    train_size = 128000
    val_size = 1280
    test_size = 200
    track_grad_norm = 2
    max_epochs = 800
    check_val_every_n_epoch = 1
    reload_dataloaders_every_n_epochs = 1
    num_workers = 64

    data_module = PathDataModule(
        function=function,
        path_len=path_len - 1,
        step_size=step_size,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_workers=num_workers,
    )

    momnet = PathVAE(
        path_len=path_len,
        dim_x=dim_x,
        kernel_sizes=kernel_sizes,
        stride=stride,
        dim_embedding=dim_embedding,
        layers_transformer=layers_transformer,
        n_heads_attention=n_heads_attention,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        use_pos_encoding=use_pos_encoding,
        dim_latent=dim_latent,
    )

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri=f"file:{METRICS_REGISTRY}"
    )
    checkpoint = ModelCheckpoint(dirpath=f"{MODEL_ACTION_REGISTRY}")
    learner = PathVAELearner(
        momnet,
        function=function,
        warmup=warmup,
        lr=lr,
        max_iters=max_iters,
        step_size=step_size,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint],
        accelerator="gpu",
        devices=1,
        logger=mlf_logger,
        track_grad_norm=track_grad_norm,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        # gradient_clip_val=10.,
        # gradient_clip_strategy="value",
    )
    trainer.fit(
        model=learner, datamodule=data_module
    )  # , ckpt_path=f"{MODEL_REGISTRY / 'epoch=99-step=119059.ckpt'}")


if __name__ == "__main__":
    sys.exit(main())
