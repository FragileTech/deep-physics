import math

import einops
import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from deep_physics.models import (
    default,
    LatentDistribution,
    PositionalEncoding,
    ResnetBlock,
    TransformerEncoder,
)


class ToSquareConv(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=1):
        super().__init__()
        dim_out = default(dim_out, dim_in)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding="same",
                ),
            )

    def forward(self, x):
        x = einops.rearrange(x, "b t x -> b 1 x t")
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        feature_maps = torch.cat(fmaps, dim=1)
        return einops.rearrange(feature_maps, "b c x t -> b x t c")


class CrossEmbedLayerPos(nn.Module):
    def __init__(
        self,
        dim_in,
        path_len,
        kernel_sizes,
        dim_out=None,
        stride=2,
        kernels_to_square=(1, 2, (2, 4), (2, 8)),
    ):
        super().__init__()
        # kernel sizes are usually for the (3,7,15) Unet base embeddings, and they represent
        # conv features at different scales. Make sure the stride is even when kernel size is
        # even, and so forth.
        # This is not Imagen so we are not using square images.
        self.to_square = ToSquareConv(dim_in=1, dim_out=path_len, kernel_sizes=kernels_to_square)
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale. The scales represent number of features (filters)
        # and decreases in powers of two as scale n_dim / (2 ** n_scale). The rest of remaining
        # dimensions are assigned to the last conv layer.
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                ),
            )

    def forward_convs(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)

    def forward(self, x):
        x_square = self.to_square(x)
        return self.forward_convs(x_square)


class Decoder(nn.Module):
    def __init__(
        self,
        dim_resnet,
        latent_dim,
        latent_channels,
        path_len,
        dim_x,
        dim_value=1,
        groups=1,
        norm=True,
        dropout=0,
    ):
        super().__init__()
        self.dim_out = path_len * dim_x
        self.latent_dim = latent_dim
        self.dim_x = dim_x
        self.dim_value = dim_value
        self.path_len = path_len
        self.out_dim_resnet = latent_channels * latent_dim * latent_dim
        self.resnet = ResnetBlock(
            dim=dim_resnet,
            dim_out=latent_channels,
            groups=groups,
            norm=norm,
        )
        self.out = nn.Sequential(
            nn.Linear(self.out_dim_resnet, self.dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x, scale_shift=None):
        z = self.resnet(x)
        z = einops.rearrange(z, "b t r c -> b (t r c)")
        return self.out(z)

    def decode(self, x):
        x_pre_decoder = einops.rearrange(
            x,
            "b (v r c) -> b v r c",
            v=self.dim_value,
            r=self.latent_dim,
            c=self.latent_dim,
        )
        x_decoder = self(x_pre_decoder)
        x_out = einops.rearrange(
            x_decoder,
            "b (t x) -> b t x",
            t=self.path_len,
            x=self.dim_x,
        )
        return torch.tanh(x_out)


class AttentionDistEncoder(nn.Module):
    def __init__(
        self,
        path_len,
        dim_x,
        kernel_sizes=(2, 4, 8),
        stride=2,
        dim_embedding=37,
        layers_transformer=2,
        n_heads_attention=2,
        dropout=0,
        dim_feedforward=64,
        use_pos_encoding=True,
        dim_latent=43,
    ):
        super().__init__()
        self.path_len = path_len
        self.dim_values = dim_x
        self.input_dim_transformer = int((path_len / 2) ** 2)
        self.dim_in_latent = dim_embedding * self.input_dim_transformer
        self.use_pos_encoding = use_pos_encoding
        self.pos_encoder = PositionalEncoding(dim_x, path_len)
        self.cross_embed = CrossEmbedLayerPos(
            dim_in=dim_x,
            path_len=path_len,
            kernel_sizes=kernel_sizes,
            dim_out=dim_embedding,
            stride=stride,
        )
        self.transformer = TransformerEncoder(
            num_layers=layers_transformer,
            input_dim=self.input_dim_transformer,
            num_heads=n_heads_attention,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.latent = LatentDistribution(in_dim=self.dim_in_latent, out_dim=dim_latent)

    def forward(self, x):
        if self.use_pos_encoding:
            x = self.pos_encoder(x)
        x = self.cross_embed(x)
        x = einops.rearrange(x, "b c w h -> b c (w h)")
        x = self.transformer(x)
        x = self.latent(x)
        return x


class PathVAE(nn.Module):
    def __init__(
        self,
        path_len,
        dim_x,
        kernel_sizes=(2, 4, 8),
        stride=2,
        dim_embedding=37,
        layers_transformer=2,
        n_heads_attention=2,
        dropout=0,
        dim_feedforward=64,
        use_pos_encoding=True,
        dim_latent=100,
    ):
        super().__init__()
        self.encoder = AttentionDistEncoder(
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
        self.dim_dec = int(np.sqrt(dim_latent))
        self.decoder = Decoder(
            dim_resnet=1,
            latent_dim=self.dim_dec,
            latent_channels=self.dim_dec,
            path_len=path_len,
            dim_x=dim_x,
        )

    def forward(self, x):
        latents = self.encoder(x)
        return self.decoder.decode(latents)

    def decode_mean(self, x):
        _ = self.forward(x)
