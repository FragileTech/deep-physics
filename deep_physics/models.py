import math

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from deep_physics.physics import action


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def exists(val):
    return val is not None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, kernel_sizes, n_channels_base=64, dim_out=None, stride=1):
        super().__init__()
        # kernel sizes are usually for the (3,7,15) Unet base embeddings, and they represent
        # conv features at different scales. Make sure the stride is even when kernel size is
        # even, and so forth.
        # This is not Imagen so we are not using square images.
        # assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale. The scales represent number of features (filters)
        # and decreases in powers of two as scale n_dim / (2 ** n_scale). The rest of remaining
        # dimensions are assigned to the last conv layer.
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # conv layer for path estimation with kernel size 1
        self.base_conv = nn.Conv2d(1, n_channels_base, 1, stride=1, padding=0)
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2)
            )

    def forward(self, x):
        # Ignore repeated potential and don't account for initial state
        # This is a trick to so we can concatenate a kernel of size 1
        xbase = x[:, :, :-1, 1:]
        base_conv_feats = self.base_conv(xbase)
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat((base_conv_feats,) + fmaps, dim=1)


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = einops.rearrange(values, "b h t v -> b t (h v)")
        # This is wrong! forgets about attention heads so the reshape fails
        # for more than one head,
        # values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        # values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, norm=True):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        groups=8,
        norm=True,
    ):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, norm=True)
        self.block2 = Block(dim_out, dim_out, groups=groups, norm=True)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h, scale_shift=None)
        return h + self.res_conv(x)


class Decoder(nn.Module):
    def __init__(
        self, dim_resnet, latent_dim, latent_channels, dim_out, groups, norm=True, dropout=0
    ):
        super().__init__()
        self.dim_out = dim_out
        self.out_dim_resnet = latent_channels * latent_dim * latent_dim
        self.resnet = ResnetBlock(
            dim=dim_resnet, dim_out=latent_channels, groups=groups, norm=norm
        )
        self.out = nn.Sequential(
            nn.Linear(self.out_dim_resnet, self.dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x, scale_shift=None):
        z = self.resnet(x)
        z = einops.rearrange(z, "b t r c -> b (t r c)")
        return self.out(z)


class LatentDistribution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.cov_mat_dim = int(self.out_dim * self.out_dim)

        self.cov_linear = nn.Linear(self.input_dim, self.cov_mat_dim)
        nn.init.eye_(self.cov_linear.weight)
        self.cov_layer = nn.Sequential(self.cov_linear, nn.ReLU())
        self.mean_layer = nn.Sequential(nn.Linear(self.input_dim, self.out_dim), nn.ReLU())
        self.dist = None

    def forward(self, x):
        z = einops.rearrange(x, "b t v -> b (t v)")
        z_pre_cov = torch.sigmoid(self.cov_linear(z))
        z_cov = einops.rearrange(z_pre_cov, "b (r c) -> b r c", r=self.out_dim, c=self.out_dim)
        z_mean = torch.tanh(self.mean_layer(z))
        self.dist = MultivariateNormal(z_mean, scale_tril=torch.tril(z_cov))
        return self.dist.rsample()


class MomentumNet(nn.Module):
    ndim_measures = 2

    def __init__(
        self,
        d_model,
        path_len,
        dim_in_ce,
        n_channels_base,
        kernel_sizes,
        dim_out_ce,
        stride_ce,
        num_layers,
        input_dim_transformer,
        num_heads,
        dim_feedforward,
        dropout,
        in_dim_latent,
        out_dim_latent,
        dim_resnet,
        latent_dim,
        latent_channels,
        groups,
        dim_out_decoder,
        dim_value,
        dim_x,
        norm=True,
    ):
        super(MomentumNet, self).__init__()
        self.path_len = path_len
        self.dim_value = dim_value
        self.dim_x = dim_x
        self.latent_dim = latent_dim
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=path_len)
        self.cross_embed = CrossEmbedLayer(
            dim_in=dim_in_ce,
            kernel_sizes=kernel_sizes,
            dim_out=dim_out_ce,
            stride=stride_ce,
            n_channels_base=n_channels_base,
        )
        self.transf_enc = TransformerEncoder(
            num_layers=num_layers,
            input_dim=input_dim_transformer,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.latent_dist = LatentDistribution(in_dim=in_dim_latent, out_dim=out_dim_latent)

        self.decoder = Decoder(
            dim_resnet=dim_resnet,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            groups=groups,
            dim_out=dim_out_decoder,
            norm=norm,
            dropout=dropout,
        )

    def encode(self, x):
        x = einops.rearrange(x, "b t v x -> b 1 (v x) t")
        x_embed = self.cross_embed(x)
        x_pre_enc = einops.rearrange(x_embed, "b v s t -> b t (v s)")
        x_post_enc = self.pos_encoding(x_pre_enc)
        x_enc = einops.rearrange(x_post_enc, "b t v -> b t v")
        x_attention = self.transf_enc(x_enc)
        return x_attention

    def decode(self, x):
        x_pre_decoder = einops.rearrange(
            x,
            "b (v r c) -> b v r c",
            v=self.dim_value,
            r=self.latent_dim,
            c=self.latent_dim,
        )
        x_decoder = self.decoder(x_pre_decoder)
        x_out = einops.rearrange(
            x_decoder,
            "b (t v x) -> b t v x",
            t=self.path_len,
            v=self.dim_value,
            x=self.dim_x,
        )
        return torch.tanh(x_out)

    def forward(self, x):
        x_attention = self.encode(x)
        x_latent = self.latent_dist(x_attention)
        return self.decode(x_latent)


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
    def __init__(self, model, function):
        super().__init__()
        self.model = model
        self.function = function
        self.mse_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        target, min_action = batch
        pred = self.model(target)
        action_loss, mse_loss, entropy = self.calculate_losses(pred, target, min_action)
        #loss = mse_loss + 0.01 * action_loss - 0.00001 * entropy
        loss = action_loss - 0.00001 * entropy
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log("action_loss", action_loss)
        self.log("mse_loss", mse_loss)
        self.log("entropy_loss", entropy)
        return loss

    def validation_step(self, batch, batch_idx):
        target, min_action = batch
        pred = self.model(target)
        action_loss, mse_loss, entropy = self.calculate_losses(pred, target, min_action)
        #loss = mse_loss + 0.1 * action_loss - 0.00001 * entropy
        loss = action_loss - 0.00001 * entropy
        self.log("val_train_loss", loss)
        self.log("val_action_loss", action_loss)
        self.log("val_mse_loss", mse_loss)
        self.log("val_entropy_loss", entropy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def calculate_losses(self, pred, target, min_action):
        mse_loss = self.mse_loss(pred, target)
        action_pred = self.calculate_action(pred, target)
        action_loss = (action_pred - min_action).mean()
        entropy = self.model.latent_dist.dist.entropy().mean()
        return action_loss, mse_loss, entropy

    def calculate_action(self, pred, target):
        bound_true, _ = split_path(target)
        _, inner_pred = split_path(pred)
        action_path_norm = reconstruct_path(bound_true, inner_pred)
        pos_norm = action_path_norm[:, :, 0]
        path = self.function.from_scaled(pos_norm)
        return action(path, self.function, return_grad=False)


class PathLearner(pl.LightningModule):
    def __init__(self, model, function, lr=1e-4, warmup=50, max_iters=2000):
        super().__init__()
        self.model = model
        self.function = function
        self.mse_loss = nn.MSELoss()
        self.lr_scheduler = None
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        target, min_action = batch
        pred = self.model(target)
        action_loss, mse_loss, entropy = self.calculate_losses(pred, target, min_action)
        loss = mse_loss + 0.01 * action_loss - 0.00001 * entropy
        #loss = action_loss  - 0.01 * entropy
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log("action_loss", action_loss)
        self.log("mse_loss", mse_loss)
        self.log("entropy_loss", entropy)
        return loss

    def validation_step(self, batch, batch_idx):
        target, min_action = batch
        pred = self.model(target)
        action_loss, mse_loss, entropy = self.calculate_losses(pred, target, min_action)
        loss = mse_loss + 0.1 * action_loss - 0.00001 * entropy
        #loss = action_loss  - 0.001 * entropy
        self.log("val_train_loss", loss)
        self.log("val_action_loss", action_loss)
        self.log("val_mse_loss", mse_loss)
        self.log("val_entropy_loss", entropy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #self.lr_scheduler = CosineWarmupScheduler(
        #    optimizer, warmup=self.warmup, max_iters=self.max_iters
        #)
        return optimizer

    def calculate_losses(self, pred, target, min_action):
        mse_loss = self.mse_loss(pred, target)
        action_pred = self.calculate_action(pred, target)
        action_loss = (action_pred - min_action).mean()
        entropy = self.model.latent_dist.dist.entropy().mean()
        return action_loss, mse_loss, entropy

    def calculate_action(self, pred, target):
        bound_true, _ = split_path(target)
        _, inner_pred = split_path(pred)
        action_path_norm = reconstruct_path(bound_true, inner_pred)
        pos_norm = action_path_norm[:, :, 0]
        path = self.function.from_scaled(pos_norm)
        return action(path, self.function, return_grad=False)
