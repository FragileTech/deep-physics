from pathlib import Path
import sys
import warnings

import einops
import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from deep_physics.functions import holder_table
from deep_physics.path_vae.pathnet import PathVAE
from deep_physics.path_vae.trainer import PathVAELearner, reconstruct_path, split_path
from deep_physics.system import PathOptimizerModule, System


def plot_landscape(function, n_points=100, **opts):
    (x_min, x_max), (y_min, y_max) = function.bounds
    xi = np.linspace(x_min, x_max, n_points)
    yi = np.linspace(y_min, y_max, n_points)
    xx, yy = np.meshgrid(xi, yi)
    ins = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
    zs, _ = function(torch.from_numpy(ins), reset_grad=True)
    zs = einops.asnumpy(zs.reshape(n_points, n_points))
    landscape = hv.QuadMesh((xx, yy, zs)).opts(**opts)
    return (landscape * hv.operation.contours(landscape)).opts(
        hv.opts.Contours(show_legend=False, cmap="gray"),
        hv.opts.QuadMesh(colorbar=True, cmap="gray"),
    )


def get_lims(df, function, eps=0.5):
    l = df[["x", "y"]].describe().loc[["min", "max"]]
    (xmin, xmax), (ymin, ymax) = l["x"].values.tolist(), l["y"].values.tolist()
    (xmin_b, xmax_b), (ymin_b, ymax_b) = function.bounds
    (xmin, xmax) = (max(xmin - eps, xmin_b), min(xmax + eps, xmax_b))
    (ymin, ymax) = (max(ymin - eps, ymin_b), min(ymax + eps, ymax_b))
    dx, dy = (xmax - xmin), (ymax - ymin)
    if dx > dy:
        (xmin, xmax), (ymin, ymax) = (xmin, xmax), (ymin - (dx - dy) / 2, ymax + (dx - dy) / 2)
    else:
        (xmin, xmax), (ymin, ymax) = (xmin - (dy - dx) / 2, xmax - (dy - dx) / 2), (ymin, ymax)
    (xmin, xmax) = (max(xmin, xmin_b), min(xmax, xmax_b))
    (ymin, ymax) = (max(ymin, ymin_b), min(ymax, ymax_b))
    xlim, ylim = (xmin, xmax), (ymin, ymax)
    # Todo: correct adding more distante to the other end when there are bounds near
    return tuple(xlim), tuple(ylim)


def plot_path(df, label="", point_color="time", line_color="black"):
    path_plot = df.hvplot.line(x="x", y="y", label=f"Path {label}", color=line_color)
    pos_plot = df.hvplot.scatter(
        x="x", y="y", label=f"Position {label}", color=point_color, size=30
    )
    pos_plot = pos_plot.opts(cmap="rainbow", colorbar=False)
    trajectory_plot = path_plot * pos_plot
    return trajectory_plot


def plot_trajectory(
    df,
    potential,
    eps=0.5,
    label="",
    step_size=1,
    landscape: bool = True,
    points_landscape=100,
    plot_potential: bool = True,
    plot_acc: bool = True,
    plot_momentum: bool = True,
    zoom_xlim: bool = False,
):
    rl = False
    trajectory_plot = plot_path(df, label=label)
    if plot_potential:
        pot_vec_plot = hv.VectorField(
            df, kdims=["x", "y"], vdims=["gva", "fvm"], label=f"F pot {label}"
        )
        pot_vec_plot = pot_vec_plot.opts(
            color="blue",
            rescale_lengths=rl,
            magnitude=hv.dim("fvm") * step_size**2,
            line_width=2,
            pivot="tail",
        )
        trajectory_plot = pot_vec_plot * trajectory_plot
    if plot_momentum:
        mom_vec_plot = hv.VectorField(
            df,
            kdims=["x", "y"],
            vdims=["pa", "pm"],
            label=f"Momentum {label}",
        )
        mom_vec_plot = mom_vec_plot.opts(
            color="green",
            line_width=2,
            rescale_lengths=rl,
            pivot="tail",
            alpha=1.0,
            magnitude=hv.dim("pm"),
        )
        trajectory_plot = mom_vec_plot * trajectory_plot
    if plot_acc:
        f_vec_plot = hv.VectorField(df, kdims=["x", "y"], vdims=["fa", "fm"], label=f"Acc {label}")
        f_vec_plot = f_vec_plot.opts(
            color="red", magnitude=hv.dim("fm"), rescale_lengths=rl, line_width=2, pivot="tip"
        )
        f_vec_plot_1 = hv.VectorField(
            df, kdims=["x", "y"], vdims=["fa", "fm"], label=f"Force {label}"
        )
        f_vec_plot_1 = f_vec_plot_1.opts(
            color="orange", magnitude=hv.dim("fm"), rescale_lengths=rl, line_width=2, pivot="tail"
        )
        trajectory_plot = f_vec_plot * trajectory_plot * f_vec_plot_1

    if landscape:
        landscape = plot_landscape(potential, n_points=points_landscape) * trajectory_plot
        return landscape
    return trajectory_plot


class PathOptimizer(param.Parameterized):
    lr = param.Number(0.1, precedence=-1)
    n_iters = param.Integer(1, bounds=(1, 5000), precedence=-1)
    minimize = param.Selector(["Action", "F=ma", "MSE"])
    path_opt = param.ClassSelector(PathOptimizerModule, precedence=-1)

    @param.depends("system", "lr", "minimize", "n_iters")
    def update(self):
        self.path_opt.lr = self.lr
        self.path_opt.n_iters = self.n_iters
        self.path_opt.minimize = self.minimize

    def __call__(self, pred, target, n_split):
        return self.path_opt(pred, target, n_split)


path_len = 20


class VAEDashboard(param.Parameterized):
    landscape = param.Boolean(False)
    potential = param.Boolean(False)
    acceleration = param.Boolean(False)
    momentum = param.Boolean(False)
    sample_index = param.Integer(0, bounds=(0, 20))
    path_select = param.ListSelector(["Target", "Predicted"], objects=["Target", "Predicted"])
    model = param.ClassSelector(PathVAELearner, instantiate=False, precedence=-1)
    target = param.ClassSelector(torch.Tensor, precedence=-1)
    pred = param.ClassSelector(torch.Tensor, precedence=-1)
    system = param.ClassSelector(System, precedence=-1)
    sample_paths = pn.widgets.Button(name="Sample paths")
    predict = pn.widgets.Button(name="Predict")
    decode_mean = param.Boolean(False)
    attach_ends = param.Boolean(False)
    n_attach = param.Integer(1, bounds=(1, int((path_len + 1) / 2)))
    full_pred = param.ClassSelector(torch.Tensor, precedence=-1)
    trigger = param.Boolean(False, precedence=-1)
    optimizer = param.ClassSelector(PathOptimizer, precedence=-1)
    base_net = param.ClassSelector(PathVAE, precedence=-1)
    run_opt = pn.widgets.Button(name="Refine")
    toggle_cheats = pn.widgets.Toggle(name="Cheat mode", value=False)
    toggle_model = pn.widgets.RadioButtonGroup(options=["MSE", "Action reg"], name="Model")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_panel, self.cheats_panel = self.create_param_dashboard()
        self.cheats_panel.height = self.param_panel.height

    @param.depends(
        "landscape",
        "potential",
        "acceleration",
        "momentum",
        "sample_index",
        "path_select",
        "trigger",
    )
    def plot_prediction(self):
        df_pred = self.system.to_df(self.pred[self.sample_index])
        df_target = self.system.to_df(self.target[self.sample_index])
        plot_target = None
        plot = None
        if "Predicted" in self.path_select:
            plot = plot_trajectory(
                df_pred,
                label="Pred",
                potential=self.system.potential,
                landscape=False,
                plot_potential=self.potential,
                plot_acc=self.acceleration,
                plot_momentum=self.momentum,
                step_size=self.system.step_size,
            )
        if "Target" in self.path_select:
            plot_target = plot_trajectory(
                df_target,
                label="Target",
                potential=self.system.potential,
                landscape=self.landscape,
                plot_potential=self.potential,
                plot_acc=self.acceleration,
                plot_momentum=self.momentum,
                step_size=self.system.step_size,
            ).opts(hv.opts.Curve(line_dash="dotted"))
            plot_target = plot_target if plot is None else plot_target * plot
        plot = plot_target if plot_target is not None else plot
        return plot.opts(hv.opts.QuadMesh(colorbar=True, cmap="blues")).opts(
            width=1480, height=800
        )

    @param.depends("sample_paths.clicks")
    def sample_new_paths(self):
        self.target, *_ = self.system.sample_trajectories(samples=100, as_numpy=False)
        self.predict_model()

    @param.depends("attach_ends", "n_attach")
    def update_attach(self):
        if self.attach_ends:
            bound_true, _ = split_path(self.target, n=self.n_attach)
            _, inner_pred = split_path(self.full_pred, n=self.n_attach)
            preds = reconstruct_path(bound_true, inner_pred, n=self.n_attach)
            self.pred = preds.detach().clone()
        else:
            self.pred = self.full_pred.detach().clone()

        self.trigger = not self.trigger

    @param.depends(
        "predict.clicks",
        "decode_mean",
    )
    def predict_model(self):
        model_in = self.system.potential.to_scaled_pos(self.target.detach())
        preds = self.model(model_in).detach()
        if self.decode_mean:
            mean_vec = self.model.model.encoder.latent.dist.mean
            preds = self.model.model.decoder.decode(mean_vec)
        self.full_pred = self.system.potential.from_scaled_pos(preds).detach().clone()
        self.update_attach()
        self.trigger = not self.trigger

    def panel(self):
        dashboard = pn.Column(
            self.plot_prediction,
            pn.Row(self.param_panel, pn.Column(self.toggle_cheats, self.cheats_panel)),
            self.sample_new_paths,
            self.predict_model,
            self.update_attach,
            self.refine_trajectory,
            self.show_cheats,
            self.select_model,
        )
        return dashboard

    @param.depends("run_opt.clicks")
    def refine_trajectory(self):
        if self.run_opt.clicks > 0:
            self.pred[self.sample_index] = self.optimizer(
                self.pred[self.sample_index].unsqueeze(0),
                self.target[self.sample_index].unsqueeze(0),
                self.n_attach,
            )
        self.trigger = not self.trigger

    @param.depends("toggle_cheats.value")
    def show_cheats(self):
        self.cheats_panel.visible = self.toggle_cheats.value
        if self.toggle_cheats.value:
            self.cheats_panel.sizing_mode = "stretch_height"
            self.toggle_cheats.visible = False

    @param.depends("toggle_model.value")
    def select_model(self):
        checkpoint = (
            "epoch=588-step=701131.ckpt"
            if self.toggle_model.value == "MSE"
            else "epoch=218-step=260763.ckpt"
        )
        self.model = PathVAELearner.load_from_checkpoint(
            str(Path(__file__).parent.parent / checkpoint),
            model=self.base_net,
            function=self.system.potential,
            step_size=self.system.step_size,
        )
        self.predict_model()

    def create_param_dashboard(self):
        widgets_dict = {
            "landscape": pn.widgets.Toggle(name="Landscape"),
            "potential": pn.widgets.Toggle(name="Potential"),
            "acceleration": pn.widgets.Toggle(name="Acceleration"),
            "momentum": pn.widgets.Toggle(name="Momentum"),
            "path_select": pn.widgets.ToggleGroup,
            "decode_mean": pn.widgets.Toggle(name="Decode mean"),
            "attach_ends": pn.widgets.Toggle(name="Attach ends"),
        }
        widgets = pn.Param(self.param, widgets=widgets_dict)
        title, landscape, potential, acceleration, momentum, sample_index = widgets[:6]
        path_select, decode_mean, attach_ends, n_attach, *_ = widgets[6:]
        param_panel = pn.Column(
            title,
            pn.Row(landscape, potential),
            pn.Row(acceleration, momentum),
            pn.Row(decode_mean, self.toggle_model),
            path_select,
            sample_index,
            self.predict,
            self.sample_paths,
        )
        opt_title, minimize = pn.Param(self.optimizer)
        cheat_panel = pn.Column(opt_title, attach_ends, n_attach, minimize, self.run_opt)
        return param_panel, cheat_panel


def main():
    path_len = 20
    dim_x = 2
    step_size = 0.25
    potential = holder_table
    kernel_sizes = (2, 4, 8)
    stride = 2
    dim_embedding = 128
    layers_transformer = 2
    n_heads_attention = 4
    dropout = 0
    dim_feedforward = 64
    use_pos_encoding = True
    dim_latent = 100
    # training
    warmup = 200
    lr = 1e-6
    max_iters = 5000

    warnings.filterwarnings("ignore")
    hv.extension("bokeh")
    pd.options.plotting.backend = "holoviews"
    pn.extension(template="material", theme="dark")
    net = PathVAE(
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
    pathvae = PathVAELearner(
        net,
        function=potential,
        warmup=warmup,
        lr=lr,
        max_iters=max_iters,
        step_size=step_size,
    )
    system = System(potential, path_len - 1, step_size)
    pos, mom, pots, pot_grads = system.sample_trajectories(samples=10, as_numpy=False)
    preds = pathvae(system.potential.to_scaled_pos(pos))
    pom = PathOptimizerModule(system=system)
    path_optimizer = PathOptimizer(path_opt=pom)
    dashboard = VAEDashboard(
        base_net=net,
        model=pathvae,
        target=pos.detach().clone(),
        pred=system.potential.from_scaled_pos(preds.detach().clone()),
        system=system,
        optimizer=path_optimizer,
    )
    pn.serve(dashboard.panel(), template="material")


if __name__ == "__main__":
    sys.exit(main())
