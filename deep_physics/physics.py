import einops
import torch


def leapfrog(
    init_pos,
    init_momentum,
    grad,
    step_size,
    function,
    force=None,
    reset_grad: bool = True,
):
    """Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    init_pos: ndarray[float, ndim=1]
        initial parameter position
    init_momentum: ndarray[float, ndim=1]
        initial momentum
    grad: float
        initial gradient value
    step_size: float
        step size
    potential: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = potential(theta)
    OUTPUTS
    -------
    new_position: ndarray[float, ndim=1]
        new parameter position
    new_momentum: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    new_potential: float
        new lnp
    """
    force = 0.0 if force is None else force
    # make half step in init_momentum
    new_momentum = init_momentum + 0.5 * step_size * (force - grad)
    # make new step in theta
    new_position = init_pos + step_size * new_momentum
    # compute new gradient
    new_potential, gradprime = function(new_position, reset_grad=reset_grad)
    # make half step in init_momentum again
    new_momentum = new_momentum + 0.5 * step_size * (force - gradprime)
    return new_position, new_momentum, gradprime, new_potential


def generate_trajectories(function, bounds, n_batch, steps=18, step_size=0.5, x0=None, p0=None):
    (x_min, x_max), (y_min, y_max) = bounds
    if x0 is None:
        x0 = torch.rand((n_batch, 2))
        x0[:, 0] = x0[:, 0] * (x_max - x_min) + x_min
        x0[:, 1] = x0[:, 1] * (y_max - y_min) + y_min

    x0.requires_grad_()
    if p0 is None:
        p0 = torch.zeros((n_batch, 2))
    v0, grad0 = function(x0)
    positions = [x0[None, :]]
    momentums = [p0[None, :]]
    potentials = [v0[None, :]]
    pot_grads = [grad0[None, :]]
    for _ in range(steps):
        x1, p1, grad1, v1 = leapfrog(
            init_pos=x0.detach(),
            init_momentum=p0,
            grad=grad0,
            step_size=step_size,
            function=function,
            force=0.0,
        )
        positions.append(x1[None, :])
        momentums.append(p1[None, :])
        potentials.append(v1[None, :])
        pot_grads.append(grad1[None, :])
        x0, p0, grad0 = x1, p1, grad1

    pos = torch.cat(positions)
    mom = torch.cat(momentums)
    pots = torch.cat(potentials)
    pot_grads = torch.cat(pot_grads)

    pos = einops.rearrange(pos, "t b x -> b t x")
    mom = einops.rearrange(mom, "t b x -> b t x")
    pots = einops.rearrange(pots, "t b -> b t 1")
    pot_grads = einops.rearrange(pot_grads, "t b x -> b t x")
    return pos, mom, pots, pot_grads


def discrete_diff(x, x0=None, keepdims: bool = True):
    deltas = x[:, 1:] - x[:, :-1]
    if not keepdims:
        return deltas
    if x0 is None:
        bs, xdim = x.shape[0], x.shape[-1]
        x0 = torch.zeros((bs, 1, xdim), device=x.device)
    return torch.cat([x0, deltas], 1)  # Shape (batch, time, xdims)


def kinetic_energy(pos, p0=None, p=None, m=1):
    p = discrete_diff(pos, p0) if p is None else p
    vec_ke = m * 0.5 * p.pow(2)
    return einops.reduce(vec_ke, "b t x -> b t", "sum")


def potential_energy(
    pos,
    potential,
    reset_grad: bool = False,
    return_grad: bool = True,
    scaled: bool = False,
):
    x = einops.rearrange(pos, "b t x -> (b t) x")
    data = potential(x, reset_grad=reset_grad, return_grad=return_grad, scaled=scaled)
    pot = einops.rearrange(data[0] if return_grad else data, "(b t) -> b t", b=pos.shape[0])
    if return_grad:
        grads = einops.rearrange(data[1], "(b t) x -> b t x", b=pos.shape[0])
        return pot, grads
    return pot


def delta_potential(
    pos,
    potential,
    dv0=None,
    keepdims: bool = True,
    reset_grad: bool = False,
    return_grad: bool = True,
    scaled: bool = False,
):
    data = potential_energy(
        pos,
        potential,
        reset_grad=reset_grad,
        return_grad=return_grad,
        scaled=scaled,
    )
    pot = data[0] if return_grad else data
    delta_pot = discrete_diff(pot.unsqueeze(-1), dv0, keepdims=keepdims).squeeze(-1)
    return (delta_pot, data[1]) if return_grad else delta_pot


def lagrangian(
    pos,
    p,
    potential,
    p0,
    dv0=None,
    reset_grad: bool = False,
    return_grad: bool = True,
    scaled: bool = False,
):
    pos = potential.scale(pos) if scaled else pos
    ke = kinetic_energy(pos, p0, p)
    data = delta_potential(
        pos,
        potential,
        dv0=dv0,
        reset_grad=reset_grad,
        return_grad=return_grad,
        scaled=False,
    )
    delta_pot = data[0] if return_grad else data
    lagrangian_ = ke - delta_pot
    return (lagrangian_, data[1]) if return_grad else lagrangian_


def action(
    pos,
    potential,
    p=None,
    p0=None,
    dv0=None,
    reset_grad: bool = False,
    return_grad: bool = True,
    scaled: bool = False,
):
    data = lagrangian(
        pos,
        p=p,
        potential=potential,
        p0=p0,
        dv0=dv0,
        return_grad=return_grad,
        reset_grad=reset_grad,
        scaled=scaled,
    )
    lagrangian_ = einops.reduce(data[0] if return_grad else data, "b t -> b", "sum")
    return (lagrangian_, data[1]) if return_grad else lagrangian_


def variate(pos, dt=0.05):
    pos = pos.detach().clone()
    pos[:, 1:-1] = pos[:, 1:-1] * (1 - dt * torch.randn_like(pos[:, 1:-1]).abs())
    return pos
