import pytest

from deep_action.physics import generate_trajectories, potential_energy
from deep_action.functions import holder_table


@pytest.fixture()
def trajectories():
    n_batch = 1000
    bounds = [(-5, 5), (-5, 5)]
    steps = 50
    step_size = 1
    function = holder_table

    pos, mom, pots, pot_grads = generate_trajectories(
        function, bounds, n_batch=n_batch, steps=steps, step_size=step_size
    )
    return (
        pos.detach().clone(),
        mom.detach().clone(),
        pots.detach().clone(),
        pot_grads.detach().clone(),
        function,
    )


def test_potential_energy(trajectories):
    pos, mom, pots, pot_grads, pot_func = trajectories
    pots = potential_energy(pos, pot_func, reset_grad=True, return_grad=False)
    pots, grad = potential_energy(
        pos.detach().clone(), pot_func, reset_grad=True, return_grad=True
    )
    pots, grad = potential_energy(pos, pot_func, reset_grad=False, return_grad=True)
