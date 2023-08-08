import numpy as np
import pytest

from pathlib import Path

from exp_decay import ExponentialDecay
from ode import ODEModel, ODEResult, solve_ode, plot_ode_solution
from exception import InvalidInitialConditionError

model = ExponentialDecay(.4)

def test_ExponentialDecay():
    u = np.array([3.2])
    dudt = model(0, u)
    assert np.isclose(dudt[0], -1.28)

def test_negative_decay_raises_ValueError_1a():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-3)

def test_negative_decay_raises_ValueError_1b():
    with pytest.raises(ValueError):
        model.decay = -1.0

def test_num_states():
    assert model.num_states == 1

def test_solve_with_different_number_of_initial_states():
    u0 = np.array([1, 2])
    with pytest.raises(InvalidInitialConditionError):
        solve_ode(model, u0, 10, .1)

@pytest.mark.parametrize("a, u0, T, dt", [
    (3, np.array([10]), 1, .1),
    (4, np.array([3]), 10, .1),
    (0, np.array([6]), 15, .2)])

def test_solve_time(a, u0, T, dt):
    model = ExponentialDecay(a)
    res = solve_ode(model, u0, T, dt)

    t_computed = res.time
    t_expected = np.arange(0, T, dt)

    assert all(t_expected == t_computed)

@pytest.mark.parametrize("a, u0, T, dt", [
    (3, np.array([4]), 1, .1),
    (1, np.array([6]), 10, .1),
    (2, np.array([5]), 15, .2)])

def test_solve_solution(a, u0, T, dt):
    model = ExponentialDecay(a)
    res = solve_ode(model, u0, T, dt)

    t = np.arange(0, T, dt)

    sol_computed = res.solution
    sol_expected = u0*np.exp(-a*t)

    relative_error = np.linalg.norm(sol_computed - sol_expected) / np.linalg.norm(sol_expected)
    diff = abs(sol_computed - sol_expected)

    assert diff.all() < relative_error

def test_ODEResult():
    t_ = 3; sol_ = (2, 3)
    t = np.zeros(t_); sol = np.zeros(sol_)
    
    res = ODEResult(time=t, solution=sol)

    assert res.num_states == 2 and res.num_timepoints == 3

def test_plot_ode_solution_saves_file():
    res = solve_ode(model, u0=np.array([4.0]), T=10, dt=.01)
    filename = Path("test_plot.png")

    if filename.is_file():
        filename.unlink()

    plot_ode_solution(res, filename=filename)

    assert filename.is_file()
    filename.unlink()
