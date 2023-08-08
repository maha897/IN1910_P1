import numpy as np
import pytest

from pendulum import Pendulum, exercise_2b, solve_pendulum

@pytest.mark.parametrize("L, theta, omega, dtomega", [
    (1.42, np.pi/6, .35, (-9.81/1.42)*np.sin(np.pi/6)), 
    (1.42, 0, 0, 0)])

def test_pendulum(L, theta, omega, dtomega):
    u = np.array([theta, omega])
    p = Pendulum(L)

    assert p(0, u)[0] == omega and p(0, u)[1] == dtomega

def test_solve_pendulum_ode_with_zero_ic():
    u0 = np.array([0, 0])
    res = exercise_2b(u0, 10, .1)
    print(res.solution)
    
    assert all(res.solution[0] == 0) and all(res.solution[1] == 0)

def test_solve_pendulum_function_zero_ic():
    results = solve_pendulum(np.array([0, 0]), 10, .1)

    assert all(results.theta == 0)
    assert all(results.omega == 0)
    assert all(results.x == 0)
    assert all(results.y == -results.pendulum.L)
