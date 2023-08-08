import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from dataclasses import dataclass

from ode import ODEModel, ODEResult, solve_ode, plot_ode_solution


class Pendulum(ODEModel):
    def __init__(self, L: float = 1, M: float = 1, g: float = 9.81) -> None: 
        """
        Constructor of Pendulum
        
        Input 
            - L: float
                rod length, default is set to 1
            - M: float 
                mass og pendulum, default is set to 1
            - g: float 
                the gravitational acceleration, default is set to 9.81
        """

        self.M = M 
        self.L = L
        self.g = g 

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Call function that returns the derivate of u at time t
        
        Input
            - t: float
                time
            - u: array
                consisting of theta and omega

        Output
            - an array of the derivative dudt
        """

        theta, omega = u
        dtheta = omega
        domega = (-self.g/self.L)*np.sin(theta)

        return np.array([dtheta, domega])
        
    @property
    def num_states(self) -> int:
        """
        Returns the number of states. u = [theta, omega] so it will return 2.

        Output
            - an int, the number of state variables
        """
        return 2


def exercise_2b(
    u0: np.ndarray,
    T: float,
    dt: float,
) -> ODEResult:
    """
    Creates an instance of Pendulum and solves ODE using solve_ode function.

    Input
        - u0: array
            the initial condition of u
        - T: float 
            end time
        - dt: float
            time steps
    
    Output
        - ODEResult containing the timepoints and solutions of ODE
    """

    model = Pendulum()
    result = solve_ode(model, u0, T, dt)

    return result


@dataclass
class PendulumResults:
    results: ODEResult
    pendulum: Pendulum

    @property
    def theta(self) -> np.ndarray:
        return self.results.solution[0]

    @property 
    def omega(self) -> np.ndarray:
        return self.results.solution[1]

    @property
    def x(self) -> np.ndarray:
        return self.pendulum.L*np.sin(self.theta)

    @property 
    def y(self) -> np.ndarray:
        return -self.pendulum.L*np.cos(self.theta)

    @property 
    def potential_energy(self) -> np.ndarray:
        return self.pendulum.g*(self.y + self.pendulum.L)

    @property
    def vx(self) -> np.ndarray:
        return np.gradient(self.x, self.results.time)

    @property 
    def vy(self) -> np.ndarray:
        return np.gradient(self.y, self.results.time)
    
    @property
    def kinetic_energy(self) -> np.ndarray:
        return (1/2)*(self.vx**2 + self.vy**2)

    @property
    def total_energy(self) -> np.ndarray:
        return self.potential_energy + self.kinetic_energy
    

def solve_pendulum(
    u0: np.ndarray,
    T: float,
    dt: float,
    pendulum: Optional[Pendulum] = Pendulum()
) -> PendulumResults:
    """
    Solves ODE and stores results in an instance of the PendulumResults.

    Input
        - u0: array
            initial condition of u
        - T: float 
            end time
        - dt: float 
            time step
        - pendulum: Pendulum (Optional)
            default is a Pendulum(L=1, M=1, g=9.81)

    Output
        - PendulumResults
    """

    result = solve_ode(pendulum, u0, T, dt)
    pendulum_results = PendulumResults(result, pendulum)

    return pendulum_results

def plot_energy(results: PendulumResults, filename: Optional[str] = None) -> None:
    """
    Plots the potential and kinetic energy in a Pendulum.
    
    Input
        - results: PendulumResults
        - filename: str

    Output
        either saves figure as filename or displays plot on screen.    
    """
    p = results.potential_energy
    k = results.kinetic_energy
    tot = results.total_energy
    t = results.results.time
 
    plt.plot(t, p, label="Potential energy")
    plt.plot(t, k, label="Kinetic energy")
    plt.plot(t, tot, label="Total energy")

    plt.title("Potential and kinetic energi in ODE system")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def exercise_2g() -> PendulumResults:
    """
    Solves ODE with u0 = [pi/6, 0.35], T = 10, dt = 0.01

    Output
        - Pendulum result
    """
    u0 = np.array([np.pi/6, .35])
    T = 10
    dt = .01
    result = solve_pendulum(u0, T, dt)

    return result


class DampenedPendulum(Pendulum):
    def __init__(self, B: float, L: float = 1, M: float = 1, g: float = 9.81) -> None:
        """
        Constructor for DampenedPendulum

        Input
            - B: float 
                damping constant
        """
        super().__init__(L, M, g)
        self.B = B

    def __call__(self, t, u):
        """
        Call function that returns the derivate of u at time t
        
        Input
            - t: float
                time
            - u: array
                consisting of theta and omega

        Output
            - an array of the derivative dudt
        """
     
        theta, omega = u
        dtheta = omega
        domega = (-self.g/self.L)*np.sin(theta) - self.B*omega

        return np.array([dtheta, domega])


def exercise_2h():
    """
    Solves ODE with u0 = [pi/6, 0.35], T = 10, and dt = 0.01.

    Output
        - energy_damped.png
    """
    u0 = np.array([np.pi/6, .35])
    T = 10
    dt = .01

    result = solve_pendulum(u0, T, dt, DampenedPendulum(1))
    plot_energy(result, "energy_damped.png")

if __name__ == "__main__":

    # Exercise 2b
    res_2b = exercise_2b(np.array([np.pi/6, .35]), 10, .01)
    state_labels = [r"$\theta$", r"$\omega$"]
    plot_ode_solution(res_2b, state_labels, "exercise_2b.png")

    # Exercise 2g
    res_2g = exercise_2g()
    plot_energy(res_2g, "energy_single.png")
    # The energy is conserved

    # Exercise 2h
    exercise_2h()
    # The energy transforms to other forms of energy and the potential and kinetic and thus total energy is dampened. 



    
