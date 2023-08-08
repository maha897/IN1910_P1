import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataclasses import dataclass
from typing import Optional

from ode import ODEModel, ODEResult, solve_ode
from pendulum import plot_energy

class DoublePendulum(ODEModel):
    def __init__(self, L1: float = 1, L2: float = 1, M1: float = 1, M2: float = 1, g: float = 9.81) -> None:
        """
        Constructor of DoublePendulum

        Input
            - L1: float
                length of first rod
            - L2: float 
                length of second rod
            - g: float 
                the gravitational acceleration, default set to 9.81  
        """
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2
        self.g = g
    
    def __call__(self, t: float, u: np.ndarray):
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
        theta1, omega1, theta2, omega2 = u

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        dtheta = theta2 - theta1

        domega1_dt = (self.L1*(omega1**2)*np.sin(dtheta)*np.cos(dtheta) + self.g*np.sin(theta2)*np.cos(dtheta) + self.L2*(omega2**2)*np.sin(dtheta) - 2*self.g*np.sin(theta1))/(2*self.L1 - self.L1*(np.cos(dtheta))**2)
        domega2_dt = (-self.L2*(omega2)**2*np.sin(dtheta)*np.cos(dtheta) + 2*self.g*np.sin(theta1)*np.cos(dtheta) - 2*self.L1*(omega1**2)*np.sin(dtheta) - 2*self.g*np.sin(theta2))/(2*self.L2 - self.L2*(np.cos(dtheta))**2)

        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

    @property 
    def num_states(self) -> int:
        """
        Returns the number of states. u = [theta1, omega1, theta2, theta2] so it will return 4.

        Output
            - an int, the number of state variables
        """
        return 4


@dataclass
class DoublePendulumResults:
    results: ODEResult
    pendulum: DoublePendulum

    @property
    def theta1(self) -> np.ndarray:
        return self.results.solution[0]

    @property 
    def omega1(self) -> np.ndarray:
        return self.results.solution[1]

    @property
    def theta2(self) -> np.ndarray:
        return self.results.solution[2]

    @property 
    def omega2(self) -> np.ndarray:
        return self.results.solution[3]

    @property
    def x1(self) -> np.ndarray:
        return self.pendulum.L1*np.sin(self.theta1)

    @property 
    def y1(self) -> np.ndarray:
        return -self.pendulum.L1*np.cos(self.theta1)

    @property
    def x2(self) -> np.ndarray:
        return self.x1 + self.pendulum.L2*np.sin(self.theta2)

    @property
    def y2(self) -> np.ndarray:
        return self.y1 - self.pendulum.L2*np.cos(self.theta2)

    @property
    def potential_energy(self) -> np.ndarray:
        P1 = self.pendulum.M1*self.pendulum.g*(self.y1 + self.pendulum.L1)
        P2 = self.pendulum.M2*self.pendulum.g*(self.y2 + self.pendulum.L1 + self.pendulum.L2)
        
        return P1 + P2

    @property
    def vx1(self) -> np.ndarray:
        return np.gradient(self.x1, self.results.time)

    @property 
    def vy1(self) -> np.ndarray:
        return np.gradient(self.y1, self.results.time)

    @property
    def vx2(self) -> np.ndarray:
        return np.gradient(self.x2, self.results.time)

    @property 
    def vy2(self) -> np.ndarray:
        return np.gradient(self.y2, self.results.time)
    
    @property
    def kinetic_energy(self) -> np.ndarray:
        K1 = (1/2)*self.pendulum.M1*(self.vx1**2 + self.vy1**2)
        K2 = (1/2)*self.pendulum.M1*(self.vx2**2 + self.vy2**2)

        return K1 + K2

    @property
    def total_energy(self) -> np.ndarray:
        return self.potential_energy + self.kinetic_energy
    

def solve_double_pendulum(
    u0: np.ndarray,
    T: float,
    dt: float = .01,
    pendulum: Optional[DoublePendulum] = DoublePendulum()
) -> DoublePendulumResults:
    """
    Solves ODE for DoublePendulum.

    Input
        - u0: array
            initial condition for u
        - T: float
            end time
        - dt: float
            time step, default is set dt = 0.01
        - pendulum: DoublePendulum, default is set to DoublePendulum(L1,L2,g=9.81)

    Output
        DoublePendulumResults
    """
    result = solve_ode(pendulum, u0, T, dt)
    pendulum_results = DoublePendulumResults(result, pendulum)

    return pendulum_results


def exercise_3d():
    """
    Solves ODE with u0 = [pi/6, 0.35, 0, 0], T =  10, dt 0.01.

    Output
        - energy_double.png
    """
    u0 = np.array([np.pi/6, .35, 0, 0])
    T = 10
    dt = .01
    res = solve_double_pendulum(u0, T, dt)

    plot_energy(res, "energy_double.png")

"""Adopted from https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
license: BSD
"""

def animate_pendulum(results: DoublePendulumResults) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(
        111, aspect="equal", autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2)
    )
    ax.grid()

    (line,) = ax.plot([], [], "o-", lw=2)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    kinetic_energy_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    potential_energy_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    def init():
        """initialize animation"""
        line.set_data([], [])
        time_text.set_text("")
        kinetic_energy_text.set_text("")
        potential_energy_text.set_text("")
        return line, time_text, kinetic_energy_text, potential_energy_text

    def animate(i):
        """perform animation step"""
        line.set_data(
            (0, results.x1[i], results.x2[i]), (0, results.y1[i], results.y2[i])
        )
        time_text.set_text(f"time = {results.results.time[i]:.1f}")
        kinetic_energy_text.set_text(
            f"kinetic energy = {results.kinetic_energy[i]:.3f} J"
        )
        potential_energy_text.set_text(
            f"potential energy = {results.potential_energy[i]:.3f} J"
        )
        return line, time_text, kinetic_energy_text, potential_energy_text

    ani = animation.FuncAnimation(
        fig, animate, frames=len(results.results.time), interval=10, blit=True, init_func=init
    )
    plt.show()


def exercise_4():
    results = solve_double_pendulum(u0=np.array([np.pi, 0.35, 0, 0]), T=40.0, dt=0.01)
    animate_pendulum(results)

if __name__ == "__main__":

    # Exercise 3d
    exercise_3d() 

    # Exercise 4
    exercise_4()
