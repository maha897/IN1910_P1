import numpy as np
from ode import ODEModel, ODEResult, plot_ode_solution, solve_ode
from scipy.integrate import solve_ivp


class ExponentialDecay(ODEModel):
    def __init__(self, a) -> None:
        """
        Constructor for ExponentianlDecay. 

        Input
            - a: float,
            the decay constant, this number cannot be negative.
        
        Raises:
            ValueError: if input value is negative.
        """
        self.decay = a
    
    @property
    def decay(self) -> float:
        return self._a

    @decay.setter
    def decay(self, value) -> None:
        if value < 0:
            raise ValueError("Constant cannot be negative.")
        self._a = value
    
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Call function that returns the derivative of u at time t. 

        Input
            - t: float
            - u: numpy array

        Output
            - a numpy array, the derivate dudt
        """

        f = lambda t, u: -self.decay*u
        return f(t, u)

    @property
    def num_states(self) -> int:
        """
        Returns number of state variables in ODE.

        Output
            - an int
        """

        return 1
    
def solve_exponential_decay(
    a: float, 
    u: np.ndarray, 
    u0: tuple, 
    T: int, 
    dt: float, 
) -> np.ndarray:
    """
    Instantiates the ExponentialDecay class and solves the ode using solve_ivp from scipy library.

    Input
        - a: float
            the exponential decay constant
        - u: array
            the u we are solving for
        - u0: tuple
            the initial condition for u
        - T: int
            end time
        - dt: float
            time step

    Output
        - an array
            Solved ode
    """

    model = ExponentialDecay(a)
    t = np.arange(0, T, dt)
    sol = solve_ivp(fun=model, t_span=(0,T), y0=u0, t_eval=t)

    return sol["t"], sol["y"]

if __name__ == "__main__":
    solve_exponential_decay(.4, np.array([3.2]), (0, ), 10, .01) # Exercise 1e

    model = ExponentialDecay(0.4)
    result = solve_ode(model, u0=np.array([4.0]), T=10.0, dt=0.01)
    plot_ode_solution(
        results=result, state_labels=["u"], filename="exponential_decay.png"
    )