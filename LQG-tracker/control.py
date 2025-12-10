import numpy as np

from dynamics import NoisyDoubleIntegrator
from estimator import KalmanFilter


class LQGController:
    system: NoisyDoubleIntegrator
    estimator: KalmanFilter

    M: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    num_steps: int

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        system: NoisyDoubleIntegrator,
        estimator: KalmanFilter,
        num_steps: int
    ):
        self.system = system
        self.estimator = estimator

        self.M = np.diag([alpha, beta])
        self.Q = self.M
        self.R = gamma
        self.num_setps = num_steps

    def step(self):
        # estimate the system state
        x_hat = self.estimator.get_estimate()
        ...
