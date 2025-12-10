import numpy as np
from dataclasses import dataclass


class NoisyDoubleIntegrator:
    sigma_wx2: float
    sigma_wv2: float

    sigma_eta2: float

    mu_x0: float
    sigma_x02: float

    mu_v0: float
    sigma_v02: float

    F: np.ndarray = np.array([[0, 1], [0, 0]])

    G: np.ndarray = np.array([[0], [1]])

    H: np.ndarray = np.array([0, 1])

    x: np.ndarray

    def __init__(self, sigma_wx2: float, sigma_wv2: float, sigma_eta2: float, mu_x0: float,
                 sigma_x02: float, mu_v0: float, sigma_v02: float):
        """
        Initialize parameters and the initial state according to the Gaussian distribution specified
        by the input parameters.
        """
        self.sigma_wx2 = sigma_wx2,
        self.sigma_wv2 = sigma_wv2,
        self.sigma_eta2 = sigma_eta2
        self.mu_x0 = mu_x0
        self.sigma_x02 = sigma_x02
        self.mu_v0 = mu_v0
        self.sigma_v02 = sigma_v02

        self.x = np.random.normal(
            loc=[self.mu_x0, self.mu_v0],
            scale=[np.sqrt(self.sigma_x02), np.sqrt(self.sigma_v02)],
        ).T

    def step(self, u: float):
        w = np.random.normal(
            loc=[0, 0],
            scale=[np.sqrt(self.sigma_wx2), np.sqrt(self.sigma_wv2)],
        ).T
        self.x = self.F @ self.x + self.G @ u + w

    def measure(self) -> np.ndarray:
        eta = np.random.normal(scale=np.sqrt(self.sigma_eta2))
        return self.H @ self.x + eta
