import numpy as np

from dynamics import NoisyDoubleIntegrator


class KalmanFilter:
    x_hat: np.ndarray
    S: np.ndarray

    F: np.ndarray
    G: np.ndarray
    H: np.ndarray
    W: np.ndarray
    ETA: np.ndarray

    def __init__(
        self, mu_x: np.ndarray, cov_x: np.ndarray, system: NoisyDoubleIntegrator
    ):
        self.x_hat = mu_x
        self.S = cov_x

        self.F = system.F
        self.G = system.G
        self.H = system.H
        self.W = np.diag([system.sigma_wx2, system.sigma_wv2])
        self.ETA = system.sigma_eta2

    def get_estimate(self) -> np.ndarray:
        return self.x_hat

    def dynamics_update(self, u: np.ndarray):
        # make dynamics update

        self.x_hat = self.F @ self.x_hat + self.G @ u
        self.S = self.F @ self.S @ self.F.T + self.W

    def measurement_update(self, y: np.ndarray):
        K = self.S @ self.H / (self.ETA + self.H @ self.S @ self.H.T)
        self.x_hat = self.x_hat + K * (y - self.H @ self.x_hat)
        self.S = (np.identity(2) - K @ self.H) @ self.S @ (
            np.identity(2) - K @ self.H
        ).T + K * self.ETA * K.T
