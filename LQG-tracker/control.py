import numpy as np

from dynamics import NoisyDoubleIntegrator
from estimator import KalmanFilter


class LQGTracker:
    system: NoisyDoubleIntegrator
    estimator: KalmanFilter

    M: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    K_series: np.ndarray
    Kov_series = np.ndarray

    r_series: np.ndarray

    num_steps: int

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        system: NoisyDoubleIntegrator,
        estimator: KalmanFilter,
        reference_trajectory: np.ndarray,
        num_steps: int,
    ):
        self.system = system
        self.estimator = estimator

        self.M = np.diag([alpha, beta])
        self.Q = self.M
        self.R = gamma

        self.r_series = reference_trajectory

        self.num_steps = num_steps

        self.initialize_gain_matrices()

    def _riccati(self, P: np.ndarray) -> np.ndarray:
        return (
            self.Q
            + self.system.F.T @ P @ self.system.F
            - (self.system.F.T @ P @ self.system.G)
            @ np.linalg.inv(self.R + self.system.G.T @ P @ self.system.G)
            @ (self.system.G.T @ P @ self.system.F)
        )

    def _gain(self, P: np.ndarray) -> np.ndarray:
        return (
            np.linalg.inv(self.R + self.system.G.T @ P @ self.system.G)
            @ self.system.G.T
            @ P
            @ self.system.F
        )

    def initialize_gain_matrices(self):
        P_series = np.zeros((self.num_steps, 2, 2))
        P_series[-1] = self.M

        for i in range(2, self.num_steps + 1):
            P_series[-i] = self._riccati(P_series[-(i - 1)])

        self.K_series = np.zeros((self.num_steps - 1, 1, 2))
        for i in range(self.num_steps - 1):
            self.K_series[i] = self._gain(P_series[i])

        print(self.K_series)
        v_series = np.zeros((self.num_steps, 2, 1))
        v_series[-1] = self.M @ self.r_series[-1]
        for i in range(2, self.num_steps + 1):
            v_series[-i] = (
                self.system.F - self.system.G @ self.K_series[-(i - 1)]
            ).T @ v_series[-(i - 1)] + self.Q @ self.r_series[-i]
        print(v_series)

        self.Kov_series = np.zeros((self.num_steps - 1))
        for i in range(len(self.Kov_series)):
            Ko = (
                np.linalg.inv(
                    self.R + self.system.G.T @ P_series[i + 1] @ self.system.G
                )
                @ self.system.G.T
            )
            self.Kov_series[i] = Ko @ v_series[i + 1]

    def _measure(self) -> tuple[np.ndarray, np.ndarray]:
        x_actual = self.system.get_actual()
        self.estimator.measurement_update(self.system.measure())
        x_hat = self.estimator.get_estimate()
        return x_actual, x_hat

    def _step(self, x_hat: np.ndarray, i: int) -> np.ndarray:
        # estimate the system state
        u_star = -self.K_series[i] @ x_hat + self.Kov_series[i]
        self.system.step(u_star)
        self.estimator.dynamics_update(u_star)
        return u_star

    def _measure_and_step(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_actual, x_hat = self._measure()
        u_star = self._step(x_hat, i)
        return x_actual, x_hat, u_star

    def get_traces(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_trace = np.zeros((self.num_steps, 2, 1))
        x_hat_trace = np.zeros((self.num_steps, 2, 1))
        u_trace = np.zeros(self.num_steps - 1)

        for i in range(self.num_steps - 1):
            x, x_hat, u_star = self._measure_and_step(i)
            x_trace[i], x_hat_trace[i], u_trace[i] = x, x_hat[:, np.newaxis], u_star
        x, x_hat = self._measure()
        x_trace[-1], x_hat_trace[-1] = x, x_hat[:, np.newaxis]

        pos_trace = x_trace[:, 0]
        v_trace = x_trace[:, 1]

        pos_hat_trace = x_hat_trace[:, 0]
        v_hat_trace = x_hat_trace[:, 1]
        return pos_trace, v_trace, pos_hat_trace, v_hat_trace, u_trace
