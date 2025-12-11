from dynamics import NoisyDoubleIntegrator
from estimator import KalmanFilter
from control import LQGTracker

import matplotlib.pyplot as plt

import numpy as np


def main():
    sigma_wx2 = 9
    sigma_wv2 = 4
    sigma_etax2 = 10000
    sigma_etav2 = 36
    mu_x0 = 200
    sigma_x02 = 40000
    mu_v0 = -1
    sigma_v02 = 900

    alpha = 5
    beta = 2

    gamma = 50

    num_steps = 200

    reference_trajectory = np.array(
        [[[mu_x0 + i * mu_v0], [mu_v0]] for i in range(num_steps)]
    )

    system = NoisyDoubleIntegrator(
        sigma_wx2,
        sigma_wv2,
        sigma_etax2,
        sigma_etav2,
        mu_x0,
        sigma_x02,
        mu_v0,
        sigma_v02,
    )
    estimator = KalmanFilter(
        mu_x=np.array([mu_x0, mu_v0]).T,
        cov_x=np.diag([sigma_x02, sigma_v02]),
        system=system,
    )
    controller = LQGTracker(
        alpha, beta, gamma, system, estimator, reference_trajectory, num_steps
    )
    pos_trace, v_trace, pos_hat_trace, v_hat_trace, u_trace = controller.get_traces()

    time = range(num_steps)
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    axes[0].plot(time, pos_trace, label="Actual position")
    axes[0].plot(time, pos_hat_trace, label="Estimated position")
    axes[0].plot(time, reference_trajectory[:, 0], label="Reference trajectory")
    axes[0].legend()
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Altitude (m)")

    axes[1].plot(time, v_trace, label="Actual velocity")
    axes[1].plot(time, v_hat_trace, label="Estimated velocity")
    axes[1].plot(time, reference_trajectory[:, 1], label="Reference velocity")
    axes[1].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m / s)")

    axes[2].plot(time[:-1], u_trace, label="Control")
    axes[2].legend()
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (m / s^2)")

    plt.suptitle("LQG Trajectory Tracking")
    plt.tight_layout()
    plt.savefig("Tracking Results.png")


if __name__ == "__main__":
    main()
