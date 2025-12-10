from dynamics import NoisyDoubleIntegrator
from estimator import KalmanFilter
from control import LQGController

import numpy as np

def main():
    sigma_wx2 = 100
    sigma_wv2 = 10
    sigma_eta2 = 10
    mu_x0 = 1000
    sigma_x02 = 100
    mu_v0 = 0
    sigma_v02 = 10

    alpha = 5
    beta = 2

    gamma = 3

    num_steps = 100
    

    system = NoisyDoubleIntegrator(sigma_wx2, sigma_wv2, sigma_eta2, mu_x0, sigma_x02, mu_v0, sigma_v02)
    estimator = KalmanFilter(mu_x = np.array([mu_x0, mu_v0]).T, cov_x = np.diag([sigma_x02, sigma_v02]), system = system)
    controller = LQGController(alpha, beta, gamma, system, estimator, num_steps)



if __name__ == "__main__":
    main()
