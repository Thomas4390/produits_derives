import numpy as np
from typing import Tuple, Any, Iterable
import pandas as pd
from numpy import ndarray
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


def get_info():
    K = np.array([80, 90, 97.5, 102.5, 110, 120])
    Put = np.array([0.1900, 0.6907, 1.6529, 3.3409, 9.8399, 19.5805])
    return pd.DataFrame(np.vstack([K, Put]).T, columns=["Strike", "Put"])


def compute_put_option(sigma: float, S0: float, K: float, T: float, r: float) -> float:
    """Compute the price of a European put option.
    :param sigma: volatility
    :param S0: initial stock price
    :param K: strike price
    :param T: maturity
    :param r: risk-free interest rate
    :return: European put option price"""

    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def compute_implied_volatility_by_bisection(
    S0: float,
    K: float,
    T: float,
    r: float,
    Put: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> float:
    """Compute the implied volatility by bisection method.
    :param S0: initial stock price
    :param K: strike price
    :param T: maturity
    :param r: risk-free interest rate
    :param Put: put price
    :param tol: maximum precision tolerance
    :param max_iter: maximum number of iterations
    :return: implied volatility"""

    # Initialisation
    sigma_min = 0.0
    sigma_max = 1.0
    sigma = (sigma_min + sigma_max) / 2
    put_estimation = compute_put_option(sigma=sigma, S0=S0, K=K, T=T, r=r)
    i = 0

    # Boucle
    while abs(put_estimation - Put) > tol and i < max_iter:
        if put_estimation > Put:
            sigma_max = sigma
        else:
            sigma_min = sigma
        sigma = (sigma_min + sigma_max) / 2
        put_estimation = compute_put_option(sigma=sigma, S0=S0, K=K, T=T, r=r)
        i += 1

    return sigma


def compute_implicit_vol_array(
    S0: float, K: ndarray, T: float, r: float, Put: ndarray
) -> ndarray:
    """Compute the implied volatility for a vector of strike prices and put prices.
    :param S0: initial stock price
    :param K: vector of strike prices
    :param T: maturity
    :param r: risk-free interest rate
    :param Put: vector of put prices
    :return: vector of implied volatilities"""
    return np.array(
        [
            compute_implied_volatility_by_bisection(S0=S0, K=K[i], T=T, r=r, Put=Put[i])
            for i in range(len(K))
        ]
    )


def CRR_Tree(S: float, K: float, T: float, r: float, sigma: float, N: int) -> float:

    """Cox-Ross-Rubinstein binomial tree for European put option.
    :param S: initial stock price
    :param K: strike price
    :param T: maturity
    :param r: risk-free interest rate
    :param sigma: volatility
    :param N: number of time steps
    :return: European put option price"""

    u = math.exp(sigma * math.sqrt(T / N))
    d = math.exp(-sigma * math.sqrt(T / N))

    p = ((math.exp(r * T / N)) - d) / (u - d)
    q = 1 - p

    discount = math.exp(-r * T / N)

    S_t = np.zeros(N + 1)
    puts = np.zeros(N + 1)

    S_t[0] = S * d**N

    for j in range(1, N + 1):
        S_t[j] = S_t[j - 1] * (u / d)

    for j in range(1, N + 1):
        puts[j] = max(K - S_t[j], 0)

    for i in range(N, 0, -1):
        for j in range(0, i):
            puts[j] = discount * (p * puts[j + 1] + q * puts[j])

    return puts[0]


# Convert matrix to dataframe
def matrix_to_dataframe(matrix: ndarray, columns: list[str]) -> pd.DataFrame:
    """Convert a matrix to a dataframe.
    :param matrix: matrix
    :param columns: list of columns names
    :return: dataframe"""

    return pd.DataFrame(matrix, columns=columns)


if __name__ == "__main__":
    pass
