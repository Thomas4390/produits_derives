import numpy as np
from typing import Tuple, Any, Iterable
import pandas as pd
from numpy import ndarray
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

def get_info():
    K  = np.array([80, 90, 97.5, 102.5, 110, 120])
    Put = np.array([0.1900, 0.6907, 1.6529, 3.3409, 9.8399, 19.5805])
    return pd.DataFrame(np.vstack([K,Put]).T, columns=['Strike','Put'])

def compute_put_option(sigma: float, S0: float, K: float, T: float, r: float) -> float:
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

def compute_implied_volatility_by_bisection(S0: float,
                                            K: float,
                                            T: float,
                                            r: float,
                                            Put: float,
                                            tol: float = 1e-6,
                                            max_iter: int = 1000) -> float:

    # Initialisation
    sigma_min = 0.0
    sigma_max = 1.0
    sigma = (sigma_min + sigma_max)/2
    Put_calcul = compute_put_option(sigma=sigma, S0=S0, K=K, T=T, r=r)
    i = 0

    # Boucle
    while abs(Put_calcul - Put) > tol and i < max_iter:
        if Put_calcul > Put:
            sigma_max = sigma
        else:
            sigma_min = sigma
        sigma = (sigma_min + sigma_max)/2
        Put_calcul = compute_put_option(sigma=sigma, S0=S0, K=K, T=T, r=r)
        i += 1

    return sigma

def compute_implicit_vol_array(S0: float, K: ndarray, T: float, r: float, Put: ndarray) -> ndarray:
    return np.array([compute_implied_volatility_by_bisection(S0=S0,
                                                             K=K[i],
                                                             T=T,
                                                             r=r,
                                                             Put=Put[i])
                     for i in range(len(K))])


def CRR_Tree(S: float, K: float, T: float, r: float, sigma: float, N: int)\
        -> float:


    u = math.exp(sigma * math.sqrt(T / N))
    d = math.exp(-sigma * math.sqrt(T / N))

    p = ((math.exp(r * T / N)) - d) / (u - d)
    q = 1 - p

    discount = math.exp(-r * T / N)

    S_t = np.zeros(N+1)
    puts = np.zeros(N+1)

    S_t[0] = S * d ** N

    for j in range(1, N + 1):
        S_t[j] = S_t[j - 1] * (u / d)

    for j in range(1, N + 1):
        puts[j] = max(K - S_t[j], 0)

    for i in range(N, 0, -1):
        for j in range(0, i):
            puts[j] = discount * (p * puts[j + 1] + q * puts[j])

    return puts[0]

# Convert matrix to dataframe
def matrix_to_dataframe(matrix: ndarray, columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=columns)


if __name__ == '__main__':
    pass



