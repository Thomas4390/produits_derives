import math
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.stats import norm
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_info():
    """Return a DataFrame with the information of the options"""

    K = np.array([80, 90, 97.5, 102.5, 110, 120])
    Put = np.array([0.1900, 0.6907, 1.6529, 3.3409, 9.8399, 19.5805])
    return pd.DataFrame(np.vstack([K, Put]).T, columns=["Strike", "Put"])


def d1(S: float, K: float, r: float, y: float, T: float, sigma: float) -> float:
    """Calculate d1 from the Black, Merton and Scholes formula

    Parameters:
        S : Underlying price
        K : Strike price
        r : Risk-free rate
        y : Dividend yield
        T : Time to maturity
        sigma : Volatility
    Returns:
        d1 : the d1 parameter of the BMS formula
    """
    return (np.log(S / K) + (r - y + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, r: float, y: float, T: float, sigma: float):
    """Calculate d2 from the Black, Merton and Scholes formula

    Parameters:
        S, K, r, y, T, sigma : as usual
    Returns:
        d2 : the d2 parameter of the BMS formula
        """
    return (np.log(S / K) + (r - y - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def delta(
    S: float,
    K: float,
    r: float,
    y: float,
    T: float,
    sigma: float,
    is_call: bool = False,
) -> float:
    """Return Black, Merton, Scholes delta of the European (call, put)

    Parameters:
        S, K, r, y, T, sigma : as usual
        is_call : True if the option is a call, False if it is a put
    Returns:
        delta : the delta of the option
        """

    _d1 = d1(S=S, K=K, r=r, y=y, T=T, sigma=sigma)
    d_sign = np.where(is_call, 1, -1)
    return d_sign * norm.cdf(d_sign * _d1)


def gamma(S: float, K: float, r: float, y: float, T: float, sigma: float) -> float:
    """Return Black, Merton, Scholes gamma of the European call or put

    Parameters:
        S, K, r, y, T, sigma : as usual
    Returns:
        gamma : the gamma of the option
        """
    _d1 = d1(S=S, K=K, r=r, y=y, T=T, sigma=sigma)
    return np.exp(-y * T) * norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def option_price(
    S: float,
    K: float,
    r: float,
    y: float,
    T: float,
    sigma: float,
    is_call: bool = False,
    ret_delta: bool = False,
) -> tuple[float | Any, Any] | float | Any:
    """Return Black, Merton, Scholes price of the European option

    Parameters:
        S, K, r, y, T, sigma : as usual
        is_call : True if the option is a call, False if it is a put
        ret_delta : True if the delta of the option is also returned
    Returns:
        premium : the premium of the option
        """

    _d1 = d1(S=S, K=K, r=r, y=y, T=T, sigma=sigma)
    _d2 = _d1 - sigma * np.sqrt(T)

    # d_sign: Sign of the the option's delta
    d_sign = np.where(is_call, 1, -1)
    delta = d_sign * norm.cdf(d_sign * _d1)
    premium = np.exp(-y * T) * S * delta - d_sign * np.exp(-r * T) * K * norm.cdf(
        d_sign * _d2
    )
    if ret_delta:
        return premium, delta
    return premium


def implied_volatility(
    opt_price: float,
    S: float,
    K: float,
    r: float,
    y: float,
    T: float,
    is_call: bool = False,
    init_vol: float = 0.6,
) -> tuple[ndarray, dict, int, str]:
    """Inverse the BMS formula numerically to find the implied volatility

    Parameters:
        opt_price : the price of the option
        S, K, r, y, T : as usual
        is_call : True if the option is a call, False if it is a put
        init_vol : the initial guess for the implied volatility
    Returns:
        sigma : the implied volatility
        """

    def pricing_error(sig):
        sig = abs(sig)
        return (
            option_price(S=S, K=K, r=r, y=y, T=T, sigma=sig, is_call=is_call)
            - opt_price
        )

    return fsolve(pricing_error, init_vol)


def antithetic_normal(n_periods: int, n_paths: int):
    """Generate antithetic standard normal shocks
    Parameters:
        n_periods : the number of time steps in the simulation
        n_paths   : the number of paths in the simulation
        """

    assert n_paths % 2 == 0, "n_paths must be an even number"
    n2 = int(n_paths / 2)
    z = np.random.normal(0, 1, (n_periods, n2))
    return np.hstack((z, -z))


def simulate_underlying(
    S0: float, r: float, y: float, sigma: float, dt: float, shocks: ndarray
) -> ndarray:
    """Simulate the GMB based on the user-provided standard normal shocks

    Parameters:
        S, r, y, sigma, : as usual
        dt     : the time step length in the simulation
        shocks : A (n_steps x n_sim) matrix of standard Normal shocks for a
                 simulation with n_steps time steps and n_sim paths

    Returns:
        S : A (n_steps+1 x n_sim) matrix with n_sim paths of the underlying simulated over n_steps time steps,
            starting at time 0
    """
    n_steps, n_sim = shocks.shape
    S = np.empty((n_steps + 1, n_sim))
    S[0, :] = S0
    for tn in range(n_steps):
        S[tn + 1, :] = S[tn, :] * np.exp(
            (r - y - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks[tn, :]
        )
    return S


def plot_implied_vol(S: ndarray, info: pd.DataFrame) -> None:
    """Plot implied vol. for question 1
    Parameters:
        S : the underlying price
        info : the dataframe containing the option data
        """
    plt.plot(info["Strike"] / S, info["Implied vol."], "o--")
    plt.xlabel(r"$\frac{K}{S_0}$", fontweight="bold")
    plt.ylabel("Sigma", fontweight="bold")
    plt.title(
        "Volatilité implicite en fonction de " + r"$\frac{K}{S_0}$", fontweight="bold"
    )
    plt.grid(linestyle="--", linewidth=0.5)
    plt.show()


def CRR_tree(
    S: float, K: float, T: float, r: float, sigma: float, Type: int = 0, N: int = 1000
) -> float:
    """Compute the call or put price with CRR tree

    Parameters:
        S, K, T, r, sigma : as usual
        Type : 0 for European, 1 for American
        N : number of steps
    Returns:
        f[0, 0] : the price of the option after CRR tree simulation
        """

    # Calcul préliminaire
    dt = T / N
    df = math.exp(-r * dt)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    # Calcul des valeurs finales du put dans l'arbre
    f = np.zeros((N + 1, N + 1))
    f[N, :] = np.maximum(
        0, K - S * (u ** np.flip(np.arange(N + 1))) * (d ** np.arange(N + 1))
    )

    # Calcul de la valeur du put américain par induction inverse
    # Européen
    if Type == 0:
        for i in range(N - 1, -1, -1):
            f[i, : i + 1] = df * (q * f[i + 1, : i + 1] + (1 - q) * f[i + 1, 1 : i + 2])

    # Américain
    else:
        for i in range(N - 1, -1, -1):
            S_j = K - S * (u ** np.flip(np.arange(i + 1)) * (d ** np.arange(i + 1)))
            f[i, : i + 1] = np.maximum(
                S_j, df * (q * f[i + 1, : i + 1] + (1 - q) * f[i + 1, 1 : i + 2])
            )

    put_value = f[0][0]

    return put_value


def CRR_tree_BD(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    Type: int = 0,
    N: int = 1000,
    ret_gamma: bool = False,
):
    """Compute the call or put price with CRR tree adjusted
    with the Broadie and Detemple correction.

    Parameters:
        S, K, T, r, sigma : as usual
        Type : 0 for European, 1 for American
        N : number of steps
        ret_gamma : if True, return the gamma of the option
    Returns:
        f[0, 0] : the price of the option after CRR tree simulation
        """

    # Calcul préliminaire
    dt = T / N
    df = math.exp(-r * dt)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    # Calcul des valeurs finales du put dans l'arbre
    f = np.zeros((N, N))
    S_end = S * (u ** np.flip(np.arange(N)) * (d ** np.arange(N)))
    f[N - 1, :] = option_price(S=S_end, K=K, r=r, y=0, T=dt, sigma=sigma, is_call=False)

    # Calcul de la valeur du put américain par induction inverse
    # Européen
    if Type == 0:
        for i in range(N - 2, -1, -1):
            f[i, : i + 1] = df * (q * f[i + 1, : i + 1] + (1 - q) * f[i + 1, 1 : i + 2])

            if i == 2:
                p_uu = f[i][0]
                p_ud = f[i][1]
                p_dd = f[i][2]

    # Américain
    else:
        for i in range(N - 2, -1, -1):
            S_j = K - S * (u ** np.flip(np.arange(i + 1)) * (d ** np.arange(i + 1)))
            f[i, : i + 1] = np.maximum(
                S_j, df * (q * f[i + 1, : i + 1] + (1 - q) * f[i + 1, 1 : i + 2])
            )

            if i == 2:
                p_uu = f[i][0]
                p_ud = f[i][1]
                p_dd = f[i][2]

    if ret_gamma:
        gamma_0 = (p_uu - 2 * p_ud + p_dd) / (((u - d) * S) ** 2)
        gamma_1 = (
            ((p_uu - p_ud) / (((u**2) * S) - S))
            - ((p_ud - p_dd) / (-((d**2) * S) + S))
        ) / (S * (u - d))
        gamma_2 = (
            ((p_uu - p_ud) / (((u**2) * S) - S))
            - ((p_ud - p_dd) / (-((d**2) * S) + S))
        ) / (0.5 * S * (u**2 - d**2))
        gamma = [gamma_0, gamma_1, gamma_2]

        return f[0][0], gamma

    else:
        return f[0][0]


def CRR_tree_df(
    S: float,
    K: pd.Series,
    T: float,
    r: float,
    sigma: pd.Series,
    Type: int = 0,
    N_Range: ndarray = np.arange(2, 101, 1),
):
    """Compute the call or put price with CRR tree for a dataframe of parameters.

    Parameters:
        S, K (series), T, r, sigma (series)
        Type : 0 for European, 1 for American
        N_Range : range of N to compute
    Returns:
        df_crr : dataframe of the price of the option after CRR tree simulation
        """

    crr_values = np.array(
        [
            [
                CRR_tree(
                    S=S, K=K.iloc[i], T=T, r=r, sigma=sigma.iloc[i], Type=Type, N=N
                )
                for N in N_Range
            ]
            for i in range(len(K))
        ]
    ).T
    columns = [f"Put_{i}" for i in range(len(K))]
    df_crr = pd.DataFrame(crr_values, columns=columns)
    df_crr.index = N_Range

    return df_crr


def CRR_tree_BD_df(
    S: float,
    K: pd.Series,
    T: float,
    r: float,
    sigma: pd.Series,
    Type: int = 0,
    N_Range: ndarray = np.arange(2, 101, 1),
    ret_gamma: bool = False,
):
    """Compute the call or put price with CRR tree adjusted
    with the Broadie and Detemple correction for a dataframe of parameters.

    Parameters:
        S, K (series), T, r, sigma (series)
        Type : 0 for European, 1 for American
        N_Range : range of N to compute
        ret_gamma : if True, return the gamma of the option
    Returns:
        df_crr : dataframe of the price of the option after CRR tree simulation
        """
    if not ret_gamma:
        crr_values = np.array(
            [
                [
                    CRR_tree_BD(
                        S=S,
                        K=K.iloc[i],
                        T=T,
                        r=r,
                        sigma=sigma.iloc[i],
                        N=N,
                        Type=Type,
                        ret_gamma=ret_gamma,
                    )
                    for N in N_Range
                ]
                for i in range(len(K))
            ]
        ).T

        columns = [f"Put_{i}" for i in range(len(K))]
        df_crr = pd.DataFrame(crr_values, columns=columns)
        df_crr.index = N_Range

        return df_crr

    else:
        crr_values = np.array(
            [
                [
                    CRR_tree_BD(
                        S=S,
                        K=K.iloc[i],
                        T=T,
                        r=r,
                        sigma=sigma.iloc[i],
                        N=N,
                        Type=Type,
                        ret_gamma=ret_gamma,
                    )[0]
                    for N in N_Range
                ]
                for i in range(len(K))
            ]
        ).T

        gamma = list(
            [
                CRR_tree_BD(
                    S=S,
                    K=K.iloc[i],
                    T=T,
                    r=r,
                    sigma=sigma.iloc[i],
                    N=N,
                    Type=Type,
                    ret_gamma=ret_gamma,
                )[1]
                for N in N_Range
            ]
            for i in range(len(K))
        )

        gamma_df = [
            pd.DataFrame(np.array(g), columns=["gamma_0", "gamma_1", "gamma_2"])
            for g in gamma
        ]
        columns = [f"Put_{i}" for i in range(len(K))]
        df_crr = pd.DataFrame(crr_values, columns=columns)
        df_crr.index = N_Range

        return df_crr, gamma_df


def plot_CRR_tree(
    df_CRR_list: list,
    N_Range: ndarray = np.arange(2, 101, 1),
    bps: float = 0.0001,
    zoom_factor: int = 20,
    cross: bool = False,
):
    """Plot the CRR tree for a list of dataframe of CRR tree.

    Parameters:
        df_CRR_list : list of dataframe of CRR tree
        N_Range : range of N to compute
        bps : bps to add to the theoretical price of BMS
        zoom_factor : zoom factor for the plot. Increase zoom_factor to zoom in
        cross : if True, plot the cross of the tree
    Returns:
        None
        """

    plt.style.use("seaborn-v0_8-deep")
    info = get_info()
    figsize = (15, 20)
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    for i in range(3):
        for j in range(2):
            k = i * 2 + j
            plt.sca(axes[i, j])
            for df_CRR in df_CRR_list:
                plt.plot(N_Range, df_CRR[0][f"Put_{k}"], df_CRR[2])

            plt.hlines(
                info["Put"].iloc[k],
                N_Range[0],
                N_Range[-1],
                linestyles="dashed",
                color="red",
            )
            plt.hlines(
                info["Put"].iloc[k] + bps,
                N_Range[0],
                N_Range[-1],
                linestyles="dotted",
                color="orange",
            )
            plt.hlines(
                info["Put"].iloc[k] - bps,
                N_Range[0],
                N_Range[-1],
                linestyles="dotted",
                color="orange",
            )
            if cross:
                data = df_CRR[0][f"Put_{k}"]
                value_up = info["Put"].iloc[k] + bps
                value_dw = info["Put"].iloc[k] - bps
                crossing = next(
                    (i for i, v in enumerate(data) if v > value_dw and v < value_up), 0
                )
                if crossing != 0:
                    plt.plot(
                        N_Range[crossing],
                        list(df_CRR[0][f"Put_{k}"])[crossing],
                        "o",
                        markersize=10,
                        markerfacecolor="red",
                    )
                    if list(df_CRR[0][f"Put_{k}"])[crossing] - info["Put"].iloc[k] > 0:
                        plt.text(
                            N_Range[crossing],
                            list(df_CRR[0][f"Put_{k}"])[crossing] + bps,
                            "N = " + str(N_Range[crossing]),
                            fontweight="bold",
                        )
                    else:
                        plt.text(
                            N_Range[crossing],
                            list(df_CRR[0][f"Put_{k}"])[crossing] - bps,
                            "N = " + str(N_Range[crossing]),
                            fontweight="bold",
                        )
            plt.ylim(
                info["Put"].iloc[k] - bps * zoom_factor,
                info["Put"].iloc[k] + bps * zoom_factor,
            )
            plt.legend([df_CRR[1] for df_CRR in df_CRR_list] + ["BMS", "+1bp", "-1bp"])
            plt.xlabel("N")
            plt.ylabel("Put")
            axes[i, j].yaxis.set_label_position("right")
            plt.title(
                f"Prix Put par CRR en fonction de N pour K={info['Strike'].iloc[k]}"
            )
    return None


def plot_gamma(
    gammas: pd.DataFrame,
    gamma_bms: pd.DataFrame,
    N_Range: ndarray = np.arange(2, 101, 1),
    bps: float = 0.0001,
    zoom_factor: int = 20,
) -> None:
    """Plot the gamma of the CRR tree for a list of dataframe of CRR tree.

    Parameters:
        gammas : list of dataframe of gamma
        gamma_bms : dataframe of gamma of BMS
        N_Range : range of N to compute
        bps : bps to add to the theoretical price of BMS
        zoom_factor : zoom factor for the plot. Increase zoom_factor to zoom in
    Returns:
        None
        """

    plt.style.use("seaborn-v0_8-deep")
    info = get_info()
    figsize = (15, 20)
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for i in range(3):
        for j in range(2):
            k = i * 2 + j
            plt.sca(axes[i, j])

            plt.plot(N_Range, gammas[k]["gamma_0"])
            plt.plot(N_Range, gammas[k]["gamma_1"])
            plt.plot(N_Range, gammas[k]["gamma_2"], ":", linewidth=3)

            plt.hlines(
                gamma_bms[k], N_Range[0], N_Range[-1], linestyles="dashed", color="red"
            )
            plt.hlines(
                gamma_bms[k] + bps,
                N_Range[0],
                N_Range[-1],
                linestyles="dotted",
                color="orange",
            )
            plt.hlines(
                gamma_bms[k] - bps,
                N_Range[0],
                N_Range[-1],
                linestyles="dotted",
                color="orange",
            )

            plt.ylim(gamma_bms[k] - bps * zoom_factor, gamma_bms[k] + bps * zoom_factor)
            plt.legend(
                [r"$\Gamma_0$", r"$\Gamma_1$", r"$\Gamma_2$", "BMS", "+1bp", "-1bp"]
            )
            plt.xlabel("N")
            plt.ylabel("Gamma")
            axes[i, j].yaxis.set_label_position("right")
            plt.title(
                f"Gamma du Put par CRR en fonction de N pour K={info['Strike'].iloc[k]}"
            )
    return None


def delta_hedging(
    S_t: ndarray,
    T: float,
    dt: float,
    r: float,
    sigma_0: float,
    info: pd.DataFrame
) -> pd.DataFrame:
    """Compute the profit of a delta hedging strategy.

    Parameters:
        S_t : array of simulated stock prices
        T : maturity
        dt : time step
        r : risk free rate
        sigma_0 : initial volatility
        info : dataframe of the puts
    Returns:
        profit : dataframe of the profit of the delta hedging strategy
        """


    # Initialisation des vecteurs de temps et du DF de profits
    S_simul = S_t.T
    S_simul_df = pd.DataFrame(S_t.T)
    t = np.arange(0, T + dt, dt)
    profit = pd.DataFrame(np.zeros((len(S_t[0]), len(info))))

    # Boucle pour itérer sur les différents puts
    for j, _ in info.iterrows():

        # Données
        IV = info["IV"][j]
        K = info["Strike"][j]

        # Calcul de la prime et du delta du put
        sigma = np.sqrt((sigma_0**2) + ((T - t) / T) * ((IV**2) - sigma_0**2))
        delta_g1 = delta(S=S_simul, K=K, r=r, y=0, T=T - t, sigma=sigma, is_call=False)
        g_1 = option_price(
            S=S_simul, K=K, r=r, y=0, T=T - t, sigma=sigma, is_call=False
        )

        # ASJ nécéssaire au delta hedging
        n_g1 = -1
        n_S = pd.DataFrame(-(n_g1 * delta_g1))
        d_nS = n_S.diff(axis=1).fillna(n_S.iloc[0, 0])

        # Calcul de la valeur du prêt
        n_B = pd.DataFrame(np.zeros((len(n_S), len(t))))
        n_B[0] = -(n_g1 * g_1[0][0] + d_nS[0] * S_simul[0][0])
        n_B[n_B.columns[1:]] = (
            -d_nS[d_nS.columns[1:]] * S_simul_df[S_simul_df.columns[1:]]
        ).cumsum(axis=1) + n_B[0][0]

        # Calcul des profit sur l'ASJ
        temp = S_simul_df.diff(axis=1).dropna(axis=1)
        temp.columns = range(temp.columns.size)
        S_profit = (temp * n_S[n_S.columns[:-1]]).sum(axis=1)

        # Calcul des profit sur la position à couvrir
        c_profit = pd.DataFrame(n_g1 * g_1).diff(axis=1).sum(axis=1)

        # Calcul des profits d'intérêts
        interest = pd.DataFrame(np.zeros((len(n_B), len(t))))
        d_f = np.exp(dt * r) - 1
        for i in range(len(t) - 1):
            interest[i + 1] = (interest[i] + n_B[i]) * d_f

        int_profit = interest.sum(axis=1)

        # Calcul des profits totaux
        profit[j] = S_profit + c_profit + int_profit

    return profit


def plot_delta_hist(profit: pd.DataFrame, info: pd.DataFrame) -> None:
    """Plot the histogram of the final value of the account for the delta hedging strategy.

    Parameters:
        profit : dataframe of the profit of the delta hedging strategy
        info : dataframe of the puts
    Returns:
        None
        """

    plt.style.use("seaborn-v0_8-deep")
    figsize = (15, 20)
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    edges = np.linspace(-1.25, 2.25, 100)

    for i in range(3):
        for j in range(2):
            k = i * 2 + j
            plt.sca(axes[i, j])
            plt.hist(profit[k], edges)
            plt.xlabel("Valeur finale du compte de marge ($)")
            plt.ylabel("")
            axes[i, j].yaxis.set_label_position("right")
            plt.axvline(x=np.mean(profit[k]), color="r", linestyle="--", linewidth=1)
            plt.legend(["Moyenne"])
            plt.title(
                f"Histogramme de la valeur finale de compte de marge pour K={info['Strike'].iloc[k]}"
            )

    plt.show()

    return None


# Définition des tic et toc pour le temps d'exécution
import time


def TicTocGenerator():

    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti


TicToc = TicTocGenerator()


def toc(tempBool=True):

    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():

    toc(False)


if __name__ == "__main__":
    pass
