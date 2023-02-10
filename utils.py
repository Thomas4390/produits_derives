import math
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import fsolve


def get_info():
    K = np.array([80, 90, 97.5, 102.5, 110, 120])
    Put = np.array([0.1900, 0.6907, 1.6529, 3.3409, 9.8399, 19.5805])
    return pd.DataFrame(np.vstack([K, Put]).T, columns=["Strike", "Put"])


def d1(S, K, r, y, T, sigma): 
    '''Calculate d1 from the Black, Merton and Scholes formula'''
    return (np.log(S/K) + (r - y + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))


def d2(S, K, r, y, T, sigma): 
    '''Calculate d2 from the Black, Merton and Scholes formula'''
    return (np.log(S/K) + (r - y - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))


def delta(S, K, r, y, T, sigma, is_call):  
    '''Return Black, Merton, Scholes delta of the European (call, put)'''
    _d1 = d1(S, K, r, y, T, sigma)    
    d_sign = np.where(is_call, 1, -1)
    return d_sign*norm.cdf(d_sign*_d1)

def gamma(S, K, r, y, T, sigma, is_call=None):
    '''Return Black, Merton, Scholes gamma of the European call or put

    Accepts is_call argument for consistency in the functions' signatures, but it is neglected
    '''
    _d1 = d1(S, K, r, y, T, sigma)
    return np.exp(-y*T)*norm.pdf(_d1)/(S*sigma*np.sqrt(T))

def option_price(S, K, r, y, T, sigma, is_call, ret_delta=False):  
    '''Return Black, Merton, Scholes price of the European option'''
    _d1 = d1(S, K, r, y, T, sigma)
    _d2 = _d1 - sigma*np.sqrt(T)
    
    # d_sign: Sign of the the option's delta
    d_sign = np.where(is_call, 1, -1)
    delta = d_sign*norm.cdf(d_sign*_d1)
    premium = np.exp(-y*T)*S*delta - d_sign*np.exp(-r*T)*K*norm.cdf(d_sign*_d2);
    if ret_delta:
        return premium, delta
    return premium


def implied_volatility(opt_price, S, K, r, y, T, is_call, init_vol=0.6):
    '''Inverse the BMS formula numerically to find the implied volatility'''
    def pricing_error(sig):
        sig = abs(sig)
        return option_price(S,K,r,y,T,sig,is_call) - opt_price
    return fsolve(pricing_error, init_vol)

def plot_implied_vol(S, info):
    '''Plot implied vol. for question 1'''
    plt.plot(info['Strike']/S, info['Implied vol.'], 'o--')
    plt.xlabel(r'$\frac{K}{S_0}$', fontweight="bold")
    plt.ylabel("Sigma", fontweight="bold")
    plt.title("Volatilité implicite en fonction de " + r'$\frac{K}{S_0}$', fontweight="bold")
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()

def CRR_tree(S, K, T, r, sigma, Type, N, ret_gamma=False):
    '''Compute the call or put price with CRR tree'''

    # Calcul préliminaire
    dt = T / N
    df = math.exp(-r*dt)
    u  = math.exp(sigma*math.sqrt(dt))
    d  = 1 / u
    q  = (math.exp(r*dt) - d) / (u - d) 

    # Calcul des valeurs finales du put dans l'arbre
    f_end = np.zeros(N + 1)
    for i in range(0, N + 1):
        S_i      = S * (u ** (N - i)) * (d ** (i))
        f_end[i] = max(K - S_i, 0)

    # Calcul de la valeur du put américain par induction inverse
    # Européen
    if Type == 0 : 
        for i in range(N, 0, -1):
            for j in range(0, i):
                f_end[j] = df * (q * f_end[j] + (1 - q) * f_end[j + 1])

    # Américain
    else:
        for i in range(N, 0, -1):
            for j in range(0, i):
                S_j      = S * (u ** (i-j-1)) * (d ** j)
                f_end[j] = max(df * (q * f_end[j] + (1 - q) * f_end[j + 1]), K - S_j)

    put_value = f_end[0]
            
    return put_value


def CRR_tree_BD(S, K, T, r, sigma, Type, N, ret_gamma=False):
    '''Compute the call or put price with CRR tree adjusted
        with the Broadie and Detemple correction.'''

    # Calcul préliminaire
    dt = T / N
    df = math.exp(-r*dt)
    u  = math.exp(sigma*math.sqrt(dt))
    d  = 1 / u
    q  = (math.exp(r*dt) - d) / (u - d) 

    # Calcul des valeurs finales du put dans l'arbre
    f_end = np.zeros(N)
    for i in range(0, N):
        S_i      = S * (u ** (N - i - 1)) * (d ** (i))
        f_end[i] = option_price(S = S_i, K = K, r = r, y = 0, T = dt, sigma = sigma, is_call = False)

    # Calcul de la valeur du put américain par induction inverse
    # Européen
    if Type == 0 : 
        for i in range(N-1, 0, -1):
            for j in range(0, i):
                f_end[j] = df * (q * f_end[j] + (1 - q) * f_end[j + 1])
                
            if i == 3:
                p_uu = f_end[0] 
                p_ud = f_end[1]
                p_dd = f_end[2]
                
    # Américain
    else:
        for i in range(N-1, 0, -1):
            for j in range(0, i):
                S_j      = S * (u ** (i-j-1)) * (d ** j)
                f_end[j] = max(df * (q * f_end[j] + (1 - q) * f_end[j + 1]), K - S_j)

            if i == 3:
                p_uu = f_end[0] 
                p_ud = f_end[1]
                p_dd = f_end[2]

    if ret_gamma:
        gamma_0 = (p_uu - 2*p_ud + p_dd) / (((u - d) * S) ** 2)
        gamma_1 = (((p_uu - p_ud) / (((u**2) * S) - S)) - ((p_ud - p_dd) / (-((d**2) * S) + S))) / (S * (u - d))
        gamma_2 = (((p_uu - p_ud) / (((u**2) * S) - S)) - ((p_ud - p_dd) / (-((d**2) * S) + S))) / (0.5 * S * (u**2 - d**2))
        gamma   = [gamma_0, gamma_1, gamma_2]
    
        return f_end[0], gamma
    
    else:
        return f_end[0]
            


def CRR_tree_df(S, K, T, r, sigma, Type, N_Range) :
    '''Iterates over all puts and conert to Data Frame'''

    crr_values = np.array(
        [
            [
                CRR_tree(S = S, K = K.iloc[i], T = T, r = r, sigma = sigma.iloc[i],
                         N = N, Type = Type)
                for N in N_Range
            ]
            for i in range(len(K))
        ]
    ).T
    columns = [f"Put_{i}" for i in range(len(K))]
    df_crr = pd.DataFrame(crr_values, columns=columns)
    df_crr.index = N_Range

    return df_crr


def CRR_tree_BD_df(S, K, T, r, sigma, Type, N_Range, ret_gamma = False) :
    '''Iterates over all puts and conert to Data Frame'''
    if not ret_gamma:
        crr_values = np.array(
            [
                [
                    CRR_tree_BD(S = S, K = K.iloc[i], T = T, r = r, sigma = sigma.iloc[i],
                                N = N, Type = Type, ret_gamma = ret_gamma)
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
                    CRR_tree_BD(S = S, K = K.iloc[i], T = T, r = r, sigma = sigma.iloc[i],
                                N = N, Type = Type, ret_gamma = ret_gamma)[0]
                    for N in N_Range
                ]
                for i in range(len(K))
            ]
        ).T
        
        gamma = list(
            
                [
                    CRR_tree_BD(S = S, K = K.iloc[i], T = T, r = r, sigma = sigma.iloc[i],
                                N = N, Type = Type, ret_gamma = ret_gamma)[1]
                    for N in N_Range
                ]
                for i in range(len(K))
            
        )

        gamma_df     = [pd.DataFrame(np.array(g), columns = ['gamma_0', 'gamma_1', 'gamma_2']) for g in gamma]
        columns      = [f"Put_{i}" for i in range(len(K))]
        df_crr       = pd.DataFrame(crr_values, columns=columns)
        df_crr.index = N_Range

        return df_crr, gamma_df


def plot_CRR_tree(df_CRR_list, N_Range, bps: float = 0.0001, zoom_factor: int = 20, cross = False):
    '''Plot results for question 2'''

    plt.style.use('seaborn-v0_8-deep')
    info      = get_info()
    figsize   = (15, 20)
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    for i in range(3):
        for j in range(2):
            k = i * 2 + j 
            plt.sca(axes[i, j])
            for df_CRR in df_CRR_list:
                plt.plot(N_Range, df_CRR[0][f'Put_{k}'], df_CRR[2])
                
            plt.hlines(info['Put'].iloc[k], N_Range[0], N_Range[-1],
                       linestyles='dashed', color='red')
            plt.hlines(info['Put'].iloc[k] + bps, N_Range[0], N_Range[-1],
                       linestyles='dotted', color='orange')
            plt.hlines(info['Put'].iloc[k] - bps, N_Range[0], N_Range[-1],
                       linestyles='dotted', color='orange')
            if cross:
                    data     = df_CRR[0][f'Put_{k}']
                    value_up = info['Put'].iloc[k] + bps
                    value_dw = info['Put'].iloc[k] - bps
                    crossing = next((i for i, v in enumerate(data) if v > value_dw and v < value_up), 0)
                    if crossing != 0:
                        plt.plot(N_Range[crossing], list(df_CRR[0][f'Put_{k}'])[crossing], 'o', markersize=10, markerfacecolor="red")
                        if list(df_CRR[0][f'Put_{k}'])[crossing] - info['Put'].iloc[k] > 0:
                            plt.text(N_Range[crossing], list(df_CRR[0][f'Put_{k}'])[crossing] + bps,
                                     "N = " + str(N_Range[crossing]), fontweight="bold")
                        else:
                            plt.text(N_Range[crossing], list(df_CRR[0][f'Put_{k}'])[crossing] - bps,
                                     "N = " + str(N_Range[crossing]), fontweight="bold")
            plt.ylim(info['Put'].iloc[k] - bps * zoom_factor,
                     info['Put'].iloc[k] + bps * zoom_factor)
            plt.legend([df_CRR[1] for df_CRR in df_CRR_list]+ ["BMS", "+1bp", "-1bp"])
            plt.xlabel("N")
            plt.ylabel("Put")
            axes[i, j].yaxis.set_label_position("right")
            plt.title(
                f"Prix Put par CRR en fonction de N pour K={info['Strike'].iloc[k]}")
    return None 


def plot_gamma(gamma, gamma_bms, N_Range, bps: float = 0.0001, zoom_factor: int = 20):
    '''Plot results for question 3 (gamma)'''

    plt.style.use('seaborn-v0_8-deep')
    info      = get_info()
    figsize   = (15, 20)
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for i in range(3):
        for j in range(2):
            k = i * 2 + j 
            plt.sca(axes[i, j])
            
            plt.plot(N_Range, gamma[k]['gamma_0'])
            plt.plot(N_Range, gamma[k]['gamma_1'])
            plt.plot(N_Range, gamma[k]['gamma_2'],':')
                
            plt.hlines(gamma_bms[k], N_Range[0], N_Range[-1],
                       linestyles='dashed', color='red')
            plt.hlines(gamma_bms[k] + bps, N_Range[0], N_Range[-1],
                       linestyles='dotted', color='orange')
            plt.hlines(gamma_bms[k] - bps, N_Range[0], N_Range[-1],
                       linestyles='dotted', color='orange')
            
            plt.ylim(gamma_bms[k] - bps * zoom_factor,
                     gamma_bms[k] + bps * zoom_factor)
            plt.legend([r'$\Gamma_0$', r'$\Gamma_1$',r'$\Gamma_2$',"BMS","+1bp", "-1bp"])
            plt.xlabel("N")
            plt.ylabel("Gamma")
            axes[i, j].yaxis.set_label_position("right")
            plt.title(
                f"Gamma du Put par CRR en fonction de N pour K={info['Strike'].iloc[k]}")
    return None 






