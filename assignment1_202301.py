import os
import sys
import numpy as np
import pandas as pd

if os.getcwd().find("dorion_francois/code/assignments") >= 0:
    sys.path.append("../..")

from jupyter_notebook import *

# import binomial_tree as crr
import black_merton_scholes as bms


def get_info():
    K = np.array([80, 90, 97.5, 102.5, 110, 120])
    Put = np.array([0.1900, 0.6907, 1.6529, 3.3409, 9.8399, 19.5805])
    return pd.DataFrame(np.vstack([K, Put]).T, columns=["Strike", "Put"])
