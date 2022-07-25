# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

import numpy as np
from NEURAL_py_EEG import utils
from NEURAL_py_EEG import NEURAL_parameters


def fd_higuchi(x, kmax="Nope"):
    """
    ---------------------------------------------------------------------
     Higuchi estimate in [1]
    ---------------------------------------------------------------------
    """
    N = len(x)
    if kmax == "Nope":
        kmax = int(np.floor(len(x) / 10))

    # what value of k to compute
    ik = 1
    k_all = np.empty(0)
    knew = 0

    while knew < kmax:
        if ik <= 4:
            knew = ik
        else:
            knew = np.floor(2 ** ((ik + 5) / 4))

        if knew <= kmax:
            k_all = np.append(k_all, knew)
        ik = ik + 1

    """
    ---------------------------------------------------------------------
    curve length for each vector:
    ---------------------------------------------------------------------
    """

    inext = 0
    L_avg = np.zeros(len(k_all))

    for k in k_all:
        L = np.zeros(k.astype(int))
        for m in range(k.astype(int)):
            ik = np.array(range(1, np.floor((N - m - 1) / k).astype(int) + 1))
            scale_factor = (N - 1) / (np.floor((N - m - 1) / k) * k)
            L[m] = np.nansum(
                np.abs(
                    x[m + np.array(ik) * k.astype(int)]
                    - x[m + (ik - 1) * k.astype(int)]
                )
            ) * (scale_factor / k)

        L_avg[inext] = np.nanmean(L)
        inext += 1

    """
    -------------------------------------------------------------------
     form log-log plot of scale v. curve length
    -------------------------------------------------------------------
    """

    x1 = np.log2(k_all)
    y1 = np.log2(L_avg)
    c = np.polyfit(x1, y1, 1)
    FD = -c[0]

    # y_fit = c[0]*x1 + c[1]
    # y_residuals = y1 - y_fit
    # r2 = 1 - np.sum(y_residuals**2) / ((N-1) * np.var(y1))

    # return FD, r2, k_all, L_avg
    return FD


def fd_katz(x, dum=0):

    """
    ---------------------------------------------------------------------
    Katz estimate in [2]
    ---------------------------------------------------------------------
    """
    N = len(x)
    p = N - 1

    # 1. line-length
    L = np.empty(N - 1)
    for n in range(N - 1):
        L[n] = np.sqrt(1 + (x[n] - x[n + 1]) ** 2)

    L = np.sum(L)

    # 2. maximum distance
    d = np.zeros(p)
    for n in range(N - 1):
        d[n] = np.sqrt((n + 1) ** 2 + (x[0] - x[n + 1]) ** 2)

    d = np.max(d)

    D = np.log(p) / (np.log(d / L) + np.log(p))

    return D


def main_fd(x, fs, feat_name="fd_higuchi", params=None):
    """
    Syntax: featx = fd_features(x, Fs, params)

    Inputs:
        x          - epoch of EEG data (size 1 x N)
        Fs         - sampling frequency (in Hz)
        params     - parameters (as dictionary)
                     see NEURAL_parameters() for examples

    Outputs:
        featx  - fractal dimension for each frequency band

    Example:
        import utils
        import fd_features
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        x = data_st['eeg_data'][data_st['ch_labels'][0]].values

        featx = fd_features.main_fd(x, Fs)

    [1] T Higuchi, â€œApproach to an irregular time series on the basis of the fractal
    theory,â€� Phys. D Nonlinear Phenom., vol. 31, pp. 277â€“283, 1988.

    [2] MJ Katz, Fractals and the analysis of waveforms. Computers in Biology and Medicine,
    vol. 18, no. 3, pp. 145â€“156. 1988
    """
    if feat_name == "FD":
        feat_name = "fd_higuchi"

    if params is None:
        params = NEURAL_parameters.NEURAL_parameters()
        if "FD" in params:
            params = params["FD"]
        else:
            raise ValueError("No default parameters found")
    elif len(params) == 0:
        params = NEURAL_parameters.NEURAL_parameters()
        if "FD" in params:
            params = params["FD"]
        else:
            raise ValueError("No default parameters found")

    freq_bands = np.array(params["freq_bands"])

    N_freq_bands = freq_bands.ndim
    if N_freq_bands == 0:
        N_freq_bands = 1

    x_orig = x.copy()

    featx = np.empty(N_freq_bands)
    for n in range(N_freq_bands):
        if N_freq_bands == 1:
            x, dum = utils.filter_butterworth_withnans(
                x_orig,
                fs,
                freq_bands[1],
                freq_bands[0],
                5,
                params["FILTER_REPLACE_ARTEFACTS"],
            )
        else:
            x, dum = utils.filter_butterworth_withnans(
                x_orig,
                fs,
                freq_bands[n][1],
                freq_bands[n][0],
                5,
                params["FILTER_REPLACE_ARTEFACTS"],
            )

        try:
            featx[n] = eval(feat_name)(x, params["qmax"])
        except:
            raise ValueError("Feature function not found: %s" % feat_name)

    return featx
