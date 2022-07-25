# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

from NEURAL_py_EEG import utils
import math
import numpy as np
from NEURAL_py_EEG import NEURAL_parameters


def mat_percentile(x, q):
    """
    Percentile function to mimic MATLAB
    :param x:
    :param q:
    :return:
    """
    q = np.array(q) / 100
    x = np.delete(x, np.where(np.isnan(x))[0])
    n = len(x)
    y = np.sort(x)
    return np.interp(q, np.linspace(1 / (2 * n), (2 * n - 1) / (2 * n), n), y)


def gen_rEEG(data, win_overlap, win_length, win_type, fs, APPLY_LOG_LINEAR_SCALE):
    """
    (list) -> list

    generate the rEEG from EEG data that has been passed in

    >gen_rEEG()

    Note important to load and use parameters correctly, win_length is 64 seconds for the other features but the
    win_length for this refers to 2 seconds by default

    """
    L_hop, L_epoch, win_epoch = utils.gen_epoch_window(
        win_overlap, win_length, win_type, fs
    )
    N = len(data)
    N_epochs = math.floor((N - (L_epoch - L_hop)) / L_hop)
    if N_epochs < 1:
        N_epochs = 1

    nw = np.array(list(range(0, L_epoch)))
    reeg = np.empty([N_epochs])
    for k in range(N_epochs):
        nf = (nw + k * L_hop) % N
        x_epoch = np.multiply(data[nf], win_epoch)

        if all(np.isnan(x_epoch)):
            reeg[k] = np.nan
        else:
            reeg[k] = np.nanmax(x_epoch) - np.nanmin(x_epoch)
    # no need to resample (as per [1])

    if APPLY_LOG_LINEAR_SCALE:
        ihigh = [idx for idx, val in enumerate(reeg) if val > 50]
        if not ihigh:
            reeg = [
                (50 * math.log(aa) / math.log(50)) if i in ihigh else aa
                for i, aa in enumerate(reeg)
            ]

    return reeg


def rEEG_mean(reeg):
    """
    (list) -> float

    pass in the reeg and return the mean value

    >rEEG_mean([2,3,4])
    3
    
    """
    return np.nanmean(reeg)


def rEEG_median(reeg):
    """
    (list) -> float

    pass in the reeg and return the median value

    >rEEG_median([2,3,4,5,6,9,9])
    5

    """
    return np.nanmedian(reeg)


def rEEG_lower_margin(reeg):
    """
    (list) -> float

    pass in the reeg and return the 5th percentile value

    >rEEG_lower_margin([2,3,4,5,6,9,9])
    2

    """

    return mat_percentile(reeg, 5)


def rEEG_upper_margin(reeg):
    """
    (list) -> float

    pass in the reeg and return the 95th percentile value

    >rEEG_upper_margin([2,3,4,5,6,9,9])
    9

    """
    return mat_percentile(reeg, 95)


def rEEG_width(reeg):
    """
    (list) -> float

    pass in the reeg and return the bandwidth

    >rEEG_width([2,3,4,5,6,9,9])
    7

    """
    return rEEG_upper_margin(reeg) - rEEG_lower_margin(reeg)


def rEEG_SD(reeg):
    """
    (list) -> float

    pass in the reeg and return the standard deviation

    >rEEG_SD([2,3,4,5,6,9,9])
    2.7603

    """
    return np.nanstd(reeg, ddof=1)


def rEEG_CV(reeg):
    """
    (list) -> float

    pass in the reeg and return the coefficient of variance

    >rEEG_CV([2,3,4,5,6,9,9])
    0.5085

    """
    return np.nanstd(reeg, ddof=1) / np.nanmean(reeg)


def rEEG_asymmetry(reeg):
    """
    (list) -> float

    pass in the reeg and return the asymmetry

    >rEEG_asymmetry([2,3,4,5,6,9,9])
    0.1429

    """
    A = np.nanmedian(reeg) - rEEG_lower_margin(reeg)
    B = rEEG_upper_margin(reeg) - np.nanmedian(reeg)
    return (B - A) / (A + B)


def main_rEEG(x, Fs, feat_name, params=None):
    """
    Syntax: featx = rEEG(x, Fs, feat_name, params)
    
    Inputs:
        x          - epoch of EEG data (size 1 x N)
        Fs         - sampling frequency (in Hz)
        feat_name  - feature type, defaults to 'rEEG_mean';
                     see full list of 'rEEG_' features in params['FEATURE_SET_ALL']
        params  - parameters (as dictionary);
                     see NEURAL_parameters() for examples
    
    Outputs:
        featx  - feature at each frequency band
    
    Example:
        import utils
        import rEEG
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        x = data_st['eeg_data'][data_st['ch_labels'][0]].values
    
        featx = rEEG.main_rEEG(x, Fs, 'rEEG_width')


    [1] D Oâ€™Reilly, MA Navakatikyan, M Filip, D Greene, & LJ Van Marter (2012). Peak-to-peak
    amplitude in neonatal brain monitoring of premature infants. Clinical Neurophysiology,
    123(11), 2139â€“53.
    """
    if params is None:
        params = NEURAL_parameters.NEURAL_parameters()
        if "rEEG" in params:
            params = params["rEEG"]
        else:
            raise ValueError("No default parameters found")
    elif len(params) == 0:
        params = NEURAL_parameters.NEURAL_parameters()
        if "rEEG" in params:
            params = params["rEEG"]
        else:
            raise ValueError("No default parameters found")

    freq_bands = params["freq_bands"]

    N_freq_bands = len(freq_bands)
    if N_freq_bands == 0:
        N_freq_bands = 1

    x_orig = x.copy()

    featx = np.empty(N_freq_bands)
    for n in range(N_freq_bands):
        if not len(freq_bands) == 0:
            x_filt, dum = utils.filter_butterworth_withnans(
                x_orig,
                Fs,
                freq_bands[n][1],
                freq_bands[n][0],
                5,
                params["FILTER_REPLACE_ARTEFACTS"],
            )
            x_orig = x.copy()
        else:
            x_filt = x.copy()

        reeg = gen_rEEG(
            x_filt,
            params["overlap"],
            params["L_window"],
            params["window_type"],
            Fs,
            params["APPLY_LOG_LINEAR_SCALE"],
        )

        try:
            featx[n] = eval(feat_name)(reeg)
        except:
            raise ValueError("Feature function not found: %s" % feat_name)

    return featx
