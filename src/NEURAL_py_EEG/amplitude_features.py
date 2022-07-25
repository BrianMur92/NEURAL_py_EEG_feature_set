# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

import numpy as np
from scipy import signal
from scipy import stats
from NEURAL_py_EEG import utils
from NEURAL_py_EEG import NEURAL_parameters


def amplitude_total_power(data):
    """
    (list) -> float

    pass in an array or list and return the signal power

    >signal_power([3,7,8,9,1])
    40.8
    """
    return np.nanmean(np.abs(data) ** 2)


def amplitude_env_mean(data):
    """
    (list) -> float

    pass in an array and return the mean value of the envelope

    >signal_envelope([3,7,8,9,1])
    50.24

    """

    return np.nanmean(np.abs(signal.hilbert(data)) ** 2)


def amplitude_SD(data):
    """
    (list) -> float

    pass in an array or list and return the standard deviation

    >signal_sd([3,7,8,9,1])
    3.4351
    """
    return np.nanstd(data, ddof=1)


def amplitude_env_SD(data):
    """
    (list) -> float

    pass in an array and return the standard deviation of the envelope

    >signal_envelope_sd([3,7,8,9,1])
    32.8035

    """

    return np.nanstd(np.abs(signal.hilbert(data)) ** 2, ddof=1)


def amplitude_skew(data):
    """
    (list) -> float

    pass in an array or list and return the absolute signal skewness

    >signal_skewness([3,7,8,9,1])
    0.4071
    """
    return np.abs(stats.skew(data, nan_policy="omit"))


def amplitude_kurtosis(data):
    """
    (list) -> float

    pass in an array or list and return the kurtosis value of data. Using Pearson's definition

    >signal_skewness([3,7,8,9,1])
    1.4904
    """
    return stats.kurtosis(data, fisher=False)


def main_amplitude(x, Fs, feat_name, params=None):
    """
    Syntax: featx = amplitude_features(x,Fs,feat_name,params)

    Inputs:
        x          - epoch of EEG data (size 1 x N)
        Fs         - sampling frequency (in Hz)
        feat_name  - feature type, defaults to 'amplitude_total_power';
                     see full list of 'amplitude_' features in params['FEATURE_SET_ALL']
        params     - parameters (as dictionary);

    Outputs:
        featx  - feature at each frequency band

    Example:
        import utils
        import amplitude_features
        
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        x = data_st['eeg_data'][data_st['ch_labels'][0]].values

        featx = amplitude_features.main_amplitude(x, Fs, 'amplitude_env_mean')
    """
    if params is None:
        params = NEURAL_parameters.NEURAL_parameters()
        if "amplitude" in params:
            params = params["amplitude"]
        else:
            raise ValueError("No default parameters found")
    elif len(params) == 0:
        params = NEURAL_parameters.NEURAL_parameters()
        if "amplitude" in params:
            params = params["amplitude"]
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

        try:
            x_filt = np.delete(x_filt, np.where(np.isnan(x_filt))[0])
            featx[n] = eval(feat_name)(x_filt)
        except:
            raise ValueError("Feature function not found: %s" % feat_name)

    return featx
