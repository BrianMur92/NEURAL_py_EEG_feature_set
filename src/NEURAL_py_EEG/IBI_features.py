# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

from NEURAL_py_EEG import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


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
    if n == 0:
        return np.nan
    y = np.sort(x)
    return np.interp(q, np.linspace(1 / (2 * n), (2 * n - 1) / (2 * n), n), y)


def estimate_IBI_lengths(anno, percentiles_all, Fs):
    """
    Estimate the max./median IBI length
    """
    min_ibi_interval = Fs / 4
    lens_anno = utils.len_cont_zeros(anno)[0]

    ishort = np.where(lens_anno < min_ibi_interval)[0]

    if ishort.size > 0:
        lens_anno = np.delete(lens_anno, ishort)

    if len(lens_anno != 0):
        pc_anno = mat_percentile(lens_anno, percentiles_all)
    else:
        pc_anno = np.zeros(1)
    return pc_anno / Fs


def main_IBI(x, Fs, feat_name):
    """
    IBI_features: burst/inter-burst interval features (assuming preterm EEG, <32 weeks GA)

    Syntax: featx = main_IBI(x, Fs, feat_name, params_st)

    Inputs:
        x          - epoch of EEG data (size 1 x N)
        Fs         - sampling frequency (in Hz)
        feat_name  - feature type, defaults to 'IBI_length_max';
                     see full list of 'IBI_' features in params['FEATURE_SET_ALL']
        params_st  - parameters (as structure);
                     see NEURAL_parameters() for examples

    Outputs:
        featx  - IBI features

    Example:
        import utils
        import IBI_features
        import matplotlib.pyplot as plt
        sys.path.append('path to ->\\py_burst_detector-master\\')
        from burst_detector import utils as bdutils
        N = 5000
        Fs = 64
        x = bdutils.gen_impulsive_noise(N)

        featx = IBI_features.main_IBI(x,Fs,'IBI_burst_number');

    [1] JM Oâ€™ Toole, GB Boylan, RO Lloyd, RM Goulding, S Vanhatalo, NJ Stevenson
    (2017). Detecting Bursts in the EEG of Very and Extremely Premature Infants using a
    Multi-Feature Approach. under review, 2017.

    """

    DBplot = 0

    bdetect_path_string = (
        "Enter path to ->" + "py_burst_detector-master/py_burst_detector-master/"
    )
    bdetect_path = Path(bdetect_path_string)

    bdetect_path_exist = os.path.isfile(
        Path(bdetect_path_string + "burst_detector/" + "eeg_burst_detector.py")
    )
    if not bdetect_path_exist:
        print("\n** ------------ **\n")
        print("Burst detector not included in path (eeg_burst_detector.py)\n\n")
        print("If installed, ensure eeg_burst_detector.py and associated files \n")
        print("are included in the python path.\n")
        print(
            "To do this, update the bdetect_path with the location of the burst detector.\n"
        )
        print("(for more on search paths see: ")
        print(
            [
                "<a href=https://docs.python.org/3/library/os.path.html>python os path</a>)\n\n"
            ]
        )
        print("If the burst detector is not installed, download from:\n")
        print(
            [
                "<a href=http://otoolej.github.io/code/py_burst_detector/> py burst detector source code </a>\n"
            ]
        )
        print("** ------------ **\n")

        return np.nan

    sys.path.append(str(bdetect_path))
    from burst_detector import eeg_burst_detector

    burst_anno, svm_out = eeg_burst_detector.eeg_bursts(x.copy(), Fs)

    if DBplot:
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(x)
        plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(burst_anno)
        plt.ylim((-0.2, 1.2))

    if feat_name == "IBI_length_max":
        """
        ---------------------------------------------------------------------
        max. (95th percentile) inter-burst interval
        ---------------------------------------------------------------------
        """
        return estimate_IBI_lengths(burst_anno, 95, Fs)

    elif feat_name == "IBI_length_median":
        """
        ---------------------------------------------------------------------
        median inter-burst interval
        ---------------------------------------------------------------------
        """
        return estimate_IBI_lengths(burst_anno, 50, Fs)

    elif feat_name == "IBI_burst_prc":
        """
        ---------------------------------------------------------------------
        percentage of bursts
        ---------------------------------------------------------------------
        """
        return (np.nansum(burst_anno) / burst_anno.size) * 100

    elif feat_name == "IBI_burst_number":
        """
        ---------------------------------------------------------------------
        number of bursts
        ---------------------------------------------------------------------
        """
        return utils.len_cont_zeros(burst_anno, 1)[0].size

    else:
        raise ValueError('\n Unknown feature "%s"; check spelling\n' % feat_name)
