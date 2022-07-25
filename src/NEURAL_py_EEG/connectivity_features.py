# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

import numpy as np
from NEURAL_py_EEG import spectral_features
import warnings
from NEURAL_py_EEG import utils
from scipy import signal
from scipy import stats
from NEURAL_py_EEG import NEURAL_parameters
import matplotlib as plt


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


def gen_cross_spectrum(x, y, Fs, param_st):
    """

    gen_cross_spectrum: cross-spectrums

    Syntax:
        [pxy,Nfreq,f_scale,fp] = connectivity_features.gen_cross_spectrum(x,y,Fs,param_st)

    Inputs:
        x,y,Fs,param_st -

    Outputs:
        [pxy,Nfreq,f_scale,fp]

    Example:

    """
    L_window = param_st["L_window"]
    window_type = param_st["window_type"]
    overlap = param_st["overlap"]
    # freq_bands = param_st['freq_bands']
    spec_method = param_st["method"]

    # remove nans
    x[np.isnan(x)] = []
    if spec_method.lower() == "bartlett-psd":
        # ---------------------------------------------------------------------
        # Bartlett PSD: same as Welch with 0 % overlap and rectangular window
        # ---------------------------------------------------------------------
        window_type = "rect"
        overlap = 0

        spec_method = "psd"

    # ---------------------------------------------------------------------
    # Welch cross - PSD
    # ---------------------------------------------------------------------
    S_x, Nfreq, f_scale, win_epoch = spectral_features.gen_STFT(
        x, L_window, window_type, overlap, Fs, 1
    )
    S_y, Nfreq, f_scale, win_epoch = spectral_features.gen_STFT(
        y, L_window, window_type, overlap, Fs, 1
    )

    S_xy = S_x * np.conjugate(S_y)  # check this

    if spec_method.lower() == "psd":
        # ---------------------------------------------------------------
        # Mean for Welch PSD
        # ---------------------------------------------------------------
        pxy = np.nanmean(S_xy, axis=0)  # Check this
    elif spec_method.lower() == "robust-psd":
        # ---------------------------------------------------------------
        # Mean for Welch PSD
        # ---------------------------------------------------------------
        pxy = np.nanmedian(S_xy, axis=0)  # Check this

    else:
        raise ValueError(
            "Unknown cross-spectral method %s - check spelling" % spec_method
        )

    # normalise (so similar to Welch's PSD)
    E_win = np.sum(np.abs(win_epoch) ** 2) / Nfreq
    pxy = pxy / (Nfreq * E_win * Fs)

    N = len(pxy)
    f_scale = Nfreq / Fs

    fp = np.array(range(N)) / f_scale

    return pxy, Nfreq, f_scale, fp


def gen_coherence(x, y, Fs, param_st):
    """
    ---------------------------------------------------------------------
     generate coherence (magnitude only) between x and y
    ---------------------------------------------------------------------
    :param x:
    :param y:
    :param Fs:
    :param param_st:
    :return:
    """
    pxx = spectral_features.gen_spectrum(x, Fs, param_st)[0]
    pyy = spectral_features.gen_spectrum(y, Fs, param_st)[0]
    pxy, dum, f_scale, fp = gen_cross_spectrum(x, y, Fs, param_st)

    c = (np.abs(pxy) ** 2) / (pxx * pyy)
    return c, pxx, pyy, f_scale, fp


def rand_phase(x, N_iter=1):
    """

    Syntax: y = connectivity_features.rand_phase(x, N_iter)

    Inputs:
        x      - input signal, length-N
        N_iter - number of random signals required

    Outputs:
        y - output signal, same amplitude spectrum but with random phase
            (size N_iter x N)

    Example:
        import numpy as np
        import connectivity_features
        import matplotlib.pyplot as plt

        N = 1000
        N_iter = 500
        x = np.random.randn(N)
        y = connectivity_features.rand_phase(x, N_iter)

        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(x)
        plt.subplot(2, 1, 2)
        plt.plot(y[1, :])
        plt.show()


    rand_phase: randomize phase (of FFT) for real-valued signal

    :param x: Input signal of length N
    :param N_iter: Number of random signals required
    :return: y - output signal, same amplitude spectrum but with random phase (size N_iter x N
    """
    if len(x) == 1:
        x = x.transpose()

    N = len(x)  # check
    Nh = np.floor(N / 2).astype(int)

    X = np.fft.fft(x)

    # ---------------------------------------------------------------------
    # 1. generate random phase
    # ---------------------------------------------------------------------
    # np.random.seed(ttt)
    if N % 2 == 1:
        rphase = np.array(-np.pi + 2 * np.pi * np.random.uniform(0, 1, [N_iter, Nh]))
        rphase = np.hstack([np.zeros([N_iter, 1]), rphase[:, :], -rphase[:, ::-1]])
    else:
        rphase = np.array(
            -np.pi + 2 * np.pi * np.random.uniform(0, 1, [N_iter, Nh - 1])
        )
        # rand is to be flipped.
        # rphase = np.array(-np.pi + 2 * np.pi * np.random.uniform(0, 1, [Nh-1, N_iter]))  # Note: to replicate MATLAB
        # rphase = np.transpose(rphase)
        rphase = np.hstack(
            [np.zeros([N_iter, 1]), rphase, np.zeros([N_iter, 1]), -rphase[:, ::-1]]
        )

    # ---------------------------------------------------------------------
    # apply to spectrum and FFT back to time domain:
    # ---------------------------------------------------------------------

    if N_iter > 1:
        Y = np.empty([N_iter, len(X)]).astype(complex)
        for i in range(N_iter):
            for j in range(len(X)):
                Y[i, j] = np.exp(complex(0, rphase[i, j])) * np.abs(X[j])

    else:
        Y = np.empty(len(X)).astype(complex)
        for j in range(len(X)):
            Y[j] = np.abs(X[j]) * np.exp(complex(0, rphase[j]))

    y = np.real(np.fft.ifft(Y, axis=1))

    return y


def connectivity_BSI(
    x,
    Fs,
    params_st,
    N_channels,
    ileft,
    iright,
    ipairs,
    N_freq_bands,
    freq_bands,
    feat_name,
):
    """
    ---------------------------------------------------------------------
     brain sysmetry index (revised version, Van Putten, 2007)

     van Putten, MJAM (2007). The revised brain symmetry index. Clinical
     Neurophysiology, 118(11), 2362â€“2367. http://doi.org/10.1016/j.clinph.2007.07.019
    ---------------------------------------------------------------------
    :param feat_name:
    :param freq_bands:
    :param N_freq_bands:
    :param ipairs:
    :param iright:
    :param ileft:
    :param N_channels:
    :param x: input epoch
    :param Fs: sampling frequency
    :param params_st: parameters
    :return:
    """

    # a) PSD estimate (Welch's periodogram)
    x_epoch = np.zeros([len(x)])
    X = np.zeros([N_channels, len(x_epoch)])
    for k in range(N_channels):
        x_epoch = x[k, :]
        x_epoch = np.delete(x_epoch, np.where(np.isnan(x_epoch))[0])

        tmp, dum, f_scale, dum, fp = spectral_features.gen_spectrum(
            x_epoch, Fs, params_st, 1
        )
        if k == 0:
            X = np.zeros([N_channels, len(tmp)])
        X[k, :] = tmp

    dum, N = X.shape  # Double check this

    if len(ileft) > 1:
        X_left = np.nanmean(
            X[ipairs[0, :].astype(int), :], axis=0
        )  # Check what the 1 does
        X_right = np.nanmean(X[ipairs[1, :].astype(int), :], axis=0)  # Again check
    else:
        X_left = X[ileft, :]
        X_right = X[iright, :]

    featx = np.zeros(N_freq_bands)
    ibandpass = []  # to suppress warning
    for n in range(N_freq_bands):
        if n == 0:
            istart = np.ceil(freq_bands[n, 0] * f_scale).astype(int) - 1
        else:
            istart = ibandpass[-1] - 1

        ibandpass = np.array(
            range(istart, np.floor(freq_bands[n][1] * f_scale).astype(int))
        )
        ibandpass = ibandpass + 1
        ibandpass[ibandpass < 0] = 0
        ibandpass[ibandpass > N - 1] = N - 1

        featx[n] = np.nanmean(
            np.abs(
                (X_left[ibandpass] - X_right[ibandpass])
                / (X_left[ibandpass] + X_right[ibandpass])
            )
        )

    return featx


def connectivity_coh(
    x,
    Fs,
    params_st,
    N_channels,
    ileft,
    iright,
    ipairs,
    N_freq_bands,
    freq_bands,
    feat_name,
):
    """
    ---------------------------------------------------------------------
     coherence (using Welch's PSD)
    ---------------------------------------------------------------------

    :param x:
    :param Fs:
    :param params_st:
    :param N_channels:
    :param ileft:
    :param iright:
    :param ipairs:
    :param N_freq_bands:
    :param freq_bands:
    :return:
    """
    # check to see if using the right PSD estimate
    # (only 'PSD' or 'bartlett-PSD' allowed)
    if params_st["method"] == "periodogram" or params_st["method"] == "robust-PSD":
        print("----------- -WARNING- ------------\n")
        print("Must use averaging PSD estimate (e.g. Welch PSD) for coherence.\n")
        print('To do so, set: feat_params_st.connectivity.method="PSD"\n')
        print("in " "NEURAL_parameters.py" " file.\n")

        if params_st["coherence_zero_level"].lower() == "analytic":
            params_st["method"] = "bartlett-PSD"

            warnings.warn(
                "Forcing PSD method for connectivity analysis to Bartlett PSD",
                DeprecationWarning,
            )

        else:
            warnings.warn(
                "Forcing PSD method for connectivity analysis to Welch PSD \
            (may need to adjust window type, window size, and overlap.)",
                DeprecationWarning,
            )

            params_st["method"] = "PSD"
        print("------------------------------------------------------------------")

    if (
        params_st["coherence_zero_level"].lower() == "analytic"
        and params_st["method"] != "bartlett-PSD"
    ):

        # check if PSD method is Bartlett or not:
        print("----------- -WARNING- ------------\n")
        print("If want to use the analytic zero-level threshold for\n")
        print("coherence (Halliday et al. 1995) then need to use Barlett PSD.\n")
        print("To do so, set: feat_params_st.connectivity.method='bartlett-PSD'\n")
        print("in 'NEURAL_parameters.py' file.\n")
        warnings.warn(
            ["Forcing PSD method for connectivity analysis to Bartlett PSD"],
            DeprecationWarning,
        )
        print("----------------------------------\n")

        params_st["method"] = "bartlett-PSD"

    # PSD and x-PSD
    dum, N_pairs = ipairs.shape
    featx_pairs = np.empty([N_freq_bands, N_pairs])
    featx_pairs.fill(np.nan)

    coh = np.zeros([N_pairs, 1])

    for p in range(N_pairs):
        x1 = x[ipairs[0, p].astype(int), :]
        x2 = x[ipairs[1, p].astype(int), :]
        x1 = np.delete(x1, np.where(np.isnan(x1))[0])
        x2 = np.delete(x2, np.where(np.isnan(x2))[0])

        # 1) estimage the coherence function
        c, pxx, pyy, f_scale, fp = gen_coherence(
            x1, x2, Fs, params_st
        )  # Check size of coh
        if p == 0:
            coh = np.empty([N_pairs, len(c)])
        coh[p, :] = c

        # 2) if estimating a zero-level threshold for the coherence:
        if params_st["coherence_zero_level"] == "surr":
            # ---------------------------------------------------------------------
            # if generating a null-hypothesis distribution from surrogate data:
            # ---------------------------------------------------------------------

            L_surr = params_st["coherence_surr_iter"]

            # Generate surrogate signals
            x1_surr = rand_phase(x1, L_surr)
            x2_surr = rand_phase(x2, L_surr)

            coh_sur = []  # to suppress warning
            for m in range(L_surr):
                c_sur, dum, dum, dum, dum = gen_coherence(
                    x1_surr[m, :], x2_surr[m, :], Fs, params_st
                )
                if m == 0:
                    coh_sur = np.empty([L_surr, len(c_sur)])
                coh_sur[m, :] = c_sur

            # estimating frequency-dependent threshold at p < Î± level of significance
            coh_threshold = np.empty([1, len(c_sur)])
            for c in range(len(c_sur)):
                coh_threshold[:, c] = mat_percentile(
                    coh_sur[:, c], 100 * (1 - params_st["coherence_zero_alpha"])
                )

        elif params_st["coherence_zero_level"] == "analytic":
            # ---------------------------------------------------------------------
            # or if using an analytic method
            # ---------------------------------------------------------------------
            # number of segments(no overlap with Bartlett PSD)
            L = np.floor(len(x1) / (Fs * params_st["L_window"]))

            coh_threshold = 1 - (params_st["coherence_zero_alpha"]) ** (1 / (L - 1))

        elif params_st["coherence_zero_level"].lower == "":
            coh_threshold = np.array([])
        else:
            raise ValueError(
                "unknown option for 'coherence_zero_level'; see NEURAL_parameters.py for details"
            )

        # 3) threshold

        if coh_threshold.size != 0:
            coh[p, coh[p, :] < coh_threshold] = 0  # Check this

        # compute the coherence (either mean, max, or max. frequency) for each frequency band
        dum, N = coh.shape
        ibandpass = []  # to suppress warning
        for n in range(N_freq_bands):
            if n == 0:
                istart = np.ceil(freq_bands[n, 0] * f_scale).astype(int) - 1
            else:
                istart = ibandpass[-1] - 1

            ibandpass = np.array(
                range(istart, np.floor(freq_bands[n][1] * f_scale).astype(int))
            )  # May need +1
            ibandpass = ibandpass + 1
            ibandpass[ibandpass < 0] = 0
            ibandpass[ibandpass > N - 1] = N - 1

            if feat_name == "connectivity_coh_mean":
                featx_pairs[n, p] = np.nanmean(coh[p, ibandpass])
            elif feat_name == "connectivity_coh_max":
                featx_pairs[n, p] = np.max(coh[p, ibandpass])
            elif feat_name == "connectivity_coh_freqmax":
                imax = np.argmax(coh[p, ibandpass])
                featx_pairs[n, p] = fp[ibandpass[imax]]

    featx = np.nanmedian(featx_pairs, 1)  # compare this
    return featx


def connectivity_corr(
    x,
    Fs,
    params_st,
    N_channels,
    ileft,
    iright,
    ipairs,
    N_freq_bands,
    freq_bands,
    feat_name,
):
    """
    ---------------------------------------------------------------------
     cross-correlation (Pearson)
    ---------------------------------------------------------------------
    :param x:
    :param Fs:
    :param params_st:
    :param N_channels:
    :param ileft:
    :param iright:
    :param ipairs:
    :param N_freq_bands:
    :param freq_bands:
    :param feat_name:
    :return:
    """
    x_orig = x.copy()
    dum, N_pairs = ipairs.shape
    featx = np.empty(N_freq_bands)
    for n in range(N_freq_bands):
        inans = dict()
        x_filt = np.empty([N_channels, x.shape[1]])
        for p in range(N_channels):
            x_filt[p, :], inans[p] = utils.filter_butterworth_withnans(
                x_orig[p, :],
                Fs,
                freq_bands[n][1],
                freq_bands[n][0],
                5,
                params_st["FILTER_REPLACE_ARTEFACTS"],
            )
            x_orig = x.copy()

        cc_pairs = np.empty(N_pairs)
        cc_pairs.fill(np.nan)

        for p in range(N_pairs):
            all_inans = np.unique(
                np.hstack(
                    [inans[ipairs[0, p].astype(int)], inans[ipairs[1, p].astype(int)]]
                )
            )

            x1 = x_filt[ipairs[0, p].astype(int), :]
            x2 = x_filt[ipairs[1, p].astype(int), :]
            if all_inans.size != 0:
                x1 = np.delete(x1, all_inans)
                x2 = np.delete(x2, all_inans)

            env1 = np.abs(signal.hilbert(x1)) ** 2
            env2 = np.abs(signal.hilbert(x2)) ** 2

            cc_pairs[p] = stats.pearsonr(env1, env2)[0]

        featx[n] = np.nanmedian(cc_pairs)

    return featx


def channel_hemispheres(channels_all):
    """
    Syntax: ipairs = channel_hemisphere_pairs(channels_all)

    Inputs:
        channels_all - list of bipolar channel names
                         e.g. ['C3-O1','C4-O2', 'F3-C3', 'F4-C4']

    Outputs:
        ileft  - indices of the channels on left hemispheres
        iright - indices of the channels on right hemispheres

    Example:
        import utils
        import connectivity_features
        import numpy as np
        
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        channel_labels = np.array(data_st['ch_labels'])
        
        ileft, iright = connectivity_features.channel_hemispheres(channel_labels)

    """
    N_channels = len(channels_all)
    ileft = np.array([])
    iright = np.array([])
    for n in range(N_channels):
        cname = channels_all[n]
        M = len(cname)
        il = np.array([])
        ir = np.array([])
        for p in range(M):
            if str.isdigit(cname[p]):
                num = int(cname[p])
                if num % 2 == 1:
                    il = np.append(il, n)
                else:
                    ir = np.append(ir, n)

        if il.size != 0 and ir.size != 0:
            raise ValueError("both odd and even in channel: %s" % cname)

        if il.size != 0:
            ileft = np.append(ileft, n)
        elif ir.size != 0:
            iright = np.append(iright, n)
        else:
            warnings.warn("left or right channel:  %s ?" % cname, DeprecationWarning)

    return ileft.astype(int), iright.astype(int)


# noinspection SpellCheckingInspection
def channel_hemisphere_pairs(channel_labels):
    """
    Inputs:
        channel_labels - list of bipolar channel names
                         e.g. ['C3-O1','C4-O2', 'F3-C3', 'F4-C4']

    Outputs:
        ipairs - indices of the channel-pairs across hemispheres
                 e.g. 'C3-O1' is paired with 'C4-O2', 'F3-C3' paired with 'F4-C4'

    Example:
        import utils
        import connectivity_features
        import numpy as np
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        channel_labels = np.array(data_st['ch_labels'])

        ipairs = connectivity_features.channel_hemisphere_pairs(channel_labels)
        channel_labels[ipairs]

    channel_hemisphere_pairs: pair left/right channels

    Syntax: ipairs=channel_hemisphere_pairs(channel_labels)
    """
    DBverbose = 0
    # N = len(channel_labels)
    ileft, iright = channel_hemispheres(channel_labels)
    N_left = len(ileft)

    ipairs = np.empty([2, N_left])
    ipairs.fill(np.nan)
    [channel_labels[xx].upper() for xx in range(len(channel_labels))]

    for n in range(N_left):
        ipairs[0, n] = ileft[n]
        ch_left = channel_labels[ileft[n]].upper()

        # change numbers from odd to even
        ch_left_match = str.replace(
            str.replace(
                str.replace(str.replace(ch_left, "1", "2"), "3", "4"), "5", "6"
            ),
            "7",
            "8",
        )  # check this

        imatch = np.array([])
        for i in range(len(iright)):
            if channel_labels[iright[i]] == ch_left_match:
                imatch = np.array(i)
                break
        # and check for reversed order
        sep = str.find(ch_left_match, "-")
        ch1 = ch_left_match[:sep]
        ch2 = ch_left_match[sep + 1 :]
        ch_left_match_rv = ch2 + "-" + ch1

        imatch_rv = np.array([])
        for i in range(len(iright)):
            if channel_labels[iright[i]] == ch_left_match_rv:
                imatch_rv = np.array(i)
                break

        if imatch.size != 0:
            ipairs[1, n] = iright[imatch]
        elif imatch_rv.size != 0:
            ipairs[1, n] = iright[imatch_rv]
        else:
            ipairs[0:1, n] = np.nan
            if DBverbose:
                print("no matching pair for channel: %s\n" % ch_left)

        # if left / right side share common electrode (e.g.Cz),
        # then should ignore
        if ch1 in ch_left or ch2 in ch_left:
            ipairs[0:1, n] = np.nan

    irem = []
    for n in range(N_left):
        if np.isnan(ipairs[:, n]).any():
            irem.append(n)

    if len(irem) != 0:
        ipairs = np.delete(ipairs, irem, axis=1)
        if DBverbose:
            print(channel_labels[ipairs])

    return ipairs.astype(int)


def main_connectivity(x, Fs, feat_name, ch_labels, params=None):
    """
    Syntax: featx = connectivity_features(x, Fs, feat_name, params)

    Inputs:
        x          - epoch of EEG data (size 1 x N)
        Fs         - sampling frequency (in Hz)
        feat_name  - feature type, defaults to 'connectivity_BSI';
                     see full list of 'connectivity_' features in params['FEATURE_SET_ALL']
        params  - parameters (as dictionary)
                     see NEURAL_parameters() for examples

    Outputs:
        featx  - feature at each frequency band

    Example:
        import utils
        import connectivity_features
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        x = data_st['eeg_data'].to_numpy().transpose()
        channel_labels = data_st['ch_labels']

        featx = connectivity_features.main_connectivity(x, Fs, 'connectivity_corr', channel_labels)


    [1] van Putten, MJAM (2007). The revised brain symmetry index. Clinical Neurophysiology,
        118(11), 2362â€“2367.
    [2] Prichard D, Theiler J (1994). Generating surrogate data for time series with several
        simultaneously measured variables. Physical Review Letters, 1994;73(7):951â€“954.
    [3] Faes L, Pinna GD, Porta A, Maestri R, Nollo G (2004). Surrogate data analysis for
        assessing the significance of the coherence function. IEEE Transactions on
        Biomedical Engineering, 51(7):1156â€“1166.
    [4] Halliday, DM, Rosenberg, JR, Amjad, AM, Breeze, P, Conway, BA, &
        Farmer, SF. (1995). A framework for the analysis of mixed time series/point
        process data--theory and application to the study of physiological tremor, single
        motor unit discharges and electromyograms. Progress in Biophysics and Molecular
        Biology, 64(2â€“3), 237â€“278.

    """
    if params is None:
        params = NEURAL_parameters.NEURAL_parameters()
        if "connectivity" in params:
            params = params["connectivity"]
        else:
            raise ValueError("No default parameters found")
    elif len(params) == 0:
        params = NEURAL_parameters.NEURAL_parameters()
        if "connectivity" in params:
            params = params["connectivity"]
        else:
            raise ValueError("No default parameters found")
    if "connectivity" in params:
        params = params["connectivity"]

    freq_bands = np.array(params["freq_bands"])

    N_channels, dum = x.shape

    if N_channels < 2:
        warnings.warn("Requires at least 2 channels", DeprecationWarning)
        return np.nan

    N_freq_bands, dum = freq_bands.shape
    if N_freq_bands == 0:
        N_freq_bands = 1

    ileft = []
    iright = []
    ipairs = []
    if N_channels > 2 and len(ch_labels) != 0:
        ileft, iright = channel_hemispheres(ch_labels)
        ipairs = channel_hemisphere_pairs(ch_labels)
    elif N_channels == 2:
        # if no channel labels then guess
        ileft = 0
        iright = 1
        ipairs = np.array([1, 2])

    if ipairs.shape[1] == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    try:
        if (
            feat_name == "connectivity_coh_mean"
            or feat_name == "connectivity_coh_max"
            or feat_name == "connectivity_coh_freqmax"
        ):

            featx = eval("connectivity_coh")(
                x,
                Fs,
                params,
                N_channels,
                ileft,
                iright,
                ipairs,
                N_freq_bands,
                freq_bands,
                feat_name,
            )
        else:
            featx = eval(feat_name)(
                x,
                Fs,
                params,
                N_channels,
                ileft,
                iright,
                ipairs,
                N_freq_bands,
                freq_bands,
                feat_name,
            )
    except:
        raise ValueError("Feature function not found: %s" % feat_name)

    return featx
