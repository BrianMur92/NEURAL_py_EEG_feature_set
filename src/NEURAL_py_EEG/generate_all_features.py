# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

from NEURAL_py_EEG import amplitude_features
from NEURAL_py_EEG import spectral_features
from NEURAL_py_EEG import connectivity_features
from NEURAL_py_EEG import rEEG
from NEURAL_py_EEG import fd_features
from NEURAL_py_EEG import utils
import numpy as np
import pandas as pd
from NEURAL_py_EEG import IBI_features
from NEURAL_py_EEG import NEURAL_parameters


def size_feature(feat_name):
    """
    size_feature: size of feature (e.g. array or 1 point)

    Syntax: N = size_feature(feat_name)

    Inputs:
        feat_name - feature name (e.g. 'spectral_power')

    Outputs:
        N - number of frequency bands

    Example:
        import generate_all_features
        N = generate_all_features.size_feature('spectral_power')
        print('Feature has %d frequency bands\n' % N)

    :param feat_name:  feature name (e.g. 'spectral_power')
    :return: N - number of frequency bands
    """

    params = NEURAL_parameters.NEURAL_parameters()
    feat_group = feat_name.split("_")

    feat_group = feat_group[0]

    if feat_group == "IBI":
        return 1

    if feat_name == "spectral_edge_frequency":
        # ---------------------------------------------------------------------
        # SPECIAL cases
        # ---------------------------------------------------------------------
        N = 1
    else:
        # ---------------------------------------------------------------------
        # feature set is size number of frequency bands
        # ---------------------------------------------------------------------
        bands = np.array(params[feat_group]["freq_bands"])
        if bands.ndim == 1:
            N = 1
        else:
            N = bands.shape[0]

    return N


def overlap_epochs(x, Fs, L_window, overlap=50, window_type="rect"):
    """
        ---------------------------------------------------------------------
        overlapping epochs in one matrix
        ---------------------------------------------------------------------

    :param x:
    :param Fs:
    :param L_window:
    :param overlap:
    :param window_type:
    :return:
    """

    L_hop, L_epoch, win_epoch = utils.gen_epoch_window(
        overlap, L_window, window_type, Fs
    )
    N = len(x)
    N_epochs = int(np.ceil((N - (L_epoch - L_hop)) / L_hop))
    if N_epochs < 1:
        N_epochs = 1
        print(
            "| WARNING: signal length is less than segment length (L_epoch - L_hop).\n"
        )
        print('| Adjust "EPOCH_LENGTH" or "EPOCH_OVERLAP" in ')
        print('"utils.py" file.\n')

    nw = np.array(list(range(L_epoch)))
    ix = np.array(list(range(N)))

    x_epochs = np.empty([N_epochs, L_epoch])
    x_epochs.fill(np.nan)

    for k in range(N_epochs):
        nf = nw + k * L_hop
        # zero-pad if outside x
        if np.max(nf) > ix[-1]:
            nf = np.delete(nf, np.where(nf > ix[-1])[0])
        i_nf = np.array(list(range(len(nf))))

        x_epochs[k, i_nf] = x[nf] * win_epoch[i_nf]

    return x_epochs


def generate_all_features(data, save_p_l=None, channel_names=None, feat_set=None):
    """
    Syntax: feat_st = generate_all_features(data, save_path, channel_names, feat_set)

    Inputs:
        data          - either EEG filename or data dictionary with EEG
                         e.g. data dictionary: data_st = utils.gen_test_EEGdata(5*60, 64, 1)
        channel_names  - channel labels to process (default, process all)
                         e.g. ['C3-O1','C4-O2','F3-C3','F4-C4']
        feat_set       - list of features to compute,
                         e.g. ['spectral_relative_power','rEEG_SD', 'connectivity_BSI']

    Outputs:
        feats_per_epochs - features estimated over all epochs
        feat_pd_names - feature names for dataframe
        feat_st - dictionary containing results - not separated into individual frequency bands
        feats_median_ch - DataFrame containing the the median over all channels. Epoch output. Separated in freq bands
        feats_median_all - DataFrame containing the the median over all channels and epochs. Separated in freq bands

    Example:
        import utils
        import generate_all_features

        # generate 5 minutes of simulated multichannel EEG data, with 64 Hz sample frequency
        data_st = utils.gen_test_EEGdata(5 * 60, 64, 1)

        # select features to compute:
        feature_set = ['spectral_relative_power', 'rEEG_SD', 'connectivity_BSI']

        # generate all features:
        feats_per_epochs, feat_pd_names, feat_st, feats_median_ch, feats_median_all =
        generate_all_features.generate_all_features(data_st, feat_set=feature_set)
    """

    params = NEURAL_parameters.NEURAL_parameters()

    # ---------------------------------------------------------------------
    # 1. load EEG data from .csv file
    # ---------------------------------------------------------------------

    if isinstance(data, dict):
        eeg_data = data["eeg_data"]
        Fs = data["Fs"]
        ch_labels = np.array(data["ch_labels"])
    else:
        eeg_data = pd.read_csv(data)
        Fs = int(data.split("_", -1)[-1][0:-4])
        ch_labels = np.array(eeg_data.columns)

    feats_per_epochs = np.array([])

    # select channels
    if not not channel_names:  # Check if channel_names is empty
        ikeep = []
        for n in range(len(channel_names)):
            it = [ind for ind, x in enumerate(ch_labels) if x == channel_names[n]]
            if not not it:
                ikeep.append(it)

        rem = [x for x in range(len(ch_labels)) if x not in ikeep]
        eeg_data = eeg_data.drop([ch_labels[rem]])
        ch_labels = ch_labels[ikeep]

    # or remove empty channels
    irem = []
    for n in range(len(ch_labels)):
        if all(np.isnan(eeg_data[ch_labels[n]])):
            irem.append(n)

    if not not irem:
        for r in range(len(irem)):
            eeg_data = eeg_data.drop([ch_labels[irem[r]]], axis=1)
        ch_labels = np.delete(ch_labels, irem)

    N_channels = len(eeg_data.columns)

    # ---------------------------------------------------------------------
    # 2. generate features
    # ---------------------------------------------------------------------
    if feat_set is None:
        feat_set = params["FEATURE_SET_ALL"]

    N_feats = len(feat_set)

    # A) iterate over features

    max_num_epochs = get_max_num_epochs(params, len(eeg_data), Fs)
    feat_pd_names = get_feature_names(params, feat_set)
    tmp1 = np.zeros([1, len(feat_pd_names)])
    tmp2 = np.zeros([max_num_epochs, len(feat_pd_names)])
    tmp1.fill(np.nan)
    tmp2.fill(np.nan)
    feats_median_all = pd.DataFrame(data=tmp1, columns=feat_pd_names)
    feats_median_ch = pd.DataFrame(
        data=tmp2[:, :98], columns=feat_pd_names[:98]
    )  # all names except IBI

    feat_st = dict()
    for n in range(N_feats):
        # print('%i / %i' % (n, N_feats-1))
        L_feature = size_feature(feat_set[n])
        feat_group = feat_set[n].split("_")
        feat_group = feat_group[0]

        # ---------------------------------------------------------------------
        # SPECTRAL and AMPLITUDE
        # (analysis on a per-channel basis and divide each channel into epochs)
        # ---------------------------------------------------------------------
        if feat_group in ["amplitude", "spectral", "rEEG", "FD"]:

            # B) iterate over all channels
            feats_channel = np.empty([N_channels, L_feature])
            feats_channel.fill(np.nan)

            for c in range(N_channels):
                x_epochs = overlap_epochs(
                    eeg_data[ch_labels[c]],
                    Fs,
                    params["EPOCH_LENGTH"],
                    params["EPOCH_OVERLAP"],
                )
                N_epochs, dum = x_epochs.shape

                if c == 0:
                    feats_per_epochs = np.empty([N_channels, N_epochs, L_feature])
                    feats_per_epochs.fill(np.nan)

                # C) iterate over epochs
                feats_epochs = np.empty([N_epochs, L_feature])
                feats_epochs.fill(np.nan)

                for e in range(N_epochs):
                    L_nans = len(np.where(np.isnan(x_epochs[e, :]))[0])
                    if (
                        100 * (L_nans / len(x_epochs[e, :]))
                        < params["EPOCH_IGNORE_PRC_NANS"]
                    ):
                        if feat_group == "spectral":
                            tmp = spectral_features.main_spectral(
                                x_epochs[e, :], Fs, feat_set[n]
                            )
                        elif feat_group == "FD":
                            tmp = fd_features.main_fd(x_epochs[e, :], Fs, feat_set[n])
                        elif feat_group == "amplitude":
                            tmp = amplitude_features.main_amplitude(
                                x_epochs[e, :], Fs, feat_set[n]
                            )
                        elif feat_group == "rEEG":
                            tmp = rEEG.main_rEEG(x_epochs[e, :], Fs, feat_set[n])
                        else:
                            raise ValueError(
                                'Incorrect "feat_group" - should not have entered here'
                            )

                        feats_epochs[e, :] = tmp
                feats_per_epochs[c, :, :] = feats_epochs
                # Median over all epochs
                feats_channel[c, :] = np.nanmedian(feats_epochs, axis=0)
            # Median over all channels
            feat_st[feat_set[n]] = np.nanmedian(feats_channel, axis=0)

            feats_median_ch = add_feat_data_to_array(
                feats_median_ch,
                np.median(feats_per_epochs, axis=0),
                feat_pd_names,
                feat_set[n],
                params,
            )
            feats_median_all = add_feat_data_to_array(
                feats_median_all,
                np.array([np.median(feats_channel, axis=0)]),
                feat_pd_names,
                feat_set[n],
                params,
            )
        # ---------------------------------------------------------------------
        # CONNECTIVITY FEATURES
        # (use over all channels but also divide into epochs)
        # ---------------------------------------------------------------------
        elif feat_group == "connectivity":
            dum_row = []
            x_epochs = []  # This is only to suppress warning
            for c in range(N_channels):
                tmp = overlap_epochs(
                    eeg_data[ch_labels[c]],
                    Fs,
                    params["EPOCH_LENGTH"],
                    params["EPOCH_OVERLAP"],
                )
                if c == 0:
                    dum_row, dumy = tmp.shape
                    x_epochs = np.empty([N_channels, dum_row, dumy])
                    x_epochs.fill(np.nan)
                x_epochs[c, :, :] = tmp

            N_epochs = dum_row

            # B) iterate over epochs:
            feats_epochs = np.empty([N_epochs, L_feature])
            feats_epochs.fill(np.nan)
            for e in range(N_epochs):
                x_ep = x_epochs[:, e, :]
                L_nans = len(np.where(np.isnan(x_ep))[0])
                if 100 * (L_nans / x_ep.size) < params["EPOCH_IGNORE_PRC_NANS"]:
                    tmp = connectivity_features.main_connectivity(
                        x_ep, Fs, feat_set[n], ch_labels
                    )
                    feats_epochs[e, :] = tmp
            # Median over all epochs
            feat_st[feat_set[n]] = np.nanmedian(feats_epochs, axis=0)
            feats_median_ch = add_feat_data_to_array(
                feats_median_ch, feats_epochs, feat_pd_names, feat_set[n], params
            )
            feats_median_all = add_feat_data_to_array(
                feats_median_all,
                np.array([np.nanmedian(feats_epochs, axis=0)]),
                feat_pd_names,
                feat_set[n],
                params,
            )

        elif feat_group == "IBI":
            # B) iterate over epochs:
            feats_channel = np.empty([N_channels, L_feature])
            feats_channel.fill(np.nan)
            for c in range(N_channels):
                feats_channel[c, :] = IBI_features.main_IBI(
                    eeg_data[ch_labels[c]].to_numpy(), Fs, feat_set[n]
                )

            # and the median over all channels
            feat_st[feat_set[n]] = np.nanmedian(feats_channel, axis=0)
            feats_median_all = add_feat_data_to_array(
                feats_median_all,
                np.array([np.nanmedian(feats_channel, axis=0)]),
                feat_pd_names,
                feat_set[n],
                params,
            )

    if save_p_l is not None:
        save_p_l = save_p_l + "_all_med"
        utils.save_pd_as_file(feats_median_all, save_p_l)
        save_p_l = save_p_l + "_ch_med"
        utils.save_pd_as_file(feats_median_ch, save_p_l)

    return feats_per_epochs, feat_pd_names, feat_st, feats_median_ch, feats_median_all


def get_feature_names(params, feat_set):
    """
    The purpose of this function to create an array that contains all the features names. This is the feature name,
    channel and frequency band. Array will be nans and one for every channel in the 'BI_MONT' param except for the
    connectivity features

    :param feat_set: The feature set being examined
    :param params: The parameters for the neural dataset
    :return: numpy string array containing feature names for all different frequency bands
    """
    all_feats = params["FEATURE_SET_ALL"]
    names = []
    for feat in all_feats:
        if feat not in feat_set:
            continue
        feat_group = feat.split("_")
        feat_group = feat_group[0]

        if feat_group == "IBI":
            bands = np.array([0.5, 30])
        elif (
            "total_freq_bands" in params[feat_group].keys()
            and feat == "spectral_edge_frequency"
        ):
            bands = np.array(params[feat_group]["total_freq_bands"])
        else:
            bands = np.array(params[feat_group]["freq_bands"])

        if bands.ndim == 1:
            N_bands = 1
        else:
            N_bands = bands.shape[0]

        for b in range(N_bands):
            if N_bands == 1:
                b1 = bands[0]
                b2 = str(int(bands[1]))
            else:
                b1 = bands[b][0]
                b2 = str(int(bands[b][1]))

            if b1 != 0.5:
                b1 = str(int(b1))
            else:
                b1 = str(b1).replace(".", "")

            names.append(feat + "__" + b1 + "_" + b2)

    return names


def get_max_num_epochs(params, N, Fs):
    """
    The purpose of this function is to find what the maximum number of epochs is so an empty nan np array can be made

        Inputs:
        params:           the parameter set
        N:                the number of samples in a single channel of an EEG
        Fs:               sampling frequency

    Outputs:
        a:                returns the number of epochs that will be generated

    NOTE: If different window sizes or overlaps are created then this function should be updated

    """
    # num_epochs = []
    L_window = params["EPOCH_LENGTH"]
    overlap = params["EPOCH_OVERLAP"]

    L_hop, L_epoch, dum = utils.gen_epoch_window(overlap, L_window, "hamm", Fs)
    a = np.ceil((N - (L_epoch - L_hop)) / L_hop)

    return int(a)


def add_feat_data_to_array(all_data, new_data, feat_pd_names, feat_name, params):
    """
    Update all_data array by appending the new data. The new data contains the data from all freq bands

    Syntax: all_data = add_feat_data_to_array(all_data, new_data, feat_pd_names, feat_name, params)

    Inputs:
        all_data:               Full data array for all features
        new_data:               New data for all feat bands of 'feat_name'
        feat_pd_names:          All the feature names (with bands and channels)
        feat_name:              Name of current feature
        params:                 Needed to get the feature group

    Outputs:
        packaged_data:    dictionary including EEG data (referential montage), sampling
                          frequency, and channel labels. Data are pandas DataFrame
    """

    feat_group = feat_name.split("_")
    feat_group = feat_group[0]
    if feat_group == "IBI":
        bands = np.array([0.5, 30])
    elif (
        "total_freq_bands" in params[feat_group].keys()
        and feat_name == "spectral_edge_frequency"
    ):
        bands = np.array(params[feat_group]["total_freq_bands"])
    else:
        bands = np.array(params[feat_group]["freq_bands"])

    if bands.ndim == 1:
        N_bands = 1
    else:
        N_bands = bands.shape[0]

    for b in range(N_bands):

        if N_bands == 1:
            b1 = bands[0]
            b2 = str(int(bands[1]))
        else:
            b1 = bands[b][0]
            b2 = str(int(bands[b][1]))

        if b1 != 0.5:
            b1 = str(int(b1))
        else:
            b1 = str(b1).replace(".", "")

        a_feat_name = feat_name + "__" + b1 + "_" + b2

        if a_feat_name not in feat_pd_names:
            raise ValueError(
                "Feature name %s does not exist - check code" % a_feat_name
            )

        # idx = [i for i, s in enumerate(feat_pd_names) if s == a_feat_name]

        # print(a_feat_name)
        if new_data.shape[0] == 1:
            all_data[a_feat_name] = [new_data[0][b]]
        else:
            all_data[a_feat_name] = new_data[:, b]

    return all_data
