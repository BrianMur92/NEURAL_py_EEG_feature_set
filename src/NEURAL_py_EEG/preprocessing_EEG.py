# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

import pandas as pd
from NEURAL_py_EEG import utils
import numpy as np
from NEURAL_py_EEG import NEURAL_parameters
from NEURAL_py_EEG.connectivity_features import channel_hemispheres
from scipy import signal
import mne


def load_edf_EEG_file(file_name, channels_to_look=None):
    """
    The purpose of this function is to read in an edf file and return a pandas dataframe containing
    the raw edf data and the sampling frequencies from all channels

    Syntax: data_out, ch_fs = load_edf_EEG_file(file_name, channels_to_look=None)

    Inputs:
        file_name:                    duration of EEG-like data in seconds (default 300 seconds)
        channels_to_look:             sampling frequency

    Outputs:
        data_out:                     pandas DataFrame containing the EEG data
        ch_fs:                        sampling frequency

    Example:
        import preprocessing_EEG
        import utils
        Fs = 256
        data_st = utils.gen_test_EEGdata(10 * 60, Fs, 1)  # 10 minutes of fake data - need to save as edf
        eeg_ref = data_st['eeg_data_ref']
        utils.save_pd_as_edf(eeg_ref, 'demo_data.edf', Fs=Fs)  # Save the fake data as edf so we can opening a real edf

        data_out, ch_fs = preprocessing_EEG.load_edf_EEG_file('demo_data.edf', channels_to_look=None)


    """
    if channels_to_look is None:
        channels_to_look = ["F3", "F4", "C3", "C4", "O1", "O2", "T3", "T4", "Cz"]
    f = mne.io.read_raw_edf(file_name, preload=True)
    data = f._data
    signal_labels = f.ch_names

    ch_names = []
    ch_fs = []
    data_out = pd.DataFrame()
    loop_index = 0

    for ch in channels_to_look:
        ch_exist = [x for x in signal_labels if ch.upper() in x.upper()]
        if len(ch_exist) > 0:
            loc = signal_labels.index(ch_exist[0])
            # data_out.insert(loop_index, ch, f.readSignal(loc))
            data_out.insert(loop_index, ch, data[loc, :] * 10 ** 6)
            ch_names.append(ch)
            tmp = f.info["sfreq"]  # [ch, f.getSampleFrequency(loc)]
            ch_fs.append(tmp)
            loop_index += 1

    # f.close
    return data_out, ch_fs


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
        print('"NEURAL_parameters.py" file.\n')

    nw = np.array(list(range(L_epoch)))
    ix = np.array(list(range(N)))

    x_epochs = np.empty([N_epochs, L_epoch])
    x_epochs.fill(np.nan)
    x_epochs_inds = np.zeros([N_epochs, L_epoch])

    for k in range(N_epochs):
        nf = nw + k * L_hop
        # zero-pad if outside x
        if np.max(nf) > ix[-1]:
            nf = np.delete(nf, np.where(nf > ix[-1])[0])
        i_nf = np.array(list(range(len(nf))))

        x_epochs[k, i_nf] = x[nf] * win_epoch[i_nf]
        x_epochs_inds[k, i_nf] = nf

    return x_epochs, x_epochs_inds.astype(int)


def art_per_channel(x, Fs, DBverbose):
    """
    Remove artefacts on a per-channel basis

    :param DBverbose:
    :param x: data from a single channel - numpy array
    :param Fs: sampling frequency
    :return: x, data with artefacts set as nan
    """
    params = NEURAL_parameters.NEURAL_parameters()

    # DBverbose = 1
    N = len(x)  # verify / check this

    amount_removed = np.array([0, 0, 0, 0], dtype=float)
    # ---------------------------------------------------------------------
    # 1. electrode - checks(continuous row of zeros)
    # ---------------------------------------------------------------------
    x_channel = x.copy()
    x_channel[np.where(x_channel != 0)] = 1
    irem = np.zeros([N])
    lens, istart, iend = utils.len_cont_zeros(x_channel, 0)
    ielec = np.array(np.where(lens >= (params["ART_ELEC_CHECK"] * Fs)))[0]

    if ielec.size != 0:
        for m in ielec:
            irun = np.array(
                range((istart[m] - 1), (iend[m] + 2), 1)
            )  # TODO check this against matlab
            irun[np.where(irun < 0)] = 0
            irun[np.where(irun > N - 1)] = N - 1
            irem[irun] = 1
            x[irun] = np.nan
    if any(irem == 1):
        print("continuous row of zeros: %.2f%%\n" % (100 * (np.sum(irem) / N)))
    amount_removed[0] = 100 * (np.sum(irem) / N)

    x_nofilt = x.copy()
    x_filt, inans = utils.filter_butterworth_withnans(x, Fs, 40, 0.1, [5, 2])

    # ---------------------------------------------------------------------
    # 2. high - amplitude artefacts
    # ---------------------------------------------------------------------
    art_coll = params["ART_TIME_COLLAR"] * Fs
    irem = np.zeros(N)

    x_hilbert = np.abs(signal.hilbert(x_filt))

    thres_upper = params["ART_HIGH_VOLT"]
    ihigh = np.array(np.where(x_hilbert > thres_upper))[0]

    if ihigh.size != 0:
        for p in range(len(ihigh)):
            irun = np.array(
                range((ihigh[p] - int(art_coll)), (ihigh[p] + int(art_coll) + 1), 1)
            )
            irun[np.where(irun < 0)] = 0
            irun[np.where(irun > N - 1)] = N - 1
            irem[irun] = 1
    x[irem == 1] = np.nan
    if any(irem == 1) and DBverbose:
        print(
            "length of high-amplitude artefacts: %.2f%%\n" % (100 * (np.sum(irem) / N))
        )
    amount_removed[1] = 100 * (np.sum(irem) / N)

    # ---------------------------------------------------------------------
    # 3. continuous constant values(i.e.artefacts)
    # ---------------------------------------------------------------------
    art_coll = params["ART_DIFF_TIME_COLLAR"] * Fs
    irem = np.zeros(N)

    x_diff_all = np.concatenate((np.diff(x), [0]))
    x_diff = x_diff_all.copy()
    x_diff[x_diff != 0] = 1
    lens, istart, iend = utils.len_cont_zeros(x_diff, 0)

    # if exactly constant for longer than.then remove:
    ielec = np.array(np.where(lens > (params["ART_DIFF_MIN_TIME"] * Fs)))[0]

    if ielec.size != 0:
        for m in ielec:
            irun = np.array(
                range((istart[m] - int(art_coll)), (iend[m] + int(art_coll) + 1), 1)
            )
            irun[np.where(irun < 0)] = 0
            irun[np.where(irun > N - 1)] = N - 1
            irem[irun] = 1
            x[irun] = np.nan
    if any(irem == 1):
        print(
            "continuous row of constant values: %.2f%%\n" % (100 * (np.sum(irem) / N))
        )
    amount_removed[2] = 100 * (np.sum(irem) / N)

    # ---------------------------------------------------------------------
    # 4. sudden jumps in amplitudes or constant values(i.e.artefacts)
    # ---------------------------------------------------------------------
    art_coll = params["ART_DIFF_TIME_COLLAR"] * Fs
    irem = np.zeros(N)
    x_diff = x_diff_all.copy()

    ihigh = np.array(np.where(np.abs(x_diff) > params["ART_DIFF_VOLT"]))[0]
    if ihigh.size != 0:
        for p in range(len(ihigh)):
            irun = np.array(
                range((ihigh[p] - int(art_coll)), (ihigh[p] + int(art_coll) + 1), 1)
            )
            irun[np.where(irun < 0)] = 0
            irun[np.where(irun > N - 1)] = N - 1
            irem[irun] = 1

    x[irem == 1] = np.nan
    if any(irem == 1) and DBverbose:
        print("length of sudden-jump artefacts: %.2f%%\n" % (100 * (np.sum(irem) / N)))
    amount_removed[3] = 100 * (np.sum(irem) / N)

    # before filtering, but should be eliminated anyway
    x[inans] = np.nan
    inans = np.where(np.isnan(x))
    x_nofilt[inans] = np.nan
    # x_nofilt[irem_muc == 1] = np.inf
    x = x_nofilt

    return x, amount_removed


def remove_artefacts(data, ch_labels, Fs, data_ref, ch_labels_ref):
    """
    remove_artefacts: simple procedure to remove artefacts

    Syntax: data = remove_artefacts(data, ch_labels, Fs, data_ref, ch_refs)

    Inputs:
        data      - EEG data, in bipolar montage; size: N_channels x N (pandas dataframe)
        ch_labels - List of bipolar channel labels,
                    e.g. ['C3-O1','C4-O2', 'F3-C3', 'F4-C4']
        Fs        - sampling frequency (in Hz)
        data_ref  - EEG data, in referential  montage; size: (N_channels+1) x N
        ch_refs   - List of referential channel labels,
                    e.g. ['C3','C4','F3','F4']

    Outputs:
        data - EEG data after processing, in bipolar montage, size: N_channels x N  (pandas dataframe)

    Example:
        import utils
        import numpy as np
        import preprocessing_EEG
        Fs = 256
        data_st = utils.gen_test_EEGdata(2*60, Fs, 1)
        N = len(data_st['eeg_data_ref'])

        # simulate artefacts:
        # 1. F3 not properly attached:
        if3 = [i for i, x in enumerate(data_st['ch_labels_ref']) if x == 'F3'][0]
        data_st['eeg_data_ref'][data_st['ch_labels_ref'][if3]] = np.random.randn(N) * 10

        # 2. electrode coupling between C4 and Cz
        ic4 = [i for i, x in enumerate(data_st['ch_labels_ref']) if x == 'C4'][0]
        icz = [i for i, x in enumerate(data_st['ch_labels_ref']) if x == 'Cz'][0]

        data_st['eeg_data_ref'][data_st['ch_labels_ref'][icz]] = data_st['eeg_data_ref'][data_st['ch_labels_ref'][ic4]]
                                                                + np.random.randn(N) * 5

        # re-generate bipolar montage:
        data_st['eeg_data'], data_st['ch_labels'] = utils.set_bi_montage(data_st['eeg_data_ref'], Fs,
                                                    data_st['ch_labels_bi'])

        # remove channels:
        eeg_art = preprocessing_EEG.remove_artefacts(data_st['eeg_data'].copy(), data_st['ch_labels'], data_st['Fs'],
                                         data_st['eeg_data_ref'], data_st['ch_labels_ref'])
    """
    params = NEURAL_parameters.NEURAL_parameters()
    DBverbose = 1

    N, N_channels = data.shape

    # ---------------------------------------------------------------------
    # 0. check in referential mode first; is there problem with one
    #    channel(e.g.Cz)
    # ---------------------------------------------------------------------
    irem_channel = np.array([], dtype=int)
    channel_names_ref = list(data_ref.columns)
    if not data_ref.empty:
        x_filt = np.zeros([data_ref.shape[1], data_ref.shape[0]])
        for ch in range(data_ref.shape[1]):
            x_filt[ch, :], dum = utils.filter_butterworth_withnans(
                data_ref[channel_names_ref[ch]].to_numpy(), Fs, 20, 0.5, 5
            )

        r = np.corrcoef(x_filt)
        np.fill_diagonal(r, np.nan)
        r_channel = np.nanmean(r, axis=0)
        del x_filt

        ilow = np.array(
            np.where(np.abs(r_channel) < params["ART_REF_LOW_CORR"])
        ).astype(int)[0]

        if ilow.size != 0:
            nn = 1
            irem_channel = np.array([], dtype=int)
            for idex in ilow:
                ch_find = ch_labels_ref[idex]
                itmp = [i for i, x in enumerate(ch_labels) if ch_find in x]
                irem_channel = np.append(irem_channel, itmp)
                nn += 1
            print(
                ":: remove channel (low ref. correlation): %s\n"
                % np.array(ch_labels)[irem_channel.astype(int)]
            )

        # if DBverbose:
        # print(r_channel)
        # print(ch_labels_ref)
        for ch in range(len(irem_channel)):
            data[ch_labels[irem_channel[ch].astype(int)]] = np.nan

    ichannels = list(range(N_channels))
    ichannels = np.delete(ichannels, irem_channel)
    N_channels = len(ichannels)

    # ---------------------------------------------------------------------
    # 1. look for electrode coupling:
    # ---------------------------------------------------------------------
    ch_labels = np.array(ch_labels)
    if N_channels > 4:
        ileft, iright = channel_hemispheres(ch_labels[ichannels])

        if len(ileft) > 1 and len(iright) > 1:
            x_means = np.zeros(N_channels)

            x_filt = np.zeros([data_ref.shape[1], data_ref.shape[0]])
            for ch in range(N_channels):
                x_filt[ch, :], dum = utils.filter_butterworth_withnans(
                    data[ch_labels[ichannels[ch]]].to_numpy(), Fs, 20, 0.5, 5
                )

            A = []  # to suppress warning
            for n in range(N_channels):
                x_means[n] = np.nanmean(np.abs(x_filt[n, :]) ** 2)

                if n == 0:
                    A = [[x_means[n], ch_labels[ichannels[n]]]]
                else:
                    A.append([x_means[n], ch_labels[ichannels[n]]])
            del x_filt

            # 1/4 of the median channel energy
            cut_off_left = np.median(x_means[ileft]) / 4
            cut_off_right = np.median(x_means[iright]) / 4

            ishort_left = np.array(np.where(x_means[ileft] < cut_off_left))
            ishort_right = np.array(np.where(x_means[iright] < cut_off_right))

            ishort = np.hstack([ileft[ishort_left], iright[ishort_right]])
            ishort = np.array(ichannels[ishort])

            if ishort.size != 0:
                print(
                    ":: remove channel (electrode coupling): %s\n" % ch_labels[ishort]
                )
                data[ch_labels[ishort][0]] = np.nan
                irem_channel = np.append(irem_channel, ishort[0])

    # if DBverbose and len(A) > 0:
    #    print(A)

    ichannels = list(range(len(data.columns)))
    ichannels = np.delete(ichannels, irem_channel)

    # all other artefacts are on a channel-by-channel basis

    irem = np.array([])

    amount_removed = np.zeros([1, 4])
    ct = 0

    for n in ichannels:
        data[ch_labels[n]], tmpp = art_per_channel(
            data[ch_labels[n]].to_numpy(), Fs, DBverbose
        )
        if ct == 0:
            amount_removed[ct, :] = tmpp
        else:
            amount_removed = np.vstack((amount_removed, tmpp))
        ct += 1
        irem = np.append(irem, np.where(np.isnan(data[ch_labels[n]])))
        if any(np.isinf(data[ch_labels[n]])):
            data[ch_labels[n]][np.where(data[ch_labels[n]] == np.inf)[0]] = np.nan

    # remove artefacts across all channels

    data.loc[np.unique(irem), ch_labels[ichannels]] = np.nan

    out = list(np.max(amount_removed, axis=0))
    if irem_channel.shape[0] != 0:
        out.append(np.unique(ch_labels[irem_channel.astype(int)]))
    else:
        out.append("None removed")

    return data, out


def LPF_zero_phase(pd_data, Fs):
    """

    (pd) -> pd

    Pass in pandas dataframe and low pass filter all the columns

    > LPF_zero_phase(data)
    data

    """
    params = NEURAL_parameters.NEURAL_parameters()
    lp = params["LP_fc"]

    cols = list(pd_data.columns)
    for i in range(len(cols)):
        data = pd_data[cols[i]].to_numpy()
        if all(np.isnan(data)):
            continue
        d_mean = np.nanmean(data)
        data = data - d_mean
        s1_filt = utils.filter_zerophase(data, Fs=Fs, LP_fc=lp, L_filt=int(4001))
        data = s1_filt + d_mean
        pd_data[cols[i]] = data

    return pd_data


def signal_downsample(pd_data, Fs, Fs_new):
    """

    (pd_data,int,int) -> pd_data

    The purpose of this function is to downsample pd_data from fs to fs_new

    > signal_downsample(pd_data,fs,fs_new)
    pd_data

    """

    cols = list(pd_data.columns)

    def isint(x):
        return x == round(x, 1)

    pd_data_downsample = pd.DataFrame()
    loop_index = 0

    if isint(Fs / Fs_new):
        idec = list(range(0, len(pd_data), round(Fs / Fs_new)))

        for i in range(len(cols)):
            data = np.array(pd_data[cols[i]])
            eeg_data = data[idec]
            pd_data_downsample.insert(loop_index, cols[i], eeg_data)
            loop_index += 1
    else:
        size_resampled = signal.resample_poly(
            np.ones(len(pd_data)), round(len(pd_data) / (Fs / Fs_new)), len(pd_data)
        ).size

        for i in range(
            len(cols)
        ):  # To find the size of resampled array so rows of all nans are sorted
            data = np.array(pd_data[cols[i]])
            if sum(np.isnan(data)) == len(data):
                down_eeg_data = np.ones(size_resampled)
                down_eeg_data.fill(np.nan)
                pd_data_downsample.insert(loop_index, cols[i], down_eeg_data)
                loop_index += 1
                continue

            down_eeg_data = signal.resample_poly(
                data, round(len(data) / (Fs / Fs_new)), len(data)
            )
            pd_data_downsample.insert(loop_index, cols[i], down_eeg_data)
            loop_index += 1

    return pd_data_downsample, Fs_new


def main_preprocessing(file, save_converted=None, save=0, Fs_new=64):
    """
    This function is used to  a) read in EEG from .edf files
                              b) remove artefacts
                              c) band-pass filter
                              d) downsample
                              e) save as .csv file

    """
    data, ch_fs = load_edf_EEG_file(file)
    Fs = ch_fs[0]  # Assuming all EEG channels

    montage_data, dum = utils.set_bi_montage(data, Fs)

    clean_data, amount_removed = remove_artefacts(
        montage_data.copy(),
        np.array(list(montage_data.columns)),
        Fs,
        data,
        np.array(list(data.columns)),
    )

    filtered_data = LPF_zero_phase(clean_data.copy(), Fs)

    down_sampled_data, Fs = signal_downsample(filtered_data.copy(), Fs, Fs_new)

    packaged_data = {
        "eeg_data": down_sampled_data,
        "Fs": Fs,
        "ch_labels": list(down_sampled_data.columns),
    }

    if save and not not save_converted:
        utils.save_pd_as_file(down_sampled_data, save_converted + "down", Fs)

    return packaged_data, amount_removed
