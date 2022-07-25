# Author:       Brian Murphy
# Date started: 30/01/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import random
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import pyedflib
from NEURAL_py_EEG import mfiltfilt


# noinspection SpellCheckingInspection
def get_window(win_length, win_type, Npad=0):
    """
    USE: win = get_window( win_length, win_type )

    INPUT:
          win_length = length of window - samples
          win_type   = type of window: { 'rect' | 'hamm' | 'hann' | 'tuke' | 'bart' }
          Npad       = (optional) zero-pad window to length Npad.

    OUTPUT:
          win = window

    EXAMPLE:
        import utils
        import matplotlib.pyplot as plt

        N = 64
        win = utils.get_window(N, 'hamm', 100)
        plt.plot(win)
        plt.show()

    (int,str) -> list

    Pass in a window length and window type string and return a window

    """

    win_type = win_type.lower()

    if win_type == "hamm" or win_type == "hamming":
        win = np.hamming(win_length)
    elif win_type == "hann" or win_type == "hanning":
        win = np.hanning(win_length)
    elif win_type == "rect" or win_type == "rectangular":
        win = np.ones(win_length)
    elif win_type == "black" or win_type == "blackman":
        win = np.blackman(win_length)
    elif win_type == "blackmanharris":
        win = signal.blackmanharris(win_length)
    elif win_type == "kaiser":
        win = np.kaiser(win_length, 5)
    elif win_type == "tuke" or win_type == "tukey":
        win = signal.tukey(win_length)
    elif win_type == "bart" or win_type == "bartlett":
        win = signal.bartlett(win_length)
    else:
        raise ValueError("Name of window function incorrect. Name: %s" % win_type)

    win = np.roll(win, math.ceil(len(win) / 2))

    if Npad != 0:
        win = pad_win(win, Npad)

    return win


def pad_win(w, Npad):
    """
    --------------------------------------------------------------------------------

     Pad window to Npad.

     Presume that positive window indices are first.

     When N is even use method described in [1]

       References:
         [1] S. Lawrence Marple, Jr., Computing the discrete-time analytic
         signal via FFT, IEEE Transactions on Signal Processing, Vol. 47,
         No. 9, September 1999, pp.2600--2603.

    --------------------------------------------------------------------------------
    """
    w_pad = np.zeros(Npad)
    N = len(w)
    Nh = np.floor(N / 2).astype(int)
    if Npad < N:
        raise ValueError("Npad is less than N")

    # Trivial case
    if N == Npad:
        return w

    # For N odd:
    if N % 2 == 1:
        n = list(range(Nh + 1))
        w_pad[n] = w[n]
        n = np.array(range(Nh))
        w_pad[Npad - n - 1] = w[N - n - 1]

        # For N even:
        # split the Nyquist frequency in two and distribute over positive
        # and negative indices.
    else:
        n = list(range(Nh))
        w_pad[n] = w[n]
        w_pad[Nh] = w[Nh] / 2

        n = np.array(range(Nh - 1))
        w_pad[Npad - n - 1] = w[N - n - 1]
        w_pad[Npad - Nh] = w[Nh] / 2
    return w_pad


# noinspection SpellCheckingInspection
def gen_epoch_window(L_overlap, L_epoch, win_type, Fs, GEN_PSD=None):
    """
    Syntax: L_hop, L_epoch, win_epoch = gen_epoch_window(L_overlap, L_epoch, win_type, Fs)

    Inputs:
        L_overlap - precentage overlap
        L_epoch   - epoch size (in seconds)
        win_type  - window type, e.g. 'hamm' for Hamming window
        Fs        - sampling frequency (in Hz)

    Outputs:
        L_hop     - hop size (in samples)
        L_epoch   - epoch size (in samples)
        win_epoch - window, of length L_epoch

    Example:
        import utils
        import matplotlib.pyplot as plt
        import numpy as np

        overlap = 50
        win_length = 2
        Fs = 64

        L_hop, L_epoch, win_epoch = utils.gen_epoch_window(overlap, win_length, 'hamm', Fs)

        print('hop length=%d; epoch length=%d' % (L_hop, L_epoch))
        plt.figure(1)
        ttime = np.array(range(len(win_epoch))) / Fs
        plt.plot(ttime, win_epoch)
        plt.show()

    (int, int, str, int) -> int, int, list

    L_hop, L_epoch, win_epoch = gen_epoch_window(win_overlap, win_length, win_type, fs)
    window

    """
    if GEN_PSD is None:
        GEN_PSD = 0

    L_hop = (100 - L_overlap) / 100
    L_epoch = math.floor(L_epoch * Fs)

    if GEN_PSD:
        # ---------------------------------------------------------------------
        # if PSD
        # ---------------------------------------------------------------------
        L_hop = np.ceil((L_epoch - 1) * L_hop)

        win_epoch = get_window(L_epoch, win_type)
        win_epoch = np.roll(win_epoch, math.floor(len(win_epoch) / 2))
    else:
        """ ---------------------------------------------------------------------
        otherwise, if using an overlap-and-add method, then for window w[n]
        âˆ‘â‚˜ w[n - mR] = 1 over all n (R = L_hop )
        
        Smith, J.O. "Overlap-Add (OLA) STFT Processing", in 
        Spectral Audio Signal Processing,
        http://ccrma.stanford.edu/~jos/sasp/Hamming_Window.html, online book, 
        2011 edition, accessed Nov. 2016.
        
        
        there are some restrictions on this:
        e.g. for Hamming window, L_hop = (L_epoch-1)/2, (L_epoch-1)/4, ... 
        ---------------------------------------------------------------------
        """
        if win_type == "hamm":
            L_hop = (L_epoch - 1) * L_hop
        elif win_type == "hann":
            L_hop = (L_epoch + 1) * L_hop
        else:
            L_hop = L_epoch * L_hop

        L_hop = math.ceil(L_hop)
        win_epoch = get_window(L_epoch, win_type)
        win_epoch = np.roll(win_epoch, math.floor(len(win_epoch) / 2))

        if win_type == "hamm" and L_epoch // 2 == 1:
            win_epoch[0] = win_epoch[0] / 2
            win_epoch[-1] = win_epoch[-1] / 2

    return L_hop, L_epoch, win_epoch


def gen_test_EEGdata(dur, Fs, include_bipolar=0, discont_activity=0):
    """
    Syntax: data_st = gen_test_EEGdata(dur, Fs, include_bipolar, discont_activity)

    Inputs:
        dur:              duration of EEG-like data in seconds (default 300 seconds)
        Fs:               sampling frequency
        include_bipolar:  include the bipolar montage aswell as the referential
        discont_activity: discontinuous-like activity of preterm EEG (bursts and inter-bursts)

    Outputs:
        packaged_data:    dictionary including EEG data (referential montage), sampling
                          frequency, and channel labels. Data are pandas DataFrame

    Example:
        import utils
        import matplotlib.pyplot as plt
        import numpy as np
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, discont_activity=1)
        x = data_st['eeg_data'][data_st['ch_labels'][0]].values

        plt.figure(1)
        ttime = np.array(range(len(x)))/Fs
        plt.plot(ttime, x)
        plt.xlabel('time (seconds)')
        plt.ylabel('$\mu$V')
        plt.show()

    """
    N = int(np.floor(dur * Fs))

    """---------------------------------------------------------------------
    generate white Gaussian noise and filter with moving average filter
    (referential montage has common signal)
    ---------------------------------------------------------------------"""

    L_ma = Fs / 4
    L_ma_noise = np.ceil(Fs / 16)
    N_channels = 9

    eeg_common = 5 * signal.lfilter(
        [1 / L_ma for a in range(int(L_ma))], 1, np.random.randn(int(N))
    )

    eeg_data_ref = (
        signal.lfilter(
            [1 / L_ma for a in range(int(L_ma))], 1, np.random.randn(N_channels, N)
        )
        + (
            signal.lfilter(
                [1 / L_ma_noise for a in range(int(L_ma_noise))],
                1,
                np.random.randn(N_channels, N),
            )
        )
        / 100
        + eeg_common
    )

    if discont_activity:
        ibursts = np.array(
            [random.randint(0, N - 1) for a in range(int(dur / 100) * Fs)]
        )
        amps = np.abs(1 + np.random.uniform(0, 1, [1, int(dur / 100) * Fs]))

        bursts = np.zeros([1, N])
        bursts[0][ibursts] = 1 * amps

        L_win = Fs * 2
        bursts_smooth = 100 * signal.lfilter(np.hamming(L_win) / L_win, 1, bursts)

        eeg_data_ref = 10 * (eeg_data_ref + (eeg_data_ref * (20 * bursts_smooth)))

    else:
        eeg_data_ref = eeg_data_ref * 50

    ch_labels_bi = [
        ["F4", "C4"],
        ["F3", "C3"],
        ["C4", "T4"],
        ["C3", "T3"],
        ["C4", "Cz"],
        ["Cz", "C3"],
        ["C4", "O2"],
        ["C3", "O1"],
    ]

    ch_labels_ref = ["C3", "C4", "Cz", "F3", "F4", "O1", "O2", "T3", "T4"]

    eeg_data_ref = pd.DataFrame(data=np.transpose(eeg_data_ref), columns=ch_labels_ref)
    packaged_data = {
        "eeg_data_ref": eeg_data_ref,
        "Fs": Fs,
        "ch_labels_ref": list(eeg_data_ref.columns),
    }

    if include_bipolar:
        packaged_data["ch_labels_bi"] = ch_labels_bi
        eeg_data, dum = set_bi_montage(eeg_data_ref, Fs, ch_labels_bi)
        packaged_data["eeg_data"] = eeg_data
        packaged_data["ch_labels"] = list(eeg_data.columns)

    return packaged_data


def set_bi_montage(sigs, Fs, bi_mont=None):
    """
    set_bbiploar_montage: Convert monopolar (referential) to bi-polar montgage

    Syntax: bi_sigs, bi_labels = set_bi_montage(sigs,channel_names)

    Inputs:
        sigs          - EEG data in referential montage - pd dataframe
        channel_names - cell of referential channel names
                         e.g. ['C3','C4','F3','F4']
        bi_mont - Bipolar montage

    Outputs:
        bi_sigs   - EEG data in referential montage
        bi_labels - cell of bipolar channel names
                    e.g. ['C3-O1','C4-O2', 'F3-C3', 'F4-C4']

    Example:
        import utils
        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)

        x_ref = data_st['eeg_data_ref']
        ch_labels_ref = data_st['ch_labels_ref']
        BI_MONT=[['F4','C4'], ['F3','C3'], ['C4','T4'], ['C3','T3'], ['C4','Cz'], ['Cz','C3'], ['C4','O2'], ['C3','O1']]

        x_bi, ch_labels_bi = utils.set_bi_montage(x_ref, ch_labels_ref, BI_MONT)
    """
    if bi_mont is None:
        bi_mont = [
            ["C3", "O1"],
            ["C4", "O2"],
            ["Cz", "C3"],
            ["C4", "Cz"],
            ["C3", "T3"],
            ["C4", "T4"],
            ["F3", "C3"],
            ["F4", "C4"],
        ]

    channel_names = list(sigs.columns)
    montage_data = pd.DataFrame()
    montage_channels = []
    loop_index = 0

    for pair in range(len(bi_mont)):
        if bi_mont[pair][0] in channel_names and bi_mont[pair][1] in channel_names:
            tmp = bi_mont[pair][0] + "_" + bi_mont[pair][1]
            montage_channels.append(tmp)
            tmp_data = sigs[bi_mont[pair][0]] - sigs[bi_mont[pair][1]]
            montage_data.insert(
                loop_index, bi_mont[pair][0] + "_" + bi_mont[pair][1], tmp_data
            )
            loop_index += 1

    return montage_data, montage_channels


def filter_zerophase(
    x, Fs=1, LP_fc=-1, HP_fc=-1, L_filt=0, win_type="hamming", DBplot=0
):
    """
    filter_zerophase: Simple implementation of zero-phase FIR filter using 'filtfilt'

    Syntax: x_filt = filter_zerophase(x, Fs, LP_fc, HP_fc, L_filt)

    Inputs:
        x      - input signal - 1 channel
        Fs     - sample frequency (Hz)
        LP_fc  - lowpass cut off (Hz)
        HP_fc  - highpass cut off (Hz)
        L_filt - length of filter (in samples) - filter order

    Outputs:
        y - filtered signal

    Example:
        import utils
        import matplotlib.pyplot as plt
        import numpy as np

        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        x = data_st['eeg_data'][data_st['ch_labels'][0]].values
        LP_fc = 20
        HP_fc = 1

        y = utils.filter_zerophase(x, Fs, LP_fc, HP_fc, 501)

        plt.figure(1)
        ttime = np.array(range(len(x))) / Fs
        plt.plot(ttime, x, 'r')
        plt.plot(ttime, y, 'b')
        plt.show()

    """
    # replace NaNs with zeros
    inans = np.array(np.where(np.isnan(x)))[0]
    if inans.size != 0:
        x[inans] = 0

    N = len(x)

    if L_filt == 0 or L_filt > (N / 4):
        L_filt = make_odd(np.round(N / 4 - 2))

    # ---------------------------------------------------------------------
    # either bandpass, low - pass, or high - pass
    # ---------------------------------------------------------------------

    if LP_fc != -1 and HP_fc != -1:
        if LP_fc > HP_fc:
            # band-pass
            filt_passband = [HP_fc / (Fs / 2), LP_fc / (Fs / 2)]
            b = signal.firwin(L_filt, filt_passband, window=win_type)
        else:
            # or band-stop
            filt_passband = [LP_fc / (Fs / 2), HP_fc / (Fs / 2)]
            b = signal.firwin(L_filt, filt_passband, window=win_type)

    elif LP_fc != -1:
        filt_passband = LP_fc / (Fs / 2)
        b = signal.firwin(L_filt, filt_passband, window=win_type)

    elif HP_fc != -1:
        L_filt = L_filt + 1
        filt_passband = HP_fc / (Fs / 2)
        b = signal.firwin(L_filt - 1, filt_passband, window=win_type)

    else:
        raise ValueError("Need to specify cutoff frequency")

    # Do the filtering

    x_filt = mfiltfilt.mat_filtfilt(b, np.array([1]), x)
    x_filt = x_filt[0:N]

    # Add the nans back in
    if inans.size != 0:
        x_filt[inans] = np.nan

    return x_filt


def make_odd(x):
    x = np.floor(x)
    if x % 2 == 0:
        x = x - 1
    return int(x)


def gen_window(win_type, L):
    # ---------------------------------------------------------------------
    # window
    # ---------------------------------------------------------------------
    win_type = win_type.lower()

    if win_type == "hamm" or win_type == "hamming":
        w = np.hamming(L)
    elif win_type == "hann" or win_type == "hanning":
        w = np.hanning(L)
    elif win_type == "rect" or win_type == "rectangular":
        w = np.ones(L)
    elif win_type == "black" or win_type == "blackman":
        w = np.blackman(L)
    elif win_type == "blackmanharris":
        w = signal.blackmanharris(L)
    elif win_type == "kaiser":
        w = np.kaiser(L, 5)
    else:
        raise ValueError("Name of window function incorrect. Name: %s" % win_type)

    return w


# noinspection SpellCheckingInspection
def filter_butterworth_withnans(
    x, Fs, F3db_lowpass, F3db_highpass, order, FILTER_REPLACE_ARTEFACTS="linear_interp"
):
    """
    Syntax: y, inans = filter_butterworth_withnans(x, Fs, F3db_lowpass, F3db_highpass, order, FILTER_REPLACE_ARTEFACTS)

    Inputs:
        x             - input signal
        Fs            - sample frequency (Hz)
        F3db_lowpass  - 3 dB lowpass cut off (Hz)
        F3db_highpass - 3 dB highpass cut off (Hz)
        order         - filter order
        FILTER_REPLACE_ARTEFACTS - what to do with NaNs?
            either replace with 0s ('zeros'), linear interpolation
            ('linear_interp', default), or cubic spline interpolation ('cubic_interp')

    Outputs:
        y - filtered signal
        inans - locations of nans

    Example:
        import utils
        import matplotlib.pyplot as plt
        import numpy as np

        Fs = 64
        data_st = utils.gen_test_EEGdata(32, Fs, 1)
        x = data_st['eeg_data'][data_st['ch_labels'][0]].values
        F3db_lowpass = 30
        F3db_highpass = 0.5

        y, inans = utils.filter_butterworth_withnans(x, Fs, F3db_lowpass, F3db_highpass, 5)

        plt.figure(1)
        ttime = np.array(range(len(x))) / Fs
        plt.plot(ttime, x)
        plt.plot(ttime, y)
        plt.show()

    """
    inans = np.array([])

    if not F3db_highpass:
        b, a = signal.butter(order, F3db_lowpass / (Fs / 2), "lowpass")
    elif not F3db_lowpass:
        b, a = signal.butter(order, F3db_highpass / (Fs / 2), "highpass")
    else:
        if isinstance(order, list):
            order_low = order[0]
            order_high = order[1]
        else:
            order_low = order
            order_high = order

        y, isnans_low = filter_butterworth_withnans(
            x, Fs, F3db_lowpass, [], order_low, FILTER_REPLACE_ARTEFACTS
        )
        y, isnans_high = filter_butterworth_withnans(
            y, Fs, [], F3db_highpass, order_high, FILTER_REPLACE_ARTEFACTS
        )
        inans = np.unique(np.concatenate([isnans_low, isnans_high]))
        return y, inans

    # remove NaNs and replace with ?
    inans = np.array(np.argwhere(np.isnan(x)))
    if inans.size != 0:
        if FILTER_REPLACE_ARTEFACTS == "zeros":
            x[inans] = 0
        elif FILTER_REPLACE_ARTEFACTS == "linear_interp":
            x = replace_start_ends_NaNs_with_zeros(x)
            x, dum = naninterp(x, "linear")
        elif (
            FILTER_REPLACE_ARTEFACTS == "cubic_interp"
            or FILTER_REPLACE_ARTEFACTS == "nans"
        ):
            x = replace_start_ends_NaNs_with_zeros(x.copy())
            x, dum = naninterp(x.copy(), "pchip")

    y = mfiltfilt.mat_filtfilt(np.array(b), np.array(a), x)

    # special case if nans
    if FILTER_REPLACE_ARTEFACTS.lower() == "nans":
        if inans.size != 0:
            y[inans] = np.nan

    return y, inans


def replace_start_ends_NaNs_with_zeros(x):
    """
    ---------------------------------------------------------------------
     replace leading or trailing NaNs with zeros (needed for naninterp)
    ---------------------------------------------------------------------
    """
    N = len(x)
    istart = np.argwhere(~np.isnan(x))[0][0]
    iend = np.argwhere(~np.isnan(x))[-1][0]

    if istart.size > 0 and istart > 0:
        x[0 : istart + 1] = 0
    if iend.size > 0 and iend < N - 1:
        x[iend + 1 : N + 1] = 0

    return x


def naninterp(X, method="linear"):
    """
    ---------------------------------------------------------------------
     fill 'gaps' in data (marked by NaN) by interpolating
    ---------------------------------------------------------------------

    """
    inan = np.argwhere(np.isnan(X))
    if inan.size == 0:
        return X, inan
    elif len(inan) == 1:
        if inan > 0:
            X[inan] = X[inan - 1]
        else:
            X[inan] = X[inan + 1]
    else:
        try:
            if method != "pchip":
                set_interp = interp1d(
                    np.transpose(np.argwhere(~np.isnan(X)))[0],
                    np.transpose(X[np.argwhere(~np.isnan(X))])[0],
                    kind=method,
                )
                X[inan] = set_interp(inan)
            else:
                set_interp = PchipInterpolator(
                    np.transpose(np.argwhere(~np.isnan(X)))[0],
                    np.transpose(X[np.argwhere(~np.isnan(X))])[0],
                    extrapolate=False,
                )
                X[inan] = set_interp(inan)

        except:
            raise ValueError("linear interpolation with NaNs")
    return X, inan


def len_cont_zeros(x, const=0):
    """
    len_cont_zeros: find length of continuous segments of zeros from binary mask x. Can contain NaNs.

    Syntax: [lens,istart,iend]=len_cont_zeros(x,conts)

    Inputs:
        x     - binary [0,1] vector - 1 row
        const - which to look for, either 0 (default) or 1

    Outputs:
        lens   - array of lengths of segments
        istart - indices: start of segments
        iend   - indices: end of segments

    Example:
        import utils
        import numpy as np

        u=np.zeros([256])
        u[49:130] = 1
        u[204:240] = 1
        u = u.astype(int)
        lens, istart, iend = utils.len_cont_zeros(u,1)

        print('%d segments of 1''s of length: \n' %len(lens) )
    """
    nan_loc = np.where(np.isnan(x))[0]
    DBplot = 0
    if np.array_equal(x, x.astype(bool)) or const not in [0, 1]:
        warnings.warn("Must be a binary signal", DeprecationWarning)

    if const == 1:
        y = np.abs(x - 1)
    else:
        y = x

    # Find run of zeros
    y = (y == 0).astype(float)
    y[nan_loc] = 0
    iedge = np.diff(np.concatenate(([0], y, [0])))
    istart = np.array(np.where(iedge == 1))[
        0
    ]  # Row zero only - input array should be 1d
    iend = np.subtract(np.where(iedge == -1), 1)[0]
    lens = np.array([iend - istart])[0]

    if DBplot:
        plt.figure(100)
        plt.plot(x)
        plt.plot(istart, np.array(x[istart]), marker="x")
        plt.plot(iend, np.array(x[iend]), marker="o", markerfacecolor="none")

    return np.array(lens), istart, iend


def save_pd_as_file(data_to_save, file_name, Fs=None):
    """
    (DataFrame, str) ->

    Pass in a pandas dataFrame and a file name. Then save as csv file in same directory

    > save_pd_as_file(pd, file_name)

    """
    if Fs is not None:
        file_name = file_name + "_Fs_" + str(Fs) + ".csv"
    else:
        file_name = file_name + ".csv"
    if isinstance(data_to_save, pd.DataFrame):
        data_to_save.to_csv(file_name, index=False)
    else:
        raise ValueError("Data needs to be pandas DataFrame to be saved!")


def save_pd_as_edf(data, file_save_name, original_edf_file=None, Fs=250):
    """

    save_pd_as_edf: Save pandas EEG dataframe as edf - nans are converted to zero.

    Syntax: save_pd_as_edf(data, file_save_name, original_edf_file, Fs)

    Inputs:
        data            - Pandas DataFrame containing the data. Column names correspond to channels
        file_save_name  - Name to save newly created EDF file as
        original_edf_file   - Original EDF files file (if present)
        Fs              - Sampling frequency (Hz)

    Outputs:

    Example:
        import utils
        Fs = 256
        data_st = utils.gen_test_EEGdata(5 * 60, Fs, 1)  # 5 minutes of fake data
        eeg_ref = data_st['eeg_data_ref']

        utils.save_pd_as_edf(eeg_ref, 'demo_data.edf', Fs=Fs)

    Note: Saving an edf using pyEDFlib will slightly change the values as it interpolates the signal from physical
          to digital values using the physical and digital maximum and minimum values (hardcoded below).
          The general shape should stay the same.
    """
    if file_save_name[-4:] != ".edf":
        file_save_name = file_save_name + ".edf"

    if original_edf_file is not None and original_edf_file[-4:] != ".edf":
        original_edf_file = original_edf_file + ".edf"

    channel_info = []
    data_list = []
    samples_in_file = []

    if original_edf_file is not None:
        f = pyedflib.EdfReader(original_edf_file)
        loc = f.getSignalLabels().index("F4")
        channels = list(data.columns)
        for ch in channels:
            ch_dict = dict()

            ch_dict["physical_min"] = f.getPhysicalMinimum(loc)
            ch_dict["physical_max"] = f.getPhysicalMaximum(loc)
            ch_dict["digital_min"] = f.getDigitalMinimum(loc)
            ch_dict["digital_max"] = f.getDigitalMaximum(loc)
            ch_dict["dimension"] = f.getPhysicalDimension(loc)
            ch_dict["sample_rate"] = f.getSampleFrequency(loc)
            ch_dict["label"] = ch
            ch_dict["transducer"] = f.getTransducer(loc)
            ch_dict["prefilter"] = f.getPrefilter(loc)

            samples_in_file.append(data.shape[0])
            x = data[ch].to_numpy()
            x[np.where(np.isnan(x))[0]] = 0
            data_list.append(x)
            channel_info.append(ch_dict)
    else:
        channels = list(data.columns)
        for ch in channels:
            ch_dict = dict()
            ch_dict["physical_min"] = -5482.288
            ch_dict["physical_max"] = 5482.288
            ch_dict["digital_min"] = -32767
            ch_dict["digital_max"] = 32767
            ch_dict["dimension"] = "uV"
            ch_dict["sample_rate"] = Fs
            ch_dict["label"] = ch
            ch_dict["transducer"] = ""
            ch_dict["prefilter"] = ""

            samples_in_file.append(data.shape[0])
            x = data[ch].to_numpy()
            x[np.where(np.isnan(x))[0]] = 0
            data_list.append(x)
            channel_info.append(ch_dict)

    w = pyedflib.EdfWriter(
        file_save_name, len(list(data.columns)), file_type=pyedflib.FILETYPE_EDFPLUS
    )
    w.setSignalHeaders(channel_info)
    w.writeSamples(data_list)
    w.close()
