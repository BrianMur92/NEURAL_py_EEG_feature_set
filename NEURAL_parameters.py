# Author:       Brian Murphy
# Date started: 17/02/2020
# Last updated: <17/02/2020 12:42:22 (BrianM)>

def NEURAL_parameters():
    """
    This function is to generate all the default NEURAL parameters

    :return: a dictionary containing all the system parameters
    """
    params = dict()

    '''
    ---------------------------------------------------------------------
    Proprocessing (lowpass filter and resample)
    ---------------------------------------------------------------------
    '''
    params['LP_fc'] = 30  # low pass filter cut-off
    params['Fs_new'] = 64  # down_sample to Fs_new

    '''
    ---------------------------------------------------------------------
    Directories - fill in directories
    ---------------------------------------------------------------------
    '''
    params['EEG_DATA_DIR'] = ''
    params['EEG_DATA_DIR_CSV'] = ''

    '''
    ---------------------------------------------------------------------
    Montage - bipolar mantage of NICU babies
    ---------------------------------------------------------------------
    '''
    params['BI_MONT'] = [['F4', 'C4'], ['F3', 'C3'], ['C4', 'T4'], ['C3', 'T3'], ['C4', 'Cz'], ['Cz', 'C3'],
                         ['C4', 'O2'], ['C3', 'O1']]

    '''
    ---------------------------------------------------------------------
    Artefacts
    ---------------------------------------------------------------------
    '''
    params['REMOVE_ART'] = 1  # simple procedure to remove artefacts; 0 to turn off

    # some default values used for preterm infants (<32 weeks of gestation)
    params['ART_HIGH_VOLT'] = 1500  # in mirco Vs
    params['ART_TIME_COLLAR'] = 10  # time collar( in seconds) around high - amplitude artefact

    params['ART_DIFF_VOLT'] = 200  # in mirco Vs
    params['ART_DIFF_TIME_COLLAR'] = 0.5  # time collar( in seconds) around fast jumps
    params['ART_DIFF_MIN_TIME'] = 0.1  # min time( in seconds) for flat(continuous) trace to be artefact

    params['ART_ELEC_CHECK'] = 1  # minimum length required for electrode check( in seconds)

    params['ART_REF_LOW_CORR'] = 0.15  # if mean correlation coefficent across referential channels
    # is < this value then remove

    # what to replace artefacts with before filtering?
    # options: 1) zeros('zeros')
    #          2) linear interpolation('linear_interp')
    #          3) cubic spline interpolation('cubic_interp')
    #          4) NaN('nans'): replace with cubic spline before filtering and then NaN's % after filtering
    params['FILTER_REPLACE_ARTEFACTS'] = 'nans'

    params['amplitude'] = dict()
    params['rEEG'] = dict()
    params['connectivity'] = dict()
    params['FD'] = dict()

    params['amplitude']['FILTER_REPLACE_ARTEFACTS'] = params['FILTER_REPLACE_ARTEFACTS']
    params['rEEG']['FILTER_REPLACE_ARTEFACTS'] = params['FILTER_REPLACE_ARTEFACTS']
    params['connectivity']['FILTER_REPLACE_ARTEFACTS'] = params['FILTER_REPLACE_ARTEFACTS']
    params['FD']['FILTER_REPLACE_ARTEFACTS'] = params['FILTER_REPLACE_ARTEFACTS']

    '''
    ---------------------------------------------------------------------
    Features
    ---------------------------------------------------------------------
    '''
    params['FEATURE_SET_ALL'] = [
        'spectral_power'
        , 'spectral_relative_power'
        , 'spectral_flatness'
        , 'spectral_diff'
        , 'spectral_entropy'
        , 'spectral_edge_frequency'
        , 'FD'
        , 'amplitude_total_power'
        , 'amplitude_SD'
        , 'amplitude_skew'
        , 'amplitude_kurtosis'
        , 'amplitude_env_mean'
        , 'amplitude_env_SD'
        , 'connectivity_BSI'
        , 'connectivity_corr'
        , 'connectivity_coh_mean'
        , 'connectivity_coh_max'
        , 'connectivity_coh_freqmax'
        , 'rEEG_mean'
        , 'rEEG_median'
        , 'rEEG_lower_margin'
        , 'rEEG_upper_margin'
        , 'rEEG_width'
        , 'rEEG_SD'
        , 'rEEG_CV'
        , 'rEEG_asymmetry'
        , 'IBI_length_max'
        , 'IBI_length_median'
        , 'IBI_burst_prc'
        , 'IBI_burst_number'
    ]

    '''
    Frequency bands
    '''
    params['FREQ_BANDS'] = [[0.5, 4], [4, 7], [7, 13], [13, 30]]

    # these bands often used for preterm infants( < 32 weeks GA):
    # params['FREQ_BANDS'] = [[0.5, 3], [3, 8], [8, 15], [15, 30]];

    '''
    ---------------------------------------------------------------------
    A.spectral features
    ---------------------------------------------------------------------
    '''

    # how to estimate the spectrum for 'spectral_flatness', 'spectral_entropy', % spectral_edge_frequency features:
    #    1) PSD: estimate power spectral density(e.g.Welch periodogram)
    #    2) robust - PSD: median(instead of mean) of spectrogram
    #    3) periodogram: magnitude of the discrete Fourier transform
    params['spectral'] = dict()
    params['spectral']['method'] = 'PSD'

    # length of time - domain analysis window and overlap:
    # (applies to 'spectral_power', 'spectral_relative_power',
    # 'spectral_flatness', and 'spectral_diff' features)
    params['spectral']['L_window'] = 2  # in seconds
    params['spectral']['window_type'] = 'hamm'  # type of window
    params['spectral']['overlap'] = 50  # overlap in percentage
    params['spectral']['freq_bands'] = params['FREQ_BANDS']
    params['spectral']['total_freq_bands'] = [params['FREQ_BANDS'][0][0], params['FREQ_BANDS'][-1][-1]]
    params['spectral']['SEF'] = 0.95  # spectral edge frequency

    # fractal dimension(FD):
    params['FD']['method'] = 'higuchi'  # method to estimate FD, either 'higuchi' or 'katz'
    params['FD']['freq_bands'] = [params['FREQ_BANDS'][0][0], params['FREQ_BANDS'][-1][-1]]
    # $$$ params['FD']['freq_bands'] = params['FREQ_BANDS']
    params['FD']['qmax'] = 6  # Higuchi method: max.value of k

    '''
    ---------------------------------------------------------------------
    B. amplitude features
    ---------------------------------------------------------------------
    '''

    # $$$ params['amplitude']['freq_bands'] = [params['FREQ_BANDS'][0][0], params['FREQ_BANDS'][-1][-1]]
    params['amplitude']['freq_bands'] = params['FREQ_BANDS']

    # for rEEG(range - EEG, similar to aEEG) from [1]
    #
    # [1] DOâ€™Reilly, MA Navakatikyan, M Filip, D Greene, & LJ Van Marter(2012).Peak - to - peak amplitude in neonatal
    # brain monitoring of premature infants.Clinical Neurophysiology, 123(11), 2139 â€“53.
    #
    # settings in [1]: window = 2 seconds; overlap = 0 %; and no log - linear scale
    params['rEEG']['L_window'] = 2  # in seconds
    params['rEEG']['window_type'] = 'rect'  # type of window
    params['rEEG']['overlap'] = 0  # overlap in percentage
    params['rEEG']['APPLY_LOG_LINEAR_SCALE'] = 0  # use this scale(either 0 or 1)
    params['rEEG']['freq_bands'] = params['FREQ_BANDS']

    '''
    ---------------------------------------------------------------------
    C. connectivity features
    ---------------------------------------------------------------------
    '''

    # how to estimate the cross spectrum for the coherence function:
    # 1) PSD: estimate power spectral density (e.g. Welch periodogram)
    # 2) bartlett-PSD: Welch periodogram with 0% overlap and rectangular window
    #    (necessary if using the analytic assessment of zero coherence, see below)
    params['connectivity']['method'] = 'bartlett-PSD'

    params['connectivity']['freq_bands'] = params['FREQ_BANDS']
    params['connectivity']['L_window'] = 8  # PSD window in seconds
    params['connectivity']['overlap'] = 75  # PSD window percentage overlap
    params['connectivity']['window_type'] = 'hamm'  # PSD window type

    # find lower coherence limit using either either a surrogate-data
    # approach [1] or an analytic threshold [2]
    # [1] Faes L, Pinna GD, Porta A, Maestri R, Nollo G (2004). Surrogate data analysis for
    #     assessing the significance of the coherence function. IEEE Transactions on
    #     Biomedical Engineering, 51(7):1156â€“1166.
    # [2] Halliday, DM, Rosenberg, JR, Amjad, AM, Breeze, P, Conway, BA, &
    #     Farmer, SF. (1995). A framework for the analysis of mixed time series/point
    #     process data--theory and application to the study of physiological tremor, single
    #     motor unit discharges and electromyograms. Progress in Biophysics and Molecular
    #     Biology, 64(2â€“3), 237â€“278.
    #
    # options for 'feat_params_st.connectivity.coherence_zero_level' are:
    # 1) 'surr' for [1]
    # 2) 'analytic' for [2]
    # 3) '' not to implement (no threshold)
    params['connectivity']['coherence_zero_level'] = 'analytic'
    # alpha value for null-hypothesis disribution cut-off:
    params['connectivity']['coherence_zero_alpha'] = 0.05
    # number of iterations required to generate null-hypothesis distribution if
    # using surrogate data approach ([2]):
    params['connectivity']['coherence_surr_iter'] = 500

    '''
    ---------------------------------------------------------------------
    Short-time analysis on EEG
    ---------------------------------------------------------------------
    '''

    params['EPOCH_LENGTH'] = 64  # seconds
    params['EPOCH_OVERLAP'] = 50  # percent

    params['EPOCH_IGNORE_PRC_NANS'] = 50  # if epoch has â‰¥ EPOCH_IGNORE_PRC_NANS (percent) then ignore

    return params
