# Author:       Brian Murphy
# Date started: 05/02/2020
# Last updated: <25/07/2022 16:40:13 (BrianM)>

from NEURAL_py_EEG import utils
from NEURAL_py_EEG import preprocessing_EEG
import numpy as np
from NEURAL_py_EEG import generate_all_features
import time


def artefact_removal_examples():
    print("Running artefact_removal_examples function")

    Fs = 256  # sampling frequency
    print("Generating EEG data using the utils.gen_test_EEGdata function")
    data_st = utils.gen_test_EEGdata(2 * 60, Fs, 1)
    N = len(data_st["eeg_data_ref"])

    # simulate artefacts:
    # 1. F3 not properly attached:
    print("Simulating artefacts:")
    print("1. F3 not properly attached")
    if3 = [i for i, x in enumerate(data_st["ch_labels_ref"]) if x == "F3"][0]
    data_st["eeg_data_ref"][data_st["ch_labels_ref"][if3]] = np.random.randn(N) * 10

    # 2. electrode coupling between C4 and Cz
    print("2. Electrode coupling between C4 and Cz")
    ic4 = [i for i, x in enumerate(data_st["ch_labels_ref"]) if x == "C4"][0]
    icz = [i for i, x in enumerate(data_st["ch_labels_ref"]) if x == "Cz"][0]

    data_st["eeg_data_ref"][data_st["ch_labels_ref"][icz]] = (
        data_st["eeg_data_ref"][data_st["ch_labels_ref"][ic4]] + np.random.randn(N) * 5
    )

    # re-generate bipolar montage:
    data_st["eeg_data"], data_st["ch_labels"] = utils.set_bi_montage(
        data_st["eeg_data_ref"], Fs, data_st["ch_labels_bi"]
    )

    # remove channels:
    print("Running artefact removal")
    eeg_art = preprocessing_EEG.remove_artefacts(
        data_st["eeg_data"].copy(),
        data_st["ch_labels"],
        data_st["Fs"],
        data_st["eeg_data_ref"],
        data_st["ch_labels_ref"],
    )[0]

    print(data_st["eeg_data"].head())
    print(eeg_art.head())
    print("The artefact_removal_examples function has finished")
    print("\n\n\n\n")


def generate_a_subset_of_features_example():
    print("Running generate_a_subset_of_features_example function")
    Fs = 64  # sampling frequency

    # generate EEG-like data (coloured Gaussian noise)
    print("Generating EEG data using the utils.gen_test_EEGdata function")
    data_st = utils.gen_test_EEGdata(5 * 60, Fs, 1)

    # define feature set (or can define in utils.NEURAL_parameters):
    feature_set = ["spectral_relative_power", "rEEG_SD", "connectivity_BSI"]
    print("Feature set: %s" % feature_set)

    # estimate features:
    print("Estimating subset of features")
    (
        feats_per_epochs,
        feat_pd_names,
        feat_st,
        feats_median_ch,
        feats_median_all,
    ) = generate_all_features.generate_all_features(data_st, feat_set=feature_set)
    print("Features extracted")
    print(feat_pd_names)
    print("The generate_a_subset_of_features_example function has finsihed")
    print("\n\n\n\n")


def generate_all_features_example():
    print("Running generate_a_subset_of_features_example function")
    Fs = 64  # sampling frequency

    # generate EEG-like data (coloured Gaussian noise)
    print("Generating EEG data using the utils.gen_test_EEGdata function")
    data_st = utils.gen_test_EEGdata(5 * 60, Fs, 1)

    # estimate features:
    print("Estimating features")
    (
        feats_per_epochs,
        feat_pd_names,
        feat_st,
        feats_median_ch,
        feats_median_all,
    ) = generate_all_features.generate_all_features(data_st)

    print("Features extracted")
    print(feat_pd_names)
    print("The generate_a_subset_of_features_example function has finished")
    print("\n\n\n\n")


def preprocessing_and_feature_extraction():
    print("Running preprocessing_and_feature_extraction function")
    Fs = 256  # sampling frequency
    # generate EEG-like data (coloured Gaussian noise)
    print("Generating EEG data using the utils.gen_test_EEGdata function")
    data_st = utils.gen_test_EEGdata(5 * 60, Fs, 1)

    print("Running artefact removal")
    eeg_art = preprocessing_EEG.remove_artefacts(
        data_st["eeg_data"].copy(),
        data_st["ch_labels"],
        data_st["Fs"],
        data_st["eeg_data_ref"],
        data_st["ch_labels_ref"],
    )[0]

    # replace the montage with the artifact free version
    data_st["eeg_data"] = eeg_art.copy()

    # filter and downsample the signal
    print("Before filtering and downsampling")
    print(data_st["eeg_data"].head())
    print("Number of rows: %s" % str(data_st["eeg_data"].shape[0]))

    data_st["eeg_data"] = preprocessing_EEG.LPF_zero_phase(
        data_st["eeg_data"].copy(), data_st["Fs"]
    )
    data_st["eeg_data"], data_st["Fs"] = preprocessing_EEG.signal_downsample(
        data_st["eeg_data"].copy(), data_st["Fs"], 64
    )

    print("After filtering and downsampling")
    print(data_st["eeg_data"].head())
    print("Number of rows: %s" % str(data_st["eeg_data"].shape[0]))

    # estimate features:
    print("Estimating features")
    (
        feats_per_epochs,
        feat_pd_names,
        feat_st,
        feats_median_ch,
        feats_median_all,
    ) = generate_all_features.generate_all_features(data_st)

    print("Features extracted")
    print(feat_pd_names)
    print("The preprocessing_and_feature_extraction function has finished")
    print("\n\n\n\n")


def load_edf_preprocessing_and_feature_extraction():
    print("Running load_edf_preprocessing_and_feature_extraction function")
    Fs = 256  # sampling frequency
    print("Generating EEG data using the utils.gen_test_EEGdata function")
    data_st = utils.gen_test_EEGdata(
        10 * 60, Fs, 1
    )  # 10 minutes of fake data - need to save as edf
    eeg_ref = data_st["eeg_data_ref"]

    print('Saving EEG data as "demo_data.edf" file')
    utils.save_pd_as_edf(
        eeg_ref, "demo_data.edf", Fs=Fs
    )  # Save the fake data as edf to simulate opening a real edf

    """
    preprocessing_EEG.main_preprocessing does the following operations to the edf file
                        a) read in EEG from .edf files
                        b) remove artefacts
                        c) band-pass filter
                        d) downsample
                        e) save as .csv file (optional by setting save_converted='name', save=1)
    There are 2 outputs:    
                        0) the data dictionary containing the bipolar EEG montage, sampling frequency and channel labels
                        1) the amount of data removed during the artifact removal stage
    """

    print(
        'Loading "demo_data.edf", performing artefact removal, low pass filtering and down-sampling'
    )
    data_st = preprocessing_EEG.main_preprocessing("demo_data.edf", Fs_new=64)[0]
    print("EDF file loaded and preprocessed")

    # estimate features:
    print("Extracting features")
    t = time.time()
    (
        feats_per_epochs,
        feat_pd_names,
        feat_st,
        feats_median_ch,
        feats_median_all,
    ) = generate_all_features.generate_all_features(data_st)

    print("Features extracted - time taken %s" % str(time.time() - t))
    print(feat_pd_names)
    print("The load_edf_preprocessing_and_feature_extraction function has finished")
    print("\n\n\n\n")

