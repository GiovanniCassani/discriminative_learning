__author__ = 'GCassani'

import os
import numpy as np
from time import strftime
from rescorla_wagner.compute_activations import compute_activations


def file_exists(output_files):

    """
    :param output_files:    a Python dictionary mapping numerical indices to file paths
    :return:                a boolean, True only if all paths in the input dictionary point to an existing file, False
                            otherwise, i.e. even if only one file from the input dictionary doesn't exist
    """

    for k in output_files:
        if not os.path.isfile(output_files[k]):
            return False

    return True


########################################################################################################################


def ndl(input_file, alpha=0.01, beta=0.01, lam=1.0, longitudinal=False):

    """
    :param input_file:          the path to a a .json file consisting of two lists of lists, the first containing
                                phonetic cues and the second containing outcome meanings; each list contains as many
                                lists as there are learning events in the input corpus (be them full utterances, single
                                words, or any intermediate representation derived from transcribed child-directed
                                speech). The first list from the list of cue representations matches the first list from
                                the list of meaning representations, both encoding the two layers of the first learning
                                event in the corpus
    :param alpha:               cue salience. For simplicity, we assume that every cue has the same salience, so
                                changing the value of this parameter does not affect the relative strength of
                                cue-outcome associations but only their absolute magnitude.
    :param beta:                learning rate. Again, we make the simplifying assumption that our simulated learners are
                                equally affected by positive and negative feedback. Changing the beta value can have a
                                significant impact on learning outcome, but 0.1 is a standard choice for this model.
    :param lam:                 maximum amount of association that an outcome can receive from all the cues. It simply
                                acts as a scaling factor, so changing its value has the same effects of changing alpha.
    :param longitudinal:        a boolean specifying whether to adopt a longitudinal design and store association
                                matrices at every 10%% of the data, to be able to analyze the time course of learning
    :return file_paths:         a dictionary mapping each time index to the file path where the matrix of cue-outcome
                                associations at that time index is stored

    This function implements Naive Discriminative Learning (NDL, see Baayen, Milin, Durdevic, Hendrix, Marelli (2011)
    for a detailed description of the model and its theoretical background). This learner uses the Rescorla-Wagner
    equations to update cue-outcome associations: it is a simple associative network with no hidden layer, that
    incrementally updates associations between cues and outcomes.

    It runs in linear time on the length of the input in number of utterances, thus if it takes 1 minute to process 1k
    utterances, it'll take 2 minutes to process 2k utterances. Moreover, the runtime also depends on the number of cues
    in which each utterances is encoded, thus the linearity is not perfect: if the first 1k utterances are longer and
    contain more cues, it'll take a bit more than a minute, while if the second 1k utterances are shorter and contain
    fewer cues, it'll take slightly less than a minute to process them.
    In details, it takes ~14 minutes to process ~550k utterances using the configuration with triphones only to encode
    input utterances, using a 2x Intel Xeon 6-Core E5-2603v3 with 2x6 cores and 2x128 Gb of RAM.
    """

    # create file paths for every required time point
    output_files = {}
    folder = os.path.dirname(input_file)

    indices = np.linspace(10, 100, 10) if longitudinal else [100]
    for idx in indices:
        output_files[idx] = os.path.join(folder, '.'.join(['_'.join(['associationMatrix', str(int(idx))]), 'npy']))

    missing_indices = []
    for idx, f_path in output_files.items():
        if not os.path.exists(f_path):
            missing_indices.append(idx)
        else:
            print(strftime("%Y-%m-%d %H:%M:%S") + ": The matrix of weights already exists at file %s." % f_path)

    if missing_indices:

        print(strftime("%Y-%m-%d %H:%M:%S") + ": I started estimating the cue-outcome associations...")

        # this function writes the matrix of association to file for every time index specified in missing_indices
        # it also writes to json files the dictionary mapping cues to their row indices in the association matrices,
        # and the dictionary mapping outcomes to their column indices in the association matrices
        compute_activations(input_file, output_files, alpha, beta, lam, missing_indices)

        print(strftime("%Y-%m-%d %H:%M:%S") + ": ... I finished estimating the cue-outcome associations.")

    return output_files
