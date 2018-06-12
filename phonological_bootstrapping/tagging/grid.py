__author__ = 'GCassani'

"""Function to perform the grid-search experiment to generate the results 
   that were presented at Psycholinguistics in Flanders 2017, in Leuven"""

import os
import numpy as np
import pandas as pd
import itertools as it
from phonological_bootstrapping.test.make import get_words_and_tags_from_test_set
from phonological_bootstrapping.helpers import compute_baselines
from phonological_bootstrapping.tagging.experiment import tag_words


def grid_search(test_file, output_folder, input_corpus, cues, outcomes, boundaries, stress_marker, reduced_vowels,
                methods, evaluations, k, flush, threshold, celex_folder, pos_mapping, longitudinal, at, a, b):

    """
    :param test_file:           the path to the file to be used as test set
    :param output_folder:       the path to the folder where the log file of every experiment will be saved
    :param input_corpus:        the path pointing to the input corpus to be used (must be a .json file)
    :param cues:                an iterable containing strings indicating which cues to be considered
                                (at least one among 'uniphones', 'diphones', 'triphones', and 'syllables')
    :param outcomes:            an iterable containing strings indicating which lexical outcomes to use, whether
                                'lemmas' or 'tokens'
    :param boundaries:          an iterable containing booleans specifying whether or not to consider word boundaries
                                in training
    :param stress_marker:       an iterable containing booleans specifying whether or not to consider stress
    :param reduced_vowels:      an iterable containing booleans specifying whether or not use reduced phonetic
                                transcriptions whenever possible
    :param methods:             an iterable containing strings specifying which method to use to process top active
                                lexical nodes given a test item (at least one among 'sum' and 'freq')
    :param evaluations:         an iterable containing strings specifying which evaluation to use when comparing
                                activation values triggered by a test item to baseline activation values
                                (at least one among 'count' and 'distr')
    :param k:                   an iterable containing integers specifying how many top active lexical nodes to
                                consider to pick the most represented PoS category
    :param flush:               an iterable containing integers specifying how many lexical nodes to flush away,
                                to get rid of nodes with very high baseline activations that would be among the top
                                active lexical nodes for all the test items
    :param threshold:           the minimum activation of an outcome to be considered in the list of top activated
                                neighbors, default is 0 and shouldn't be lowered, but can be increased.
    :param celex_folder:        the path to the folder containing Celex data
    :param pos_mapping:         the path to the file containing the mapping from CHILDES to Celex PoS tags
    :param longitudinal:        a boolean specifying whether to use or not a longitudinal design
    :param at:                  the threshold at which to compute precision of discrimination
    :param a:                   the alpha parameter of the Rescorla Wagner model
    :param b:                   the beta parameter of the Rescorla Wagner model
    :return summary_table.txt:      a Pandas data frame containing as many rows as there are different parametrizations
                                defined by the unique combinations of all the values in the input arguments and the
                                columns specified in the body of the function when the data frame is initialized,
                                containing summary statistics about each model on a PoS tagging experiment
    :return error_analysis:     a Pandas data frame with the same number of rows as summary_table.txt, but with columns
                                indicating the classification and mis-classification patterns by PoS tag for each model
    """

    time = 10 if longitudinal else 1
    rows = int(np.prod([len(cues), len(outcomes), len(stress_marker), len(reduced_vowels), len(boundaries),
                        len(methods), len(evaluations), len(k), len(flush), time]))

    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Test_set", "Corpus", "Boundaries", "Cues", "Outcomes", "Stress", "Vowels",
                                          "Method", "Evaluation", "K", "F", "Time",
                                          "Accuracy", "Majority_baseline", "Entropy", "Entropy_baseline",
                                          "PoS", "Frequency"])

    test_set = {}

    # get the items in the test set and compute majority baseline and entropy of the PoS tags distribution
    test_words, tags = get_words_and_tags_from_test_set(test_file)
    test_set['filename'] = os.path.basename(test_file)
    test_set['items'] = test_words
    majority_base, entropy_base = compute_baselines(tags)

    row_id = 0
    parametrizations = it.product(cues, boundaries, stress_marker, reduced_vowels,
                                  outcomes, methods, evaluations, k, flush)
    for parametrization in parametrizations:
        cue, boundary, marker, r, outcome, method, evaluation, k_value, f_value = parametrization
        uniphones = True if cue == 'uniphones' else False
        diphones = True if cue == 'diphones' else False
        triphones = True if cue == 'triphones' else False
        syllables = True if cue == 'syllables' else False
        vowels = 'reduced' if r else 'full'
        sm = "stress" if marker else 'no-stress'
        bound = 'yes' if boundary else 'no'
        training = os.path.splitext(os.path.basename(input_corpus))[0]

        logs, f1, h, pos, freq = tag_words(input_corpus, test_set, celex_folder, pos_mapping, output_folder,
                                           stress_marker=marker, boundaries=boundary, reduced=r, outcomes=outcome,
                                           uniphones=uniphones, diphones=diphones, triphones=triphones,
                                           syllable=syllables, at=at, threshold=threshold, alpha=a, beta=b, lam=1.0,
                                           method=method, evaluation=evaluation, flush=f_value, k=k_value,
                                           longitudinal=longitudinal)
        for time_idx in f1:
            summary_table.loc[row_id] = pd.Series({"Test_set": test_set['filename'], "Corpus": training, "Cues": cue,
                                                   "Outcomes": outcome, "Stress": sm, "Boundaries": bound,
                                                   "Vowels": vowels, "Method": method, "Evaluation": evaluation,
                                                   "K": k_value, "F": f_value, "Time": time_idx,
                                                   "Accuracy": f1[time_idx], "Majority_baseline": majority_base,
                                                   "Entropy": h[time_idx], "Entropy_baseline": entropy_base,
                                                   "PoS": pos[time_idx], "Frequency": freq[time_idx]})
            row_id += 1

        print()
        print()
        print("%" * 120)
        print("%" * 120)
        print()
        print()

    return summary_table
