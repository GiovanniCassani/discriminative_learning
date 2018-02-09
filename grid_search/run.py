__author__ = 'GCassani'

"""Function to perform the grid-search experiment to generate the results 
   that were presented at Psycholinguistics in Flanders 2017, in Leuven"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from grid_search.make_test_set import get_words_and_tags_from_test_set
from phonetic_bootstrapping.experiment.helpers import compute_baselines
from phonetic_bootstrapping.experiment.phonetic_bootstrapping import phonetic_bootstrapping


def run_grid_search(test_files, training_corpora, cues, outcomes, stress_marker, reduced, methods, evaluations, k,
                    flush, celex_folder, pos_mapping, longitudinal):

    """
    :param test_files:          an iterable containing the paths to the test sets to be evaluated
    :param training_corpora:    an iterable containing the paths to the corpora to be used for training
    :param cues:                an iterable containing strings indicating which cues to be considered
                                (at least one among 'uniphones', 'diphones', 'triphones', and 'syllables')
    :param outcomes:            an iterable containing strings indicating which lexical outcomes to use, whether
                                'lemmas' or 'tokens'
    :param stress_marker:       an iterable containing booleans specifying whether or not to consider stress
    :param reduced:             an iterable containing booleans specifying whether or not use reduced phonetic
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
    :param celex_folder:        the path to the folder containing Celex data
    :param pos_mapping:         the path to the file containing the mapping from CHILDES to Celex PoS tags
    :param longitudinal:        a boolean specifying whether to use or not a longitudinal design
    :return summary_table:      a Pandas data frame containing as many rows as there are different parametrizations
                                defined by the unique combinations of all the values in the input arguments and the
                                columns specified in the body of the function when the data frame is initialized
    """

    time = 10 if longitudinal else 1
    rows = np.prod([len(test_files), len(training_corpora), len(cues), len(outcomes), len(stress_marker),
                    len(reduced), len(methods), len(evaluations), len(k), len(flush), time])

    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Test_set", "Corpus", "Cues", "Outcomes", "Stress", "Vowels",
                                          "Method", "Evaluation", "K", "F", "Time",
                                          "Accuracy", "Majority_baseline", "Entropy", "Entropy_baseline",
                                          "PoS", "Frequency"])

    """
    Code to use the parallelizer:
    
    from itertools import product
    from multiprocessing import Pool
    
    uniphones = ['True', 'False'] if 'uniphones' in cues else ['False']
    diphones = ['True', 'False'] if 'diphones' in cues else ['False']
    triphones = ['True', 'False'] if 'triphones' in cues else ['False']
    syllables = ['True', 'False'] if 'syllables' in cues else ['False']
    
    # the order of parameters for the phonetic_bootstrapping() function is the following:
    # input_file, test_items_path, celex_dir, pos_mapping, method, evaluation, k, flush, ambiguous, new, separator,
    # reduced, outcomes, uni_phones, di_phones, tri_phones, syllable, stress_marker, alpha, beta, lam, longitudinal
    all_parametrizations = product(training_corpora, test_sets, celex_folder, pos_mapping, methods, evaluations, k,
                                   flush, 'unambiguous', 'new', '~', reduced, outcomes, uniphones, diphones, triphones,
                                   syllables, stress_marker, 0.001, 0.001, 1.0, longitudinal)
                         
    for parametrization in all_parametrizations:
        sm = "stress" if parametrization[4] else 'no-stress'
        vowels = 'reduced' if parametrization[5] else 'full'

        Pool.starmap(phonetic_bootstrapping, all_parametrizations, nodes=8)
    """

    test_sets = defaultdict(dict)
    for test in test_files:
        # get the items in a test set and compute majority baseline and entropy of the PoS tags distribution; the
        # dictionary test_sets maps each file name to the words in it (only the phonological form and target PoS tag),
        # and to the corresponding majority baseline and entropy baseline
        test_words, tags = get_words_and_tags_from_test_set(test)
        test_sets[test]['filename'] = os.path.basename(test)
        test_sets[test]['items'] = test_words
        majority_baseline, entropy_baseline = compute_baselines(tags)
        test_sets[test]['majority'] = majority_baseline
        test_sets[test]['entropy'] = entropy_baseline

    a, b = [0.001, 0.001]  # set parameters for the discriminative learning model
    row_id = 0
    for test in test_files:

        majority_baseline = test_sets[test]['majority']
        entropy_baseline = test_sets[test]['entropy']

        for corpus in training_corpora:
            for cue in cues:
                for outcome in outcomes:
                    for marker in stress_marker:
                        for r in reduced:
                            for method in methods:
                                for evaluation in evaluations:
                                    for k_value in k:
                                        for f_value in flush:

                                            uniphones = True if cue == 'uniphones' else False
                                            diphones = True if cue == 'diphones' else False
                                            triphones = True if cue == 'triphones' else False
                                            syllables = True if cue == 'syllables' else False
                                            vowels = 'reduced' if r else 'full'
                                            sm = "stress" if marker else 'no-stress'
                                            training = os.path.splitext(os.path.basename(corpus))[0]

                                            log, f1, h, pos, freq = phonetic_bootstrapping(corpus, test_sets[test],
                                                                                           celex_folder, pos_mapping,
                                                                                           method=method, k=k_value,
                                                                                           reduced=r, flush=f_value,
                                                                                           stress_marker=marker,
                                                                                           uni_phones=uniphones,
                                                                                           di_phones=diphones,
                                                                                           tri_phones=triphones,
                                                                                           syllable=syllables,
                                                                                           outcomes=outcome,
                                                                                           alpha=a, beta=b, lam=1.0,
                                                                                           longitudinal=longitudinal)

                                            for time_idx in f1:
                                                summary_table.loc[row_id] = pd.Series({"Test_set":
                                                                                           test_sets[test]['filename'],
                                                                                       "Corpus": training, "Cues": cue,
                                                                                       "Outcomes": outcome,
                                                                                       "Stress": sm, "Vowels": vowels,
                                                                                       "Method": method,
                                                                                       "Evaluation": evaluation,
                                                                                       "K": k_value, "F": f_value,
                                                                                       "Time": time_idx,
                                                                                       "Accuracy": f1[time_idx],
                                                                                       "Majority_baseline":
                                                                                           majority_baseline,
                                                                                       "Entropy": h[time_idx],
                                                                                       "Entropy_baseline":
                                                                                           entropy_baseline,
                                                                                       "PoS": pos[time_idx],
                                                                                       "Frequency": freq[time_idx]})
                                            row_id += 1

    return summary_table
