__author__ = 'GCassani'

"""Function to perform the grid-search experiment to generate the results 
   that were presented at Psycholinguistics in Flanders 2017, in Leuven"""

import numpy as np
import pandas as pd
from phonetic_bootstrapping.experiment.phonetic_bootstrapping import phonetic_bootstrapping


def run_grid_search(test_sets, training_corpora, cues, outcomes, stress_marker, reduced, methods, evaluations, k, flush,
                    celex_folder, pos_mapping, longitudinal):

    """
    :param test_sets:           an iterable containing the paths to the test sets to be evaluated
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
    rows = np.prod([len(test_sets), len(training_corpora), len(cues), len(outcomes), len(stress_marker),
                    len(reduced), len(methods), len(evaluations), len(k), len(flush), time])

    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Test_set", "Corpus", "Cues", "Outcomes", "Stress", "Vowels",
                                          "Method", "Evaluation", "K", "F", "Time",
                                          "Accuracy", "Entropy", "PoS", "Frequency"])
    row_id = 0
    for test in test_sets:
        for corpus in training_corpora:
            for cue in cues:
                for outcome in outcomes:
                    for marker in stress_marker:
                        for r in reduced:
                            for method in methods:
                                for evaluation in evaluations:
                                    for k_value in k:
                                        for f_value in flush:

                                            a, b = [0.001, 0.001] if corpus == 'words' else [0.00001, 0.00001]
                                            uniphones = True if cue == 'uniphones' else False
                                            diphones = True if cue == 'diphones' else False
                                            triphones = True if cue == 'triphones' else False
                                            syllables = True if cue == 'syllables' else False
                                            vowels = 'reduced' if r else 'full'
                                            sm = "stress" if marker else 'no-stress'

                                            f1, h, pos, freq = phonetic_bootstrapping(corpus, test, celex_folder,
                                                                                      pos_mapping, method=method,
                                                                                      evaluation=evaluation, k=k_value,
                                                                                      flush=f_value, reduced=r,
                                                                                      stress_marker=marker,
                                                                                      uni_phones=uniphones,
                                                                                      di_phones=diphones,
                                                                                      tri_phones=triphones,
                                                                                      syllable=syllables,
                                                                                      outcomes=outcomes,
                                                                                      alpha=a, beta=b, lam=1.0,
                                                                                      longitudinal=longitudinal)

                                            for time_idx in f1:
                                                summary_table.loc[row_id] = pd.Series({"Test_set": test,  "Cues": cue,
                                                                                       "Outcomes": outcome,
                                                                                       "Stress": sm, "Vowels": vowels,
                                                                                       "Method": method,
                                                                                       "Evaluation": evaluation,
                                                                                       "K": k_value, "F": f_value,
                                                                                       "Time": time_idx,
                                                                                       "Accuracy": f1[time_idx],
                                                                                       "Entropy": h[time_idx],
                                                                                       "PoS": pos[time_idx],
                                                                                       "Frequency": freq[time_idx]})
                                            row_id += 1

    return summary_table
