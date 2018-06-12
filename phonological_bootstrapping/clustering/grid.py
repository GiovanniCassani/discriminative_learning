__author__ = 'GCassani'

"""Function to evaluate clustering in a matrix using a kNN approach"""

import os
import numpy as np
import pandas as pd
import itertools as it
from phonological_bootstrapping.clustering.experiment import cluster_words


def grid_search(corpus, output_folder, pos_mapping, celex_folder, outcomes, cues, boundaries, stress_marker,
                reduced_vowels, distances, nn, at, longitudinal, a, b):

    """
    :param corpus:          the corpus to be used for training
    :param output_folder:   the path of the folder where
    :param pos_mapping:     the path to the file mapping CHILDES PoS tags to Celex tags
    :param celex_folder:    the path to the directory where the celex data are stored
    :param outcomes:        an iterable of strings indicating which types of outcomes to consider
    :param cues:            an iterable of strings indicating which phonological cues to consider
    :param boundaries:      an iterable of booleans indicating whether to consider word boundaries or not
    :param stress_marker:   an iterable of booleans indicating whether to consider stress or discard it
    :param reduced_vowels:  an iterable of booleans indicating whether to use reduced phonetic forms from Celex or not
    :param distances:       an iterable indicating which distances to consied to compute clustering
    :param nn:              an iterable indicating the number of neighbors to consider for clustering
    :param at:              the number of neighbors to consider when assessing discrimination's precision
    :param longitudinal:    a boolean indicating whether to use a longitudinal approach or not
    :param a:               the alpha parameter of the Rescorla Wagner learning model
    :param b:               the beta parameter of the Rescorla Wagner learning model
    :return summary_table:  a Pandas dataframe storing clustering accuracy information over all the possible
                            parametrizations resulting from the combination of all the values in the input iterables
    """

    time_points = np.linspace(10, 100, 10) if longitudinal else [100]
    rows = int(np.prod([len(cues), len(outcomes), len(stress_marker), len(reduced_vowels),
                        len(boundaries), len(nn), len(time_points)]))
    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Corpus", "Boundaries", "Cues", "Outcomes", "Stress",
                                          "Vowels", "K", "Precision", "Time", "Distance",
                                          "Accuracy", "Baseline_acc", "Entropy", "Baseline_entr"])

    row_id = 0
    parametrizations = it.product(cues, boundaries, stress_marker, reduced_vowels, outcomes, distances, nn)
    for parametrization in parametrizations:

        cue, boundary, marker, r, outcome, distance, neighbours = parametrization
        uniphones = True if cue == 'uniphones' else False
        diphones = True if cue == 'diphones' else False
        triphones = True if cue == 'triphones' else False
        syllables = True if cue == 'syllables' else False
        vowels = 'reduced' if r else 'full'
        sm = "stress" if marker else 'no-stress'
        bound = 'yes' if boundary else 'no'
        training = os.path.splitext(os.path.basename(corpus))[0]

        accuracies = cluster_words(corpus, output_folder, celex_folder, pos_mapping, distance=distance,
                                   outcomes=outcome, stress_marker=marker, boundaries=boundary, reduced=r,
                                   uniphones=uniphones, diphones=diphones, triphones=triphones, syllables=syllables,
                                   at=at, nn=neighbours, a=a, b=b, longitudinal=longitudinal)

        for time_idx in accuracies:
            accuracy = accuracies[time_idx]['accuracy']
            baseline_acc = accuracies[time_idx]['baseline_acc']
            entropy = accuracies[time_idx]['entropy']
            baseline_entr = accuracies[time_idx]['baseline_entr']
            summary_table.loc[row_id] = pd.Series({"Corpus": training, "Cues": cue,"Outcomes": outcome, "Stress": sm,
                                                   "Boundaries": bound, "Vowels": vowels, "K": neighbours,
                                                   "Time": int(time_idx), "Distance": distance, "Precision": at,
                                                   "Accuracy": accuracy, "Baseline_acc": baseline_acc,
                                                   "Entropy": entropy, "Baseline_entr": baseline_entr})
            row_id += 1

        print()
        print("%" * 120)
        print("%" * 120)
        print()
        print()

    return summary_table
