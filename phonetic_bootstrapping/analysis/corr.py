__author__ = 'GCassani'

"""Function to compute pairwise correlations and print them"""

import numpy as np
from phonetic_bootstrapping.analysis.plot import scatter


def compute_correlations(statistics, measures, items, pos=None):

    """
    :param statistics:
    :param measures:
    :param items:
    :param pos:
    :return:
    """

    data = np.zeros([len(measures), len(items)])
    for idx, key in enumerate(items):
        for measure in measures:
            if measure == 'frequencies':
                value = statistics[measure][key]
            else:
                value = statistics[measure][pos][key] if pos else statistics[measure][key]
            data[measures[measure], idx] = value

    correlations = np.corrcoef(data)

    return correlations, data


########################################################################################################################


def print_correlations(correlations, measures):

    """
    :param measures:
    :param correlations:
    :return:
    """

    done = []
    for measure1, id1 in measures.items():
        for measure2, id2 in measures.items():
            pair = {measure1, measure2}
            if pair not in done and id1 != id2:
                print("\t%s ~ %s: %0.4f" %
                      (measure1, measure2, correlations[measures[measure1], measures[measure2]]))
                done.append(pair)

    return done


########################################################################################################################


def outcome_correlations(outcome_statistics, measures, plot_path):

    """
    :param outcome_statistics:
    :param measures:
    :param plot_path:
    :return:
    """

    measure2id = {m: i for i, m in enumerate(measures)}

    shared = set()
    for measure in measures:
        shared = set.intersection(shared, set(outcome_statistics[measure].keys())) if shared \
            else set(outcome_statistics[measure].keys())

    # compute and print pairwise correlations across all shared outcomes for all pairs of measures
    correlations, outcome_data = compute_correlations(outcome_statistics, measure2id, shared)
    print_correlations(correlations, measure2id)
    scatter(measure2id, outcome_data, plot_path, 'outcome_correlations.pdf')


########################################################################################################################


def cue_correlations(cue_statistics, measures, cues):

    """
    :param cue_statistics:
    :param measures:
    :param cues:
    :return:
    """

    measure2id = {m: i for i, m in enumerate(measures)}

    for pos_tag in cue_statistics['pos_tags']:
        print("%s" % pos_tag)

        correlations, pos_data = compute_correlations(cue_statistics, measure2id, cues, pos=pos_tag)
        print_correlations(correlations, measure2id)
