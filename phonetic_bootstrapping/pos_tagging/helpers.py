__author__ = 'GCassani'

"""Functions that process test items and activation matrices to make categorization possible"""

import operator
import numpy as np
from collections import defaultdict
from scipy.stats import contingency
from matrix.statistics import activations


def dict2numpy(dict1, dict2):

    """
    :param dict1:   a dictionary with numerical values
    :param dict2:   a dictionary with numerical values
    :return a:      a 2-by-n numpy array, where n is equal to the length of the union of the keys of the two input
                    dictionaries, that contains the values from the first dictionary in the first row and the values
                    from the second dictionary in the second row, with columns corresponding to keys. If a key is not
                    present in a dictionary, the corresponding value is assumed to be 0
    :return ids:    a dictionary mapping column indices in the numpy array to the keys of the two input dictionaries;
                    indices are 0 based
    """

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    keys = keys1.union(keys2)
    ids = defaultdict(str)
    a = np.zeros(shape=(2, len(keys)))

    for i, key in enumerate(keys):
        ids[i] = key
        if key in keys1:
            a[0, i] = dict1[key]
            if key in keys2:
                a[1, i] = dict2[key]
            else:
                a[1, i] = 0
        else:
            a[0, i] = 0

    return a, ids


########################################################################################################################


def std_res(observed, expected):

    """
    :param observed:    an 2-by-n numpy array containing the observed frequencies
    :param expected:    an 2-by-n numpy array containing the expected frequencies under the null hypothesis
    :return res:        the standardized Pearson's residuals indicating which dimensions in the observed data show
                        the stronger deviation from the expected frequencies
    """

    n = observed.sum()
    rsum, csum = contingency.margins(observed)
    v = csum * rsum * (n - rsum) * (n - csum) / float(n ** 3)
    res = (observed - expected) / np.sqrt(v)

    return res


########################################################################################################################


def sort_lexical_nodes_from_matrix(nphones, associations_matrix, cues, outcomes, to_filter):

    """
    :param nphones:             an iterable containing strings. Each string represent a phoneme sequence.
    :param associations_matrix: a NumPy array, with cues as rows and outcomes as columns.
    :param cues:                a dictionary mapping cues to their respective row indices. Keys are strings, values are
                                integers.
    :param outcomes:            a dictionary mapping outcomes to their respective column indices. Keys are strings,
                                values are integers.
    :param to_filter:           an iterable containing strings indicating prohibited outcomes, i.e. outcomes that
                                should not be considered; if empty, all outcomes are considered
    :return sorted_nodes:       a list of ordered tuples, whose first element is a string and second element is a
                                number. The string is an outcome from the associations matrix and the number is the sum
                                of all activations involving the word and all the input cues (n-phones). The ordering is
                                done according to the second element in the tuples, i.e. the number: this means that the
                                first tuple in the output list will contain the word that received the highest amount of
                                activation given the input n-phones.
    """

    # get the row indices of all the input nphones (duplicates are counted as many times as they occur)
    # if an input cue didn't appear during training, go ahead and ignore it
    cue_mask = []
    for cue in nphones:
        try:
            cue_mask.append(cues[cue])
        except KeyError:
            pass

    # get the summed activation for each outcome given the active n-phones from the input
    alphas = activations(associations_matrix, cue_mask)

    # reverse the input dictionary providing outcome to column index mapping and get the column index to outcome mapping
    # this is needed because we want to know which outcome does the i-th column correspond to
    ids = dict(zip(outcomes.values(), outcomes.keys()))

    # zip together outcomes and their respective total activations, matching on indices, and then sort the resulting
    # array, which turns it automatically into a list. If an outcome is in the list of items to filter out, do not
    # include it in the final sorted list
    outcomes_alphas = []
    for i in range(alphas.shape[0]):
        if ids[i] not in to_filter:
            outcomes_alphas.append((ids[i], alphas[i]))
    dtype = [('word', 'S50'), ('total_v', float)]
    outcomes_alphas = np.sort(np.array(outcomes_alphas, dtype=dtype), order='total_v')
    sorted_nodes = sorted(outcomes_alphas, key=operator.itemgetter(1), reverse=True)

    return sorted_nodes


########################################################################################################################


def differences(dict1, dict2):

    """
    :param dict1:   a dictionary with numerical values
    :param dict2:   a dictionary with numerical values
    :return diff:   a dictionary using as keys the union of the keys from the input dictionaries and as values the
                    difference between the values for a same key in the two input dictionary, doing dict1 - dict2.
                    The function handles keys missing from one of the two dictionaries assuming missing keys have a
                    value of 0 (zero).
    """

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    keys = keys1.union(keys2)
    diff = defaultdict(float)

    # loop through all the keys from the two dictionaries, if a key exists in both, then subtract the value in the
    # second dictionary from the value in the first dictionary for the same key; if the key only exists in the first
    # dictionary, simply assign that value to the dictionary of differences; if the key only exists in the second
    # dictionary, take its negative, since it boils down to 0 minus the value
    for key in keys:
        if key in keys1:
            if key in keys2:
                diff[key] = dict1[key] - dict2[key]
            else:
                diff[key] = dict1[key]
        else:
            diff[key] = -dict2[key]

    return diff
