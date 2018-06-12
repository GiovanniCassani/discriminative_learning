__author__ = 'GCassani'

"""Functions to load and handle activation matrices"""

import os
import json
import operator
import numpy as np


def load(filename):

    """
    :param filename:        a string indicating the path to a .npy file containing a NumPy array
    :return weight_matrix:  the NumPy array loaded from the input file
    :return cue_ids:        a dictionary mapping strings to row indices: the data are loaded automatically and the
                            dictionary has as many entries as there are rows in the weight_matrix
    :return outcome_ids:    a dictionary mapping strings to column indices: the data are loaded automatically and the
                            dictionary has as many entries as there are columns in the weight_matrix
    """

    d = os.path.dirname(filename)
    cue_file = os.path.join(d, "cueIDs.json")
    outcome_file = os.path.join(d, "outcomeIDs.json")

    cue_ids = json.load(open(cue_file, "r"))
    outcome_ids = json.load(open(outcome_file, "r"))
    weight_matrix = np.load(filename)

    return weight_matrix, cue_ids, outcome_ids


########################################################################################################################


def rearrange(matrix, f, axis='r', reverse=False):

    """
    :param matrix:      a numpy array containing numerical values
    :param f:           a function operating on numerical values whose outcome is used to rearrange the desired matrix
                        dimensions
    :param axis:        a string specifying whether rows ('r') or columns ('c') are to be rearranged; any other value
                        will cause an error.
    :param reverse:     a boolean specifying whether to sort in ascending (False) or descending (True) order - default
                        is ascending.
    :return matrix:     a numpy array having the same dimensionality of the input ones, with either rows or columns
                        rearranged according to the output of the specified function in the desired order
    """

    outcome = f(matrix, axis=axis)

    sorted_ids = [i[0] for i in sorted(outcome.items(), key=operator.itemgetter(1), reverse=reverse)]

    if axis == 'r':
        matrix = matrix[sorted_ids, :]
    elif axis == 'c':
        matrix = matrix[:, sorted_ids]
    else:
        ValueError("Please specify the axis on which marginals are computed: either 'r' or 'c'.")

    return matrix


########################################################################################################################


def group_outcomes(matrix, outcomes2ids):

    """
    :param matrix:          a NumPy 2d array
    :param outcomes2ids:    a dictionary mapping strings to column indices of the input matrix. Each string consists of
                            a word and a part of speech tag separated by a pipe symbol ('|')
    :return: sorted_matrix: the input matrix, with columns regrouped so that words from a same pos tag are net to each
                            other
    :return outcomes2ids:   a dictionary mapping outcomes from the input dictionary to the corresponding columns of the
                            output matrix
    """

    # sort outcomes according to their PoS
    sorted_by_pos = sorted(outcomes2ids.items(), key=lambda tup: tup[0].split('|')[1])
    # get the column indices of the outcomes keeping the new order
    sorted_indices = [ii for outcome, ii in sorted_by_pos]
    # reorder the columns in the input matrix using the sorted indices: the column corresponding to the first index
    # will be the first of the sorted_matrix
    sorted_matrix = matrix[:, sorted_indices]
    # map outcomes to their new column indices, since the outcome at the first column in the original matrix does not
    # point to the first matrix anymore
    outcomes2ids = {}
    for idx, outcome in enumerate(sorted_by_pos):
        outcomes2ids[outcome[0]] = idx

    return sorted_matrix, outcomes2ids
