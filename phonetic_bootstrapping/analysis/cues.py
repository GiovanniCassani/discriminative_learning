__author__ = 'GCassani'

"""Function to compute diagnostic measures about phonetic cues from the phonological bootstrapping experiment"""

import os
from collections import defaultdict
from matrix.nodes import write_ranked_nodes
from matrix.statistics import median_absolute_deviation, activations, norm
from phonetic_bootstrapping.experiment.helpers import store_dict
from phonetic_bootstrapping.analysis.plot import plot_ranks


def pos_filter(input_dict, target):

    """
    :param input_dict:          a dictionary mapping strings to column indices in the weight_matrix
    :param target:              a string indicating the PoS tag of interest. Keys in the dictionary are filtered
                                according to the PoS tag they belong to (marked as a capital letter at the end of the
                                string, after a tilde ('~') that separates it from the word form)
    :return mask:               a list containing the indices matching the target PoS tag
    """

    mask = []
    for k in input_dict:
        category = k.split("|")[-1]
        if category == target:
            mask.append(input_dict[k])

    return mask


########################################################################################################################


def cue_measures(weight_matrix, row_ids, col_ids, plots_folder):

    """
    :param weight_matrix:   array-like structure
    :param row_ids:         a dictionary mapping row numerical indices to strings
    :param col_ids:         a dictionary mapping column numerical indices to strings
    :param plots_folder:    the path to the folder where plots and .txt files are created
    :return cue_values:     a dictionary of dictionaries, where first-level keys are strings identifying PoS tags and
                            second order keys are outcome strings belonging to each PoS tag.
    """

    # check whether the provided folder path points to an existing folder, and create it if it doesn't already exist
    # then checks that the path ends with a slash, and add one if it doesn't
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)

    cue_values = defaultdict(dict)
    measures = ["MAD", "activations", "1-norm", "2-norm"]

    # get all the Part-of-Speech tags from the column identifiers (PoS tags are assumed to be the last element of each
    # outcome identifier, after a vertical bar ('|')
    pos_tags = {outcome.split("|")[1] for outcome in col_ids}
    cue_values['pos_tags'] = pos_tags

    # for every PoS tag, get the columns matching outcomes that share a given PoS tag and the column indices that
    # identify them; compute the MAD of each row vector; rank cues according to their MAD values given a specific PoS
    # tag; plot the MAD values against the rank of the cue, separately for each PoS tag
    for pos in pos_tags:

        outcome_indices = pos_filter(col_ids, pos)

        # get the MAD value for each row over the columns belonging to a given PoS tag; store values in a dictionary
        # using row indices as keys.
        med_abs_dev = median_absolute_deviation(weight_matrix, outcome_indices, axis=1)
        pos_mad = store_dict(med_abs_dev)

        # get the 1- and 2-norm value for each row vector over the columns belonging to a given PoS tag; store values in
        # two dictionaries using row indices as keys.
        norms1 = norm(weight_matrix, outcome_indices, axis=1)
        pos_norm1 = store_dict(norms1)
        norms2 = norm(weight_matrix, outcome_indices, axis=1, p=2)
        pos_norm2 = store_dict(norms2)

        # sum the total activation of each cue over all the outcomes belonging to a same PoS tag, then average it by
        # the number of outcomes belonging to that PoS tag; store values in a dictionary using row indices as keys
        category_avg_activations = activations(weight_matrix, outcome_indices, axis=1) / len(outcome_indices)
        pos_avg_act = store_dict(category_avg_activations)

        for fun in measures:

            if fun == 'MAD':
                values = pos_mad
            elif fun == '1-norm':
                values = pos_norm1
            elif fun == '2-norm':
                values = pos_norm2
            else:
                values = pos_avg_act

            name = "_".join([fun, pos])

            # rank rows according to their value of fun computed over the words belonging to the current PoS tag
            ranked = write_ranked_nodes(values, row_ids, plots_folder, name)

            # plot values of each column against the rank of the column according to its  value
            scatter_path = os.path.join(plots_folder, ".".join(["_".join([name, 'scatter']), 'pdf']))
            plot_ranks(ranked, output_path=scatter_path, yname=fun,
                       xname='Rank', figname=fun)

            # flip the row identifiers dictionary so that strings are keys and indices are values
            # then map MAD/activation values to strings for a specific PoS tag
            # finally create a key in the output dictionary for the PoS being considered, nested within a dictionary
            # specifying the function being computed,
            # finally store the dictionary with MAD/activation values as value of the PoS key under the function dict
            reversed_mapping = dict(zip(row_ids.values(), row_ids.keys()))
            pos_values = {}
            for cue_id in values:
                cue_string = reversed_mapping[cue_id]
                pos_values[cue_string] = values[cue_id]
            cue_values[fun][pos] = pos_values

    return cue_values
