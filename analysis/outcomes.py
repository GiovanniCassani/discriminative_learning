__author__ = 'GCassani'

"""Function to compute diagnostic measures about lexical outcomes from the phonetic bootstrapping experiment"""

import os
from collections import defaultdict
from matrix.nodes import write_ranked_nodes
from analysis.plot import plot_ranks
from phonological_bootstrapping.helpers import store_dict
from matrix.statistics import median_absolute_deviation, activations, norm


def outcome_measures(weight_matrix, col_ids, plots_folder):

    """
    :param weight_matrix:   array-like structure
    :param col_ids:         a dictionary mapping column numerical indices to strings
    :param plots_folder:    the path to the folder where plots and .txt files are created
    :return outcome_values: a dictionary of dictionaries, where each inner dictionary maps outcome strings to
                            corresponding values, here MADs and total activation values
    """

    measures = ['MAD', 'activations', '1-norm', '2-norm']
    outcome_values = defaultdict(dict)

    # check whether the provided folder path points to an existing folder, and create it if it doesn't already exist
    # then checks that the path ends with a slash, and add one if it doesn't
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)

    # use all rows when computing MADs
    indices = range(weight_matrix.shape[0])

    # get the median absolute deviation for each column over all rows and store values in a dictionary
    # using columns indices as keys.
    med_abs_dev = median_absolute_deviation(weight_matrix, indices)
    mad_values = store_dict(med_abs_dev)

    # get the 1- and 2-norm for each column vector over all rows and store values in two dictionaries using columns
    # indices as keys.
    norms1 = norm(weight_matrix, indices)
    norm1_values = store_dict(norms1)
    norms2 = norm(weight_matrix, indices, p=2)
    norm2_values = store_dict(norms2)

    # compute total activation values for each column vector
    cue_indices = range(weight_matrix.shape[0])
    activation_values = activations(weight_matrix, cue_indices)
    outcome_activations = store_dict(activation_values)

    for fun in measures:

        if fun == 'MAD':
            values = mad_values
        elif fun == 'activations':
            values = outcome_activations
        elif fun == '1-norm':
            values = norm1_values
        else:
            values = norm2_values

        ranked = write_ranked_nodes(values, col_ids, plots_folder, fun)

        # plot values of each column against the rank of the column according to its  value
        scatter_path = os.path.join(plots_folder, ".".join(["_".join([fun, 'scatter']), 'pdf']))
        plot_ranks(ranked, output_path=scatter_path, yname=fun,
                   xname='Rank', figname=fun)

        reversed_mapping = dict(zip(col_ids.values(), col_ids.keys()))
        outcome_dict = {}
        for idx in values:
            string_form = reversed_mapping[idx]
            outcome_dict[string_form] = values[idx]

        outcome_values[fun] = outcome_dict

    return outcome_values
