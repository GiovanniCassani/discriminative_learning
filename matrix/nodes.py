__author__ = 'GCassani'

"""Functions to re-arrange information about cues and outcomes to improve diagnostics and visualization,
   sorting them according to a given measure, such as frequency, entropy, activation, or MAD"""

import os
import operator


def rank_nodes(dict1, dict2, filter_vec=None):

    """
    :param dict1:       a python dictionary
    :param dict2:       a python dictionary whose values are used to access keys from dict1
    :param filter_vec:  default None, meaning that all keys from the argument mapping are evaluated.
                        If an iterable is passed, this must contain items that are also values of the mapping
                        dictionary (the requirement that its values are also keys of the nodes dictionary always holds)
    :return ranked:     a Python dictionary whose keys are the keys from dict2 and whose values are the values from
                        dict1. This dictionary is ranked according to the values.

    EXAMPLE:
    dict1 contains numerical indices as keys mapping to entropy values.
    dict2 contains strings indicating words as keys mapping to numerical indices referring to row or column indices in
    an activation matrix
    filter_vec contains a subset of all the numerical indices from dict1.
    ranks maps those strings from dict2 that match indices from filter_vec to entropy values corresponding to the same
    indices in dict1. Indices are not returned: however, if a filter is passed the indices in it are known already, and
    if no filter is passed then all indices are evaluated.
    """

    ranked = {}

    reversed_mapping = dict(zip(dict2.values(), dict2.keys()))

    if filter_vec:
        for k in filter_vec:
            name = reversed_mapping[k]
            ranked[name] = dict1[k]
    else:
        for k in dict1:
            name = reversed_mapping[k]
            ranked[name] = dict1[k]

    ranked = sorted(ranked.items(), key=operator.itemgetter(1), reverse=True)

    return ranked


########################################################################################################################


def write_ranked_nodes(values, ids, plots_folder, name):

    """
    :param values:          a dictionary mapping numerical indices to numerical values
    :param ids:             a dictionary mapping numerical indices to strings
    :param plots_folder:    a string indicating the path to a folder
    :param name:            a string indicating the name of the function used to generate the numerical values in the
                            values dictionary
    :return ranked:         a Python dictionary whose keys are strings from the ids dictionary and whose values are the
                            values from the values dictionary. This dictionary is ranked according to the values.

    Rank columns according to their value and print to file, specifying which function generated the values used to
    generate the ranking.
    """

    ranked = rank_nodes(values, ids)
    ranked_path = os.path.join(plots_folder, ".".join(["_".join([name, 'list']), 'txt']))
    for el in ranked:
        with open(ranked_path, 'a+') as f:
            f.write("\t".join([el[0], str(el[1])]))
            f.write("\n")

    return ranked
