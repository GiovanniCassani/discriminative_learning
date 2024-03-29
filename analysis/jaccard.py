__author__ = 'GCassani'

"""Function to compute the Jaccard distance between lexical items and the top active cues in the activation matrix
   estimated with the ndl function given the test lexical item"""

import os
import operator
import numpy as np
from time import strftime
from analysis.plot import plot_ranks
from corpus.encode.item import encode_item
from celex.utilities.dictionaries import tokens2ids
from corpus.encode.words.phonology import get_phonological_form


def get_top_active_cues(weight_matrix, column, n, reversed_cue_ids):

    """
    :param weight_matrix:       a NumPy array, containing numerical values
    :param column:              an integer specifying which column from the input array needs to be considered
    :param n:                   an integer specifying how many elements (i.e. row indices) from the column vector are
                                considered
    :param reversed_cue_ids:    a dictionary mapping row indices to strings
    :return top_active_cues:    a set of strings, including all the strings from the input dictionary matching the row
                                indices selected from the specified column in the input NumPy array
    """

    top_active_cues_ids = set()
    top_active_cues = set()

    # sort all values in the relevant column in descending order, to get most active cues first
    sorted_cue_activations = np.argsort(weight_matrix[:, column])[::-1]

    # consider the n top active cues but keep adding cues if their activation values is the same as the n-th cue:
    # this has the purpose of avoiding that one cue out of many gets selected as the n-th out of other criteria
    # also make sure that cues have higher than 0 activations
    for i in range(len(sorted_cue_activations)):
        if weight_matrix[sorted_cue_activations[i], column] > 0:
            if i < n:
                top_active_cues_ids.add(sorted_cue_activations[i])
            else:
                if weight_matrix[sorted_cue_activations[i], column] == \
                        weight_matrix[sorted_cue_activations[i - 1], column]:
                    top_active_cues_ids.add(sorted_cue_activations[i])
                else:
                    break
        else:
            break

    for identifier in top_active_cues_ids:
        try:
            top_active_cues.add(reversed_cue_ids[identifier])
        except KeyError:
            continue

    return top_active_cues


########################################################################################################################


def jaccard(weight_matrix, cues2ids, outcomes2ids, celex_dict, plots_folder='',
            stress_marker=True, uniphone=False, diphone=False, triphone=True, syllable=False, boundaries=True):

    """
    :param weight_matrix:           the matrix of cue-outcome association estimated using the ndl model
    :param cues2ids:                a dictionary mapping strings to row indices in the weight_matrix
    :param outcomes2ids:            a dictionary mapping strings to column indices in the weight_matrix
    :param celex_dict:              the dictionary extracted from the celex database
    :param plots_folder:            a string indicating the path to a folder, where all the plots and files generated by
                                    the function will be stored. The function checks if the folder already exists, and
                                    if it doesn't the function creates it
    :param uniphone:                a boolean indicating whether single phonemes are to be considered while encoding
                                    column identifiers
    :param diphone:                 a boolean indicating whether sequences of two phonemes are to be considered while
                                    encoding column identifiers
    :param triphone:                a boolean indicating whether sequences of three phonemes are to be considered while
                                    encoding column identifiers
    :param syllable:                a boolean indicating whether syllables are to be considered while encoding column
                                    identifiers
    :param stress_marker:           a boolean indicating whether stress markers from the phonological representations of
                                    Celex need to be preserved or can be discarded
    :param boundaries:              a boolean specifying whether to consider or not word boundaries
    :return jaccard_coefficients:   a dictionary mapping outcome surface forms (strings) to the Jaccard coefficient
                                    computed between the gold-standard and most active cues as estimated from the input
                                    matrix. Gold-standard cues are extracted from the outcome phonological form
                                    according the specified encoding; moreover, a vector of length k (where k is the
                                    number of gold-standard cues) is filled with the top k cues for the outcome being
                                    considered, looking at raw activation values. `The Jaccard coefficient is the
                                    proportion between the intersection of the two vectors and their union, telling how
                                    many cues are shared proportionally to how many unique cues there are. The higher
                                    the number, the higher the overlap and the better the network was able to
                                    discriminate the good cues for an outcome.
    """

    # specify the string that identifies plots generated by this function in their file names
    f_name = 'jaccard'

    token_indices = tokens2ids(celex_dict)

    jaccard_coefficients = {}
    true_cues = {}
    active_cues = {}
    ids2cues = dict(zip(cues2ids.values(), cues2ids.keys()))
    total_items = len(outcomes2ids)
    check_points = {int(np.floor(total_items / 100 * n)): n for n in np.linspace(5, 100, 20)}

    # consider each outcome separately
    for idx, outcome in enumerate(outcomes2ids):

        column_id = outcomes2ids[outcome]
        wordform, pos = outcome.split('|')
        celex_entry = (wordform, pos, wordform)
        word_phon = get_phonological_form(celex_entry, celex_dict, token_indices)
        if boundaries:
            word_phon = '+' + word_phon + '+'

        if isinstance(word_phon, str):

            # get the relevant phonological cues
            nphones = encode_item(word_phon, stress_marker=stress_marker, uniphones=uniphone,
                                  diphones=diphone, triphones=triphone, syllables=syllable)

            # get the top active phonological cues from the input association matrix given the outcome being considered
            top_active_cues = get_top_active_cues(weight_matrix, column_id, len(nphones), ids2cues)

            # compute the Jaccard coefficient and store correct and predicted cues for every outcome
            set_inters = len(set.intersection(top_active_cues, set(nphones)))
            set_union = len(set.union(top_active_cues, set(nphones)))
            jaccard_coefficients[outcome] = set_inters / set_union
            true_cues[outcome] = nphones
            active_cues[outcome] = top_active_cues

        if idx+1 in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": %d%% of the outcomes have been processed to estimate the Jaccard coefficient."
                  % check_points[idx+1])

    if plots_folder:

        # check whether the provided folder path points to an existing folder,
        # and create it if it doesn't already exist
        if not os.path.isdir(plots_folder):
            os.makedirs(plots_folder)

        ranked_path = os.path.join(plots_folder, '.'.join(["_".join([f_name, 'list']), 'txt']))
        sorted_coeffs = sorted(jaccard_coefficients.items(), key=operator.itemgetter(1), reverse=True)

        scatter_path = os.path.join(plots_folder, ".".join(["_".join([f_name, 'scatter']), 'pdf']))
        plot_ranks(sorted_coeffs, output_path=scatter_path, yname='Jaccard coeff',
                   xname='Rank', figname='Jaccard coefficient for each outcome')

        # write to file each outcome together with the correct and predicted phonological cues
        with open(ranked_path, 'a+') as f_name:
            for outcome in sorted_coeffs:
                outcome = outcome[0]
                jaccard_coeff = outcome[1]
                true = true_cues[outcome]
                top_active = active_cues[outcome]
                f_name.write("\t".join([outcome, str(jaccard_coeff), str(true), str(top_active)]))
                f_name.write("\n")

    return jaccard_coefficients
