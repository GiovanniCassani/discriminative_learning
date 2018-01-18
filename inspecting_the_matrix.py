__author__ = 'GCassani'

import json
import os
import operator
import argparse
import ndl
import numpy as np
import matplotlib.pyplot as plt
import preprocess_corpus as prc
import celex_processing as clx
from phonetic_bootstrapping import store_dict
from collections import defaultdict
from time import strftime

"""
All the functions in this module assume that the cue-outcome matrix of associations has been estimated using the
vectorized implementation of the NDL. Thus, it assumes a .npy matrix and two .json dictionaries containing row and
column indices. If you estimated the cue-outcome associations with the standard NDL implementation, you should write
your own function to convert them to the compatible format, or re-run the estimation process using the vectorized
implementation - which is always recommended because it works in linear time and is neater.
This module offers several routines to inspect the matrix of associations and evaluate what has been learned, what
properties of the input might have driven the process and in general it helps to make sense of the matrix itself. This
module consists of the following functions, listed together with a short description of what they do. Check the
documentation of each function for details about input arguments and output structures.

- write_ranked_nodes    : write the information contained in a dictionary mapping numerical indices to numerical values
                            to a file
- plot_matrix           : plots a confusion matrix
- plot_ranks            : plots a rank scatterplot given an array of tuples where the second element is a number
- load_matrix           : loads the NumPy matrix of associations, together with the json files with the cue and outcome
                            indices indicating corresponding rows and columns in the matrix
- category filter       : gives the column indices of outcomes belonging to a certain PoS tag
- rearrange             : rearrange the rows/columns in a matrix by sorting them according to the value of a chosen
                            function, that is passed as input
- rank_nodes            : returns a Python dictionary whose keys are the keys from the second input dictionary and whose
                            values are the values from the first input dictionary, ranking key:value pairs according to
                            the value; in essence it maps outcome strings to real values, using outcome column indices
                            to establish the mapping
- encode_column_ids     : gives back the nphones of which outcome strings, identified by their column indices,
                            consist of
- get_top_active_cues   : gives a set of strings, including all the strings from the input dictionary that match the row
                            indices selected from the specified column in the input NumPy matrix
- outcome_measures      : a function that computes several measures concerning column vectors in the matrix
                            of associations
- cue_measures          : a function that computes several measures concerning row vectors in the matrix of
                            associations, separately for each subset of outcome belonging to a same PoS tag
- jaccard               : a function that computes the Jaccard coefficients between the gold standard set of nphones of
                            which an outcome consist of and the nphones that are most active in the matrix of
                            associations for the same outcome
- check_input_arguments : runs a bunch of checks on the arguments passed as input and raises errors when appropriate
- main                  : a function that calls outcome_measures, cue_measures, and jaccard when the module is called
                            from command line
"""


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
    ranked_path = plots_folder + "_".join([name, 'list']) + '.txt'
    for el in ranked:
        with open(ranked_path, 'a+') as f:
            f.write("\t".join([el[0], str(el[1])]))
            f.write("\n")

    return ranked


########################################################################################################################


def plot_matrix(weight_matrix, figname='Figure title', output_path=''):

    """
    :param weight_matrix:   a NumPy array
    :param figname:         a string indicating the plot title - default is 'Figure title'
    :param output_path:     a string indicating where to save the plot. If no path is provided (default), the plot is
                            shown in the current window
    """

    fig = plt.figure()
    im = plt.imshow(weight_matrix, aspect='auto', interpolation='nearest')
    fig.colorbar(im)
    plt.xlabel('Outcomes')
    plt.ylabel('Cues')
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')

    plt.title(figname)

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


########################################################################################################################


def plot_ranks(l, xname='IV', yname='DV', figname='Figure title', output_path=''):

    """
    :param l:               a list of tuples, where the second element of each tuple is numerical
    :param xname:           a string indicating the label of the x axis - default is IV (Independent Variable)
    :param yname:           a string indicating the label of the y axis - default is DV (Dependent Variable)
    :param figname:         a string indicating the plot title - default is 'Figure title'
    :param output_path:     a string indicating where to save the plot. If no path is provided (default), the plot is
                            shown in the current window
    """

    x_vec = np.linspace(1, len(l), len(l))
    y_vec = np.zeros((len(l)))
    for i in range(len(l)):
        y_vec[i] = l[i][1]

    fig = plt.figure()
    plt.scatter(x_vec, y_vec, alpha=0.75)
    axes = plt.gca()
    axes.set_xlim([0, len(l) + 1])
    ymin = y_vec.min() - y_vec.min() / float(10)
    ymax = y_vec.max() + y_vec.max() / float(10)
    axes.set_ylim([ymin, ymax])
    plt.title(figname)
    plt.xlabel(xname)
    plt.ylabel(yname)

    plt.tick_params(
        axis='both',
        which='minor',
        bottom='off',
        top='off',
        left='off',
        right='off')

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


########################################################################################################################


def load_matrix(filename):

    """
    :param filename:        a string indicating the path to a .npy file containing a NumPy array
    :return weight_matrix:  the NumPy array loaded from the input file
    :return cue_ids:        a dictionary mapping strings to row indices: the data are loaded automatically and the
                            dictionary has as many entries as there are rows in the weight_matrix
    :return outcome_ids:    a dictionary mapping strings to column indices: the data are loaded automatically and the
                            dictionary has as many entries as there are columns in the weight_matrix
    """

    d = os.path.dirname(filename)
    f = os.path.splitext(os.path.basename(filename))[0]
    e = 'json'

    # first strip the bit of the filename after the last underscore, substitute it with wither 'cueIDs' or 'outcomeIDs'
    # depending on the desired file, glue back together using underscores, then join the complete filename and the new
    # extension, and finally glue together the path to the filename (now complete of its extension
    cue_file = "/".join([d, ".".join(["_".join(["_".join(f.split("_")[:-1]), "cueIDs"]), e])])
    outcome_file = "/".join([d, ".".join(["_".join(["_".join(f.split("_")[:-1]), "outcomeIDs"]), e])])

    cue_ids = json.load(open(cue_file, "r"))
    outcome_ids = json.load(open(outcome_file, "r"))
    weight_matrix = np.load(filename)

    return weight_matrix, cue_ids, outcome_ids


########################################################################################################################


def category_filter(input_dict, target):

    """
    :param input_dict:          a dictionary mapping strings to column indices in the weight_matrix
    :param target:              a string indicating the PoS tag of interest. Keys in the dictionary are filtered
                                according to the PoS tag they belong to (marked as a capital letter at the end of the
                                string, after a tilde ('~') that separates it from the word form)
    :return mask:               a list containing the indices matching the target PoS tag
    """

    mask = []
    for k in input_dict:
        category = k.split("~")[-1]
        if category == target:
            mask.append(input_dict[k])

    return mask


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
    dict1 contains numerical indices as keys and entropies as values.
    dict2 contains strings indicating words as keys and numerical indices as values.
    filter_vec contains a subset of all the numerical indices from dict1.
    ranks maps those strings from dict2 that match indices from filter_vec to entropy values corresponding to the same
    indices in dict1. Indices are not returned: however, if a filter is passed the filters are known already, and if no
    filter is passed then all indices are evaluated.
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


def encode_column_ids(word_phon, uniphone=True, diphone=False, triphone=False, syllable=False, stress_marker=True):

    """
    :param word_phon:       a string corresponding to the phonological transcription of a word extracted from the CELEX
                            database, encoded using the DISC character set
    :param uniphone:        a boolean indicating whether single phonemes are to be considered while encoding column
                            identifiers
    :param diphone:         a boolean indicating whether sequences of two phonemes are to be considered while encoding
                            column identifiers
    :param triphone:        a boolean indicating whether sequences of three phonemes are to be considered while encoding
                            column identifiers
    :param syllable:        a boolean indicating whether syllables are to be considered while encoding column
                            identifiers
    :param stress_marker:   a boolean indicating whether stress markers from the phonological representations of Celex
                            need to be preserved or can be discarded
    :return nphones:        a set of strings, each being a sub-string extracted from the input string, of the specified
                            length (determined by the value of the input boolean parameters
    """

    vowels = clx.vowels()

    word_phon = word_phon.decode("utf-8").encode("utf-8")
    word_phon = '+' + word_phon.translate(None, "\"") + '+'

    uniphones = []
    diphones = []
    triphones = []
    syllables = []

    table = str.maketrans(dict.fromkeys("'-"))
    if stress_marker:
        word_phon = prc.recode_stress(word_phon, vowels)
    else:
        word_phon = word_phon.translate(table)
    if syllable:
        syllables = word_phon.split("-")
    word_phon = word_phon.translate(table)

    if uniphone:
        uniphones = prc.get_nphones(word_phon, n=1)
    if diphone:
        diphones = prc.get_nphones(word_phon, n=2)
    if triphone:
        triphones = prc.get_nphones(word_phon, n=3)
    nphones = set(uniphones + diphones + triphones + syllables)

    return nphones


########################################################################################################################


def get_top_active_cues(weight_matrix, col, n, reversed_cue_ids):

    """
    :param weight_matrix:       a NumPy array, containing numerical values
    :param col:                 an integer specifying which column from the input array needs to be considered
    :param n:                   an integer specifying how many elements (i.e. row indices) from the column vector are
                                considered
    :param reversed_cue_ids:    a dictionary mapping row indices to strings
    :return top_active_cues:    a set of strings, including all the strings from the input dictionary matching the row
                                indices selected from the specified column in the input NumPy array
    """

    top_active_cues_ids = set()
    top_active_cues = set()

    # sort column values in descending order, to get most active cues first
    sorted_col = np.argsort(weight_matrix[:, col])[::-1]

    # consider the n top active cues but keep adding cues if their activation values is the same as the n-th cue:
    # this has the purpose of avoiding that one cue out of many gets selected as the n-th out of other criteria
    for i in range(len(sorted_col)):
        if i < n:
            top_active_cues_ids.add(sorted_col[i])
        else:
            if weight_matrix[sorted_col[i], col] == weight_matrix[sorted_col[i-1], col]:
                top_active_cues_ids.add(sorted_col[i])
            else:
                break

    for identifier in top_active_cues_ids:
        try:
            top_active_cues.add(reversed_cue_ids[identifier].decode("utf-8").encode("utf-8"))
        except KeyError:
            continue

    return top_active_cues


########################################################################################################################


def outcome_measures(weight_matrix, col_ids, plots_folder):

    """
    :param weight_matrix:   array-like structure
    :param col_ids:         a dictionary mapping column numerical indices to strings
    :param plots_folder:    the path to the folder where plots and .txt files are created
    :return outcome_values: a dictionary of dictionaries, where each inner dictionary maps outcome strings to
                            corresponding values, here MADs and total activation values
    """

    functions = ['outcomeMAD', 'outcomeAct', 'outcome1norm', 'outcome2norm']
    outcome_values = defaultdict(dict)

    # check whether the provided folder path points to an existing folder, and create it if it doesn't already exist
    # then checks that the path ends with a slash, and add one if it doesn't
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)
    if not plots_folder.endswith("/"):
        plots_folder += "/"

    # use all rows when computing MADs
    indices = range(weight_matrix.shape[0])

    # get the median absolute deviation for each column over all rows and store values in a dictionary
    # using columns indices as keys.
    med_abs_dev = ndl.median_absolute_deviation(weight_matrix, indices)
    mad_values = store_dict(med_abs_dev)

    # get the 1- and 2-norm for each column vector over all rows and store values in two dictionaries using columns
    # indices as keys.
    norms1 = ndl.norm(weight_matrix, indices)
    norm1_values = store_dict(norms1)
    norms2 = ndl.norm(weight_matrix, indices, p=2)
    norm2_values = store_dict(norms2)

    # compute total activation values for each column vector
    cue_indices = range(weight_matrix.shape[0])
    activation_values = ndl.activations(weight_matrix, cue_indices)
    outcome_activations = store_dict(activation_values)

    # get the total activation for an outcome over all cues

    for fun in functions:

        if fun == 'outcomeMAD':
            values = mad_values
        elif fun == 'outcomeAct':
            values = outcome_activations
        elif fun == 'outcome1norm':
            values = norm1_values
        else:
            values = norm2_values

        ranked = write_ranked_nodes(values, col_ids, plots_folder, fun)

        # plot values of each column against the rank of the column according to its  value
        scatter_path = plots_folder + "_".join([fun, 'scatter']) + '.pdf'
        plot_ranks(ranked, output_path=scatter_path, yname=fun,
                   xname='Rank', figname=fun)

        reversed_mapping = dict(zip(col_ids.values(), col_ids.keys()))
        outcome_dict = {}
        for idx in values:
            string_form = reversed_mapping[idx]
            outcome_dict[string_form] = values[idx]

        outcome_values[fun] = outcome_dict

    return outcome_values


########################################################################################################################


def cue_measures(weight_matrix, row_ids, col_ids, plots_folder):

    """
    :param weight_matrix:   array-like structure
    :param row_ids:         a dictionary mapping row numerical indices to strings
    :param col_ids:         a dictionary mapping column numerical indices to strings
    :param plots_folder:    the path to the folder where plots and .txt files are created
    :return cue_mads:       a dictionary of dictionaries, where first-level keys are strings identifying PoS tags and
                            second order keys are outcome strings belonging to each PoS tag. Values are MAD values for
                            each outcome string under a specific PoS tag.
    """

    # check whether the provided folder path points to an existing folder, and create it if it doesn't already exist
    # then checks that the path ends with a slash, and add one if it doesn't
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)
    if not plots_folder.endswith("/"):
        plots_folder += "/"

    cue_values = defaultdict(dict)

    functions = ["cueMAD", "cueAct", "cue1norm", "cue2norm"]

    # get all the Part-of-Speech tags from the column identifiers (PoS tags are assumed to be the last element of each
    # outcome identifier, after a tilde ('~')
    pos_tags = {outcome.split("~")[1] for outcome in col_ids}

    # for every PoS tag, get the columns matching outcomes that share a given PoS tag and the column indices that
    # identify them; compute the MAD of each row vector; rank cues according to their MAD values given a specific PoS
    # tag; plot the MAD values against the rank of the cue, separately for each PoS tag
    for pos in pos_tags:

        outcome_indices = category_filter(col_ids, pos)

        # get the MAD value for each row over the columns belonging to a given PoS tag; store values in a dictionary
        # using row indices as keys.
        med_abs_dev = ndl.median_absolute_deviation(weight_matrix, outcome_indices, axis=1)
        pos_mad = store_dict(med_abs_dev)

        # get the 1- and 2-norm value for each row vector over the columns belonging to a given PoS tag; store values in
        # two dictionaries using row indices as keys.
        norms1 = ndl.norm(weight_matrix, outcome_indices, axis=1)
        pos_norm1 = store_dict(norms1)
        norms2 = ndl.norm(weight_matrix, outcome_indices, axis=1, p=2)
        pos_norm2 = store_dict(norms2)

        # sum the total activation of each cue over all the outcomes belonging to a same PoS tag, then average it by
        # the number of outcomes belonging to that PoS tag; store values in a dictionary using row indices as keys
        category_avg_activations = ndl.activations(weight_matrix, outcome_indices, axis=1) / len(outcome_indices)
        pos_avg_act = store_dict(category_avg_activations)

        for fun in functions:

            if fun == 'cueMAD':
                values = pos_mad
            elif fun == 'cue1norm':
                values = pos_norm1
            elif fun == 'cue2norm':
                values = pos_norm2
            else:
                values = pos_avg_act

            name = "_".join([fun, pos])

            # rank rows according to their value of fun computed over the words belonging to the current PoS tag
            ranked = write_ranked_nodes(values, row_ids, plots_folder, name)

            # plot values of each column against the rank of the column according to its  value
            scatter_path = plots_folder + "_".join([name, 'scatter']) + '.pdf'
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


########################################################################################################################


def jaccard(weight_matrix, row_ids, column_ids, celex_dir, plots_folder, reduced=True,
            stress_marker=False, uniphone=True, diphone=False, triphone=False, syllable=False):

    """
    :param weight_matrix:           a NumPy array
    :param row_ids:                 a dictionary mapping strings to row indices in the weight_matrix
    :param column_ids:              a dictionary mapping strings to column indices in the weight_matrix
    :param celex_dir:               a string specifying the path to the folder where the information extracted from the
                                    CELEX database will be stored or is retrieved if already exists
    :param plots_folder:            a string indicating the path to a folder, where all the plots and files generated by
                                    the function will be stored. The function checks if the folder already exists, and
                                    if it doesn't the function creates it
    :param reduced:                 a boolean specifying whether reduce phonological form should be always preferred
                                    when available. This parameter depends on how the training corpus was extracted in
                                    the first place: if the corpus was extracted with reduced variants being considered,
                                    setting this parameter to False here does not make any sense.
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
    f_name = 'Jaccard'

    # check whether the provided folder path points to an existing folder, and create it if it doesn't already exist
    # then checks that the path ends with a slash, and add one if it doesn't
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)
    if not plots_folder.endswith("/"):
        plots_folder += "/"
    if not celex_dir.endswith("/"):
        celex_dir += "/"

    celex_dict = clx.get_celex_dictionary(celex_dir, reduced)

    tokens2ids = prc.get_token_identifier_mappings(celex_dict)

    jaccard_coefficients = {}
    true_cues = {}
    active_cues = {}
    reversed_cue_ids = dict(zip(row_ids.values(), row_ids.keys()))

    # consider each outcome separately
    for el in column_ids:

        # get the column id of the outcome being considered
        col_idx = column_ids[el]

        # get the phonological form of the outcome from CELEX
        wordform, pos = el.split('~')
        outcome = (wordform, pos, wordform)
        word_phon = prc.get_phonological_form(outcome, celex_dict, tokens2ids)

        if isinstance(word_phon, str):

            # get the relevant phonological units from the outcome phonological form
            nphones = encode_column_ids(word_phon, stress_marker=stress_marker, uniphone=uniphone,
                                        diphone=diphone, triphone=triphone, syllable=syllable)

            # get the top active phonological cues from the input association matrix given the outcome being considered
            top_active_cues = get_top_active_cues(weight_matrix, col_idx, len(nphones), reversed_cue_ids)

            # compute the Jaccard coefficient and store correct and predicted cues for every outcome
            inters = len(nphones.intersection(top_active_cues))
            union = float(len(nphones.union(top_active_cues)))
            jaccard_coefficients[el] = inters / union
            true_cues[el] = nphones
            active_cues[el] = top_active_cues

    ranked_path = plots_folder + "_".join([f_name, 'list']) + '.txt'
    sorted_coeffs = sorted(jaccard_coefficients.items(), key=operator.itemgetter(1), reverse=True)

    scatter_path = plots_folder + "_".join([f_name, 'scatter']) + '.pdf'
    plot_ranks(sorted_coeffs, output_path=scatter_path, yname='Jaccard coeff',
               xname='Rank', figname='Jaccard coefficient for each outcome')

    # write to file each outcome together with the correct and predicted phonological cues
    for el in sorted_coeffs:
        outcome = el[0]
        jaccard_coeff = el[1]
        true = true_cues[outcome]
        top_active = active_cues[outcome]
        with open(ranked_path, 'a+') as f_name:
            f_name.write("\t".join([outcome, str(jaccard_coeff), str(true), str(top_active)]))
            f_name.write("\n")

    return jaccard_coefficients


########################################################################################################################


def check_input_arguments(args, parser):

    """
    :param args:    the result of an ArgumentParser structure
    :param parser:  a parser object, created using argparse.ArgumentParser
    """

    # check corpus file
    if not os.path.exists(args.corpus) or not args.corpus.endswith(".json"):
        raise ValueError("There are problems with the corpus file you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    # check association matrix
    if not os.path.exists(args.associations) or not args.associations.endswith(".npy"):
        raise ValueError("There are problems with the association matrix file you provided: either the path does not "
                         "exist or the file extension is not .npy. Provide a valid path to a .npy file.")

    # check extension of the output file for cue frequency counts
    if not args.cue_file.endswith(".txt"):
        raise ValueError("Indicate the path to a .txt file to store cue frequency counts.")

    # check extension of the output file for outcome frequency counts
    if not args.outcome_file.endswith(".txt"):
        raise ValueError("Indicate the path to a .txt file to store outcome frequency counts.")

    # check that at least one phonetic encoding is specified
    if not (args.uni or args.di or args.tri or args.syl):
        parser.error('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    # check CELEX folder
    if not os.path.exists(args.celex_dir):
        raise ValueError("The provided folder does not exist. Provide the path to an existing folder.")

    # check if plot folder exists, and make it in case it does not
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Analyze the structure of an association matrix derived with the NDL.')

    parser.add_argument('-i', '--input_corpus', required=True, dest='corpus',
                        help='Give the path to the input corpus (.json),'
                             'encoded in phonetic cues and lexical outcomes.')
    parser.add_argument('-a', '--association_matrix', required=True, dest='associations',
                        help='Give the path to the cue-outcome associations matrix, computed using the NDL.')
    parser.add_argument('-c', '--cue_file', required=True, dest='cue_file',
                        help='Give the path to a .txt file to store cue frequencies from the input corpus.')
    parser.add_argument('-o', '--outcome_file', required=True, dest='outcome_file',
                        help='Give the path to a .txt file to store outcome frequencies from the input corpus.')
    parser.add_argument('-p', "--plot_path", required=True, dest="plot_path",
                        help="Specify the path to the folder where plots will be stored.")
    parser.add_argument('-C', "--Celex_dir", required=True, dest="celex_dir",
                        help="Specify the path to the folder containing CELEX files or dictionary.")
    parser.add_argument('-r', '--reduced', action='store_true', dest='reduced',
                        help='Specify if reduced phonological forms are to be considered.')
    parser.add_argument('-g', '--grammatical', action='store_true', dest='grammatical',
                        help='Specify if grammatical meanings, e.g. plural, past, are to be considered.')
    parser.add_argument("-u", "--uniphones", action="store_true", dest="uni",
                        help="Specify if uniphones need to be encoded.")
    parser.add_argument("-d", "--diphones", action="store_true", dest="di",
                        help="Specify if diphones need to be encoded.")
    parser.add_argument("-t", "--triphones", action="store_true", dest="tri",
                        help="Specify if triphones need to be encoded.")
    parser.add_argument("-s", "--syllables", action="store_true", dest="syl",
                        help="Specify if syllables need to be encoded.")
    parser.add_argument("-m", "--stress_marker", action="store_true", dest="stress",
                        help="Specify if stress need to be encoded.")

    args = parser.parse_args()

    check_input_arguments(args, parser)

    # make sure that all folder paths end with a slash
    if not args.celex_dir.endswith("/"):
        args.celex_dir += "/"
    if not args.plot_path.endswith("/"):
        args.plot_path += "/"

    # compute and store cue and outcome frequency counts
    cue_freqs = ndl.frequency(args.corpus, 'cues')
    if not os.path.exists(args.cue_file):
        with open(args.cue_file, 'a+') as c_f:
            for item in sorted(cue_freqs.items(), key=operator.itemgetter(1), reverse=True):
                c_f.write("\t".join([item[0], str(item[1])]))
                c_f.write("\n")
    outcome_freqs = ndl.frequency(args.corpus, 'outcomes')
    if not os.path.exists(args.outcome_file):
        with open(args.outcome_file, 'a+') as o_f:
            for item in sorted(outcome_freqs.items(), key=operator.itemgetter(1), reverse=True):
                o_f.write("\t".join([item[0], str(item[1])]))
                o_f.write("\n")
    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished computing cue and outcome frequency counts.")

    # print information about the current input parameters concerning cues and outcomes
    prc.encoding_features(args.corpus, uni_phones=args.uni, di_phones=args.di, tri_phones=args.tri,
                          syllable=args.syl, stress_marker=args.stress,
                          grammatical=args.grammatical)

    # load the matrix of associations, get column and row indices, and
    mat, r_ids, c_ids = load_matrix(args.associations)

    # cue_acts and cue_mads are dictionaries of dictionaries, where each key is a PoS tag, and it is
    # mapped to all cues in a particular model, in turn mapped to the average activation, or median
    # absolute deviation value of each cue, computed separately for each PoS tag over all the outcomes that belong to
    # each PoS tag
    cue_values = cue_measures(mat, r_ids, c_ids, args.plot_path)
    cue_mad = cue_values['cueMAD']
    cue_act = cue_values['cueAct']
    cue_norm1 = cue_values['cue1norm']
    cue_norm2 = cue_values['cue2norm']
    print(strftime("%Y-%m-%d %H:%M:%S") +
          ": Finished computing cue Median Absolute Deviation, average activation and 1-norm values for each PoS tag.")

    # outcome_act, outcome_norm, and outcome_mads are dictionaries mapping outcome surface forms to total activation,
    # vector norm, or median absolute deviation values computed over all cues
    outcome_values = outcome_measures(mat, c_ids, args.plot_path)
    outcome_mad = outcome_values['outcomeMAD']
    outcome_act = outcome_values['outcomeAct']
    outcome_1norm = outcome_values['outcome1norm']
    outcome_2norm = outcome_values['outcome2norm']

    # jaccard_coeff is a dictionary mapping outcome surface forms to the Jaccard coefficient computed between the vector
    # of gold standard cues extracted from the outcome and the vector of the top active cues extracted from the
    # input matrix (see the documentation of the function for further details)
    jaccard_coeff = jaccard(mat, r_ids, c_ids, args.celex_dir, args.plot_path, stress_marker=args.stress,
                            uniphone=args.uni, diphone=args.di, triphone=args.tri, syllable=args.syl)
    print(strftime("%Y-%m-%d %H:%M:%S") +
          ": Finished computing Jaccard coefficients, Median Absolute Deviations and total activation values " +
          "for outcomes, comparing true to top active cues.")
    print()
    print()

    # Compute several correlations among the different measures, for both cues and outcomes
    print("CORRELATIONS:")
    print("\tOutcomes:")

    # compute pairwise correlations between frequency, MAD, activation, and Jaccard coefficients for the outcomes
    # in the input corpus, then print them to standard output
    outcomes = set(outcome_act.keys())
    outc_summary_row_ids = {'freq': 0,
                            'act': 1,
                            'jacc': 2,
                            'mad': 3,
                            '1norm': 4,
                            '2norm': 5}

    outcome_summary = np.zeros([len(outc_summary_row_ids), len(outcomes)])
    # outcome_summary_col_ids = {}

    shared = set(outcome_freqs.keys()) & set(outcome_act.keys()) & set(outcome_mad.keys()) \
             & set(outcome_1norm.keys()) & set(jaccard_coeff.keys())

    for idx, key in enumerate(shared):
        outcome_summary[outc_summary_row_ids['freq'], idx] = outcome_freqs[key]
        outcome_summary[outc_summary_row_ids['act'], idx] = outcome_act[key]
        outcome_summary[outc_summary_row_ids['jacc'], idx] = jaccard_coeff[key]
        outcome_summary[outc_summary_row_ids['mad'], idx] = outcome_mad[key]
        outcome_summary[outc_summary_row_ids['1norm'], idx] = outcome_1norm[key]
        outcome_summary[outc_summary_row_ids['2norm'], idx] = outcome_2norm[key]
        # outcome_summary_col_ids[idx] = key

    correlations = np.corrcoef(outcome_summary)
    print("\tFrequency ~ Activation: %0.4f" % correlations[outc_summary_row_ids['freq'], outc_summary_row_ids['act']])
    print("\tFrequency ~ Jaccard: %0.4f" % correlations[outc_summary_row_ids['freq'], outc_summary_row_ids['jacc']])
    print("\tFrequency ~ MAD: %0.4f" % correlations[outc_summary_row_ids['freq'], outc_summary_row_ids['mad']])
    print("\tFrequency ~ 1-norm: %0.4f" % correlations[outc_summary_row_ids['freq'], outc_summary_row_ids['1norm']])
    print("\tFrequency ~ 2-norm: %0.4f" % correlations[outc_summary_row_ids['freq'], outc_summary_row_ids['2norm']])
    print("\tJaccard ~ Activation: %0.4f" % correlations[outc_summary_row_ids['jacc'], outc_summary_row_ids['act']])
    print("\tJaccard ~ MAD: %0.4f" % correlations[outc_summary_row_ids['jacc'], outc_summary_row_ids['mad']])
    print("\tJaccard ~ 1-norm: %0.4f" % correlations[outc_summary_row_ids['jacc'], outc_summary_row_ids['1norm']])
    print("\tJaccard ~ 2-norm: %0.4f" % correlations[outc_summary_row_ids['jacc'], outc_summary_row_ids['2norm']])
    print("\tActivation ~ MAD: %0.4f" % correlations[outc_summary_row_ids['act'], outc_summary_row_ids['mad']])
    print("\tActivation ~ 1-norm: %0.4f" % correlations[outc_summary_row_ids['act'], outc_summary_row_ids['1norm']])
    print("\tActivation ~ 2-norm: %0.4f" % correlations[outc_summary_row_ids['act'], outc_summary_row_ids['2norm']])
    print("\tMAD ~ 1-norm: %0.4f" % correlations[outc_summary_row_ids['mad'], outc_summary_row_ids['1norm']])
    print("\tMAD ~ 2-norm: %0.4f" % correlations[outc_summary_row_ids['mad'], outc_summary_row_ids['2norm']])
    print("\t1-norm ~ 2-norm: %0.4f" % correlations[outc_summary_row_ids['1norm'], outc_summary_row_ids['2norm']])

    # plot pairwise correlations and save them to file
    subplots = [(0, 0, 'freq', 'act'),
                (1, 0, 'freq', 'jacc'),
                (2, 0, 'freq', 'mad'),
                (3, 0, 'freq', '1norm'),
                (4, 0, 'freq', '2norm'),
                (1, 1, 'act', 'jacc'),
                (2, 1, 'act', 'mad'),
                (3, 1, 'act', '1norm'),
                (4, 1, 'act', '2norm'),
                (2, 2, 'jacc', 'mad'),
                (3, 2, 'jacc', '1norm'),
                (4, 2, 'jacc', '2norm'),
                (3, 3, 'mad', '1norm'),
                (4, 3, 'mad', '2norm'),
                (4, 4, '1norm', '2norm')]

    f_outcome_corr, axarr = plt.subplots(len(outc_summary_row_ids) - 1, len(outc_summary_row_ids) - 1)
    f_outcome_corr.suptitle('Outcomes: correlations')
    for subpl in subplots:
        r, c, x_name, y_name = subpl
        x = outcome_summary[outc_summary_row_ids[x_name]]
        y = outcome_summary[outc_summary_row_ids[y_name]]
        axarr[r, c].scatter(x, y)
        xlow = x.min() - x.min() / float(10)
        xhigh = x.max() + x.max() / float(10)
        axarr[r, c].set_xlim([xlow, xhigh])
        ylow = y.min() - y.min() / float(10)
        yhigh = y.max() + y.max() / float(10)
        axarr[r, c].set_ylim([ylow, yhigh])
        if r == len(outc_summary_row_ids) - 1:
            axarr[r, c].set_xlabel(x_name)
        else:
            axarr[r, c].set_xlabel('')
        if c == 0:
            axarr[r, c].set_ylabel(y_name)
        else:
            axarr[r, c].set_ylabel('')
        axarr[r, c].set_xticklabels([])
        axarr[r, c].set_yticklabels([])

    axarr[0, 1].axis('off')
    axarr[0, 2].axis('off')
    axarr[0, 3].axis('off')
    axarr[0, 4].axis('off')
    axarr[1, 2].axis('off')
    axarr[1, 3].axis('off')
    axarr[1, 4].axis('off')
    axarr[2, 3].axis('off')
    axarr[2, 4].axis('off')
    axarr[3, 4].axis('off')
    f_outcome_corr.savefig(args.plot_path + 'outcome_correlations.pdf')
    plt.close(f_outcome_corr)

    print()
    print("\tCues:")

    # compute pairwise correlations between cue frequency, MAD, and activation values separately for each PoS tag
    # then print to standard output
    cue_summary_row_ids = {'freq': 0,
                           'act': 1,
                           'mad': 2,
                           '1norm': 3,
                           '2norm': 4}

    for pos_tag in cue_act:

        cues = set(cue_act[pos_tag].keys())
        cue_summary = np.zeros([len(cue_summary_row_ids), len(cues)])

        for idx, cue in enumerate(cues):
            cue_summary[cue_summary_row_ids['freq'], idx] = cue_freqs[cue]
            cue_summary[cue_summary_row_ids['mad'], idx] = cue_mad[pos_tag][cue]
            cue_summary[cue_summary_row_ids['act'], idx] = cue_act[pos_tag][cue]
            cue_summary[cue_summary_row_ids['1norm'], idx] = cue_norm1[pos_tag][cue]
            cue_summary[cue_summary_row_ids['2norm'], idx] = cue_norm2[pos_tag][cue]

        correlations = np.corrcoef(cue_summary)

        print("\t%s" % pos_tag)
        print("\t\tFrequency ~ MAD (within class): %0.4f" %
              correlations[cue_summary_row_ids['freq'], cue_summary_row_ids['mad']])
        print("\t\tFrequency ~ Activation (within class): %0.4f" %
              correlations[cue_summary_row_ids['freq'], cue_summary_row_ids['act']])
        print("\t\tFrequency ~ 1-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['freq'], cue_summary_row_ids['1norm']])
        print("\t\tFrequency ~ 2-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['freq'], cue_summary_row_ids['2norm']])
        print("\t\tMAD ~ Activation (within class): %0.4f" %
              correlations[cue_summary_row_ids['mad'], cue_summary_row_ids['act']])
        print("\t\tMAD ~ 1-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['mad'], cue_summary_row_ids['1norm']])
        print("\t\tMAD ~ 2-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['mad'], cue_summary_row_ids['2norm']])
        print("\t\tActivation ~ 1-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['act'], cue_summary_row_ids['1norm']])
        print("\t\tActivation ~ 2-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['act'], cue_summary_row_ids['2norm']])
        print("\t\t1-norm ~ 2-norm (within class): %0.4f" %
              correlations[cue_summary_row_ids['1norm'], cue_summary_row_ids['2norm']])

    print()
    print("#" * 100)


########################################################################################################################


if __name__ == '__main__':

    main()
