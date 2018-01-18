__author__ = 'GCassani'

import os
import json
import argparse
import numpy as np
from collections import Counter
from time import strftime


"""
This module uses the Rescorla-Wagner update rule to incrementally update a matrix of cue-outcome associations. This is
implemented in a vectorized implementation that relies on NumPy arrays and runs much faster. Furthermore, this module 
provides basic tools to analyse the matrix of associations. This module consists of the following functions, listed 
together with a short description of what they do. Check the documentation of each function for details about input 
arguments and output structures.

- norm                              : computes the p-norm of the vectors in a matrix, where p is a positive integer; it
                                        works on both rows and columns, and allows to specify which vectors to consider
- median_absolute_deviations        : computes the Median Absolute Deviation of the vectors in a matrix; it works on
                                        both rows and columns, and allows to specify the vectors to consider
- activations                       : computes the total activation of a vector, over the specified dimensions; when
                                        operating on columns, it sums activations over the specified rows, and vice
                                        versa
- frequency                         : computes cue or outcome frequency counts over the input corpus
- get_cues_and_outcomes             : maps each cue to an numerical index, maps each outcome to a numerical index.
                                        Indices the row (cues) and column (outcomes) corresponding to each cue/outcome
- file_exist                        : check whether files storing the desired matrix of associations already exists
- ndl_load                          : loads information contained in files when file_exist returns True
- ndl_compute                       : estimates the cue-outcome associations when file_exist returns False
- discriminative_learner_vectorized : the perceptron that learns the cue-outcome association matrix (this implementation
                                        uses NumPy arrays)
- main                              : a function that runs the discriminative learner (in the chosen version) when the
                                        module is called from command line

"""


def norm(weight_matrix, indices, axis=0, p=1):

    """
    :param weight_matrix:   a NumPy matrix
    :param indices:         a vector of numerical indices indicating which rows or columns to consider
    :param axis:            0 for column-wise vector norms, 1 for row-wise vector norms
    :param p:               1 to get the absolute length of the vector; 2 to get its Euclidean length
    :return vector_norms:   the p-norm of the vectors, computed according to the specification of p

    The function computes the vector norms from the input matrix, according to the order specified by the parameter p,
    along the dimension specified by the parameter axis. Indices is an array-like structure that can operate on rows or
    columns, depending on the value passed to the argument axis:
    - if axis=0, i.e. norms are computed for column vectors, indices is interpreted as indicating the rows to be
        considered in the computation of the column vector norms. Vector norms are computed for all the columns, but
        only considering the rows whose indices are specified in the input vector.
    - if axis=1, i.e. norms are computed for row vectors, indices is interpreted as indicating the columns to be
        considered in the computation of the row vector norms. Vector norms are computed for all the rows, but only
        considering the columns whose indices are specified in the input vector.
        
    """

    if axis == 0:
        vector_norms = np.linalg.norm(weight_matrix[indices, :], ord=p, axis=axis)
    else:
        vector_norms = np.linalg.norm(weight_matrix[:, indices], ord=p, axis=axis)
    return vector_norms


########################################################################################################################


def median_absolute_deviation(weight_matrix, indices, axis=0):

    """
    :param weight_matrix:   a NumPy matrix
    :param indices:         a vector of numerical indices indicating which rows or columns to consider
    :param axis:            0 for column-wise MADs, 1 for row-wise MADs
    :return med_abs_dev:    an array of MAD values computed over the specified dimension for the specified subset of
                            rows/columns

    The function computes the Median Absolute Deviations (MADs) from the input matrix, along the dimension specified by
    the parameter axis. Indices is an array-like structure that can operate on rows or columns, depending on the value
    passed to the argument axis:
    - if axis=0, i.e. MADs are computed for column vectors, indices is interpreted as indicating the rows to be
        considered in the computation of the column vector MADs. MADs are computed for all column vectors, but only
        considering the rows whose indices are specified in the input vector.
    - if axis=1, i.e. MADs are computed for row vectors, indices is interpreted as indicating the columns to be
        considered in the computation of the row vector MADs. MADs are computed for all the rows, but only considering 
        the columns whose indices are specified in the input vector.
    """

    if axis == 0:
        median = np.median(weight_matrix[indices, :], axis=axis)
        med_abs_dev = np.median(np.absolute(weight_matrix[indices, :] - median), axis=axis)
    else:
        median = np.median(weight_matrix[:, indices], axis=axis)
        med_abs_dev = np.median(np.absolute(weight_matrix[:, indices] - np.vstack(median)), axis=axis)

    return med_abs_dev


########################################################################################################################


def activations(weight_matrix, indices, axis=0):

    """
    :param weight_matrix:   a NumPy matrix
    :param indices:         a vector of numerical indices indicating which rows or columns to consider
    :param axis:            0 for column-wise summed activations, 1 for row-wise summed activations
    :return alphas:         a vector of activation values computed over the specified dimension for the specified subset
                            of rows/columns

    The function computes the summed activation values from the input matrix, along the dimension specified by
    the parameter axis. Indices is an array-like structure that can operate on rows or columns, depending on the value
    passed to the argument axis:
    - if axis=0, i.e. activation values are computed over column vectors, indices is interpreted as indicating the rows
        to be considered in the computation of the column activation values. Activation values are computed for all 
        column vectors, but only considering the rows whose indices are specified in the input vector.
    - if axis=1, i.e. activation values are computed over row vectors, indices is interpreted as indicating the columns
        to be considered in the computation of the row activation values. Activation values are computed for all 
        row vectors, but only considering the columns whose indices are specified in the input vector.
    """
    if axis == 0:
        alphas = np.sum(weight_matrix[indices, :], axis=axis)
    else:
        alphas = np.sum(weight_matrix[:, indices], axis=axis)

    return alphas


########################################################################################################################


def frequency(corpus_file, target):

    """
    :param corpus_file:     a string specifying the path to the corpus to be used as input: the file is assumed to be a
                            json file consisting of two lists of lists, the first containing cues and the second
                            outcomes. Each list consists of lists, one for each learning event.
    :param target:          a string specifying whether a frequency list should be derived for cues ('cues') or outcomes
                            ('outcomes'); any other value will give an error.
    :return frequencies:    a dictionary where strings (cues or outcomes) are used as keys and the number of utterances
                            they occur in as values (it slightly differs from raw frequency counts because even if a cue
                            or outcome occurs more than once in a sentence, its frequency count is only updated once
    """

    corpus = json.load(open(corpus_file, 'r+'))

    frequencies = Counter()

    for i in range(len(corpus[0])):

        if target == 'cues':
            cues = set(corpus[0][i])
            for target_cue in cues:
                frequencies[target_cue] += 1
        elif target == 'outcomes':
            outcomes = set(corpus[1][i])
            for target_outcome in outcomes:
                frequencies[target_outcome] += 1
        else:
            ValueError("Please specify the target items to be counted: either 'cues' or 'outcomes'.")

    return frequencies


########################################################################################################################


def get_cues_and_outcomes(input_file):

    """
    :param input_file:      a string indicating the path to the corpus to be considered: it is assumed to be a .json
                            file consisting of two lists of lists: the first encodes learning events into their
                            consistuent phonetic cues, the second encodes the same learning events into their meaning
                            units. Each learning event is a list nested in the two main lists.
                            outcomes; each of the two may consists of multiple, comma-separated strings
    :return cue2ids:        a dictionary mapping each of the strings found in the cues fields to a numerical index
    :return outcome2ids:    a dictionary mapping each of the strings found in the outcomes fields to a numerical index
    """

    outcomes = set()
    cues = set()

    corpus = json.load(open(input_file, 'r+'))

    for i in range(len(corpus[0])):
        trial_cues = set(corpus[0][i])
        cues.update(trial_cues)
        trial_outcomes = set(corpus[1][i])
        outcomes.update(trial_outcomes)

    cues2ids = {k: idx for idx, k in enumerate(cues)}
    outcomes2ids = {k: idx for idx, k in enumerate(outcomes)}

    return cues2ids, outcomes2ids


########################################################################################################################


def file_exists(output_files):

    """
    :param output_files:    a Python dictionary mapping numerical indices to file paths
    :return:                a boolean, True only if all paths in the input dictionary point to an existing file, False
                            otherwise, i.e. even if only one file from the input dictionary doesn't exist
    """

    for k in output_files:
        if not os.path.isfile(output_files[k]):
            return False

    return True


########################################################################################################################


def ndl_load(output_files, weight_matrices):

    """
    :param output_files:        a Python dictionary mapping numerical indices to strings
    :param weight_matrices:     a Python dictionary
    :return weight_matrices:    a Python dictionary mapping numerical indices to large NumPy arrays containing
                                cue-outcome associations computed using the Rescorla-Wagner model of learning: cues are
                                rows and columns are outcomes
    :return cues2ids:           a Python dictionary mapping cues to row indices in the weight matrix
    :return outcomes2ids:       a Python dictionary mapping outcomes to column indices in the weight matrix
    """

    filename, ext = os.path.splitext(os.path.basename(output_files[100]))
    dirname = os.path.dirname(output_files[100])
    cue_file = dirname + '/' + '_'.join(filename.split('_')[:-2]) + '_cueIDs.json'
    cues2ids = json.load(open(cue_file, "r"))
    outcome_file = dirname + '/' + '_'.join(filename.split('_')[:-2]) + '_outcomeIDs.json'
    outcomes2ids = json.load(open(outcome_file, "r"))

    for k in output_files:
        weight_matrices[k] = np.load(output_files[k])

    return weight_matrices, cues2ids, outcomes2ids


########################################################################################################################


def ndl_compute(input_file, alpha, beta, lam, indices, weight_matrices):

    """
    :param input_file:          the path to a a .json file consisting of two lists of lists, the first containing
                                phonetic cues and the second containing outcome meanings; each list contains as many
                                lists as there are learning events in the input corpus (be them full utterances, single
                                words, or any intermediate representation derived from transcribed Child-directed
                                speech). The first list from the list of cue representations matches the first list from
                                the list of meaning representations, both encoding the two layers of the first learning
                                event in the corpus
    :param alpha:               cue salience. For simplicity, we assume that every cue has the same salience, so
                                changing the value of this parameter does not affect the relative strength of
                                cue-outcome associations but only their absolute magnitude.
    :param beta:                learning rate. Again, we make the simplifying assumption that our simulated learners are
                                equally affected by positive and negative feedback. Changing the beta value can have a
                                significant impact on learning outcome, but 0.1 is a standard choice for this model.
    :param lam:                 maximum amount of association that an outcome can receive from all the cues. It simply
                                acts as a scaling factor, so changing its value has the same effects of changing alpha.
    :param indices:             a list of numbers indicating when to store the matrix of associations to file. The
                                numbers indicate percentages of the input corpus.
    :param weight_matrices:     a Python dictionary mapping numerical indices indicating the percentage of learning
                                trials from the input corpus used to NumPy arrays containing cue-outcome associations
                                computed using the Rescorla-Wagner model of learning: cues are rows and columns are
                                outcomes. If the longitudinal parameter is set to False, the dictionary contains one
                                index, 100, and one NumPy array containing cue-outcome associations estimated over the
                                full corpus. If the longitudinal parameter is set to True, the dictionary contains 20
                                indices and as many NumPy arrays, each estimated on an increasing number of learning
                                trials from the input corpus (5%, 10%, 15%, and so on)
    """

    # get two dictionaries mapping each cue and each outcome from the input corpus to a unique numerical index
    cues2ids, outcomes2ids = get_cues_and_outcomes(input_file)
    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": number of cues and outcomes in the input corpus estimated.")
    print()

    # create an empty matrix with as many rows as there are cues and as many columns as there are outcomes in
    # input corpus. The indices extracted before will point to a row for cues and to a column for outcomes
    weight_matrix = np.zeros((len(cues2ids), len(outcomes2ids)))

    # compute the learning rate once and for all, since alpha doesn't change and beta is constant for all cues
    learning_rate = alpha * beta

    print(strftime("%Y-%m-%d %H:%M:%S") + ": started estimating the cue-outcome associations.")

    corpus = json.load(open(input_file, 'r+'))
    # get the total number of learning trials and the line indexes corresponding to each 5% of the corpus to
    # print the advance in processing the input corpus each time an additional 5% of the learning trials is
    # processed
    total_utterances = len(corpus[0])
    check_points = {int(np.floor(total_utterances / 100 * n)): n for n in indices}

    for i in range(len(corpus[0])):
        # get the cues and outcomes in the learning trial
        trial_cues = set(corpus[0][i])
        trial_outcomes = set(corpus[1][i])

        # Create a masking vector for outcomes: the vector contains 0s for all outcomes that don't occur in the
        # learning trial and the lambda value for all outcomes that don't.
        outcome_mask = np.zeros(len(outcomes2ids))
        for outcome in trial_outcomes:
            outcome_mask[outcomes2ids[outcome]] = lam

        # create a masking vector for the cues: this vector contains as many elements as there are cues in the
        # learning trial. If a cue occurs more than once, its corresponding index will appear more than once
        cue_mask = []
        for cue in trial_cues:
            cue_mask.append(cues2ids[cue])

        # compute the total activation for each outcome given the cues in the current learning trial. In order
        # to select the cues that are present in the learning trial - and only those - the cue masking vector is
        # used: it subsets the weight matrix using the indices appended to it, and a row is considered as many
        # times as its corresponding index occurs in the current trial. Then, a sum is performed column-wise
        # returning the total activation for all outcomes. The total activation for unknown outcomes, those that
        # are yet to be experienced, will be 0.
        total_v = np.sum(weight_matrix[cue_mask], axis=0)
        if total_v[total_v > lam].any():
            print(strftime("%Y-%m-%d %H:%M:%S") + ": Something went wrong with utterance number %d:" % i)
            print(trial_cues, trial_outcomes)
            print('The total amount of activation for the current learning instance exceeded the '
                  'chosen value for the lambda parameter: the amount of activation was set to lambda.')
            total_v[total_v > lam] = lam

        # compute the change in activation for each outcome using the outcome masking vector (that has a value
        # of 0 in correspondence of all absent outcomes and a value of lambda in correspondence of all present
        # outcomes). Given that yet to be experienced outcomes have a total activation of 0 and a lambda value
        # of 0, no change in association is needed for cue-outcome associations involving these outcomes. On the
        # contrary, known but not present outcomes have a lambda value of 0 but a total activation higher or
        # lower, resulting in a change of association.
        delta_a = (outcome_mask - total_v) * learning_rate

        # get rid of duplicates in the cue masking vector since cue associations are only updated once,
        # regardless of how many times a cue occurs in the current trial. Then sum the vector of changes in
        # association to the weight matrix: each value in delta_a is summed to all values in the corresponding
        # column of the weight_matrix.
        cue_mask = list(set(cue_mask))
        weight_matrix[cue_mask] += delta_a

        # for every additional 5% of processed learning trials, print to console the progress made by the
        # function
        if i+1 in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") + ": %d%% of the input corpus has been processed."
                  % check_points[i+1])
            weight_matrices[check_points[i+1]] = weight_matrix

    return weight_matrices, cues2ids, outcomes2ids


########################################################################################################################


def discriminative_learner(input_file, alpha=0.01, beta=0.01, lam=1.0, longitudinal=False):

    """
    :param input_file:          the path to a a .json file consisting of two lists of lists, the first containing
                                phonetic cues and the second containing outcome meanings; each list contains as many
                                lists as there are learning events in the input corpus (be them full utterances, single
                                words, or any intermediate representation derived from transcribed Child-directed
                                speech). The first list from the list of cue representations matches the first list from
                                the list of meaning representations, both encoding the two layers of the first learning
                                event in the corpus
    :param alpha:               cue salience. For simplicity, we assume that every cue has the same salience, so
                                changing the value of this parameter does not affect the relative strength of
                                cue-outcome associations but only their absolute magnitude.
    :param beta:                learning rate. Again, we make the simplifying assumption that our simulated learners are
                                equally affected by positive and negative feedback. Changing the beta value can have a
                                significant impact on learning outcome, but 0.1 is a standard choice for this model.
    :param lam:                 maximum amount of association that an outcome can receive from all the cues. It simply
                                acts as a scaling factor, so changing its value has the same effects of changing alpha.
    :param longitudinal:        a boolean specifying whether to adopt a longitudinal design and store association
                                matrices at every 5% of the data, to be able to analyze the time course of learning
    :return weight_matrices:    a Python dictionary mapping numerical indices indicating the percentage of learning
                                trials from the input corpus used to NumPy arrays containing cue-outcome associations
                                computed using the Rescorla-Wagner model of learning: cues are rows and columns are
                                outcomes. If the longitudinal parameter is set to False, the dictionary contains one
                                index, 100, and one NumPy array containing cue-outcome associations estimated over the
                                full corpus. If the longitudinal parameter is set to True, the dictionary contains 20
                                indices and as many NumPy arrays, each estimated on an increasing number of learning
                                trials from the input corpus (5%, 10%, 15%, and so on)
    :return cues2ids:           a Python dictionary mapping cues to row indices in the weight matrix
    :return outcomes2ids:       a Python dictionary mapping outcomes to column indices in the weight matrix

    This function implements Naive Discriminative Learning (NDL, see Baayen, Milin, Durdevic, Hendrix, Marelli (2011)
    for a detailed description of the model and its theoretical background). This learner uses the Rescorla-Wagner
    equations to update cue-outcome associations: it is a simple associative network with no hidden layer, that
    incrementally updates associations between cues and outcomes.

    It runs in linear time on the length of the input in number of utterances, thus if it takes 1 minute to process 1k
    utterances, it'll take 2 minutes to process 2k utterances. Moreover, the runtime also depends on the number of cues
    in which each utterances is encoded, thus the linearity is not perfect: if the first 1k utterances are longer and
    contain more cues, it'll take a bit more than a minute, while if the second 1k utterances are shorter and contain
    fewer cues, it'll take slightly less than a minute to process them.
    In details, it takes ~14 minutes to process ~550k utterances using the configuration with triphones only to encode
    input utterances, using a 2x Intel Xeon 6-Core E5-2603v3 with 2x6 cores and 2x128 Gb of RAM.
    """

    weight_matrices = {}
    output_files = {}
    file_path = os.path.splitext(input_file)[0]
    if longitudinal:
        indices = np.linspace(5, 100, 20)
    else:
        indices = [100]

    # derive the filename for the output file(s)
    for idx in indices:
        output_files[idx] = file_path + '_associationMatrix_' + str(int(idx)) + '.npy'

    # check if all the files already exist; if they do, load them, together with the corresponding .json files
    # containing cue and outcome indices; if they don't, compute the matrix of cue-outcome associations and store it to
    # file, together with the mapping between cues and row indices, and between outcomes and column indices.
    # If the longitudinal parameter is set to true, do this for all time points
    exist = file_exists(output_files)
    if exist:
        weight_matrices, cues2ids, outcomes2ids = ndl_load(output_files, weight_matrices)
        print("The matrix of weights for the desired model already exists and has been loaded.")

    else:
        weight_matrices, cues2ids, outcomes2ids = ndl_compute(input_file, alpha, beta, lam, indices, weight_matrices)

        print(strftime("%Y-%m-%d %H:%M:%S") + ": I finished estimating the cue-outcome associations.")

        # save the weights matrix using the save method for NumPy arrays and the dictionaries mapping cues and
        # outcomes to their row and column indices respectively to two different .json files

        for k in weight_matrices:
            if os.path.exists(output_files[k]):
                print("The file %s already exists." % output_files[k])
            else:
                np.save(output_files[k], weight_matrices[k])

        cues_indices = file_path + '_cueIDs.json'
        json.dump(cues2ids, open(cues_indices, 'w'))
        outcome_indices = file_path + '_outcomeIDs.json'
        json.dump(outcomes2ids, open(outcome_indices, 'w'))

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished writing to files.")

    return weight_matrices, cues2ids, outcomes2ids


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Process arguments to create Celex dictionary.')

    parser.add_argument("-i", "--input_file", required=True, dest="in_file",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-L", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to work in a longitudinal design or not (default: False).")
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01,
                        help="Specify the value of the alpha parameter.")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01,
                        help="Specify the value of the beta parameter.")
    parser.add_argument("-l", "--lambda", dest="lam", default=1.0,
                        help="Specify the value of the lambda parameter.")

    args = parser.parse_args()

    if not os.path.exists(args.in_file) or not args.in_file.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    discriminative_learner(args.in_file, alpha=float(args.alpha), beta=float(args.beta), lam=float(args.lam),
                           longitudinal=args.longitudinal)

########################################################################################################################


if __name__ == '__main__':

    main()
