__author__ = 'GCassani'

import operator
import numpy as np
from collections import Counter
from matrix.matrix import group_outcomes
import phonological_bootstrapping.clustering.similarities as sim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _fit_k(matrix, targets, how_many=100):

    """
    :param matrix:      a NumPy 2d array with observations to be classified as rows and dimensions to use for
                        classification as columns
    :param targets:     a list containing the correct classes of the items to be categorized, in the same order as the
                        rows of the input matrix
    :param how_many:    how many dimensions of the input matrix to use to evaluate classification on a subset of it
    :return:            the accuracy of the LDA classifier using the full matrix and the subset based on the value of
                        the parameter how_many
    """

    # create a LDA classifier
    clf = LinearDiscriminantAnalysis()

    # compute the variance of each column, indicating how much the association values of a cue vary over the tokens
    # then take the ones with the highest variance (in the desired number)
    col_vars = np.var(matrix, axis=0)
    useful_dimensions = np.argpartition(col_vars, -how_many)[-how_many:]

    # fit a LDA classifier using the whole cue-outcome matrix, get its accuracy
    clf.fit(matrix, targets)
    accuracy = clf.score(matrix, targets)

    # fit a LDA classifier only using the cues with the highest variance, the get its accuracy
    matrix_subset = matrix[:, useful_dimensions]
    clf.fit(matrix_subset, targets)
    accuracy_subset = clf.score(matrix_subset, targets)

    return accuracy, accuracy_subset, matrix_subset


########################################################################################################################


def _fit_t(matrix, targets, threshold=0.00005):

    """
    :param matrix:          a NumPy 2d array with observations to be classified as rows and dimensions to use for
                            classification as columns
    :param targets:         a list containing the correct classes of the items to be categorized, in the same order as
                            the rows of the input matrix
    :param threshold:       the minimum variance a dimension needs to have to be considered
    :return:                the accuracy of the LDA classifier using the full matrix, the subset based on the value
                            of the parameter threshold_cues, and the number of useful cues and useful tokens given the
                            thresholds
    """

    # create a LDA classifier
    clf = LinearDiscriminantAnalysis()

    # compute the variance of each column, indicating how much the association values of a cue vary over the tokens
    # then take the ones with the highest variance (higher than the threshold)
    col_vars = np.var(matrix, axis=0)
    useful_dimensions = np.where(col_vars > threshold)[0]

    # fit a LDA classifier using the whole cue-outcome matrix, get its accuracy
    clf.fit(matrix, targets)
    accuracy = clf.score(matrix, targets)

    # fit a LDA classifier only using the cues with the highest variance, the get its accuracy
    matrix_subset = matrix[:, useful_dimensions]
    clf.fit(matrix_subset, targets)
    accuracy_subset = clf.score(matrix_subset, targets)

    return accuracy, accuracy_subset, matrix_subset, len(useful_dimensions)


########################################################################################################################


def compute_majority_baseline(target_outcomes):

    """
    :param target_outcomes:     a dictionary mapping tokens to numerical indices; tokens consist of a string followed
                                by the corresponding PoS tag, separated by a pipe symbol ('|')
    :return:                    the list of PoS tags of the target outcomes and the accuracy that would be obtained by
                                classifying all tokens with the most frequent tag in the set
    """

    target_tags = []
    for outcome, ii in sorted(target_outcomes.items(), key=operator.itemgetter(1)):
        target_tags.append(outcome.split('|')[1])
    baseline = Counter(target_tags).most_common(1)[0][1] / len(target_tags)

    return target_tags, baseline


########################################################################################################################


def check_row_variances(matrix):

    """
    :param matrix:  a NumPy 2d array
    :return:        the input array without any row that has variance 0 and the indices of rows that have variance 0
    """

    # compute the variance for each observation and remove those that have no variance over the dimensions of the
    # subset matrix
    row_vars = np.var(matrix, axis=1)
    exclude = np.where(row_vars == 0)[0]
    matrix = matrix[np.nonzero(row_vars)[0], :]

    return matrix, exclude


########################################################################################################################


def check_col_variances(matrix):

    """
    :param matrix:  a NumPy 2d array
    :return:        the input array without any column that has variance 0 and the indices of columns that have
                    variance 0
    """

    # compute the variance for each column and remove those that have no variance over the rows of the
    # subset matrix
    col_vars = np.var(matrix, axis=0)
    exclude = np.where(col_vars == 0)[0]
    matrix = matrix[:, np.nonzero(col_vars)[0]]

    return matrix, exclude


########################################################################################################################


def realign_targets(target_outcomes, exclude):

    """
    :param target_outcomes: a dictionary mapping tokens to numerical indices, with tokens consisting of the word form
                            and the PoS tag, separated by a pipe symbol ('|')
    :param exclude:         an iterable containing indices indicating tokens with 0 variance in the matrix of
                            cue-outcome associations
    :return:                a dictionary mapping tokens to numerical indices, in the same form of the input, but
                            without all words whose index is in exclude. Indices are realigned considering those that
                            were removed: suppose that exclude contains indices 350 and 512 with target_outcomes
                            containing 600 tokens. In the output dictionary, the words that were mapped to indices 350
                            and 512 have been deleted, and the tokens that mapped to 351 and 513 in the input now map
                            to 350 and 511 respectively, while the token which mapped to 600 now maps to 598.
    """

    targets = {}
    ii = 0
    for outcome, jj in sorted(target_outcomes.items(), key=operator.itemgetter(1)):
        if jj not in exclude:
            targets[outcome] = ii
            ii += 1

    return targets


########################################################################################################################


def subset_experiment(associations, target_outcomes, how_many_cues=100, how_many_tokens=100):

    """
    :param associations:        the cue-outcome association matrix (cues are rows, outcomes are columns)
    :param target_outcomes:     a dictionary mapping outcomes to their indices
    :param how_many_cues:       the number of phonological cues to consider as dimensions for the restricted LDA that
                                attempts to classify tokens based on their activation patterns over phonological cues
    :param how_many_tokens:     the number od tokens to consider as dimensions for the restricted LDA that attempts to
                                classify tokens based on their correlation with other tokens, with correlation being
                                computed over the activations over phonological cues
    :return:                    the accuracy on the full LDA on the cue-outcome matrix
                                the accuracy on the restricted LDA on the subset cue-outcome matrix
                                the majority baseline on the cue-outcome matrix
                                the accuracy on the full LDA on the outcome-outcome correlation matrix
                                the accuracy on the restricted LDA on the subset outcome-outcome correlation matrix
                                the majority baseline on the outcome-outcome correlation matrix
    """

    target_tags, phon_baseline = compute_majority_baseline(target_outcomes)

    # filter the outcomes that are not among the targets and re-group tokens so that tokens from a same lexical
    # category have closer indices, then transpose the matrix to have cues as columns and tokens as rows
    associations, target_outcomes = group_outcomes(associations, target_outcomes)
    associations = associations.T
    rows, cols = associations.shape
    k = cols if how_many_cues > cols else how_many_cues
    phon_accuracy, phon_accuracy_subset, associations_subset = _fit_k(associations, target_tags, how_many=k)
    print("The LDA on cue-outcome activations has been completed.")

    # get rid of observation with variance 0 over the dimensions in the subset association matrix
    associations_subset, exclude = check_row_variances(associations_subset)
    if exclude.size:
        target_outcomes = realign_targets(target_outcomes, exclude)
        target_tags, distr_baseline = compute_majority_baseline(target_outcomes)
    else:
        distr_baseline = phon_baseline

    # use the trimmed matrix to compute token to token correlations, get the tokens with the highest variances
    similarities = sim.pairwise_corr(associations_subset, target='rows')
    if np.isnan(similarities).any():
        similarities = np.nan_to_num(similarities)

    rows, cols = similarities.shape
    k = cols if how_many_tokens > cols else how_many_tokens
    distr_accuracy, distr_accuracy_subset, similarity_subset = _fit_k(similarities, target_tags, how_many=k)
    print("The LDA on outcome-outcome correlation similarities has been completed.")

    return phon_accuracy, phon_accuracy_subset, phon_baseline, distr_accuracy, distr_accuracy_subset, distr_baseline


########################################################################################################################


def threshold_experiment(associations, target_outcomes, cues_threshold=0.00005, tokens_threshold=0.0008):

    """
    :param associations:        the cue-outcome association matrix (cues are rows, outcomes are columns)
    :param target_outcomes:     a dictionary mapping outcomes to their indices
    :param cues_threshold:      the minimum variance of a cue to be considered as a relevant dimension for the
                                restricted LDA run on the cue-outcome association matrix
    :param tokens_threshold:    the minimum variance of a token to be considered as a relevant dimension for the
                                restricted LDA run on the outcome-outcome correlation matrix
    :return:
    """

    target_tags, phon_baseline = compute_majority_baseline(target_outcomes)

    # filter the outcomes that are not among the targets and re-group tokens so that tokens from a same lexical
    # category have closer indices, then transpose the matrix to have cues as columns and tokens as rows
    associations, target_outcomes = group_outcomes(associations, target_outcomes)
    associations = associations.T
    phon_accuracy, phon_accuracy_subset, associations_subset, useful_cues = _fit_t(associations, target_tags,
                                                                                   threshold=cues_threshold)
    print("The LDA on cue-outcome activations has been completed.")

    # get rid of observation with variance 0 over the dimensions in the subset association matrix
    associations_subset, exclude = check_row_variances(associations_subset)
    if exclude.size:
        target_outcomes = realign_targets(target_outcomes, exclude)
        target_tags, distr_baseline = compute_majority_baseline(target_outcomes)
    else:
        distr_baseline = phon_baseline

    # use the trimmed matrix to compute token to token correlations, get the tokens with the highest variances
    similarities = sim.pairwise_corr(associations_subset, target='rows')
    if np.isnan(similarities).any():
        similarities = np.nan_to_num(similarities)

    distr_accuracy, distr_accuracy_subset, similarity_subset, useful_tokens = _fit_t(similarities, target_tags,
                                                                                     threshold=tokens_threshold)
    print("The LDA on outcome-outcome correlation similarities has been completed.")

    return phon_accuracy, phon_accuracy_subset, phon_baseline, useful_cues, \
           distr_accuracy, distr_accuracy_subset, distr_baseline, useful_tokens
