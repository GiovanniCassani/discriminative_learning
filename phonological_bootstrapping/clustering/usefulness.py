_author__ = 'GCassani'

"""Functions to quantify the usefulness of phonological cues and tokens, using variance as a proxy of usefulness,
   indicating the amount of information carried by each dimension"""

import json
import numpy as np
import progressbar
from time import strftime
from matrix.matrix import group_outcomes
from collections import defaultdict, Counter
import phonological_bootstrapping.clustering.similarities as sim
from phonological_bootstrapping.clustering.lda import check_col_variances, realign_targets


def recursive_defaultdict():

    return defaultdict(recursive_defaultdict)


########################################################################################################################


def _update_dict(target1, target2, dictionary):

    """
    :param target1:     a string
    :param target2:     another string
    :param dictionary:  a dictionary of dictionaries where target1 is a first-level key and target2 is a
                        second-level key
    """

    if target2 in dictionary[target1]:
        dictionary[target1][target2] += 1
    else:
        dictionary[target1][target2] = 1


########################################################################################################################


def _get_avg_conditional_probability(target, co_occurrences, frequencies, reverse=False):

    """
    :param target:          a string which is a key in the co_occurrences dictionary
    :param co_occurrences:  a dictionary mapping strings to other strings and the count of how often the two strings
                            co-occurred in the corpus
    :param frequencies:     a dictionary mapping strings to their frequency count
    :param reverse:         a boolean indicating whether to compute the conditional probability given the target and not
                            of the target (default to False)
    :return:                the average conditional probability of the target string given all the co-occurring strings
                            (unless the parameter Reverse is set to True, in which case the average conditional
                            probability of the co-occurring elements given the target is returned)
    """

    conditional_probabilities = []

    if len(co_occurrences[target]):
        for k in co_occurrences[target]:
            f = frequencies[target] if reverse else frequencies[k]
            cp = co_occurrences[target][k] / f
            if cp > 1:
                conditional_probabilities.append(1)
            else:
                conditional_probabilities.append(cp)
        avg = sum(conditional_probabilities) / len(conditional_probabilities)

    else:
        avg = np.nan

    return avg


########################################################################################################################


def get_cue_variances(association_matrix, target_outcomes):

    """
    :param association_matrix:  the matrix of associations estimated using the Rescorla-Wagner update rule
    :param target_outcomes:     a dictionary mapping strings to column indices of the input matrix. Each string
                                consists of a word and a part of speech tag separated by a pipe symbol ('|')
    :return row_variances:      a NumPy array containing the row variances
    :return association_matrix: the input matrix, with only the relevant columns
    """

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started computing row variances...")
    # compute the variances over rows, i.e. the variance of phonological cues
    association_matrix, target_outcomes = group_outcomes(association_matrix, target_outcomes)
    row_variances = np.var(association_matrix, axis=1)
    print("Row variances: ", type(row_variances), row_variances.shape)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished computing row_variances")
    print()

    return row_variances, association_matrix, target_outcomes


########################################################################################################################


def get_token_variances(association_matrix, target_tokens, row_variances, how_many_cues=1000):

    """
    :param association_matrix:  the matrix of associations estimated using the Rescorla-Wagner update rule
    :param target_tokens:       a dictionary mapping strings to column indices of the input matrix. Each string
                                consists of a word and a part of speech tag separated by a pipe symbol ('|')
    :param row_variances:       a NumPy array with the row variances
    :param how_many_cues:       the number of rows with highest variance to use to subset the input association matrix
    :return token_variances:    a dictionary mapping each target outcome to its variance over the token-to-token
                                pairwise correlations estimated from the input association_matrix
    """

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started subsetting the association matrix...")
    # get the dimensionality of the matrix and check that the desired number of phonological cues is not larger than
    # the dimensionality of the matrix, then get the k dimensions with largest variance
    rows, cols = association_matrix.shape
    k = rows if how_many_cues > rows else how_many_cues
    useful_dimensions = np.argpartition(row_variances, -k)[-k:]
    matrix_subset = association_matrix[useful_dimensions, :]
    print("Matrix subset: ", type(matrix_subset), matrix_subset.shape)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished subsetting the association matrix.")
    print()

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started computing correlation similarity matrix and token variances...")
    # make sure to get rid of all words whose variance over the remaining columns is 0, then compute the pairwise
    # correlation between each word's association vector over phonological cues and finally compute the variance of
    # each token
    associations_subset, exclude = check_col_variances(matrix_subset)
    if exclude.size:
        target_tokens = realign_targets(target_tokens, exclude)
    print("Matrix subset (after removing columns with 0 variance): ",
          type(associations_subset), associations_subset.shape)
    similarities = sim.pairwise_corr(associations_subset)
    print("Similarity matrix: ", type(similarities), similarities.shape)
    if np.isnan(similarities).any():
        similarities = np.nan_to_num(similarities)
    column_variances = np.var(similarities, axis=0)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished computing correlation similarity matrix and token variances.")
    print()

    # map each token to its variance computed over token-to-token pairwise correlations derived from the cue-token
    # association matrix
    token_variances = {}
    for token in target_tokens:
        token_idx = target_tokens[token]
        token_variances[token] = column_variances[token_idx]

    return token_variances


########################################################################################################################


def compute_distributional_predictors(corpus_path, indices):

    """
    :param corpus_path:             a string indicating the path to the encoded corpus used as input to the
                                    Rescorla-Wagner model
    :param indices:                 an iterable of integers between 1 and 100, indicating at which percentage of the
                                    input corpus statistics have to be stored
    :return outcome_statistics:     a dictionary of dictionaries mapping each outcome to four statistics computed at
                                    each specified time point, including:
                                    - an outcome frequency, i.e. in how many learning events it occurs
                                    - an outcome lexical diversity, i.e. the number of other tokens occurring in the
                                        learning events where the outcome occurs
                                    - an outcome phonological diversity, i.e. the number of different phonological cues
                                        occurring in the same learning events as the outcome
                                    - an outcome predictability, i.e. the conditional probability of an outcome given a
                                        phonological cue, averaged over all the phonological cues occurring in the same
                                        learning events as the outcome
    :return cue_statistics:         a dictionary of dictionaries mapping each cue to four statistics computed at each
                                    specified time point, including:
                                    - a cue frequency, i.e. how often it occurs in the corpus
                                    - a cue lexical diversity, i.e. the number of different tokens occurring in the
                                        learning events in which the cue occurs
                                    - a cue phonological diversity, i.e. the number of different phonological cues that
                                        occurs in the learning events where the cue occurs
                                    - a cue predictability, i.e. the conditional probability of a cue given an outcome,
                                        averaged over all the outcomes occurring in the learning events where the cue
                                        also occurs
    """

    corpus = json.load(open(corpus_path, "r"))

    # initialize a bunch of dictionaries to keep track of frequencies and co-occurrences
    token_statistics = recursive_defaultdict()
    token_frequencies = Counter()
    token_lexical_diversity = defaultdict(dict)
    token_phonological_diversity = defaultdict(dict)
    cue_statistics = recursive_defaultdict()
    cue_frequencies = Counter()
    cue_lexical_diversity = defaultdict(dict)
    cue_phonological_diversity = defaultdict(dict)

    # get the line index at which to store distributional statistics
    total = len(corpus[0])
    time_points = {np.ceil(total / float(100) * n): n for n in indices}

    bar = progressbar.ProgressBar(max_value=len(corpus[0]))
    for ii in bar(range(len(corpus[0]))):

        # get the phonological cues and the lexical outcomes in the current learning trial
        tokens = corpus[1][ii]
        phonological_cues = corpus[0][ii]

        # loop through all tokens, update frequency counts and co-occurrences with other tokens and with the
        # phonological cues in the same learning trial
        for t in tokens:
            token_frequencies[t] += 1
            other_tokens = set(tokens) - {t}
            for other_token in other_tokens:
                _update_dict(t, other_token, token_lexical_diversity)

            for c in phonological_cues:
                _update_dict(t, c, token_phonological_diversity)

        # repeat looping through all phonological cues
        for c in phonological_cues:
            cue_frequencies[c] += 1
            other_cues = set(phonological_cues) - {c}
            for other_cue in other_cues:
                _update_dict(c, other_cue, cue_phonological_diversity)

            for t in tokens:
                _update_dict(c, t, cue_lexical_diversity)

        # at the specified line indices, store statistics for tokens and phonological cues (frequency, lexical and
        # phonological diversity, and conditional probability)
        if ii + 1 in time_points:

            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": Started storing statistics at %d%% of the corpus..." % time_points[ii + 1])

            for t in token_frequencies:

                # frequency, lexical and phonological diversity of tokens
                token_statistics[t]['freq'][time_points[ii + 1]] = np.log10(token_frequencies[t])
                token_statistics[t]['lexdiv'][time_points[ii + 1]] = np.log10(len(token_lexical_diversity[t]))
                token_statistics[t]['phondiv'][time_points[ii + 1]] = np.log10(len(token_phonological_diversity[t]))

                # average conditional probability of a token given the co-occurring phonological cues
                token_cues_predictability = _get_avg_conditional_probability(t, token_phonological_diversity,
                                                                             cue_frequencies)
                token_statistics[t]['p_token_cues'][time_points[ii + 1]] = token_cues_predictability

                # average conditional probability of a token given the co-occurring tokens
                token_tokens_predictability = _get_avg_conditional_probability(t, token_lexical_diversity,
                                                                               token_frequencies)
                token_statistics[t]['p_token_tokens'][time_points[ii + 1]] = token_tokens_predictability

                # average predictive power of a token with respect to the co-occurring tokens
                tokens_token_predictability = _get_avg_conditional_probability(t, token_lexical_diversity,
                                                                               token_frequencies, reverse=True)
                token_statistics[t]['p_tokens_token'][time_points[ii + 1]] = tokens_token_predictability

                # average predictive power of a token with respect to the co-occurring phonological cues
                cues_token_predictability = _get_avg_conditional_probability(t, token_phonological_diversity,
                                                                             token_frequencies, reverse=True)
                token_statistics[t]['p_cues_token'][time_points[ii + 1]] = cues_token_predictability

            for c in cue_frequencies:

                # frequency, lexical and phonological diversity of phonological cues
                cue_statistics[c]['freq'][time_points[ii + 1]] = np.log10(cue_frequencies[c])
                cue_statistics[c]['lexdiv'][time_points[ii + 1]] = np.log10(len(cue_lexical_diversity[c]))
                cue_statistics[c]['phondiv'][time_points[ii + 1]] = np.log10(len(cue_phonological_diversity[c]))

                # average conditional probability of a cue given the co-occurring tokens
                cue_tokens_predictability = _get_avg_conditional_probability(c, cue_lexical_diversity,
                                                                             token_frequencies)
                cue_statistics[c]['p_cue_tokens'][time_points[ii + 1]] = cue_tokens_predictability

                # average conditional probability of a cue given the co-occurring cues
                cue_cues_predictability = _get_avg_conditional_probability(c, cue_phonological_diversity,
                                                                           cue_frequencies)
                cue_statistics[c]['p_cue_cues'][time_points[ii + 1]] = cue_cues_predictability

                # average predictive power of a cue with respect to all the co-occurring cues
                cues_cue_predictability = _get_avg_conditional_probability(c, cue_phonological_diversity,
                                                                           cue_frequencies, reverse=True)
                cue_statistics[c]['p_cues_cue'][time_points[ii + 1]] = cues_cue_predictability

                # average predictive power of a cue with respect to all the co-occurring tokens
                tokens_cue_predictability = _get_avg_conditional_probability(c, cue_lexical_diversity,
                                                                             cue_frequencies, reverse=True)
                cue_statistics[c]['p_tokens_cue'][time_points[ii + 1]] = tokens_cue_predictability

            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": ...finished storing statistics at %d%% of the corpus." % time_points[ii + 1])
            print()

    return token_statistics, cue_statistics
