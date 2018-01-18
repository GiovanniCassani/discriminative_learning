from __future__ import division
import os
import operator
import random
import ndl
import json
import argparse
import numpy as np
import preprocess_corpus as prc
import celex_processing as clx
from time import strftime
from collections import defaultdict, Counter
from scipy.stats import contingency, chi2_contingency
from scipy.stats import entropy

__author__ = 'GCassani'


"""
This module performs a full phonological bootstrapping experiment using Naive Discriminative Learning as implemented in
the ndl.py module. It encodes a corpus in the desired cues and outcomes, learns the matrix of associations, gets a set
of test items and categorizes them according to their lexical category exploiting the cues of each test item and their
associations to the outcomes encountered during training. This module consists of the following functions, listed
together with a short description of what they do. Check the documentation of each function for details about input
arguments and output structures.

- sort_lexical_nodes_from_matrix    : returns a list of outcome strings together with the total activation computed from
                                        the matrix of associations, given the input nphones, ranking the outcomes by
                                        their activation given the nphones. It works with the vectorized implementation
                                        (and requires cues and outcomes ids to work.
- store_dict                        : maps elements of an array to their positional indices in the array itself
- print_weights                     : print cue-outcome associations for the standard implementation
- read_in_test_items                : store phonetically encoded test items from a .txt file into a set
- get_pos_tag                       : takes an outcome (word-PoS), and extracts its PoS tag
- get_top_pos                       : gets two dictionaries mapping the same set of keys to different values and looks 
                                        for the key from dict1 having the highst value, using dict2 to resolve ties
- category_activation               : takes the top active outcomes given some cues and sums activation values for all
                                        words belonging to a same PoS, for all PoS tags represented in the list
- pos_frequency_and_activation      : gets a list of tuples, consisting of a string (word-PoS) and a float, and an 
                                        integer and returns two dictionaries, one mapping PoS tags to their frequency 
                                        across the first k tuples and the other mapping PoS tags to the sum of 
                                        activations from words belonging to a same PoS tag across the first k tuples
- testing                           : performs the test phase of a phonological bootstrapping experiment; it returns an
                                        accuracy score, reflecting how well words from the test set could be categorized
                                        into lexical categories using the knowledge obtained during training and encoded
                                        in the matrix of associations
- phonetic_bootstrapping            : performs a full phonological bootstrapping experiment, learning cue-outcome
                                        associations from a corpus and categorizing words from a test set
- main                              : a general function to run the function phonetic_bootstrapping when the module is
                                        called from command line

"""


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
    alphas = ndl.activations(associations_matrix, cue_mask)

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


def store_dict(array):

    """
    :param array:   an array like structure
    :return d:      a dictionary mapping indices from the array to corresponding values from the array
    """

    d = {}
    for idx, val in enumerate(array):
        d[idx] = val
    return d


########################################################################################################################


def print_weights(weight_matrix, n):

    """
    :param weight_matrix:   a dictionary of dictionaries, containing strings as first- and second-level keys and
                            floatings as values
    :param n:               the number of decimals to round floatings
    """

    for k1 in sorted(weight_matrix.keys()):
        for k2 in sorted(weight_matrix[k1].keys()):
            # print each key1-key2 pair with the corresponding number, rounding according to the specified number of
            # decimals
            print('\t'.join([k1, k2, str(round(weight_matrix[k1][k2], n))]))


########################################################################################################################


def read_in_test_items(test_items_path, stress_marker=False):

    """
    :param test_items_path: the path to a .txt file containing one string per line. Each string needs to consist of two
                            parts, joined by a vertical bar ('|'): the phonological form of the word to the left, the 
                            PoS tag to the right
    :param stress_marker:   a boolean indicating whether stress information from the test items should be preserved or 
                            not. It is assumed that test items are all encoded for stress: setting this argument to
                            False cause to discard the stress information. Secondary stress, marked with ", is always 
                            deleted. It is assumed that stress is encoded as '
    :return test_items:     a set containing the elements from the file
    """

    test_items = set()

    with open(test_items_path, 'r+') as f:
        for line in f:
            item = line.strip()
            word = item.replace("\"", "")
            if not stress_marker:
                word.replace("'", "")
            test_items.add(word)

    return test_items


########################################################################################################################


def get_pos_tag(word):

    """
    :param word:        a string, with a vertical bar ('|') dividing the wordform from the PoS tag. i.e. a single, 
                        capital letter that marks the lexical category to which the word belong. 
    :return pos:        a string indicating the PoS tag extracted from the input word. If no PoS tag can be retrieved 
                        from the word, the function returns None
    """

    if not isinstance(word, str):
        word = word.decode('UTF-8')

    try:
        pos = word.split('|')[1]
        return pos
    # if the word being considered doesn't have a PoS tag, append None to the list of PoS tags
    except IndexError:
        return None


########################################################################################################################


def get_top_pos_from_dict(dict1, dict2):

    """
    :param dict1:       a dictionary mapping PoS tags to their frequency distribution across the k top active outcomes, 
                        with k specified by the user. This dictionary is the first output of the function 
                        pos_frequency_and_activation()
    :param dict2:       a dictionary mapping PoS tags to their total activation across the k top active outcomes, with 
                        k specified by the user. The activation of each outcome tagged with a same PoS is summed to 
                        obtain a total for each PoS. This dictionary is the second output of the function 
                        pos_frequency_and_activation()
    :return top_key1:   the single key from dict1 with highest value. In case of ties, meaning that two or more keys 
                        from dict1 have the same value, the values for the same keys from dict2 are used to resolve the 
                        tie. The two dictionaries must have the same keys
    """

    # initialize an empty string variable to store the key with highest value in dict1 and a numeric variable to
    # keep track of the highest value so far in dict1
    top_key1 = ''
    highest_v = 0

    # loop through dict1 descending order
    for k, v in sorted(dict1.items(), key=operator.itemgetter(1), reverse=True):

        # if the value of the current key is higher than the value being stored, use the current value
        # instead and update value of the variable for the most common key1
        if v > highest_v:
            highest_v = v
            top_key1 = k

        # if there is a tie, meaning that the following key has the same valuet of the most frequent one being
        # currently stored, the corresponding values from dict2 are retrieved
        elif v == highest_v:
            most_common_pos_activation = dict2[top_key1]
            curr_pos_activation = dict2[k]

            # if the value from dict2 for the key1 being considered is higher than the value from dict2 for the key1
            # being stored as the most common, than update the information storing the current key1 as most
            # common. The value1 is the same, so no need to update it. If, on the contrary, the value from dict2 of the
            # key1 being considered is lower then the value from dict2 of the key1 being stored as the most common,
            # nothing needs to be updated
            if curr_pos_activation > most_common_pos_activation:
                top_key1 = k
            # if, however, the two key1 also tie in their value2, then pick one at random. Again, value1 is the same,
            # so no need to update it
            elif curr_pos_activation == most_common_pos_activation:
                top_key1 = random.choice([k, top_key1])

        # as soon as the value1 of a key1 is lower than the current value1 being stored, return the key1 being stored as
        # the most common
        else:
            return top_key1, highest_v

    # in the unlikely scenario in which all key1 have the same value1 and, when the function runs out of key1 to
    # evaluate, return the key1 that either has the highest value2 or the one that got selected at random if all PoS
    # tags also tied when value2 are considered
    return top_key1, highest_v


########################################################################################################################


def category_activation(input_list, target_category):

    """
    :param input_list:      an iterable containing tuples. Each tuple consists of a string and a floating; the string,
                            in turn, consists of two parts, a word form and a PoS tag, joined by a vertical bar ('|')
    :param target_category: a string indicating a PoS tag
    :return category_alpha: a number indicating the total activation that the target category received from all the
                            elements in the input iterable. First, the PoS tag of each tuple is matched against the
                            target one: if they're the same, the second element of the tuple, the floating, is summed to
                            the category_alpha
    """

    category_alpha = 0

    for item in input_list:
        word = item[0].decode('UTF-8') if not isinstance(item[0], str) else item[0]
        category = word.split('|')[1]
        if category == target_category:
            category_alpha += item[1]

    return category_alpha


########################################################################################################################


def pos_frequency_and_activation(l, k):

    """
    :param l:       an iterable containing tuples. Each tuple consists of a string and a floating; the string, in
                    turn, consists of two parts, separated by a vertical bar ('|')
    :param k:       a number indicating how many tuples from the input iterable to consider
    :return freq:   a dictionary mapping all the rightmost parts of the strings from the tuples (the part after the 
                    bar) to their frequency in the k first tuples    
    :return act:    a dictionary mapping all the rightmost parts of the strings from the tuples (the part after the 
                    bar) to the sum of the numerical part of each tuple matching the key
                    
    This function is used here to compute the frequency distribution of PoS tags in the k top active outcomes, provided
    in the input argument l, and the PoS activation across the same outcomes. The first dictionary maps PoS tags to 
    their frequency counts, the second maps PoS tags to their total activation, i.e. the sum of activation values from
    all the words tagged with a same PoS.
    """

    freq = Counter([get_pos_tag(o[0]) for o in l[:k]])
    act = defaultdict(float)
    for pos in freq:
        act[pos] = category_activation(l[:k], pos)

    return freq, act


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


def assign_pos_tag(item_activations, freq_baseline, act_baseline, k=100,
                   evaluation='distr', method='freq', stats=True):

    """
    :param item_activations:    an iterable containing tuples. Each tuple consists of a string and a floating; the 
                                string, in turn, consists of two parts, separated by a vertical bar ('|') 
    :param freq_baseline:       a dictionary mapping strings (PoS tags) to counts (their frequency of occurrence among 
                                the k top active outcomes at baseline, i.e. given all cues)
    :param act_baseline:        a dictionary mapping strings (PoS tags) to activations, obtained by summing the 
                                activation  value at baseline for all words tagged with a same PoS tag.
    :param method:              a string indicating the way in which the function looks at top active outcomes; two 
                                options are available:
                                - 'freq' makes the function compute the distribution of PoS tags over the k top active 
                                    nodes (see the explanation of the parameter k) and rank PoS tags according to their
                                    frequency among the top active cues
                                - 'sum' makes the function compute the sum of activation from all outcomes belonging to
                                    a given PoS tag within the k top active outcomes given the input cues, and rank PoS
                                    tags according to their total activation among the top active cues
    :param evaluation:          a string indicating how to compare baseline activations to item-triggered ones; two 
                                options are available:
                                - 'count', simply tag the test item with the PoS tag that either was more frequent or 
                                    had highest summed activation within the top active outcomes; frequency or 
                                    activation are returned and can be correlated to reaction times
                                - 'distr', compare the frequency counts or summed activations generated by a specific 
                                    test item to the frequency counts or summed activations at baseline and tag the test
                                    item with the PoS tag receiving highest support by the change in the distribution of
                                    frequencies or summed activations (a statistic is returned, Chi-squared for 
                                    frequency distributions and t-test for summed activations, whose value can be 
                                    correlated to reaction times)
    :param stats:               if True, makes the function assign a PoS tag to a test item based on the result of a 
                                statistical test (Chi-squared for frequencies, t-test for activations): with the 
                                Chi-squared, the PoS tag is chosen whose Pearson standardised residual is highest for 
                                the item-triggered frequency distribution; with the t-test, the PoS tag is chosen that..
                                if False, the PoS tag with the highest positive difference between item-triggered and 
                                baseline frequency/activation is chosen
                                CAVEAT: this parameter only makes sense if the 'distr' option is chosen for the 
                                            'evaluation'parameter
    :param k:                   an integer specifying how many elements to consider from the baseline activations and 
                                the activations triggered by a specific test item. By default, the top 100 outcomes are
                                considered, and compared according to the chosen combination of method and eval 
    :return pos:                the PoS tag selected as the best candidate given the activation triggered by the 
                                test item
    :return value:              the value used to decide (according to the specifications chosen for the parameters
                                'method', 'evaluation', and 'stats'
    """

    # compute frequency distribution over PoS tags and PoS summed activation over the k most active outcomes given
    # the test item being considered
    pos_freq_item, pos_act_item = pos_frequency_and_activation(item_activations, k)

    if evaluation == 'distr':
        if method == 'freq':
            if stats:
                # convert dictionaries to numpy array, with item frequencies in the first row and baseline
                # frequencies in the second; then perform a chi-squared test to see if the two frequency
                # distributions are independent, get the table with expected frequencies under the null hypothesis
                # of independence and use them to compute Pearson standardised residuals. Take the highest value
                # from the first row (indicating a higher frequency than expected for the item-triggered frequency
                # distribution) and choose the corresponding PoS tag
                observed_freq, col_ids = dict2numpy(pos_freq_item, freq_baseline)
                chi2_freq, p_freq, dof_freq, expected_freq = chi2_contingency(observed_freq)
                res = std_res(observed_freq, expected_freq)
                pos = col_ids[np.argmax(res[0])]
                value = np.max(res[0])
            else:
                # find the PoS with the largest positive difference between item measures and baseline and store
                # the difference (frequency gain in one case, activation gain in the other); look for the largest
                # positive difference in the frequency table, and use the activation differences to resolve ties
                diff_freq = differences(pos_freq_item, freq_baseline)
                diff_act = differences(pos_act_item, act_baseline)
                pos, value = get_top_pos_from_dict(diff_freq, diff_act)
        else:
            diff_freq = differences(pos_freq_item, freq_baseline)
            diff_act = differences(pos_act_item, act_baseline)
            pos, value = get_top_pos_from_dict(diff_act, diff_freq)
    else:
        # find the PoS with highest frequency/summed activation in the item activation vector, and store the
        # frequency/activation
        if method == 'freq':
            pos, value = get_top_pos_from_dict(pos_freq_item, pos_act_item)
        else:
            pos, value = get_top_pos_from_dict(pos_act_item, pos_freq_item)

    return pos, value


########################################################################################################################


def testing(test_items, logfile, weights_matrix, cues2ids, outcomes2ids, method='freq', evaluation='count',
            stats=False, k=100, flush=0, uni_phones=True, di_phones=False, tri_phones=False, syllable=False,
            stress_marker=False):

    """
    :param test_items:      an iterable containing strings. Each string is the phonological form of a word
    :param logfile:         the path to a .txt file, where the function prints information about the processes it runs
                            and their outcome
    :param weights_matrix:  a NumPy array containing the matrix of cue-outcome associations estimated via the ndl
                            module; rows represent cues, columns represent outcomes.
    :param cues2ids:        a Python dictionary mapping cues to row indices in the weight matrix
    :param outcomes2ids:    a Python dictionary mapping outcomes to column indices in the weight matrix
    :param method:          a string indicating the way in which the function looks at top active outcomes; two 
                            options are available:
                            - 'freq' makes the function compute the distribution of PoS tags over the k top active 
                                nodes (see the explanation of the parameter k) and rank PoS tags according to their
                                frequency among the top active cues
                            - 'sum' makes the function compute the sum of activation from all outcomes belonging to
                                a given PoS tag within the k top active outcomes given the input cues, and rank PoS
                                tags according to their total activation among the top active cues
    :param evaluation:      a string indicating how to compare baseline activations to item-triggered ones; two 
                            options are available:
                            - 'count', simply tag the test item with the PoS tag that either was more frequent or 
                                had highest summed activation within the top active outcomes; frequency or 
                                activation are returned and can be correlated to reaction times
                            - 'distr', compare the frequency counts or summed activations generated by a specific 
                                test item to the frequency counts or summed activations at baseline and tag the test
                                item with the PoS tag receiving highest support by the change in the distribution of
                                frequencies or summed activations (a statistic is returned, Chi-squared for 
                                frequency distributions and t-test for summed activations, whose value can be 
                                correlated to reaction times)
    :param stats:           if True, makes the function assign a PoS tag to a test item based on the result of a 
                            statistical test (Chi-squared for frequencies, t-test for activations): with the 
                            Chi-squared, the PoS tag is chosen whose Pearson standardised residual is highest for the
                            item-triggered frequency distribution; with the t-test, the PoS tag is chosen that...
                            if False, the PoS tag with the highest positive difference between item-triggered and 
                            baseline frequency/activation is chosen
                            CAVEAT: this parameter only makes sense if the 'distr' option is chosen for the 'evaluation'
                                        parameter
    :param k:               an integer specifying how many elements to consider from the baseline activations and 
                            the activations triggered by a specific test item. By default, the top 100 outcomes are
                            considered, and compared according to the chosen combination of method and eval
    :param flush:           specify whether (and how many) top active outcome at baseline to flush away from 
                            subsequent computations. It may be the case that whatever the input cues, the same high 
                            frequency outcomes come out as being the most active. It may then make sense to not 
                            consider them when evaluating the distribution of lexical categories over the most 
                            active outcomes given an input item
    :param uni_phones:      a boolean indicating whether single phonemes are to be considered while encoding input
                            utterances
    :param di_phones:       a boolean indicating whether sequences of two phonemes are to be considered while
                            encoding input utterances
    :param tri_phones:      a boolean indicating whether sequences of three phonemes are to be considered while
                            encoding input utterances
    :param syllable:        a boolean indicating whether syllables are to be considered while encoding input
                            utterances
    :param stress_marker:   a boolean indicating whether stress markers from the input phonological representation need
                            to be preserved or can be discarded
    :return accuracy:       the proportion of items from the input iterable that could be categorized correctly - the
                            PoS tag of the words receiving highest activation given the phonetic cues in each test item
                            matched the PoS tag attached to the test item itself
    :return entr:           the normalized entropy of the distribution of PoS tags chosen by the model when tagging test
                            items. If a model always chooses the same PoS tag, the entropy will be minimal; ideally, a 
                            good model doesn't behave like a majority baseline model, even though this might result in
                            high accuracy scores
    :return most_freq:      the PoS tag that is applied most frequently by a model. Useful to spot anomalous 
                            over-extension of a PoS tag 
    :return freq:           the frequency with which the most frequent PoS tag applied by the model is actually applied
    """

    if type(weights_matrix) is not dict:
        ValueError("Unrecognized input structure: .")

    to_filter = set()
    baseline_activations = sort_lexical_nodes_from_matrix(cues2ids.keys(), weights_matrix, cues2ids,
                                                          outcomes2ids, to_filter)

    # if top active outcomes at baseline need to be flushed away, store flushed outcomes in a set and store the other
    # outcomes in a list of tuples
    if flush:
        for outcome in baseline_activations[:flush]:
            to_filter.add(outcome[0])
        baseline_activations = baseline_activations[flush:]

    # compute baseline frequency distribution over PoS tags and PoS summed activation over the k most active outcomes
    # given all input cues at once
    pos_freq_baseline, pos_act_baseline = pos_frequency_and_activation(baseline_activations, k)
    with open(logfile, 'a+') as log_f:
        log_f.write("\t".join([str(baseline_activations[:k])]))
        log_f.write("\n\n")

    hits = 0
    total = 0
    celex_vowels = clx.vowels()

    chosen_tags = []

    for item in test_items:
        if not isinstance(item, str):
            ValueError("The input items must consist of either strings or unicode objects: check your input file!")

        # split the test token from its Part-of-Speech and encode it in nphones
        word, target_pos = item.split('|')
        word = '+' + word + '+'
        word_nphones = prc.encode_item(word, celex_vowels, uni_phones=uni_phones, di_phones=di_phones,
                                       tri_phones=tri_phones, syllable=syllable, stress_marker=stress_marker)

        # sort all words from the input matrix of associations according to the total activations that the phonetic cues
        # in the test items have to each outcome; then get the top active nodes according to the specified method
        item_activations = sort_lexical_nodes_from_matrix(word_nphones, weights_matrix, cues2ids,
                                                          outcomes2ids, to_filter)

        # get the most likely PoS tag for the test item
        top_pos, value = assign_pos_tag(item_activations, pos_freq_baseline, pos_act_baseline,
                                        evaluation=evaluation, method=method, k=k, stats=stats)
        chosen_tags.append(top_pos)

        # print the test item (with the correct PoS tag), the PoS tag assigned by the model, the statistic computed to
        # assign the chosen PoS (frequency/activation count or difference) and the k top active nodes given the test
        # item
        with open(logfile, 'a+') as log_f:
            log_f.write("\t".join([item, top_pos, str(value), str(item_activations[:k])]))
            log_f.write("\n")

        # compare the predicted and true PoS tag and increment the count of hits if they match
        if top_pos == target_pos:
            hits += 1

        total += 1

    freq_counts = Counter(chosen_tags)
    most_freq, freq = sorted(freq_counts.items(), key=operator.itemgetter(1), reverse=True)[0]
    freq_counts = list(freq_counts.values())
    while len(freq_counts) < 9:
        freq_counts.append(0)

    entr = entropy(freq_counts, base=len(freq_counts))

    # compute the accuracy dividing the number of correctly categorized test items by the total number of test items
    accuracy = hits / float(total)
    return accuracy, entr, most_freq, freq


########################################################################################################################


def phonetic_bootstrapping(input_file, test_items_path, logfile, celex_dict_file, separator='~',
                           method='freq', evaluation='count', k=100, flush=0,
                           reduced=False, grammatical=False,
                           uni_phones=True, di_phones=False, tri_phones=False,
                           syllable=False, stress_marker=False,
                           alpha=0.00001, beta=0.00001, lam=1.0, longitudinal=False):

    """
    :param input_file:          a .json file containing transcripts of child-caregiver interactions extracted from the
                                CHILDES database. The json file consists of two lists of lists, of the same length, both
                                contain utterances but encoded differently. The first encodes each utterance as a list
                                of tokens; the second encodes each utterance as a list of lemmas and Part-of-Speech
                                tags, joined by a vertical bar ('|')
    :param test_items_path:     the path to a .txt file containing one string per line. Each string needs to consist of
                                two parts, joined by a vertical bar ('|')
    :param logfile:             the path to a .txt file, where the function prints information about the processes it
                                runs and their outcome
    :param celex_dict_file:     a string specifying the path to the CELEX dictionary file, in .json format
    :param separator:           the character that separates the word baseform from its PoS tag in the input corpus
    :param method:              a string indicating the way in which the function looks at top active outcomes; two 
                                options are available:
                                - 'freq' makes the function compute the distribution of PoS tags over the k top active 
                                    nodes (see the explanation of the parameter k) and rank PoS tags according to their
                                    frequency among the top active cues
                                - 'sum' makes the function compute the sum of activation from all outcomes belonging to
                                    a given PoS tag within the k top active outcomes given the input cues, and rank PoS
                                     tags according to their total activation among the top active cues
    :param evaluation:          a string indicating how to compare baseline activations to item-triggered ones; two 
                                options are available:
                                - 'count', simply tag the test item with the PoS tag that either was more frequent or 
                                    had highest summed activation within the top active outcomes; frequency or 
                                    activation are returned and can be correlated to reaction times
                                - 'distr', compare the frequency counts or summed activations generated by a specific 
                                    test item to the frequency counts or summed activations at baseline and tag the test
                                    item with the PoS tag receiving highest support by the change in the distribution of
                                    frequencies or summed activations (a statistic is returned, Chi-squared for 
                                    frequency distributions and t-test for summed activations, whose value can be 
                                    correlated to reaction times)
    :param k:                   an integer specifying how many elements to consider from the baseline activations and
                                the activations triggered by a specific test item. By default, the top 100 outcomes are
                                considered, and compared according to the chosen combination of method and eval
    :param flush:               specify whether (and how many) top active outcome at baseline to flush away from 
                                subsequent computations. It may be the case that whatever the input cues, the same high 
                                frequency outcomes come out as being the most active. It may then make sense to not 
                                consider them when evaluating the distribution of lexical categories over the most 
                                active outcomes given an input item 
    :param reduced:             a boolean specifying whether reduced phonological forms should be extracted from Celex
                                whenever possible (if set to True) or if standard phonological forms should be preserved
                                (if False)
    :param grammatical:         a boolean specifying whether inflectional and grammatical meanings should be extracted
                                from input utterances (these include meanings such as PLURAL, PRESENT, THIRD PERSON, and
                                so son). If False, only lexical meanings are extracted from input utterances, using the
                                lemmas provided in the input corpus
    :param uni_phones:          a boolean indicating whether single phonemes are to be considered while encoding input
                                utterances
    :param di_phones:           a boolean indicating whether sequences of two phonemes are to be considered while
                                encoding input utterances
    :param tri_phones:          a boolean indicating whether sequences of three phonemes are to be considered while
                                encoding input utterances
    :param syllable:            a boolean indicating whether syllables are to be considered while encoding input
                                utterances
    :param stress_marker:       a boolean indicating whether stress markers from the phonological representations of
                                Celex need to be preserved or can be discarded
    :param alpha:               a number indicating cue salience. For simplicity, we assume that every cue has the same
                                salience, so changing the value of this parameter does not affect the relative strength
                                of of cue-outcome associations but only their absolute magnitude
    :param beta:                a number indicating the learning rate for positive and negative situations. Again, we
                                make the simplifying assumption that our simulated learners are equally affected by
                                positive and negative feedback. Changing the beta value can have a significant impact on
                                the learning outcome, but 0.1 is a standard choice for this model. If the number of
                                learning trials or the number of different cues in a learning trial are very large, both
                                beta and alpha need to be lowered considerably
    :param lam:                 maximum amount of association that an outcome can receive from all the cues. It simply
                                acts as a scaling factor, so changing its value has the same effects of changing alpha
    :param longitudinal:        a boolean specifying whether to work in a longitudinal setting or not
    :return accuracies:         the categorization accuracy on PoS tagging using phonetic cues, computed for every
                                matrix of associations (if the longitudinal parameter is set to False, there will only
                                be one matrix; if the same parameter is set to True, there will be 20 matrices, with 20
                                accuracy scores)
    :return entropies:          the normalized entropy of the distribution of the PoS tags assigned by the model,
                                computed for every matrix of associations (cf accuracies)
    :return most_frequents:     the PoS tag that was applied the most by the model, for every matrix of associations
                                (cf accuracies)
    :return frequencies:        the frequency count of the most frequent PoS tag applied by the model, for every matrix
                                of associations (cf accuracies)
    """

    encoded_corpus = prc.encode_corpus(input_file, celex_dict_file, separator=separator, uni_phones=uni_phones,
                                       di_phones=di_phones, tri_phones=tri_phones, syllable=syllable,
                                       stress_marker=stress_marker, grammatical=grammatical, reduced=reduced)

    weight_matrices, cues2ids, outcomes2ids = ndl.discriminative_learner(encoded_corpus, alpha=alpha,
                                                                         beta=beta, lam=lam,
                                                                         longitudinal=longitudinal)

    print()
    print("#####" * 20)
    print()

    # read the test items into a Python ietrable
    test_words = read_in_test_items(test_items_path)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Start test phase, using %s as test set."
          % os.path.basename(test_items_path))

    # for each test item, compute the items from the matrix of weights that are most activated given the cues in the
    # item, get the PoS tag that is most present among the most active lexical nodes and check whether the predicted PoS
    # tag matches the gold-standard one provided along the test item. Return a global score indicating the accuracy on
    # the test set
    accuracies = []
    entropies = []
    most_frequents = []
    frequencies = []
    for idx in weight_matrices:
        accuracy, entr, most_freq, freq = testing(test_words, logfile, weight_matrices[idx], cues2ids, outcomes2ids,
                                                  method=method, evaluation=evaluation, flush=flush, k=k,
                                                  uni_phones=uni_phones, di_phones=di_phones, tri_phones=tri_phones,
                                                  syllable=syllable, stress_marker=stress_marker)
        accuracies.append(accuracy)
        entropies.append(entr)
        most_frequents.append(most_freq)
        frequencies.append(freq)

        print("Accuracy: %0.5f" % accuracy)

        with open(logfile, 'a+') as l:
            l.write("\n")
            l.write("Accuracy on the test set %s is: %0.5f" % (os.path.basename(test_items_path), accuracy))
            l.write("\n")
            l.write("The entropy of the applied PoS tags is: %0.5f" % entr)
            l.write("\n")
            l.write("The PoS tag '%s' was applied most frequently, with a frequency of %0.1f" % (most_freq, freq))
            l.write("\n")

    return accuracies, entropies, most_frequents, frequencies

########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Run a full phonetic bootstrapping experiment using the NDL model.')

    parser.add_argument("-I", "--input_file", required=True, dest="training_corpus",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-T", "--test_items", required=True, dest="test_items_file",
                        help="Specify the path to the file containing test items (encoded as .txt).")
    parser.add_argument("-C", "--Celex_dict", required=True, dest="celex_dict_file",
                        help="Specify the directory containing the CELEX files.")
    parser.add_argument("-S", "--separator", dest="sep", default='~',
                        help="Specify the character separating lemma and PoS tag in the input corpus.")
    parser.add_argument("-q", "--method", dest="method", default="freq",
                        help="Specify whether to look at frequency ('freq') or total activation ('sum') of PoS tags.")
    parser.add_argument("-e", "--evaluation", dest="evaluation", default="count",
                        help="Specify whether to consider counts ('count') or compare distributions ('distr').")
    parser.add_argument("-f", "--flush", dest="flush", default=0,
                        help="Specify whether (and how many) top active outcome words to flush away from computations.")
    parser.add_argument("-k", "--threshold", dest="threshold", default=100,
                        help="Specify how many top active nodes are considered to cluster PoS tags.")
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
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01,
                        help="Specify the value of the alpha parameter.")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01,
                        help="Specify the value of the beta parameter.")
    parser.add_argument("-l", "--lambda", dest="lam", default=1,
                        help="Specify the value of the lambda parameter.")
    parser.add_argument("-L", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()

    # make sure that at least one phonetic feature is provided
    if not (args.uni or args.di or args.tri or args.syl):
        parser.error('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    # make sure the argument passed for celex_dir is a valid folder
    if not os.path.exists(args.celex_dict_file) or not args.celex_dict_file.endswith(".json"):
        raise ValueError("There are problems with the CELEX dictionary file  you provided: either the path does not "
                         "exist or the file extension is not .json. Provide a valid path to a .json file.")

    # make sure the training corpus exists and is a .json file
    if not os.path.exists(args.training_corpus) or not args.training_corpus.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")
    # make sure the files with the test items exists and is a .txt file
    if not os.path.exists(args.test_items_file) or not args.test_items_file.endswith(".txt"):
        raise ValueError("There are problems with the test items file you provided: either the path does not exist or"
                         "the file extension is not .txt. Provide a valid path to a .txt file.")

    encoding_string = prc.encoding_features(args.training_corpus, reduced=args.reduced, uni_phones=args.uni,
                                            di_phones=args.di, tri_phones=args.tri, syllable=args.syl,
                                            stress_marker=args.stress, grammatical=args.grammatical, verbose=False)

    # create the log file in the same folder where the corpus is located, using encoding information in the file name,
    # including corpus, phonetic features, method, evaluation, f and k values
    corpus_dir = os.path.dirname(args.training_corpus)
    filename = os.path.splitext(os.path.basename(args.training_corpus))[0]
    test_file = os.path.splitext(os.path.basename(args.test_items_file))[0]
    log_file = "/".join([corpus_dir, ".".join(["_".join(['logfile', filename, test_file, encoding_string, args.method,
                                                         args.evaluation, ''.join(['k', str(args.threshold)]),
                                                         ''.join(['f', str(args.flush)])]), 'txt'])])

    # run the experiments using the input parameters
    phonetic_bootstrapping(args.training_corpus, args.test_items_file, log_file, args.celex_dict_file,
                           separator=args.sep, reduced=args.reduced, k=int(args.threshold), method=args.method,
                           evaluation=args.evaluation, flush=int(args.flush), grammatical=args.grammatical,
                           uni_phones=args.uni, di_phones=args.di, tri_phones=args.tri, syllable=args.syl,
                           stress_marker=args.stress, alpha=float(args.alpha), beta=float(args.beta),
                           lam=float(args.lam), longitudinal=args.longitudinal)

    with open(log_file, 'a+') as log_f:
        log_f.write("\n")
        log_f.write("#" * 100)
        log_f.write("\n")
        log_f.write("\n")


########################################################################################################################


if __name__ == '__main__':

    main()
