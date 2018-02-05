__author__ = 'GCassani'

"""Helper function to identify and process words to be used as test items in the grid search experiments"""

import json
from collections import Counter
from scipy.stats import entropy


def align_tag_set(words, pos_mapping):

    """
    :param words:           an iterable containing words extracted from a corpus containing transcripts from the CHILDES
                            database. Each item in the iterable consists of a word form and a Part-of-Speech tag, the
                            two being joined by a vertical bar ('|')
    :param pos_mapping:     a dictionary created using the function get_pos_category_mapping() from the module
                            preprocess_corpus.py. The dictionary maps PoS tags from the MOR tagset used in CHILDES to
                            the PoS tags used in the Celex database.
    :return filtered_words: a set of words from the input list or set whose PoS tag exists in the input dictionary (some
                            PoS tags from the MOR tagset could not be included in the dictionary to exclude specific
                            words from the corpus) and whose PoS tag has been changed according to the mapping specified
                            in the input dictionary.
    """

    filtered_words = set()

    for word in words:
        token, pos = word.split('|')
        if pos in pos_mapping:
            new_pos = pos_mapping[pos]
            filtered_words.add('|'.join([token, new_pos]))

    return filtered_words


########################################################################################################################


def add_word_to_test_set(test_set, word, pos_tag, celex_lemmas):

    """
    :param test_set:        a set of tuples (can also be empty)
    :param word:            a string indicating the orthographic form of a word
    :param pos_tag:         a string indicating the PoS tag (from Celex) of that word
    :param celex_lemmas:    a dictionary mapping orthographic forms to the PoS tags with which each orthographic form
                            is tagged in the Celex database; furthermore, each PoS tag is mapped to the phonological
                            form of that word when belonging to that PoS tag
    :return test_set:       the input set, with the input word added as a tuple consisting of the orthographic form and
                            the phonetic form
    """

    orthographic = '|'.join([word, pos_tag])
    phonetic = '|'.join([celex_lemmas[word][pos_tag], pos_tag])
    test_set.add((orthographic, phonetic))

    return test_set


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


def get_stats_from_existing_logfile(log_file):

    """
    :param log_file:    the log file with the experiment outcome and summary statistic for a certain parametrization
    :return f1:         the proportion of items from the test set that were categorized correctly in the
                        parametrization corresponding to the input logfile
    :return h:          the normalized entropy of the distribution of PoS tags chosen by the model when tagging test
                        items in the parametrization corresponding to the input logfile
    :return pos:        the PoS tag that was applied most frequently by the model with the parametrization
                        corresponding to the input logfile
    :return freq:       the frequency with which the most frequent PoS tag applied by the model is actually applied in
                        the parametrization corresponding to the input logfile
    """

    log_dict = json.load(open(log_file, "r"))
    hits = 0
    chosen_pos = []
    total = len(log_dict)

    for item in log_dict:
        word, pos = item.split("|")
        predicted = log_dict[item]['predicted']
        if pos == predicted:
            hits += 1
        chosen_pos.append(predicted)

    f1 = hits/total
    chosen_pos_freqs = Counter(chosen_pos)
    pos_frequencies = list(chosen_pos_freqs.values())
    h = entropy(pos_frequencies, base=len(chosen_pos_freqs))
    pos, freq = chosen_pos_freqs.most_common(1)[0]

    return f1, h, pos, freq
