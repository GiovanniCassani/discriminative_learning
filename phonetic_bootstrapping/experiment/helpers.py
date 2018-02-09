__author__ = 'GCassani'

"""Helper function to identify and process words to be used as test items in the grid search experiments"""

import operator
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
        token, pos = word.split('~')
        if pos in pos_mapping:
            new_pos = pos_mapping[pos]
            filtered_words.add('|'.join([token, new_pos]))

    return filtered_words


########################################################################################################################


def derive_orthography_and_phonology(wordform, pos_tag, celex_lemmas):

    """
    :param wordform:        a string indicating the orthographic form of a word
    :param pos_tag:         a string indicating the PoS tag (from Celex) of that word
    :param celex_lemmas:    a dictionary mapping orthographic forms to the PoS tags with which each orthographic form
                            is tagged in the Celex database; furthermore, each PoS tag is mapped to the phonological
                            form of that word when belonging to that PoS tag
    :return word:           a tuple containing the input wordform in its orthographic and phonetic form
    """

    orthographic = '|'.join([wordform, pos_tag])
    phonetic = '|'.join([celex_lemmas[wordform][pos_tag], pos_tag])
    word = (orthographic, phonetic)
    return word


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


def compute_baselines(tags):

    """
    :param tags:                a list of strings, representing the PoS tags of the words in the test set
    :return majority_baseline:  the accuracy that would be achieved on the test set by always predicting the most
                                frequent PoS tag in the set
    :return entropy_baseline:   the entropy of the distribution of PoS tags as found in the test set
    """

    tag_frequencies = Counter(tags)
    most_frequent = sorted(tag_frequencies.items(), key=operator.itemgetter(1), reverse=True)[0][1]
    majority_baseline = most_frequent / len(tags)
    entropy_baseline = entropy(list(tag_frequencies.values()), base=len(list(tag_frequencies)))

    return majority_baseline, entropy_baseline


########################################################################################################################


def compute_summary_statistics(log_dict):

    """
    :param log_dict:    the dictionary containing the test items together with the categorization outcomes
    :return f1:         the proportion of items from the test set that were categorized correctly in the
                        parametrization corresponding to the input logfile
    :return h:          the normalized entropy of the distribution of PoS tags chosen by the model when tagging test
                        items in the parametrization corresponding to the input logfile
    :return pos:        the PoS tag that was applied most frequently by the model with the parametrization
                        corresponding to the input logfile
    :return freq:       the frequency with which the most frequent PoS tag applied by the model is actually applied in
                        the parametrization corresponding to the input logfile
    """

    hits = 0
    chosen_pos = []
    total = len(log_dict)
    tags = set()

    for item in log_dict:
        word, pos = item.split("|")
        tags.add(pos)
        predicted = log_dict[item]['predicted']
        if pos == predicted:
            hits += 1
        chosen_pos.append(predicted)

    f1 = hits/total if total > 0 else 0
    chosen_pos_freqs = Counter(chosen_pos)
    pos_frequencies = list(chosen_pos_freqs.values())
    while len(pos_frequencies) < len(tags):
        pos_frequencies.append(0)
    h = entropy(pos_frequencies, base=len(chosen_pos_freqs))
    pos, freq = chosen_pos_freqs.most_common(1)[0]

    return f1, h, pos, freq
