__author__ = 'GCassani'

from collections import defaultdict, Counter


def map_indices_to_test_words(neighbours_file):

    """
    :param neighbours_file: the path to the file containing neighbours for the target words
    :return ids2words:      a dictionary mapping column indices to target words
    """

    ids2words = {}
    with open(neighbours_file, 'r') as f:
        first_line = f.readline()
        targets = first_line.strip().split('\t')
        for idx, target in enumerate(targets):
            ids2words[idx] = target

    return ids2words


########################################################################################################################


def map_words_to_pos(tokens_file):

    """
    :param tokens_file:     the path to the file containing summary information about tokens, lemmas, pos tags, and
                            triphones
    :return wordform2pos:   a dictionary mapping each wordform to all the tags with which it's been encountered in the
                            corpus
    """

    wordform2pos = defaultdict(set)
    with open(tokens_file, 'r') as f:
        for line in f:
            word = line.strip().split('\t')[0]
            pos = line.strip().split('\t')[2]
            wordform2pos[word].add(pos)

    return wordform2pos
