__author__ = 'GCassani'

"""Function to extract single words from an input corpus"""

import json


def single_words(corpus_name):

    """
    :param corpus_name:     the file pointing to a .json file to be used as input corpus
    :return encoded_corpus: the input file recoded as a list of lists, each consisting of two lists, the first contains
                            the token, the second contains the corresponding lemma
    """

    corpus = json.load(open(corpus_name, 'r+'))

    encoded_corpus = [[], []]

    # for every utterance in the input corpus, remove words with a PoS tag that doesn't belong to the
    # dictionary of PoS mappings; then map valid words to the right PoS tag as indicated by the PoS dictionary
    for i in range(len(corpus[0])):
        for j in range(len(corpus[0][i])):
            encoded_corpus[0].append([corpus[0][i][j]])
            encoded_corpus[1].append([corpus[1][i][j]])

    return encoded_corpus


########################################################################################################################


def get_words_from_corpus(corpus):

    """
    :param corpus:  a list consisting of two lists, which in turn consists of several lists. Each inner, lowest level
                    list contains a string, consisting of two parts separated by a vertical bar ('|'): to the left is
                    the word, to the right the Part-of-Speech to which the word belongs
    :return words:  a set containing all the unique strings from all the lists nested in the second first-order list
    """

    words = set()

    for i in range(len(corpus[0])):
        outcomes = set(corpus[1][i])
        for outcome in outcomes:
            words.add(outcome)

    return words
