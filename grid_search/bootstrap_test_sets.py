__author__ = 'GCassani'

"""Function to create balanced, smaller test sets from a list of words"""

import os
import random
from collections import defaultdict


def organize_words_by_pos_tag(test_set, target_tags):

    """
    :param test_set:        the path to a .txt file containing two strings per line. The first string consists of the
                            orthographic form of the word and the associated PoS tag, separated by a vertical bar ('\')
                            while the second string consists of the phonological form of the word and the associated
                            PoS tag, separated by a vertical bar ('|'). The two strings are tab ("\t") separated.
    :param target_tags:     an iterable containing strings indicating which PoS tags to consider
    :return words_by_pos:   a dictionary mapping each PoS tag to a set containing all the words from the input file
                            tagged with the PoS tag (each word consists of the orthographic form and the phonetic form,
                            both tagged with the appropriate PoS tag and separated by a tab ('\")
    """

    words_by_pos = defaultdict(list)

    with open(test_set, "r") as f:
        for line in f:
            ortho, phon = line.strip().split("\t")
            pos = ortho.split("|")[1]
            if pos in target_tags:
                words_by_pos[pos].append("\t".join([ortho, phon]))

    return words_by_pos


########################################################################################################################


def write_sampled_words_to_file(words_by_pos, output_file, k=1000):

    """
    :param words_by_pos:    a dictionary mapping each PoS tag to a set containing all the words from the input file
                            tagged with the PoS tag
    :param output_file:     the path of the file where the sampled test items will be written to
    :param k:               the number of words per category to be considered
    """

    with open(output_file, "a") as o_f:
        for tag, words in words_by_pos.items():
            test_words = random.sample(words, k)
            for word in test_words:
                o_f.write(word)
                o_f.write("\n")


########################################################################################################################


def bootstrap_test_sets(test_set, output_folder, n=36, k=1000):

    """
    :param test_set:        the test set from which smaller test set are going to be bootstrapped
    :param output_folder:   the folder where the bootstrapped test sets will be written to file
    :param n:               the number of test sets to bootstrap
    :param k:               the number of words per category to be considered
    """

    test_set_name, ext = os.path.splitext(os.path.basename(test_set))
    target_tags = {"A", "B", "N", "V"}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    words_by_pos = organize_words_by_pos_tag(test_set, target_tags)

    for i in range(n):
        output_file = os.path.join(output_folder,
                                   "_".join([test_set_name, "".join(["sample", str(i+1), ext])]))
        write_sampled_words_to_file(words_by_pos, output_file, k)
