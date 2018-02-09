__author__ = 'GCassani'

"""Function to create lists of words to be used as test items in the phonetic bootstrapping grid search experiment"""


import os
import json
import numpy as np
from time import strftime
from collections import defaultdict
from celex.get import get_celex_dictionary
from corpus.chunk.words import get_words_from_corpus
from corpus.encode.utilities import get_pos_mapping
from celex.utilities.dictionaries import lemmas2phon
from phonetic_bootstrapping.experiment.helpers import align_tag_set, derive_orthography_and_phonology


def make_test_set(corpus_file, celex_dir, pos_mapping, output_file,
                  ambiguous=False, new=False, reduced=False, stress_marker=False):

    """
    :param corpus_file:     the path to a .json file consisting of two lists, which in turn consists of several lists.
                            Each inner list contains strings
    :param celex_dir:       the path to the Celex directory
    :param pos_mapping:     a .txt file mapping CHILDES PoS tags to CELEX tags
    :param output_file:     the file where the test items will be printed
    :param ambiguous:       if True, words are returned that are tagged with multiple PoS tags in CELEX and all have
                            the same pronunciation
    :param new:             if True, words are returned that are listed in the CELEX dictionary but never appear in the
                            input corpus
    :param reduced:         a boolean specifying whether reduced phonological forms should be extracted from Celex
                            whenever possible (if set to True) or if standard phonological forms should be preserved
                            (if False)
    :param stress_marker:   a boolean indicating whether stress information from the test items should be preserved or
                            not. It is assumed that test items are all encoded for stress: setting this argument to
                            False causes the algorithm to discard the stress information. Secondary stress, marked with
                            ("), is always deleted. It is assumed that stress is encoded as (')
    :return test_set:       a set containing tuples (orthographic form, phonological form): orthographic forms - and
                            corresponding phonological forms - are added based on the input arguments ambiguous and new
    """

    corpus = json.load(open(corpus_file, 'r+'))
    celex_dict = get_celex_dictionary(celex_dir, reduced=reduced)
    pos_tags = get_pos_mapping(pos_mapping)

    corpus_words = get_words_from_corpus(corpus)
    filtered_words = align_tag_set(corpus_words, pos_tags)
    celex_lemmas = lemmas2phon(celex_dict)
    lemmas_with_pos = {'|'.join([k, p]) for k in celex_lemmas for p in celex_lemmas[k]}

    if new:
        # only look at words that are in CELEX but not in the corpus, without considering PoS tags
        candidates = lemmas_with_pos - (lemmas_with_pos.intersection(filtered_words))
    else:
        # only look at words that are both in CELEX and in the corpus, without considering PoS tags
        candidates = lemmas_with_pos.intersection(filtered_words)

    ambiguous_words = set()
    unambiguous_words = set()

    candidates_copy = {el.split('|')[0] for el in candidates}
    total_words = len(candidates_copy) - 1
    check_points = {int(np.floor(total_words / 100 * n)): n for n in np.linspace(20, 100, 5)}

    for idx, word in enumerate(candidates_copy):
        pos_tags = set(celex_lemmas[word].keys())
        if len(pos_tags) > 1:
            phonetic_forms = defaultdict(set)
            for tag in pos_tags:
                # when the same orthographic form corresponds to two different phonetic forms (that differ depending on
                # the PoS tag), this dictionary has multiple keys, each paired to the PoS tag matching the phonetic form
                phonetic_forms[celex_lemmas[word][tag]].add(tag)

            if len(set(phonetic_forms.keys())) == 1:
                # if the set of phonetic forms has cardinality 1, then the word is pronounced the same regardless of the
                # PoS tag; add the word with all its PoS tags to the set of ambiguous words
                for tag in pos_tags:
                    test_item = derive_orthography_and_phonology(word, tag, celex_lemmas)
                    ambiguous_words.add(test_item)

            else:
                # if the cardinality the set of phonetic forms is not 1, than there are pronunciation differences for
                # the same orthographic form depending on the PoS tag; if this is the case, find possible pairs that are
                # ambiguous and the forms that are unambiguous, and add each to the test set according to the parameter
                # choice
                if len(pos_tags) == len(set(phonetic_forms.keys())):
                    # if there are as many different phonetic forms as there are PoS tags, it means that there is no
                    # ambiguity once the PoS tag is considered and that the phonetic form of a words maps uniquely to
                    # a given combination of orthographic form and PoS tag; thus, add each word-PoS combination together
                    # with the corresponding phonological form to the set of unambiguous words
                    for tag in pos_tags:
                        test_item = derive_orthography_and_phonology(word, tag, celex_lemmas)
                        unambiguous_words.add(test_item)
                else:
                    # if there are fewer phonetic forms than PoS tags, than there is some ambiguity, meaning that two or
                    # more combinations of orthographic form and PoS tag are pronounced the same, making them
                    # indistinguishable from a purely phonetic perspective. There might be unambiguous forms, but it is
                    # not guaranteed as there could be pairs of ambiguous orthographic forms: phonetic forms that pair
                    # to more than one PoS tag are ambiguous, those that pair to a single PoS tag are not.
                    for form, pos_tags in phonetic_forms.items():
                        if len(pos_tags) == 1:
                            test_item = derive_orthography_and_phonology(word, list(pos_tags)[0], celex_lemmas)
                            unambiguous_words.add(test_item)
                        else:
                            for tag in pos_tags:
                                test_item = derive_orthography_and_phonology(word, tag, celex_lemmas)
                                ambiguous_words.add(test_item)

        else:
            test_item = derive_orthography_and_phonology(word, list(pos_tags)[0], celex_lemmas)
            unambiguous_words.add(test_item)

        if idx in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") + ": %d%% of target words have been evaluated." % check_points[idx])

    test_set = set()
    words_to_evaluate = ambiguous_words if ambiguous else unambiguous_words
    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started writing to file...")
    with open(output_file, "a+") as o:
        for word in words_to_evaluate:
            if word[0] in candidates:
                phon = word[1].replace("\"", "")
                if not stress_marker:
                    phon.replace("'", "")
                test_set.add(phon)
                o.write("\t".join([word[0], phon]))
                o.write("\n")
    print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished writing to file.")

    return test_set


########################################################################################################################


def get_words_and_tags_from_test_set(test_file):

    """
    :param test_file:   the path to the file containing the words to be used as test items.

    :return words:
    :return tags:
    """

    # read the test items into a Python iterable
    if os.path.exists(test_file):
        words = read_in_test_items(test_file)
    else:
        raise ValueError('Please provide a valid file to be used as test set.')

    tags = []
    for word in words:
        tags.append(word.split("|")[1])

    return words, tags


########################################################################################################################


def read_in_test_items(test_items_path):

    """
    :param test_items_path: the path to a .txt file containing two strings per line. The first string consists of the
                            orthographic form of the word and the associated PoS tag, separated by a vertical bar ('\')
                            while the second string consists of the phonological form of the word and the associated
                            PoS tag, separated by a vertical bar ('|'). The two strings are tab ("\t") separated.
    :return test_items:     a set containing the elements from the file
    """

    test_items = []

    with open(test_items_path, 'r+') as in_file:
        for line in in_file:
            test_items.append(line.strip().split("\t")[1])

    return test_items
