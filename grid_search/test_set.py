__author__ = 'GCassani'

"""Function to create lists of words to be used as test items in the phonetic bootstrapping grid search experiment"""


import json
from collections import defaultdict
from celex.get import get_celex_dictionary
from corpus.chunk.words import get_words_from_corpus
from corpus.encode.utilities import get_pos_mapping
from celex.utilities.dictionaries import lemmas2phon
from phonetic_bootstrapping.experiment.helpers import align_tag_set, add_word_to_test_set


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

    for word in candidates_copy:
        pos_tags = celex_lemmas[word].keys()
        if len(pos_tags) > 1:
            phonetic_forms = defaultdict(set)
            for tag in pos_tags:

                # when the same orthographic form corresponds to two different phonetic forms (that differ depending on
                # the PoS tag), this dictionary has multiple keys, each paired to the PoS tag matching the phonetic form
                phonetic_forms[celex_lemmas[word][tag]].add(tag)

            if len(phonetic_forms.keys()) == 1:
                # if the set of phonetic forms has cardinality 1, then the word is pronounced the same regardless of the
                # PoS tag; add the word with all its PoS tags to the set of ambiguous words
                for tag in pos_tags:
                    ambiguous_words = add_word_to_test_set(ambiguous_words, word, tag, celex_lemmas)
            else:
                # if the cardinality the set of phonetic forms is not 1, than there are pronunciation differences for
                # the same orthographic form depending on the PoS tag; if this is the case, find possible pairs that are
                # ambiguous and the forms that are unambiguous, and add each to the test set according to the parameter
                # choice
                if len(pos_tags) == len(phonetic_forms.keys()):
                    # if there are as many different phonetic forms as there are PoS tags, it means that there is no
                    # ambiguity once the PoS tag is considered and that the phonetic form of a words maps uniquely to
                    # a given combination of orthographic form and PoS tag; thus, add each word-PoS combination together
                    # with the corresponding phonological form to the set of unambiguous words
                    for tag in pos_tags:
                        unambiguous_words = add_word_to_test_set(unambiguous_words, word, tag, celex_lemmas)
                else:
                    # if there are fewer phonetic forms than PoS tags, than there is some ambiguity, meaning that two or
                    # more combinations of orthographic form and PoS tag are pronounced the same, making them
                    # indistinguishable from a purely phonetic perspective. There might be unambiguous forms, but it is
                    # not guaranteed as there could be pairs of ambiguous orthographic forms: phonetic forms that pair
                    # to more than one PoS tag are ambiguous, those that pair to a single PoS tag are not.
                    for form, pos_tags in phonetic_forms.items():
                        if len(pos_tags) == 1:
                            unambiguous_words = add_word_to_test_set(unambiguous_words, word, list(pos_tags)[0],
                                                                     celex_lemmas)
                        else:
                            for tag in pos_tags:
                                ambiguous_words = add_word_to_test_set(ambiguous_words, word, tag, celex_lemmas)

        else:
            unambiguous_words = add_word_to_test_set(unambiguous_words, word, list(pos_tags)[0], celex_lemmas)

    test_set = set()
    words_to_evaluate = ambiguous_words if ambiguous else unambiguous_words
    for word in words_to_evaluate:
        with open(output_file, "a+") as o:
            if word[0] in candidates:
                phon = word[1].replace("\"", "")
                if not stress_marker:
                    phon.replace("'", "")
                test_set.add((word[0], phon))
                o.write(phon)
                o.write("\n")

    return test_set


########################################################################################################################


def read_in_test_items(test_items_path):

    """
    :param test_items_path: the path to a .txt file containing one string per line. Each string needs to consist of two
                            parts, joined by a vertical bar ('|'): the phonological form of the word to the left, the
                            PoS tag to the right
    :return test_items:     a set containing the elements from the file
    """

    test_items = set()

    with open(test_items_path, 'r+') as in_file:
        for line in in_file:
            test_items.add(line.strip())

    return test_items
