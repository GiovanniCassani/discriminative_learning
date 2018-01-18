__author__ = 'GCassani'

import os
import argparse
import json
import preprocess_corpus as prc
from collections import defaultdict


def lemma_dict_from_celex(celex_dict):

    """
    :param celex_dict:  a dictionary created using the celex_processing.py module
    :return lemma_dict: a dictionary mapping orthographic forms to the PoS tags with which each orthographic form is 
                        tagged in the Celex dataset; furthermore, each PoS tag is mapped to the phonological form of 
                        that word when belonging to that PoS tag
    """

    lemma_dict = defaultdict(dict)

    for k in celex_dict['lemmas']:
        word = celex_dict['lemmas'][k]['surface'].decode("utf-8").encode("utf-8")
        pos = celex_dict['lemmas'][k]['pos'].decode("utf-8").encode("utf-8")
        phon = celex_dict['lemmas'][k]['lemma_phon'].decode("utf-8").encode("utf-8")
        if pos not in {'UNK', '?'}:
            lemma_dict[word][pos] = phon
    return lemma_dict


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


########################################################################################################################


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
    :param test_set: 
    :param word: 
    :param pos_tag: 
    :param celex_lemmas: 
    :return: 
    """

    orthographic = '|'.join([word, pos_tag])
    phonetic = '|'.join([celex_lemmas[word][pos_tag], pos_tag])
    test_set.add((orthographic, phonetic))

    return test_set


########################################################################################################################


def get_test_words(corpus_file, celex_dict, pos_mapping, ambiguous=False, new=False):

    """
    :param corpus_file: the path to a .json file consisting of two lists, which in turn consists of several lists. Each 
                        inner list contains strings 
    :param celex_dict:  the path to .json file containing a dictionary obtained using the celex_processing.py module
    :param pos_mapping: a .txt file mapping CHILDES PoS tags to CELEX tags
    :param ambiguous:   if True, words are returned that are tagged with multiple PoS tags in CELEX and all have the 
                        same pronunciation
    :param new:         if True, words are returned that are listed in the CELEX dictionary but never appear in the 
                        input corpus
    :return test_set:   a set containing tuples (orthographic form, phonological form): orthographic forms - and 
                        corresponding phonological forms - are added based on the input arguments ambiguous and new
    """

    corpus = json.load(open(corpus_file, 'r+'))
    celex = json.load(open(celex_dict, 'r+'))
    pos_tags = prc.get_pos_category_mapping(pos_mapping)

    corpus_words = get_words_from_corpus(corpus)
    filtered_words = align_tag_set(corpus_words, pos_tags)
    celex_lemmas = lemma_dict_from_celex(celex)
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
                    for f, pos_tags in phonetic_forms.items():
                        if len(pos_tags) == 1:
                            unambiguous_words = add_word_to_test_set(unambiguous_words, word, list(pos_tags)[0],
                                                                     celex_lemmas)
                        else:
                            for tag in pos_tags:
                                ambiguous_words = add_word_to_test_set(ambiguous_words, word, tag, celex_lemmas)

        else:
            unambiguous_words = add_word_to_test_set(unambiguous_words, word, pos_tags[0], celex_lemmas)

    test_set = set()

    if ambiguous:
        for el in ambiguous_words:
            if el[0] in candidates:
                test_set.add(el)
    else:
        for el in unambiguous_words:
            if el[0] in candidates:
                test_set.add(el)

    return test_set


########################################################################################################################


def check_input_arguments(args):

    """
    :param args:    the result of an ArgumentParser structure: it must contain the arguments 'corpus', 'pos', 
                    'outcome_file', and 'celex_dict'
    """

    # check corpus file
    if not os.path.exists(args.corpus) or not args.corpus.endswith(".json"):
        raise ValueError("There are problems with the corpus file you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    # check extension of the output file for cue frequency counts
    if not args.pos.endswith(".txt"):
        raise ValueError("There are problems with the file with the PoS tag mapping you provided: either the path does "
                         "not exist or the file extension is not .txt. Provide a valid path to a .txt file.")

    # check extension of the output file for outcome frequency counts
    if not args.outcome_file.endswith(".txt"):
        raise ValueError("Indicate the path to a .txt file to store the words for the desired test_set.")

    # check CELEX dictionary
    if not os.path.exists(args.celex_dict) or not args.celex_dict.endswith(".json"):
        raise ValueError("There are problems with the celex dictionary file you provided: either the path does not "
                         "exist or the file extension is not .json. Provide a valid path to a .json file.")


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Get a list of test items from the input corpus and Celex.')

    parser.add_argument('-i', '--input_corpus', required=True, dest='corpus',
                        help='Give the path to an input corpus (.json), encoded in phonetic cues and lexical outcomes.')
    parser.add_argument('-c', '--celex_dict', required=True, dest='celex',
                        help='Give the path to the CELEX dict, encoded as .json.')
    parser.add_argument('-p', '--pos_tags', required=True, dest='pos',
                        help='Give the path to a .txt file mapping CHILDES pos tags to CELEX tags.')
    parser.add_argument('-o', '--output_file', required=True, dest='output_file',
                        help='Specify the path where the selected words will be printed.')
    parser.add_argument('-a', '--ambiguous', action="store_true", dest='ambiguous',
                        help='Specify whether phonetically ambiguous words are to be returned.')
    parser.add_argument('-n', "--new", action="store_true", dest="new",
                        help="Specify whether words that appear in CELEX but not in the corpus are to be returned.")

    args = parser.parse_args()

    test_set = get_test_words(args.corpus, args.celex, args.pos, args.ambiguous, args.new)

    for el in test_set:
        with open(args.output_file, "a+") as o:
            o.write(el[1])
            o.write("\n")


########################################################################################################################


if __name__ == '__main__':

    main()
