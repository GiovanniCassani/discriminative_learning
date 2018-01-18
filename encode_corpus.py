__author__ = 'GCassani'

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from time import strftime


"""
This module recodes an input corpus consisting of full utterances into smaller sequences of words, from single words to
multi-word expressions and prosodic-like chunks. It consists of the following functions, listed together with a short
description of what they do. Check the documentation of each function for details about input arguments and output
structures; all functions assume the same input, i.e. transcripts from CHILDES consisting of two lists of lists. The
first contains utterances encoded as lists of tokens, the second contains the same utterances, in the same order,
encoded as lists of lemmas.

- single_words          : recode the input corpus into lists of single words, essentially breaking down each utterance
                            into the constituents tokens (first list) and lemmas (second list)
- multiword_expressions : recode the input corpus into lists of multi-word expressions, defined on the basis of
                            conditional probabilities (in line with the Chunk-Based Learner [McCauley & Christiansen,
                            2014])
- prosodic_chunks       : recode the input corpus into lists of prosodic-like chunks. It exploits the CBL to hypothesize
                            distributionally motivated chunks and then merge them to satisfy the constraint that an
                            utterance must be recoded in at least as many prosodic-like chunks as there are words
                            carrying primary stress in the utterance itself
- full_utterances       : gives back the corpus as it is, keeping the organization in full utterances as derived from
                            CHILDES transcripts
- main                  : a function that runs one or more of the other functions and is called when the module is run
                            from command line

All functions return the corpus in the same form of the input, i.e. a json file consisting of two lists of lists, the
first encoding tokens at the desired granularity (single words, multi-word expressions, prosodic-like chunks, full
utterances) and the second encoding lemmas at the same granularity.

"""


def single_words(corpus_name):

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


def multiword_expressions(corpus):

    return


########################################################################################################################


def prosodic_chunks(corpus):

    return


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Recode utterances from the corpus.')

    parser.add_argument("-i", "--input_file", required=True, dest="in_file",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-s", "--size", dest="size", default='utterances',
                        help="Specify granularity of the resulting encoding.")

    args = parser.parse_args()

    encoding_options = ['utterances', 'prosodic', 'multiword', 'words']

    if not args.size in encoding_options:
        raise ValueError("The encoding option you selected is not available. Choose among 'utterances', 'prosodic',"
                         "'multiword', or 'words'. Check the documentation for information about each encoding.")

    if not os.path.exists(args.in_file) or not args.in_file.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    encoded_corpus = [[], []]
    if args.size == 'words':
        encoded_corpus = single_words(args.in_file)
    elif args.size == 'prosodic':
        pass
    elif args.size == 'multiword':
        pass
    else:
        encoded_corpus = json.load(open(args.in_file, 'r+'))

    basename, ext = os.path.splitext(args.in_file)
    output_file = "".join(["_".join([basename, args.size]), ext])

    json.dump(encoded_corpus, open(output_file, 'w'))

########################################################################################################################


if __name__ == '__main__':

    main()