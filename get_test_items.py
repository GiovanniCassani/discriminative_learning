__author__ = 'GCassani'

"""Function to get the test items (can be called from command line)"""

import os
import argparse
from grid_search.test_set import make_test_set


def check_input_arguments(args):

    """
    :param args:    the result of an ArgumentParser structure: it must contain the arguments 'corpus', 'pos',
                    'outcome_file', and 'celex_dict'
    """

    # check corpus file
    if not os.path.exists(args.input_corpus) or not args.input_corpus.endswith(".json"):
        raise ValueError("There are problems with the corpus file you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    # check extension of the output file for cue frequency counts
    if not args.pos.endswith(".txt"):
        raise ValueError("There are problems with the file with the PoS tag mapping you provided: either the path does "
                         "not exist or the file extension is not .txt. Provide a valid path to a .txt file.")

    # check extension of the output file for outcome frequency counts
    if not args.output_file.endswith(".txt"):
        raise ValueError("Indicate the path to a .txt file to store the words for the desired test_set.")


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Get a list of test items from the input corpus and Celex.')

    parser.add_argument('-I', '--input_corpus', required=True, dest='input_corpus',
                        help='Give the path to an input corpus (.json), encoded in phonetic cues and lexical outcomes.')
    parser.add_argument('-C', '--Celex_dir', required=True, dest='celex',
                        help='Give the path to the Celex directory.')
    parser.add_argument('-M', '--pos_tags', required=True, dest='pos',
                        help='Give the path to a .txt file mapping CHILDES pos tags to CELEX tags.')
    parser.add_argument('-O', '--output_file', required=True, dest='output_file',
                        help='Specify the path where the selected words will be printed.')
    parser.add_argument('-a', '--ambiguous', action="store_true", dest='ambiguous',
                        help='Specify whether phonetically ambiguous words are to be returned.')
    parser.add_argument('-n', "--new", action="store_true", dest="new",
                        help="Specify whether words that appear in CELEX but not in the corpus are to be returned.")

    args = parser.parse_args()

    check_input_arguments(args)

    make_test_set(args.input_corpus, args.celex, args.pos, args.output_file, args.ambiguous, args.new)


########################################################################################################################


if __name__ == '__main__':

    main()
