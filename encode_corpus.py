__author__ = 'GCassani'

"""Function to encode the corpus into phonetic cues and lexical outcomes (can be called from command line)"""

import os
import argparse
from corpus.encoder import corpus_encoder


def check_arguments(args, parser):

    if not (args.uni or args.di or args.tri or args.syl):
        parser.error('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    if not os.path.exists(args.celex_dir):
        raise ValueError("The Celex directory you provided doesn't exist. Provide a valid path.")

    if not os.path.exists(args.pos_mapping):
        raise ValueError("The file containing the mapping from CHILDES to Celex PoS tags isn't valid."
                         "Please provide a valid path.")

    if not os.path.exists(args.in_file) or not args.in_file.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Process arguments to create Celex dictionary.')

    parser.add_argument("-I", "--input_file", required=True, dest="in_file",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-C", "--Celex_dir", required=True, dest="celex_dir",
                        help="Specify the path to the Celex directory.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path to the file containing the mapping between CHILDES and Celex PoS tags.")
    parser.add_argument("-S", "--separator", dest="sep", default='~',
                        help="Specify the character separating lemma and PoS tag in the input corpus.")
    parser.add_argument("-o", "--outcomes", dest="outcomes", default='tokens',
                        help="Specify whethere to use 'lemmas' or 'tokens' (default) as lexical outcomes.")
    parser.add_argument("-u", "--uniphones", action="store_true", dest="uni",
                        help="Specify if uniphones need to be encoded.")
    parser.add_argument("-d", "--diphones", action="store_true", dest="di",
                        help="Specify if diphones need to be encoded.")
    parser.add_argument("-t", "--triphones", action="store_true", dest="tri",
                        help="Specify if triphones need to be encoded.")
    parser.add_argument("-s", "--syllables", action="store_true", dest="syl",
                        help="Specify if syllables need to be encoded.")
    parser.add_argument("-m", "--stress_marker", action="store_true", dest="stress",
                        help="Specify if stress need to be encoded.")
    parser.add_argument("-r", "--reduced", action="store_true", dest="reduced",
                        help="Specify if reduced vowels are to be considered when extracting CELEX phonetic forms.")
    parser.add_argument("-b", "--boundaries", action="store_true", dest="boundaries",
                        help="Specify whether word boundaries are to be considered when training on utterances.")

    args = parser.parse_args()

    check_arguments(args, parser)

    corpus_encoder(args.in_file, args.celex_dir, args.pos_mapping, separator=args.sep,
                   uniphones=args.uni, diphones=args.di, triphones=args.tri, syllables=args.syl,
                   stress_marker=args.stress, reduced=args.reduced, outcomes=args.outcomes,
                   boundaries=args.boundaries)


########################################################################################################################


if __name__ == '__main__':

    main()
