__author__ = 'GCassani'

"""Function to estimate cue-outcome connections given an input corpus encoded as lists of cues matched to sets of
   outcomes using naïve discriminative learning (can be called from command line)"""

import os
import argparse
from rescorla_wagner.ndl import ndl


def main():

    parser = argparse.ArgumentParser(description='Process arguments to create Celex dictionary.')

    parser.add_argument("-I", "--input_corpus", required=True, dest="input_corpus",
                        help="Specify the corpus to be used as input, consisting of lists of phonetic cues"
                             "paired to sets of lexical outcomes (the file needs to be encoded as .json).")
    parser.add_argument("-L", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to work in a longitudinal design or not (default: False).")
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01,
                        help="Specify the value of the alpha parameter.")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01,
                        help="Specify the value of the beta parameter.")
    parser.add_argument("-l", "--lambda", dest="lam", default=1.0,
                        help="Specify the value of the lambda parameter.")

    args = parser.parse_args()

    if not os.path.exists(args.input_corpus) or not args.input_corpus.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    ndl(args.input_corpus, longitudinal=args.longitudinal,
        alpha=float(args.alpha), beta=float(args.beta), lam=float(args.lam))


########################################################################################################################


if __name__ == '__main__':

    main()
