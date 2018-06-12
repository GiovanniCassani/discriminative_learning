__author__ = 'GCassani'

"""Function to run the phonetic bootstrapping experiment (can be called from command line)"""

import os
import argparse
from collections import defaultdict
from phonological_bootstrapping.test.make import get_words_and_tags_from_test_set
from phonological_bootstrapping.helpers import compute_baselines
from phonological_bootstrapping.tagging.experiment import tag_words


def check_arguments(args, parser):

    # make sure that at least one phonetic feature is provided
    if not (args.uni or args.di or args.tri or args.syl):
        parser.error('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    # make sure the argument passed for celex_dir is a valid folder
    if not os.path.exists(args.celex_dir):
        raise ValueError("The path you provided for the Celex directory doesn't exist! Provide a valid one.")

    # make sure the training corpus exists and is a .json file
    if not os.path.exists(args.input_corpus) or not args.input_corpus.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    # make sure the files with the test items exists and is a .txt file
    if not os.path.exists(args.test_items_file) or not args.test_items_file.endswith(".txt"):
        raise ValueError("There are problems with the test items file you provided: either the path does not exist or"
                         "the file extension is not .txt. Provide a valid path to a .txt file.")


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Run a full phonetic bootstrapping experiment using the NDL model.')

    parser.add_argument("-I", "--input_corpus", required=True, dest="input_corpus",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-T", "--test_file", required=True, dest="test_file",
                        help="Specify the path to the file containing test items (encoded as .txt)."
                             "If the file doesn't exist, one is created according to the values of the"
                             "parameters -N and -A.")
    parser.add_argument("-C", "--Celex_dir", required=True, dest="celex_dir",
                        help="Specify the directory containing the CELEX files.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path to the file containing the mapping from CHILDES to Celex PoS tags.")
    parser.add_argument("-O", "--output_folder", required=True, dest="output_folder",
                        help="Specify the path to the folder where experiment's outcomes will be stored.")
    parser.add_argument("-S", "--separator", dest="sep", default='~',
                        help="Specify the character separating lemma and PoS tag in the input corpus.")
    parser.add_argument("-N", "--new", action="store_true", dest="new",
                        help="Specify if words from Celex but not in the corpus are to be used as test items"
                             "(only if the -T argument doesn't yet exist).")
    parser.add_argument("-A", "--ambiguous", action="store_true", dest="ambiguous",
                        help="Specify whether syntactically ambiguous or unambiguous words are to be used "
                             "as test items (only if the -T argument doesn't yet exist).")
    parser.add_argument("-q", "--method", dest="method", default="freq",
                        help="Specify whether to look at frequency ('freq') or total activation ('sum') of PoS tags.")
    parser.add_argument("-e", "--evaluation", dest="evaluation", default="count",
                        help="Specify whether to consider counts ('count') or compare distributions ('distr').")
    parser.add_argument("-f", "--flush", dest="flush", default=0,
                        help="Specify whether (and how many) top active outcome words to flush away from computations.")
    parser.add_argument("-k", "--threshold", dest="threshold", default=100,
                        help="Specify how many top active nodes are considered to cluster PoS tags.")
    parser.add_argument('-r', '--reduced', action='store_true', dest='reduced',
                        help='Specify if reduced phonological forms are to be considered.')
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
    parser.add_argument("-o", "--outcomes", dest="outcomes", default='tokens',
                        help="Specify which lexical outcomes to use, 'tokens' (default) or 'lemmas'.")
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01,
                        help="Specify the value of the alpha parameter.")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01,
                        help="Specify the value of the beta parameter.")
    parser.add_argument("-l", "--lambda", dest="lam", default=1,
                        help="Specify the value of the lambda parameter.")
    parser.add_argument("-L", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")
    parser.add_argument("-B", "--boundaries", action="store_true", dest="boundaries",
                        help="Specify whether to consider word boundaries during training (default: False).")

    args = parser.parse_args()

    check_arguments(args, parser)

    test_set = defaultdict(dict)
    test_words, tags = get_words_and_tags_from_test_set(args.test_file)
    test_set['filename'] = os.path.basename(args.test_file)
    test_set['items'] = test_words
    majority_baseline, entropy_baseline = compute_baselines(tags)
    test_set['majority'] = majority_baseline
    test_set['entropy'] = entropy_baseline

    # run the experiments using the input parameters
    tag_words(args.input_corpus, test_set, args.celex_dir, args.pos_mapping, args.output_folder,
              method=args.method, evaluation=args.evaluation, k=int(args.threshold), flush=int(args.flush),
              separator=args.sep, reduced=args.reduced, stress_marker=args.stress, outcomes=args.outcomes,
              uniphones=args.uni, diphones=args.di, triphones=args.tri, syllable=args.syl,
              longitudinal=args.longitudinal, boundaries=args.boundaries,
              alpha=float(args.alpha), beta=float(args.beta), lam=float(args.lam))


########################################################################################################################


if __name__ == '__main__':

    main()
