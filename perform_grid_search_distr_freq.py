__author__ = 'GCassani'

"""Function to perform the grid search experiment (can be called from command line)"""

import os
import argparse
from grid_search.run import run_grid_search


def main():

    parser = argparse.ArgumentParser(description="Run a grid search to explore all possible parameters.")

    parser.add_argument("-c", "--corpus_folder", required=True, dest="corpus_folder",
                        help="Specify the folder where the training corpora are located.")
    parser.add_argument("-T", "--test_set", required=True, dest="test_set",
                        help="The path to the file containing the test items."
                             "If a the path to a folder is passed, all the files in it are considered as test sets.")
    parser.add_argument("-C", "--Celex_folder", required=True, dest="celex_folder",
                        help="Specify the folder where the Celex data are located.")
    parser.add_argument("-O", "--output_file", required=True, dest="out_file",
                        help="Specify the path of the output file.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path of the file containing the mapping from CHILDES to Celex PoS tags.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()

    # hardcoded values, specific to the experiment I ran
    stress_marker = [True]
    reduced = [False]
    flush = [0, 50, 100, 200]
    k = [25, 50, 75, 100]
    evaluations = ['distr']
    methods = ['freq']
    training_corpora = [os.path.join(args.corpus_folder, "aggregate_utterances.json"),
                        os.path.join(args.corpus_folder, "aggregate_words.json")]
    cues = ['triphones', 'syllables']
    outcomes = ['lemmas', 'tokens']

    if os.path.isdir(args.test_set):
        test_sets = os.listdir(args.test_set)
    else:
        if os.path.isfile(args.test_set):
            test_sets = [args.test_set]
        else:
            raise ValueError("Please provide a valid path for the test file(s).")

    print(test_sets)

    summary_table = run_grid_search(test_sets, training_corpora, cues, outcomes, stress_marker,
                                    reduced, methods, evaluations, k, flush,
                                    args.celex_folder, args.pos_mapping, args.longitudinal)

    summary_table.to_csv(args.out_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
