_author__ = 'GCassani'

"""Function to perform the PoS tagging experiment (can be called from command line)"""

import os
import argparse
from grid_search.run import run_grid_search


def main():

    parser = argparse.ArgumentParser(description="Run a grid search to explore all possible parameters.")

    parser.add_argument("-c", "--corpus_folder", required=True, dest="corpus_folder",
                        help="Specify the folder where the training corpora are located.")
    parser.add_argument("-T", "--test_set_folder", required=True, dest="test_set_folder",
                        help="The path to the folder containing all the files to be used as test sets.")
    parser.add_argument("-C", "--Celex_folder", required=True, dest="celex_folder",
                        help="Specify the folder where the Celex data are located.")
    parser.add_argument("-O", "--output_file", required=True, dest="out_file",
                        help="Specify the path of the .csv output file where summary statistics will be written to.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path of the file containing the mapping from CHILDES to Celex PoS tags.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()

    # values were picked on the basis of a grid search
    flush = [100]
    k = [50]
    evaluations = ['count']
    methods = ['freq']
    reduced_vowels = [False]

    # experimental contrasts: training regime, cues, outcomes, stress
    training_corpora = [os.path.join(args.corpus_folder, "aggregate_utterances.json"),
                        os.path.join(args.corpus_folder, "aggregate_words.json")]
    cues = ['triphones', 'syllables']
    outcomes = ['lemmas', 'tokens']
    stress_marker = [True, False]

    if os.path.isdir(args.test_set_folder):
        test_sets = os.listdir(args.test_set_folder)
        test_set_files = [os.path.join(args.test_set_folder, test_set) for test_set in test_sets]
    else:
        raise ValueError("Please provide a valid path for the folder containing the test files.")

    summary_table = run_grid_search(test_set_files, training_corpora, cues, outcomes, stress_marker,
                                    reduced_vowels, methods, evaluations, k, flush,
                                    args.celex_folder, args.pos_mapping, args.longitudinal)

    summary_table.to_csv(args.out_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
