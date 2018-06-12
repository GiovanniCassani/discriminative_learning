_author__ = 'GCassani'

"""Function to perform the PoS tagging experiment (can be called from command line)"""

import os
import argparse
import numpy as np
import phonological_bootstrapping.tagging.error_analysis as err
from phonological_bootstrapping.tagging.grid import grid_search


def main():

    parser = argparse.ArgumentParser(description="Evaluate PoS tagging accuracy using test sets of new, unknown words")

    parser.add_argument("-c", "--corpus", required=True, dest="corpus",
                        help="Specify the path to the training corpus (encoded as .json).")
    parser.add_argument("-T", "--test_set_folder", required=True, dest="test_set_folder",
                        help="The path to the folder containing all the files to be used as test sets.")
    parser.add_argument("-C", "--Celex_folder", required=True, dest="celex_folder",
                        help="Specify the folder where the Celex data are located.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path of the file containing the mapping from CHILDES to Celex PoS tags.")
    parser.add_argument("-O", "--output_folder", required=True, dest="output_folder",
                        help="Specify the path of the folder where the logfiles will be stored together with"
                             "the summary tables.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")
    parser.add_argument("-p", "--precision", dest="precision", default=5,
                        help="Specify how many outcomes to consider to compute precision.")

    args = parser.parse_args()

    precision = int(args.precision)

    # for words
    # a, b = [0.01, 0.01]
    # for utterances
    a, b = [0.001, 0.001]
    threshold = 0
    flush = [0]
    k = [50]
    methods = ['freq']
    reduced_vowels = [False]
    evaluations = ['count']

    # experimental contrasts: cues, stress
    outcomes = ['tokens']
    cues = ['triphones', 'syllables']
    boundaries = [True]
    stress_marker = [True, False]

    if os.path.isdir(args.test_set_folder):
        test_sets = os.listdir(args.test_set_folder)
        test_set_files = [os.path.join(args.test_set_folder, test_set) for test_set in test_sets]
    else:
        raise ValueError("Please provide a valid path for the folder containing the test files.")

    log_folder = os.path.join(args.output_folder, "log_files/tagging/")
    summary_file = os.path.join(args.output_folder, "PoStagging_summary.csv")
    error_file = os.path.join(args.output_folder, "PoStagging_errors.csv")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    for test_file in test_set_files:

        summary_table = grid_search(test_file, log_folder, args.corpus, cues, outcomes, boundaries, stress_marker,
                                    reduced_vowels, methods, evaluations, k, flush, threshold, args.celex_folder,
                                    args.pos_mapping, args.longitudinal, precision, a, b)

        # write summary statistics to file
        if os.path.exists(summary_file):
            summary_table.to_csv(summary_file, sep='\t', index=False, mode="a", header=False)
        else:
            summary_table.to_csv(summary_file, sep='\t', index=False)

    time_indices = np.linspace(10, 100, 10) if args.longitudinal else [100]
    corpora = [os.path.splitext(os.path.basename(args.corpus))[0]]
    categorization_outcomes, categorization_columns = err.compute_error_analysis(log_folder, test_sets,
                                                                                 corpora, boundaries, outcomes,
                                                                                 cues, stress_marker, reduced_vowels,
                                                                                 methods, evaluations, k, flush,
                                                                                 precision, time_indices)

    error_analysis_dataset = err.update_error_analysis_dataset(categorization_outcomes, categorization_columns)
    error_analysis_dataset.to_csv(error_file, sep='\t', index=False, mode="w", header=True)


########################################################################################################################


if __name__ == '__main__':

    main()
