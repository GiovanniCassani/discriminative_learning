_author__ = 'GCassani'

"""Function to perform the error analysis on any PoS tagging experiment, feeding the folder containing the test sets
   of interest and the folder where the log files of the experiment are located."""

import os
import argparse
import numpy as np
import phonological_bootstrapping.tagging.error_analysis as err


def main():

    parser = argparse.ArgumentParser(description="Run a grid search to explore all possible parameters.")

    parser.add_argument("-T", "--test_set_folder", required=True, dest="test_set_folder",
                        help="The path to the folder containing all the files to be used as test sets.")
    parser.add_argument("-L", "--logs_folder", required=True, dest="logs_folder",
                        help="Specify the folder where the log files resulting from a PoS tagging "
                             "experiment are located.")
    parser.add_argument("-O", "--output_file", required=True, dest="out_file",
                        help="Specify the path of the .csv output file where summary statistics will be written to.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to adopt a longitudinal design (default: False).")

    args = parser.parse_args()

    if os.path.isdir(args.test_set_folder):
        test_sets = os.listdir(args.test_set_folder)
    else:
        raise ValueError("Please provide a valid path for the folder containing the test files.")

    flush = [0]
    k = [50]
    at = 5
    methods = ['freq']
    evaluations = ['count']
    reduced_vowels = [False]
    corpora = ['aggregate_words']
    outcomes = ['tokens']
    boundaries = [True]
    cues = ['triphones', 'syllables']
    stress_marker = [True, False]
    time_indices = np.linspace(10, 100, 10) if args.longitudinal else [100]

    categorization_outcomes, categorization_columns = err.compute_error_analysis(args.logs_folder, test_sets,
                                                                                 corpora, boundaries, outcomes, cues,
                                                                                 stress_marker, reduced_vowels,
                                                                                 methods, evaluations,
                                                                                 k, flush, at, time_indices)

    error_analysis_dataset = err.update_error_analysis_dataset(categorization_outcomes, categorization_columns)
    error_analysis_dataset.to_csv(args.out_file, sep='\t', index=False, mode="w", header=True)


########################################################################################################################


if __name__ == '__main__':

    main()
