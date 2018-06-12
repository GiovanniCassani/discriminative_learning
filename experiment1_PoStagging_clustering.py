_author__ = 'GCassani'

"""Function to perform the clustering experiment (can be called from command line)"""

import os
import argparse
import numpy as np
import phonological_bootstrapping.clustering.error_analysis as err
from phonological_bootstrapping.clustering.grid import grid_search


def main():

    parser = argparse.ArgumentParser(description="Assess whether words from the same category cluster together"
                                                 "on the basis of the sound sequences they consist of.")

    parser.add_argument("-c", "--corpus", required=True, dest="corpus",
                        help="Specify the path to the training corpus (encoded as .json).")
    parser.add_argument("-C", "--Celex_folder", required=True, dest="celex_folder",
                        help="Specify the folder where the Celex data are located.")
    parser.add_argument("-O", "--output_folder", required=True, dest="output_folder",
                        help="Specify the path of the folder where the logfiles will be stored together with"
                             "the summary tables.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path of the file containing the mapping from CHILDES to Celex PoS tags.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")
    parser.add_argument("-p", "--precision", dest="precision", default=5,
                        help="Specify how many outcomes to consider to compute precision.")

    args = parser.parse_args()

    precision = int(args.precision)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    log_folder = os.path.join(args.output_folder, "log_files/clustering/")
    summary_file = os.path.join(args.output_folder, "PoSclustering_summary.csv")
    error_file = os.path.join(args.output_folder, "PoSclustering_errors.csv")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # for words
    # a, b = [0.01, 0.01]

    # for utterances
    a, b = [0.001, 0.001]

    nn = [50]
    distances = ['correlations']
    reduced_vowels = [False]

    # experimental contrasts: cues, stress
    outcomes = ['tokens']
    cues = ['triphones', 'syllables']
    boundaries = [True]
    stress_marker = [True, False]

    summary_table = grid_search(args.corpus, log_folder, args.pos_mapping, args.celex_folder, outcomes, cues,
                                boundaries, stress_marker, reduced_vowels, distances, nn, precision,
                                args.longitudinal, a, b)

    # write summary statistics to file
    if os.path.exists(summary_file):
        summary_table.to_csv(summary_file, sep='\t', index=False, mode="a",
                             header=False)
    else:
        summary_table.to_csv(summary_file, sep='\t', index=False)

    time_indices = np.linspace(10, 100, 10) if args.longitudinal else [100]
    corpora = [os.path.splitext(os.path.basename(args.corpus))[0]]
    categorization_outcomes, categorization_columns = err.compute_error_analysis(log_folder, corpora, boundaries,
                                                                                 outcomes, cues, stress_marker,
                                                                                 reduced_vowels, distances, nn,
                                                                                 precision, time_indices)

    error_analysis_dataset = err.update_error_analysis_dataset(categorization_outcomes, categorization_columns)
    error_analysis_dataset.to_csv(error_file, sep='\t', index=False, mode="w", header=True)


########################################################################################################################


if __name__ == '__main__':

    main()
