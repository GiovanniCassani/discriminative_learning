_author__ = 'GCassani'

"""Function to perform LDA classification experiments varying thresholds and corpora"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import itertools as it
from matrix.matrix import load
from rescorla_wagner.ndl import ndl
from corpus.encoder import corpus_encoder
from celex.get import get_celex_dictionary
from analysis.discrimination import find_discriminated
from phonological_bootstrapping.clustering.lda import subset_experiment


def main():

    parser = argparse.ArgumentParser(description="Assess whether words from the same category cluster together"
                                                 "first considering their sound patterns and then how they correlate"
                                                 "to each other based on their contexts of occurrence")

    parser.add_argument("-c", "--corpus", required=True, dest="corpus",
                        help="Specify the path to the training corpus (encoded as .json).")
    parser.add_argument("-C", "--celex_folder", required=True, dest="celex_folder",
                        help="Specify the folder where the Celex data are located.")
    parser.add_argument("-O", "--output_folder", required=True, dest="output_folder",
                        help="Specify the path of the folder where the logfiles will be stored together with"
                             "the summary tables.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path of the file containing the mapping from CHILDES to Celex PoS tags.")
    parser.add_argument("-p", "--precision", dest="precision", default=5,
                        help="Specify the number of outcomes to consider when computing discrimination's precision.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()

    cues = ['triphones']
    outcomes = ['tokens']
    stress_marker = [True]
    boundaries = [True]
    reduced_vowels = [False]
    distances = ['correlation']
    number_of_cues = [100, 500, 1000]
    number_of_tokens = [50, 250,  500]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    summary_file = os.path.join(args.output_folder, "LDAk_summary.csv")
    # error_file = os.path.join(args.output_folder, "PoStagging_errors.csv")

    parametrizations = it.product(cues, outcomes, stress_marker, boundaries, reduced_vowels, distances,
                                  number_of_cues, number_of_tokens)

    time_points = np.linspace(10, 100, 10) if args.longitudinal else [100]
    rows = int(np.prod([len(cues), len(outcomes), len(stress_marker), len(reduced_vowels), len(boundaries),
                        len(time_points), len(distances), len(number_of_cues), len(number_of_tokens)]))
    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Corpus", "Cues", "Outcomes", "Stress", "Vowels", "Precision", "Time",
                                          "Distance", "numCues", "numTokens", "Phon_acc", "Phon_acc_subset",
                                          "Phon_baseline", "Distr_acc", "Distr_acc_subset", "Distr_baseline"])

    ii = 0
    for parametrization in parametrizations:

        print(parametrization)

        cue, outcome, stress, boundary, reduced, distance, how_many_cues, how_many_tokens = parametrization

        uniphones = True if cue == 'uniphones' else False
        diphones = True if cue == 'diphones' else False
        triphones = True if cue == 'triphones' else False
        syllables = True if cue == 'syllables' else False
        vowels = 'reduced' if reduced else 'full'
        sm = "stress" if stress else 'no-stress'
        bound = 'yes' if boundary else 'no'
        training = os.path.splitext(os.path.basename(args.corpus))[0]

        encoded_corpus = corpus_encoder(args.corpus, args.celex_folder, args.pos_mapping, separator='~',
                                        stress_marker=stress, reduced=reduced, uniphones=uniphones,
                                        diphones=diphones, triphones=triphones, syllables=syllables,
                                        outcomes=outcome, boundaries=boundary)

        corpus_dir = os.path.dirname(encoded_corpus)

        a, b = [0.001, 0.001] if training == 'aggregate_utterances' else [0.01, 0.01]
        file_paths = ndl(encoded_corpus, alpha=a, beta=b, lam=1, longitudinal=args.longitudinal)

        celex_dict = get_celex_dictionary(args.celex_folder, reduced=reduced)

        for idx, file_path in file_paths.items():

            matrix, cues2ids, outcomes2ids = load(file_path)

            # get the column ids of all perfectly discriminated outcomes at the current time point
            # perfectly discriminated outcomes are considered to be those whose jaccard coefficient
            # between true phonetic cues and most active phonetic cues for the outcome is 1
            discriminated_file = os.path.join(corpus_dir,
                                              '.'.join(['discriminatedOutcomes',
                                                        str(int(idx)),
                                                        ''.join(['at', args.precision]),
                                                        'json']))

            if not os.path.exists(discriminated_file):
                discriminated = find_discriminated(matrix, cues2ids, outcomes2ids, celex_dict,
                                                   stress_marker=stress_marker, uniphones=uniphones,
                                                   diphones=diphones, triphones=triphones,
                                                   syllables=syllables, boundaries=boundaries, at=int(args.precision))
                json.dump(discriminated, open(discriminated_file, 'w'))
            else:
                discriminated = json.load(open(discriminated_file, 'r'))

            print()
            print("The discriminated outcomes have been identified (file: %s)." % discriminated_file)

            accuracies = subset_experiment(matrix, discriminated,
                                           how_many_cues=how_many_cues, how_many_tokens=how_many_tokens)

            summary_table.loc[ii] = pd.Series({"Corpus": training, "Cues": cue,"Outcomes": outcome, "Stress": sm,
                                               "Boundaries": bound, "Vowels": vowels, "Time": int(idx),
                                               "Distance": distance, "Precision": args.precision,
                                               "numCues": how_many_cues, "numTokens": how_many_tokens,
                                               "Phon_acc": accuracies[0], "Phon_acc_subset": accuracies[1],
                                               "Distr_acc": accuracies[3], "Distr_acc_subset": accuracies[4],
                                               "Phon_baseline": accuracies[2], "Distr_baseline": accuracies[5]})
            ii += 1

    if os.path.exists(summary_file):
        summary_table.to_csv(summary_file, sep='\t', index=False, mode="a", header=False)
    else:
        summary_table.to_csv(summary_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
