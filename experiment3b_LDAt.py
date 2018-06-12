_author__ = 'GCassani'

"""Function to perform LDA classification experiments varying thresholds and corpora"""

import os
import json
import operator
import argparse
import numpy as np
import pandas as pd
import itertools as it
from matrix.matrix import load
from rescorla_wagner.ndl import ndl
from corpus.encoder import corpus_encoder
from celex.get import get_celex_dictionary
from analysis.discrimination import find_discriminated
from phonological_bootstrapping.clustering.lda import threshold_experiment


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
    parser.add_argument("--cue_threshold", dest="cue_threshold", default='high',
                        help="Specify whether to choose a 'high' (i.e. strict) the threshold on relevant cues or"
                             "a low (i.e. lax) one.")
    parser.add_argument("--token_threshold", dest="token_threshold", default='low',
                        help="Specify whether to choose a 'high' (i.e. strict) the threshold on relevant tokens or"
                             "a low (i.e. lax) one.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()

    # thresholds have been determined manually according to the following criteria:
    # - high thresholds were set to yield around 100 dimensions at time t100
    # - low thresholds were set to yield around 100 dimensions at time t10
    # Whenever a threshold value didn't yield any dimension because the value was too stringent, the threshold was
    # lowered until at least 1 dimension was available at all time points. First, I adjusted the threshold on
    # phonological cues' variance, then on tokens' variance. Practically, it was always possible to set thresholds to
    # yield around 100 dimensions at the specified time points, except for tokens' variance in the low variance setting,
    # wheere the threshold yielding around 100 dimensions at t10 quickly left the model without any dimension.
    # Therefore, these models start with considerably high dimensionalities, and finishes with almost no dimensionans.

    thresholds = {
        'aggregate_utterances_at5_c_low': 0.00000001,                   # 99 cues at t10 (799 at t100)
        'aggregate_utterances_at5_c_high': 0.00000075,                  # 105 cues at t100 (2 cues at t10)
        'aggregate_utterances_at5_t_low': 0.02,                         # 741 tokens at t10 (3 tokens at t100)
        'aggregate_utterances_at5_t_high': 0.04,                        # 125 tokens at 100 (843 tokens at t10)

        'aggregate_utterances_at25_c_low': 0.000000005,                 # 98 cues at t10 (802 cues at t100)
        'aggregate_utterances_at25_c_high': 0.00000025,                 # 156 cues at t100 (2 cues at t10)
        'aggregate_utterances_at25_t_low': 0.015,                       # 1632 tokens at t10 (19 tokens at t100)
        'aggregate_utterances_at25_t_high': 0.033,                      # 140 tokens at t100 (1731 at t10)

        'aggregate_words_at5_c_low': 0.000025,                          # 101 cues at t10 (318 cues at t100)
        'aggregate_words_at5_c_high': 0.00005,                          # 131 cues at t100 (40 cues at t10)
        'aggregate_words_at5_t_low': 0.0325,                            # 834 tokens at t10 (3 tokens at t100)
        'aggregate_words_at5_t_high': 0.07,                             # 95 tokens at t100 (1004 tokens at t10)

        'aggregate_words_at25_c_low': 0.0000075,                        # 117 cues at t10 (416 cues at t100)
        'aggregate_words_at25_c_high': 0.00002,                         # 124 cues at t100 (33 cues at t10)
        'aggregate_words_at25_t_low': 0.0295,                           # 1850 tokens at t10 (7 tokens at t100)
        'aggregate_words_at25_t_high': 0.085                            # 113 tokens at t100 (2581)
    }

    cues = ['triphones']
    outcomes = ['tokens']
    stress_marker = [True]
    boundaries = [True]
    reduced_vowels = [False]
    distances = ['correlation']

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    summary_file = os.path.join(args.output_folder, "LDAt_summary.csv")
    # error_file = os.path.join(args.output_folder, "PoStagging_errors.csv")

    parametrizations = it.product(cues, outcomes, stress_marker, boundaries, reduced_vowels, distances)

    time_points = np.linspace(10, 100, 10) if args.longitudinal else [100]
    rows = int(np.prod([len(cues), len(outcomes), len(stress_marker), len(reduced_vowels), len(boundaries),
                        len(time_points), len(distances)]))
    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Corpus", "Cues", "Outcomes", "Stress", "Vowels", "Precision",
                                          "Time", "Distance", "tCues", "numCues", "tTokens", "numTokens",
                                          "Phon_acc", "Phon_acc_subset", "Phon_baseline",
                                          "Distr_acc", "Distr_acc_subset", "Distr_baseline"])

    ii = 0
    for parametrization in parametrizations:

        cue, outcome, stress, boundary, reduced, distance = parametrization

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

        # precision at 25
        a, b = [0.001, 0.001] if training == 'aggregate_utterances' else [0.01, 0.01]
        c = thresholds['_'.join([training, ''.join(['at', args.precision]), 'c', args.cue_threshold])]
        t = thresholds['_'.join([training, ''.join(['at', args.precision]), 't', args.token_threshold])]

        file_paths = ndl(encoded_corpus, alpha=a, beta=b, lam=1, longitudinal=args.longitudinal)

        celex_dict = get_celex_dictionary(args.celex_folder, reduced=reduced)

        for idx, file_path in sorted(file_paths.items(), key=operator.itemgetter(0)):

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
                                                   syllables=syllables, boundaries=boundaries, at=args.precision)
                json.dump(discriminated, open(discriminated_file, 'w'))
            else:
                discriminated = json.load(open(discriminated_file, 'r'))

            print()
            print("The discriminated outcomes have been identified (file: %s)." % discriminated_file)

            accuracies = threshold_experiment(matrix, discriminated, cues_threshold=c, tokens_threshold=t)

            summary_table.loc[ii] = pd.Series({"Corpus": training, "Cues": cue,"Outcomes": outcome, "Stress": sm,
                                               "Boundaries": bound, "Vowels": vowels, "Time": int(idx),
                                               "Distance": distance, "Precision": args.precision,
                                               "tCues": args.cue_threshold, "numCues": accuracies[3],
                                               "tTokens": args.token_threshold, "numTokens": accuracies[7],
                                               "Phon_acc": accuracies[0], "Phon_acc_subset": accuracies[1],
                                               "Distr_acc": accuracies[4], "Distr_acc_subset": accuracies[5],
                                               "Phon_baseline": accuracies[2], "Distr_baseline": accuracies[6]})
            ii += 1

    if os.path.exists(summary_file):
        summary_table.to_csv(summary_file, sep='\t', index=False, mode="a", header=False)
    else:
        summary_table.to_csv(summary_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
