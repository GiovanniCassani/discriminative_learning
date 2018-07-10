_author__ = 'GCassani'

"""Function to carry out the simulations to compute usefulness of phonological cues and tokens as dimensions for
   language learning"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import itertools as it
from time import strftime
from matrix.matrix import load
from rescorla_wagner.ndl import ndl
from corpus.encoder import corpus_encoder
from celex.get import get_celex_dictionary
from analysis.discrimination import find_discriminated
import phonological_bootstrapping.clustering.usefulness as usf


def main():

    parser = argparse.ArgumentParser(description="Compute the variance of each phonological cue and token, as a proxy "
                                                 "to identify the amount of information they carry")

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
    number_of_cues = [100, 1000]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    cues_variance_file = os.path.join(args.output_folder, "words_cues_variance.csv")
    tokens_variance_file = os.path.join(args.output_folder, "words_tokens_variance.csv")

    parametrizations = it.product(cues, outcomes, stress_marker, boundaries, reduced_vowels)

    time_points = np.linspace(10, 100, 10) if args.longitudinal else [100]

    cues_table = pd.DataFrame(index=[], columns=["Corpus", "Cues", "Outcomes", "Stress", "Vowels", "Precision", "Time",
                                                 "Phonological_cue",
                                                 "Variance",
                                                 "Frequency",
                                                 "Lexical_diversity",
                                                 "Phonological_diversity",
                                                 "Cue|Cues_predictability",
                                                 "Cue|Tokens_predictability",
                                                 "Cues|Cue_predictability",
                                                 "Tokens|Cue_predictability"])

    tokens_table = pd.DataFrame(index=[], columns=["Corpus", "Cues", "Outcomes", "Stress", "Vowels", "Precision",
                                                   "Time", "numCues", 'Token',
                                                   "Variance",
                                                   "Frequency",
                                                   "Lexical_diversity",
                                                   "Phonological_diversity",
                                                   "Token|Tokens_predictability",
                                                   "Token|Cues_predictability",
                                                   "Tokens|Token_predictability",
                                                   "Cues|Token_predictability"])

    ii = 0
    jj = 0

    for parametrization in parametrizations:

        print(parametrization)

        cue_type, outcome, stress, boundary, reduced = parametrization

        uniphones = True if cue_type == 'uniphones' else False
        diphones = True if cue_type == 'diphones' else False
        triphones = True if cue_type == 'triphones' else False
        syllables = True if cue_type == 'syllables' else False
        vowels = 'reduced' if reduced else 'full'
        sm = "stress" if stress else 'no-stress'
        bound = 'yes' if boundary else 'no'
        training = os.path.splitext(os.path.basename(args.corpus))[0]

        encoded_corpus = corpus_encoder(args.corpus, args.celex_folder, args.pos_mapping, separator='~',
                                        stress_marker=stress, reduced=reduced, uniphones=uniphones,
                                        diphones=diphones, triphones=triphones, syllables=syllables,
                                        outcomes=outcome, boundaries=boundary)

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Started computing distributional statistics from the corpus...")
        token_statistics, cue_statistics = usf.compute_distributional_predictors(encoded_corpus, time_points)
        print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished computing distributional statistics from the corpus.")
        print()

        corpus_dir = os.path.dirname(encoded_corpus)

        a, b = [0.001, 0.001] if training == 'aggregate_utterances' else [0.01, 0.01]
        file_paths = ndl(encoded_corpus, alpha=a, beta=b, lam=1, longitudinal=args.longitudinal)

        celex_dict = get_celex_dictionary(args.celex_folder, reduced=reduced)

        for idx, file_path in file_paths.items():

            idx = int(idx)
            matrix, cues2ids, outcomes2ids = load(file_path)

            # get the column ids of all perfectly discriminated outcomes at the current time point
            # perfectly discriminated outcomes are considered to be those whose jaccard coefficient
            # between true phonetic cues and most active phonetic cues for the outcome is 1
            discriminated_file = os.path.join(corpus_dir, '.'.join(['discriminatedOutcomes',
                                                                    str(idx),
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
            print(strftime("%Y-%m-%d %H:%M:%S") + ": The discriminated outcomes have been identified (file: %s)."
                  % discriminated_file)
            print()

            row_variances, matrix, discriminated = usf.get_cue_variances(matrix, discriminated)
            cue_variances = {}
            for cue in cues2ids:
                cue_idx = cues2ids[cue]
                cue_variances[cue] = row_variances[cue_idx]

            print(strftime("%Y-%m-%d %H:%M:%S") + ": Started storing cue variances...")
            for cue in cue_variances:
                if len(cue_statistics[cue]['freq']) == 10:

                    frequency = cue_statistics[cue]['freq'][idx]
                    lexical_diversity = cue_statistics[cue]['lexdiv'][idx]
                    phonological_diversity = cue_statistics[cue]['phondiv'][idx]

                    # average conditional probability of a cue given the co-occurring cues
                    cue_cues_predictability = cue_statistics[cue]['p_cue_cues'][idx]

                    # average predictive power of a cue with respect to all the co-occurring cues
                    cues_cue_predictability = cue_statistics[cue]['p_cues_cue'][idx]

                    # average conditional probability of a cue given the co-occurring tokens
                    cue_tokens_predictability = cue_statistics[cue]['p_cue_tokens'][idx]

                    # average predictive power of a cue with respect to all the co-occurring tokens
                    tokens_cue_predictability = cue_statistics[cue]['p_tokens_cue'][idx]

                    cues_table.loc[ii] = pd.Series({"Corpus": training, "Cues": cue_type, "Outcomes": outcome,
                                                    "Stress": sm, "Boundaries": bound, "Vowels": vowels, "Time": idx,
                                                    "Precision": int(args.precision), "Phonological_cue": cue,
                                                    "Variance": cue_variances[cue], "Frequency": frequency,
                                                    "Lexical_diversity": lexical_diversity,
                                                    "Phonological_diversity": phonological_diversity,
                                                    "Cue|Cues_predictability": cue_cues_predictability,
                                                    "Cues|Cue_predictability": cues_cue_predictability,
                                                    "Cue|Tokens_predictability": cue_tokens_predictability,
                                                    "Tokens|Cue_predictability": tokens_cue_predictability})
                ii += 1

            print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished storing cue variances.")
            print()

            for how_many_cues in number_of_cues:

                print("Number of cues: ", how_many_cues)

                token_variances = usf.get_token_variances(matrix, discriminated, row_variances,
                                                          how_many_cues=how_many_cues)

                print(strftime("%Y-%m-%d %H:%M:%S") + ": Started storing token variances...")
                for token in token_variances:
                    if len(token_statistics[token]['freq']) == 10:

                        frequency = token_statistics[token]['freq'][idx]
                        lexical_diversity = token_statistics[token]['lexdiv'][idx]
                        phonological_diversity = token_statistics[token]['phondiv'][idx]

                        # average conditional probability of a token given the co-occurring tokens
                        token_tokens_predictability = token_statistics[token]['p_token_tokens'][idx]

                        # average predictive power of a token with respect to the co-occurring tokens
                        tokens_token_predictability = token_statistics[token]['p_tokens_token'][idx]

                        # average conditional probability of a token given the co-occurring phonological cues
                        token_cues_predictability = token_statistics[token]['p_token_cues'][idx]

                        # average predictive power of a token with respect to the co-occurring phonological cues
                        cues_token_predictability = token_statistics[token]['p_cues_token'][idx]

                        tokens_table.loc[jj] = pd.Series({"Corpus": training, "Cues": cue_type,
                                                          "Outcomes": outcome, "Stress": sm, "Boundaries": bound,
                                                          "Vowels": vowels, "Time": idx, "numCues": how_many_cues,
                                                          "Precision": int(args.precision), "Token": token,
                                                          "Variance": token_variances[token],
                                                          "Frequency": frequency,
                                                          "Lexical_diversity": lexical_diversity,
                                                          "Phonological_diversity": phonological_diversity,
                                                          "Token|Tokens_predictability": token_tokens_predictability,
                                                          "Tokens|Token_predictability": tokens_token_predictability,
                                                          "Token|Cues_predictability": token_cues_predictability,
                                                          "Cues|Token_predictability": cues_token_predictability})
                    jj += 1
                print(strftime("%Y-%m-%d %H:%M:%S") + ": ...finished storing token variances.")

                print()
                print('-' * 100)
                print()

            print()
            print()
            print('=' * 100)
            print('=' * 100)
            print()
            print()

        print()
        print()
        print()
        print('#' * 100)
        print('#' * 100)
        print('#' * 100)
        print()
        print()
        print()

    if os.path.exists(cues_variance_file):
        cues_table.to_csv(cues_variance_file, sep='\t', index=False, mode="a", header=False)
    else:
        cues_table.to_csv(cues_variance_file, sep='\t', index=False)

    if os.path.exists(tokens_variance_file):
        tokens_table.to_csv(tokens_variance_file, sep='\t', index=False, mode="a", header=False)
    else:
        tokens_table.to_csv(tokens_variance_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
