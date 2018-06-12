_author__ = 'GCassani'

"""Function to track the lexical development of each model (can be called from command line)"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import itertools as it
from matrix.matrix import load
from rescorla_wagner.ndl import ndl
from analysis.jaccard import jaccard
from corpus.encoder import corpus_encoder
from celex.get import get_celex_dictionary
from analysis.precision import precision_at


def get_cumulative_vocabulary(corpus_file, indices):

    """
    :param corpus_file: the path to the corpus (encoded as .json) encoded using the corpus_encoder() function from
                        the module corpus.encoder
    :param indices:     an iterable indicating at which percentages of the corpus to store vocabulary estimates
    :return vocabulary: a dictionary mapping indices to corresponding vocabulary estimates from the corpus
    """

    corpus = json.load(open(corpus_file, 'r'))
    outcomes = set()
    vocabulary = {}

    total = len(corpus[0])
    check_points = {np.floor(total / float(100) * n): n for n in indices}

    for ii in range(len(corpus[0])):
        for outcome in set(corpus[1][ii]):
            outcomes.add(outcome)
        if ii+1 in check_points:
            vocabulary[check_points[ii+1]] = len(outcomes)

    return vocabulary


########################################################################################################################


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
    parser.add_argument("-p", "--precision", dest="precision", default=5,
                        help="Specify the number of outcomes to consider when computing discrimination's precision.")
    parser.add_argument("-l", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()
    at = args.precision

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    summary_file = os.path.join(args.output_folder, "lexicalDevelopment_summary.csv")

    a, b = [0.01, 0.01]
    reduced_vowels = [False]
    boundaries = [True]
    outcomes = ['tokens']
    cues = ['triphones', 'syllables']
    stress_marker = [True]

    time_points = np.linspace(10, 100, 10) if args.longitudinal else [100]
    rows = int(np.prod([len(cues), len(outcomes), len(stress_marker), len(reduced_vowels),
                        len(boundaries), len(time_points)]))
    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Corpus", "Boundaries", "Cues", "Outcomes", "Stress", "Vowels", "Time", "At",
                                          "Discriminated", "@".join(["Precision", str(at)]), "Jaccard@1", "Total"])

    row_id = 0
    parametrizations = it.product(reduced_vowels, boundaries, outcomes, cues, stress_marker)
    for parametrization in parametrizations:

        r, boundary, outcome, cue, marker = parametrization
        uniphones = True if cue == 'uniphones' else False
        diphones = True if cue == 'diphones' else False
        triphones = True if cue == 'triphones' else False
        syllables = True if cue == 'syllables' else False
        vowels = 'reduced' if r else 'full'
        sm = "stress" if marker else 'no-stress'
        bound = 'yes' if boundary else 'no'
        training = os.path.splitext(os.path.basename(args.corpus))[0]
        celex_dict = get_celex_dictionary(args.celex_folder, reduced=r)

        encoded_corpus = corpus_encoder(args.corpus, args.celex_folder, args.pos_mapping, separator='~',
                                        stress_marker=marker, reduced=r, outcomes=outcome, boundaries=boundary,
                                        uniphones=uniphones, diphones=diphones, triphones=triphones,
                                        syllables=syllables)

        cumulative_vocabulary = get_cumulative_vocabulary(encoded_corpus, time_points)
        print()
        print("The cumulative vocabulary for the file %s has been estimated" % encoded_corpus)
        print()

        file_paths = ndl(encoded_corpus, alpha=a, beta=b, lam=1, longitudinal=args.longitudinal)

        for idx, file_path in file_paths.items():
            matrix, cues2ids, outcomes2ids = load(file_path)

            # get the Jaccard coefficient for each outcome and select those with a coefficient of 1, meaning that the
            # model would choose all and only the correct cues when expressing an the outcome; get the number of such
            # outcomes
            print()
            jaccard_coefficients = jaccard(matrix, cues2ids, outcomes2ids, celex_dict, stress_marker=marker,
                                           uniphone=uniphones, diphone=diphones, triphone=triphones, syllable=syllables,
                                           boundaries=boundaries)
            jaccard_one = {}
            for token in outcomes2ids:
                if jaccard_coefficients[token] == 1:
                    jaccard_one[token] = outcomes2ids[token]
            n_jaccard = len(jaccard_one)
            print()

            # get the outcomes that are correctly discriminated given the cues they consist of: in detail, take an
            # outcome, encode it in its phonetic cues, check which outcomes are most active given such cues, check
            # whether the correct one is among the top ones (how many is indicated by the parameter 'at'; store all
            # outcomes where the correct one is among the top active ones given the cues in it
            print()
            precise = precision_at(matrix, outcomes2ids, cues2ids, celex_dict, stress_marker=marker,
                                   uniphone=uniphones, diphone=diphones, triphone=triphones, syllable=syllables,
                                   boundaries=boundaries, at=at)
            n_precise = len(precise)
            print()

            # repeat but only for the outcomes with a Jaccar coefficient of 1, to quantify two-way discrimination
            print()
            discriminated = precision_at(matrix, jaccard_one, cues2ids, celex_dict, stress_marker=marker,
                                         uniphone=uniphones, diphone=diphones, triphone=triphones, syllable=syllables,
                                         boundaries=boundaries, at=at)
            n_discriminated = len(discriminated)
            print()

            vocabulary_estimate = cumulative_vocabulary[int(idx)]

            summary_table.loc[row_id] = pd.Series({"Corpus": training, "Cues": cue, "Outcomes": outcome, "Stress": sm,
                                                   "Boundaries": bound, "Vowels": vowels, "Time": int(idx), "At": at,
                                                   "Discriminated": n_discriminated, "Total": vocabulary_estimate,
                                                   "@".join(["Precision", str(at)]): n_precise, "Jaccard@1": n_jaccard})
            row_id += 1

    if os.path.exists(summary_file):
        summary_table.to_csv(summary_file, sep='\t', index=False, mode="a",
                             header=False)
    else:
        summary_table.to_csv(summary_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
