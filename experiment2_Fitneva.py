_author__ = 'GCassani'

"""Function to perform the experiment on the dataset from Fitneva et al 2009"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from phonetic_bootstrapping.experiment.phonetic_bootstrapping import phonetic_bootstrapping


def read_in_fitneva_dataset(path):

    """
    :param path:        the path where the file is located
    :return test_items: a dictionary mapping each phonetically encoded test item to the corresponding orthographic form
                        ['orthography'], its PoS tag ['pos'], the typicality value ['typicality'], the proportion of
                        subjects that judged it to be a noun ['proportion_noun'] and a verb ['proportion_verb'] -
                        the sum of proportion_noun and proportion_verb is 1
    :return test_set:   a dictionary mapping the keys 'filename' and 'items' respectively to the basename of the file
                        and to the set of phonological forms of the test items
    """

    test_items = defaultdict(dict)
    test_set = {}

    items = set()
    with open(path, "r") as f:
        next(f)
        for line in f:
            ortho, phon, pos, typ, prop_n, prop_v = line.strip().split("\t")
            test_items[phon]['orthography'] = ortho
            test_items[phon]['pos'] = pos
            test_items[phon]['typicality'] = typ
            test_items[phon]['proportion_noun'] = prop_n
            test_items[phon]['proportion_verb'] = prop_v
            items.add(phon)

    test_set['filename'] = os.path.basename(path)
    test_set['items'] = items

    return test_items, test_set


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Run a grid search to explore all possible parameters.")

    parser.add_argument("-c", "--corpus", required=True, dest="corpus",
                        help="Specify the .json file containing the input corpus.")
    parser.add_argument("-T", "--test_file", required=True, dest="test_file",
                        help="The path to the file containing the items from Fitneva et al's experiment.")
    parser.add_argument("-C", "--Celex_folder", required=True, dest="celex_folder",
                        help="Specify the folder where the Celex data are located.")
    parser.add_argument("-O", "--output_file", required=True, dest="out_file",
                        help="Specify the path of the .csv output file where summary statistics will be written to.")
    parser.add_argument("-M", "--pos_mapping", required=True, dest="pos_mapping",
                        help="Specify the path of the file containing the mapping from CHILDES to Celex PoS tags.")

    args = parser.parse_args()

    # experimental contrasts: training regime, cues, outcomes, stress
    cues = ['triphones', 'syllables']
    outcomes = ['lemmas', 'tokens']
    stress_marker = [True, False]

    a, b = [0.001, 0.001]  # learning rates for discriminative learning
    training = os.path.splitext(os.path.basename(args.corpus))[0]

    # store the test items in a dictionary and pre-allocate a Pandas dataframe with enough rows to store the
    # item-level results of the necessary simulations
    test_items, test_set = read_in_fitneva_dataset(args.test_file)
    rows = np.prod([len(test_items), len(cues), len(outcomes), len(stress_marker)])
    summary_table = pd.DataFrame(index=np.arange(0, rows),
                                 columns=["Corpus", "Cues", "Outcomes", "Stress",
                                          "Word", "Phon_form", "Phon_PoS", "Typicality", "Proportion_N", "Proportion_V",
                                          "Predicted_PoS", "Frequency_N", "Frequency_V"])

    r = 0
    for cue in cues:
        for outcome in outcomes:
            for marker in stress_marker:
                triphones = True if cue == 'triphones' else False
                syllables = True if cue == 'syllables' else False
                sm = "stress" if marker else 'no-stress'

                log, f1, h, pos, freq = phonetic_bootstrapping(args.corpus, test_set, args.celex_folder,
                                                               args.pos_mapping, method='freq', evaluation='count',
                                                               k=50, flush=100, reduced=False, uni_phones=False,
                                                               di_phones=False, stress_marker=marker,
                                                               syllable=syllables, tri_phones=triphones,
                                                               outcomes=outcome, alpha=a, beta=b, lam=1.0,
                                                               longitudinal=False)

                for item in log:
                    summary_table.loc[r] = pd.Series({"Corpus": training, "Cues": cue, "Outcomes": outcome,
                                                      "Stress": sm, "Word": test_items[item]['orthography'],
                                                      "Phon_form": item, "Phon_PoS": test_items[item]['pos'],
                                                      "Typicality": test_items[item]['typicality'],
                                                      "Proportion_N": test_items[item]['proportion_noun'],
                                                      "Proportion_V": test_items[item]['proportion_verb'],
                                                      "Predicted_PoS": log[item]['predicted'],
                                                      "Frequency_N": log[item]['n_freq'],
                                                      "Frequency_V": log[item]['v_freq']})
                    r += 1

    summary_table.to_csv(args.out_file, sep='\t', index=False)


########################################################################################################################


if __name__ == '__main__':

    main()
