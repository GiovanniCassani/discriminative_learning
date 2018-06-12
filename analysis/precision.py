__author__ = 'GCassani'

"""Function to estimate the precision of the discrimination learning: the function checks whether given
   the cues in an outcome, the outcome is among the k top active outcomes"""

import json
import operator
import numpy as np
from time import strftime
from corpus.encode.item import encode_item
from celex.utilities.dictionaries import tokens2ids
from phonological_bootstrapping.tagging.helpers import compute_outcomes_activations
from corpus.encode.words.phonology import get_phonological_form


def precision_at(matrix, outcomes2ids, cues2ids, celex_dict, at=5, output_file='', boundaries=True,
                 stress_marker=True, uniphone=False, diphone=False, triphone=True, syllable=False):

    """
    :param matrix:          the matrix of cue-target_outcome association estimated using the ndl model
    :param outcomes2ids:    a dictionary mapping outcomes to their column indices in the matrix
    :param cues2ids:        a dictionary mapping cues to their row indices in the matrix
    :param celex_dict:      the dictionary extracted from the celex database
    :param at:              the number of top active outcomes where to look for the correct target_outcome
    :param output_file:     the path to the .json file where the output dictionary will be dumped
    :param boundaries:      a boolean specifying whether to consider or not word boundaries
    :param uniphone:        a boolean indicating whether to encode outcomes using uniphones
    :param diphone:         a boolean indicating whether to encode outcomes using diphones
    :param triphone:        a boolean indicating whether to encode outcomes using triphones
    :param syllable:        a boolean indicating whether to encode outcomes using syllables
    :param stress_marker:   a boolean indicating whether to preserve or discard stress markers from the phonological
                            representations of the target_outcome
    :return discriminated:  a dictionary mapping all target outcomes which were categorized correctly to their column
                            indices in the association matrix
    """

    total_items = len(outcomes2ids)
    check_points = {int(np.floor(total_items / 100 * n)): n for n in np.linspace(5, 100, 20)}

    discriminated = {}
    correct = 0
    token_indices = tokens2ids(celex_dict)

    # consider each outcome separately
    for idx, target_outcome in enumerate(outcomes2ids):

        wordform, pos = target_outcome.split('|')
        celex_entry = (wordform, pos, wordform)
        word_phon = get_phonological_form(celex_entry, celex_dict, token_indices)
        if boundaries:
            word_phon = '+' + word_phon + '+'

        if isinstance(word_phon, str):

            # get the relevant phonological cues
            nphones = encode_item(word_phon, stress_marker=stress_marker, uniphones=uniphone,
                                  diphones=diphone, triphones=triphone, syllables=syllable)

            # compute activations of all outcomes given such cues
            outcome_alphas = compute_outcomes_activations(nphones, matrix, cues2ids, outcomes2ids, set())

            # sort outcomes according to their activations and get the k top active ones
            sorted_outcome_alphas = sorted(outcome_alphas.items(), key=operator.itemgetter(1), reverse=True)[:at]

            # make sure that no outcome among the top active ones has a negative activation
            if not sorted_outcome_alphas[-1][1] > 0:
                sorted_outcome_alphas = [(outcome, act) for outcome, act in sorted_outcome_alphas if act > 0]

            for active_outcome, alpha in sorted_outcome_alphas:
                if active_outcome == target_outcome:
                    correct += 1
                    discriminated[target_outcome] = outcomes2ids[target_outcome]
                    break

        if idx + 1 in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": %d%% of the outcomes have been processed to estimate the precision of discrimination."
                  % check_points[idx + 1])

    print("precision at %i: %.4f" % (at, correct/total_items))

    if output_file:
        json.dump(discriminated, open(output_file, 'w'))
    else:
        return discriminated
