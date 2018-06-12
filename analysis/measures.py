__author__ = 'GCassani'

"""Functions to compute and extract several measures from cue-outcome association matrices and learning trials"""

import os
import operator
from time import strftime
from celex.get import get_celex_dictionary
from corpus.cues_outcomes import frequency
from analysis.jaccard import jaccard
from analysis.cues import cue_measures
from analysis.outcomes import outcome_measures


def get_frequencies(corpus, cue_file, outcome_file):

    """
    :param corpus:
    :param cue_file:
    :param outcome_file:
    :return:
    """

    # compute and store cue and outcome frequency counts
    cue_freqs = frequency(corpus, 'cues')
    outcome_freqs = frequency(corpus, 'outcomes')

    if not os.path.exists(cue_file):
        with open(cue_file, 'a+') as c_f:
            for item in sorted(cue_freqs.items(), key=operator.itemgetter(1), reverse=True):
                c_f.write("\t".join([item[0], str(item[1])]))
                c_f.write("\n")

    if not os.path.exists(outcome_file):
        with open(outcome_file, 'a+') as o_f:
            for item in sorted(outcome_freqs.items(), key=operator.itemgetter(1), reverse=True):
                o_f.write("\t".join([item[0], str(item[1])]))
                o_f.write("\n")

    print()
    print(": ".join([strftime("%Y-%m-%d %H:%M:%S"), "Finished computing cue and outcome frequency counts."]))

    return cue_freqs, outcome_freqs


########################################################################################################################


def get_cue_and_outcome_measures(associations, row_ids, col_ids, celex_dir, plot_path,
                                 uniphones, diphones, triphones, syllable, stress_marker, reduced=False):

    """
    :param associations:
    :param row_ids:
    :param col_ids:
    :param celex_dir:
    :param plot_path:
    :param uniphones:
    :param diphones:
    :param triphones:
    :param syllable:
    :param stress_marker:
    :return:
    """

    celex_dict = get_celex_dictionary(celex_dir, reduced)
    jaccard_values = jaccard(associations, row_ids, col_ids, celex_dict, plots_folder=plot_path,
                             stress_marker=stress_marker, uniphone=uniphones, diphone=diphones,
                             triphone=triphones, syllable=syllable)
    outcome_values = outcome_measures(associations, col_ids, plot_path)
    cue_values = cue_measures(associations, row_ids, col_ids, plot_path)

    return cue_values, outcome_values, jaccard_values
