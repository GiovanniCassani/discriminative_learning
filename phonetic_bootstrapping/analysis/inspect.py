__author__ = 'GCassani'

"""Function to analyze the cue-outcome association matrix learned from phonetic cues"""

import os
from time import strftime
from matrix.matrix import load
from corpus.encode.utilities import encoding_features
from phonetic_bootstrapping.analysis.measures import get_frequencies, get_cue_and_outcome_measures
from phonetic_bootstrapping.analysis.corr import outcome_correlations, cue_correlations


def inspect_the_matrix(input_corpus, associations, celex_dir, plot_path,
                       uniphones, diphones, triphones, syllable, stress_marker):

    """
    :param input_corpus:
    :param associations:
    :param celex_dir:
    :param plot_path:
    :param uniphones:
    :param diphones:
    :param triphones:
    :param syllable:
    :param stress_marker:
    :return:
    """

    encoding_features(input_corpus, uni_phones=uniphones, di_phones=diphones, tri_phones=triphones,
                      syllable=syllable, stress_marker=stress_marker)

    d = os.path.dirname(input_corpus)
    plot_path = os.path.join(plot_path)
    cue_file = os.path.join(d, "cueFreqs.txt")
    outcome_file = os.path.join(d, "outcomeFreqs.txt")
    cue_frequencies, outcome_frequencies = get_frequencies(input_corpus, cue_file, outcome_file)

    # load the matrix of associations, get column and row indices
    matrix, row_ids, col_ids = load(associations)

    cue_values, outcome_values, jaccard = get_cue_and_outcome_measures(matrix, row_ids, col_ids, celex_dir, plot_path,
                                                                       uniphones, diphones, triphones,
                                                                       syllable, stress_marker)

    outcome_values['jaccard'] = jaccard
    outcome_values['frequencies'] = outcome_frequencies
    cue_values['frequencies'] = cue_frequencies

    print(": ".join([strftime("%Y-%m-%d %H:%M:%S"), "Finished computing statistics for cues and outcomes."]))

    cue_measures = ['MAD', 'activations', '1-norm', '2-norm', 'frequencies']
    cues = set(cue_frequencies.keys())
    cue_correlations(cue_values, cue_measures, cues)

    outcome_measures = cue_measures + ['jaccard']
    outcome_correlations(outcome_values, outcome_measures, plot_path)
