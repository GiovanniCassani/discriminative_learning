__author__ = 'GCassani'

"""Function to identify outcomes that the model can fully discriminate"""

from analysis.jaccard import jaccard
from analysis.precision import precision_at


def find_discriminated(matrix, cues2ids, outcomes2ids, celex_dict, stress_marker=True, uniphones=False,
                       diphones=False, triphones=True, syllables=False, boundaries=True, at=5):

    """
    :param matrix:          the matrix of cue-target_outcome association estimated using the ndl model
    :param outcomes2ids:    a dictionary mapping outcomes to their column indices in the matrix
    :param cues2ids:        a dictionary mapping cues to their row indices in the matrix
    :param celex_dict:      the dictionary extracted from the celex database
    :param at:              the number of top active outcomes where to look for the correct target_outcome
    :param boundaries:      a boolean specifying whether to consider or not word boundaries
    :param uniphones:       a boolean indicating whether to encode outcomes using uniphones
    :param diphones:        a boolean indicating whether to encode outcomes using diphones
    :param triphones:       a boolean indicating whether to encode outcomes using triphones
    :param syllables:       a boolean indicating whether to encode outcomes using syllables
    :param stress_marker:   a boolean indicating whether to preserve or discard stress markers from the phonological
                            representations of the target_outcome
    :return discriminated:  a dictionary mapping all target outcomes which were categorized correctly to their column
                            indices in the association matrix
    """

    jaccard_coeffs = jaccard(matrix, cues2ids, outcomes2ids, celex_dict, stress_marker=stress_marker,
                             uniphone=uniphones, diphone=diphones, triphone=triphones, syllable=syllables,
                             boundaries=boundaries)
    discriminated = {}
    for outcome in outcomes2ids:
        if jaccard_coeffs[outcome] == 1:
            discriminated[outcome] = outcomes2ids[outcome]
    discriminated = precision_at(matrix, discriminated, cues2ids, celex_dict, stress_marker=stress_marker,
                                 uniphone=uniphones, diphone=diphones, triphone=triphones, syllable=syllables,
                                 boundaries=boundaries, at=at)

    return discriminated
