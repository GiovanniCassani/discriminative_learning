__author__ = 'GCassani'

"""Function to create a log file with an unambiguous name pointing to the specification of the model being run"""

import os
from corpus.encode.utilities import encoding_features


def make_log_file(training_corpus, output_folder, ext, dist='cosine', nn=10, at=5, time=100, reduced=False,
                  uniphones=False, diphones=True, triphones=False, syllables=False, stress_marker=True,
                  outcomes='tokens', boundaries=False):

    """
    :param training_corpus: the path to the file used as input corpus for the experiment
    :param output_folder:   the path to the folder where the logfile will be created
    :param ext:             the extension of the file
    :param dist:            the distance used to retrieve neighbors
    :param nn:              an integer specifying how many neighbors to consider
    :param time:            the percentage of the input corpus used to compute cue-outcome activation values
    :param reduced:         a boolean specifying whether reduced phonological forms should be extracted from Celex
                            whenever possible (if set to True) or if standard phonological forms should be preserved
                            (if False)
    :param uniphones:       a boolean indicating whether to consider single phonemes as cues
    :param diphones:        a boolean indicating whether to consider sequences of two phonemes as cues
    :param triphones:       a boolean indicating whether to consider sequences of three phonemes as cues
    :param syllables:       a boolean indicating whether to consider syllables as cues
    :param stress_marker:   a boolean indicating whether stress markers from the phonological representations of
                            Celex need to be preserved or can be discarded
    :param outcomes:        a string indicating which outcomes to use, whether 'tokens' (default) or 'lemmas'
    :param boundaries:      a boolean specifying whether word boundaries are to be considered when training on full
                            utterances
    :return log_file:       the path of the file where the log of the experiment will be written
    """

    encoding = encoding_features(training_corpus, reduced=reduced, verbose=False, outcomes=outcomes,
                                 uniphones=uniphones, diphones=diphones, triphones=triphones, syllables=syllables,
                                 stress_marker=stress_marker, boundaries=boundaries)

    # create the log file in the same folder where the corpus is located, using encoding information in the file name,
    # including corpus, phonetic features, method, evaluation, f and k values
    filename = os.path.splitext(os.path.basename(training_corpus))[0]
    log_file = os.path.join(output_folder,
                            ".".join(['logfile', filename, encoding, ''.join(['t', str(int(time))]), dist,
                                      ''.join(['nn', str(nn)]), ''.join(['at', str(at)]), ext]))
    return log_file
