__author__ = 'GCassani'

"""Function to collect cues and outcomes in a corpus and their frequency counts"""

import json
from collections import Counter


def get_cues_and_outcomes(input_file):

    """
    :param input_file:      a string indicating the path to the corpus to be considered: it is assumed to be a .json
                            file consisting of two lists of lists: the first encodes learning events into their
                            consistuent phonetic cues, the second encodes the same learning events into their meaning
                            units. Each learning event is a list nested in the two main lists.
                            outcomes; each of the two may consists of multiple, comma-separated strings
    :return cue2ids:        a dictionary mapping each of the strings found in the cues fields to a numerical index
    :return outcome2ids:    a dictionary mapping each of the strings found in the outcomes fields to a numerical index
    """

    outcomes = set()
    cues = set()

    corpus = json.load(open(input_file, 'r+'))

    for i in range(len(corpus[0])):
        trial_cues = set(corpus[0][i])
        cues.update(trial_cues)
        trial_outcomes = set(corpus[1][i])
        outcomes.update(trial_outcomes)

    cues2ids = {k: idx for idx, k in enumerate(sorted(cues))}
    outcomes2ids = {k: idx for idx, k in enumerate(sorted(outcomes))}

    return cues2ids, outcomes2ids


########################################################################################################################


def frequency(corpus_file, target):

    """
    :param corpus_file:     a string specifying the path to the corpus to be used as input: the file is assumed to be a
                            json file consisting of two lists of lists, the first containing cues and the second
                            outcomes. Each list consists of lists, one for each learning event.
    :param target:          a string specifying whether a frequency list should be derived for cues ('cues') or outcomes
                            ('outcomes'); any other value will give an error.
    :return frequencies:    a dictionary where strings (cues or outcomes) are used as keys and the number of utterances
                            they occur in as values (it slightly differs from raw frequency counts because even if a cue
                            or outcome occurs more than once in a sentence, its frequency count is only updated once
    """

    corpus = json.load(open(corpus_file, 'r+'))

    frequencies = Counter()

    for i in range(len(corpus[0])):

        if target == 'cues':
            cues = set(corpus[0][i])
            for target_cue in cues:
                frequencies[target_cue] += 1
        elif target == 'outcomes':
            outcomes = set(corpus[1][i])
            for target_outcome in outcomes:
                frequencies[target_outcome] += 1
        else:
            ValueError("Please specify the target items to be counted: either 'cues' or 'outcomes'.")

    return frequencies
