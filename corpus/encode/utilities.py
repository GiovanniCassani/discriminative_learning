__author__ = 'GCassani'

"""Helper functions to pre-process the corpus"""


def get_pos_mapping(input_file):

    """
    :param input_file:  a .txt fil containing two elements per line separated by a white space. The first element of
                        each line is used as a key in a dictionary, and the second value in the line is used as the
                        corresponding value. If there duplicates among the first elements (the keys), the output
                        dictionary will contain the corresponding second element (the value) from the last line in
                        which the first element occurred.
    :return out_dict:   a dictionary containing elements from the input file arranged in the specified way
    """

    out_dict = {}

    with open(input_file, 'r') as f:
        for line in f:
            pos = line.rstrip("\n").split()
            out_dict[pos[0]] = pos[1]

    return out_dict


########################################################################################################################


def encoding_features(corpus_name, reduced=True, uniphones=True, diphones=False, triphones=False, syllables=False,
                      stress_marker=False, outcomes='tokens', boundaries=False, log='', verbose=True):

    """
    :param corpus_name:         a string indicating the name of the corpus being processed
    :param reduced:             a boolean indicating whether reduced phonological forms are extracted from CELEX or not
    :param uniphones:           a boolean indicating whether single phonemes are to be considered while encoding input
                                utterances
    :param diphones:            a boolean indicating whether sequences of two phonemes are to be considered while
                                encoding input utterances
    :param triphones:           a boolean indicating whether sequences of three phonemes are to be considered while
                                encoding input utterances
    :param syllables:           a boolean indicating whether syllables are to be considered while encoding input
                                utterances
    :param stress_marker:       a boolean indicating whether stress markers from the phonological representations of
                                Celex need to be preserved or can be discarded
    :param outcomes:            a string indicating which outcomes to use, whether 'tokens' (default) or 'lemmas'
    :param boundaries:          a boolean specifying whether word boundaries are considered when training on full
                                utterances
    :param log:                 the path to a file where the log is printed. Default is empty string, meaning that no
                                file is provided and everything is printed to standard output.
    :param verbose:             a boolean indicating whether to print information to screen (default is True)
    :return encoding_string:    a string that tells which parameters where used to encode the corpus; it can be appended
                                to file names to unequivocally determine which parameters were used to create a certain
                                file and derived measures.
    """

    desired_cues = []
    encoding_string = ''

    if reduced:
        encoding_string += 'r'
    else:
        encoding_string += 'f'

    if boundaries:
        encoding_string += 'b'
    else:
        encoding_string += 'c'

    if uniphones:
        desired_cues.append('uniphones')
        encoding_string += 'u'
    if diphones:
        desired_cues.append('diphones')
        encoding_string += 'd'
    if triphones:
        desired_cues.append('triphones')
        encoding_string += 't'
    if syllables:
        desired_cues.append('syllables')
        encoding_string += 's'

    if outcomes == 'tokens':
        encoding_string += 'k'
    elif outcomes == 'lemmas':
        encoding_string += 'l'
    else:
        raise ValueError("Unrecognized specification concerning lexical outcomes. "
                         "Please, choose either 'tokens' or 'lemmas'.")

    if stress_marker:
        desired_cues.append('with stress marker')
        encoding_string += 'm'
    else:
        desired_cues.append('without stress marker')
        encoding_string += 'n'

    num_hash = 120
    desired_cues = ", ".join(desired_cues)
    padding_cues = " " * (num_hash - 15 - len(desired_cues))
    padding_outcomes = " " * (num_hash - 19 - len(outcomes))
    padding_corpus = " " * (num_hash - 17 - len(corpus_name))
    if log:
        with open(log, "w+") as log_file:
            log_file.write("\n\n")
            log_file.write("#" * num_hash)
            log_file.write("\n")
            log_file.write("#####  CORPUS: " + corpus_name + padding_corpus + "##")
            log_file.write("\n")
            log_file.write("#####  CUES: " + desired_cues + padding_cues + "##")
            log_file.write("\n")
            log_file.write("#####  OUTCOMES: " + outcomes + padding_outcomes + "##")
            log_file.write("\n")
            log_file.write("#" * num_hash)
            log_file.write("\n\n")
    else:
        if verbose:
            print()
            print("#" * num_hash)
            print("#####  CORPUS: " + corpus_name + padding_corpus + "##")
            print("#####  CUES: " + desired_cues + padding_cues + "##")
            print("#####  OUTCOMES: " + outcomes + padding_outcomes + "##")
            print("#" * num_hash)
            print()

    return encoding_string
