__author__ = 'GCassani'

"""Function to encode the full corpus for the study on non-word semantics"""

import os
import json
import numpy as np
from time import strftime
from corpus.encode.item import encode_item
from celex.get import get_celex_dictionary
from celex.utilities.dictionaries import tokens2ids
from corpus.encode.utilities import get_pos_mapping, encoding_features
from corpus.encode.words.phonology import get_phonetic_encoding, concatenate_phonological_representations


def encode_corpus(corpus, celex_dict, tokens2identifiers, pos_dict,
                  separator='~', uni_phones=False, di_phones=True, tri_phones=False, syllable=False,
                  stress_marker=True, boundaries=False):

    """
    :param corpus:              a .json object to be used as input corpus, consisting of two aligned lists of lists,
                                meaning that a second-order list in each first order list refers to a same utterance;
                                the first list contains utterances encoded as lists of tokens, the second list contains
                                utterances encoded as lists of lemmas and PoS tags
    :param celex_dict:          the path to the Celex dictionary to be used to recode the utterances into phonetic cues
                                and lexical outcomes
    :param tokens2identifiers:  a dictionary mapping a token surface form from Celex to all token ids linked to it
    :param pos_dict:            a dictionary mapping CHILDES PoS tags to corresponding Celex PoS tags
    :param separator:           a string indicating the character separating lemmas from PoS tags in the input corpus
    :param uni_phones:          a boolean indicating whether uni-phones are relevant phonetic cues
    :param di_phones:           a boolean indicating whether di-phones are relevant phonetic cues
    :param tri_phones:          a boolean indicating whether tri-phones are relevant phonetic cues
    :param syllable:            a boolean indicating whether syllables are relevant phonetic cues
    :param stress_marker:       a boolean indicating whether to discard or not the stress marker from the Celex phonetic
                                transcriptions
    :return all_cue, all_outcomes:  two lists of lists, where each inner list contains the cues and the outcomes for
                                    each learning event respectively
    """

    total = len(corpus[0])
    check_points = {np.floor(total / float(100) * n): n for n in np.linspace(5, 100, 20)}

    all_cues, all_outcomes = [[], []]

    # for every utterance in the input corpus, remove words with a PoS tag that doesn't belong to the
    # dictionary of PoS mappings; then map valid words to the right PoS tag as indicated by the PoS dictionary

    for i in range(len(corpus[0])):
        words = []
        for j in range(len(corpus[0][i])):
            lemma, pos_tag = corpus[1][i][j].split(separator)
            if pos_tag in pos_dict:
                token = corpus[0][i][j]
                new_tag = pos_dict[pos_tag]
                words.append((token, new_tag, lemma))

        # if there are valid words in the utterance, encode it
        if 0 < len(words) <= 12:

            # get the phonetic encoding of the words in the current learning trial:
            # if they can all be encoded using Celex, a list is returned, other wise a tuple is
            phonological_representations = get_phonetic_encoding(words, celex_dict, tokens2identifiers)

            # if a phonological representation could be found for all words in the utterance, proceed
            if isinstance(phonological_representations, list):

                utterance = concatenate_phonological_representations(phonological_representations,
                                                                     boundaries=boundaries)
                table = str.maketrans(dict.fromkeys('"'))
                utterance = utterance.translate(table)

                n_phones = encode_item(utterance, uniphones=uni_phones, diphones=di_phones,
                                       triphones=tri_phones, syllables=syllable, stress_marker=stress_marker)

                outcomes = []
                for word in words:
                    token, pos, lemma = word
                    outcomes.append('|'.join([token, lemma, pos]))

                # append the phonetic representation of the current learning event to the list of phonetic
                # representations for the whole corpus, and the lexical meanings of the current learning event to
                # the list of lexical meanings for the whole corpus
                all_cues.append(n_phones)
                all_outcomes.append(outcomes)

        if i in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": %d%% of the input corpus has been processed and encoded in the desired way." % check_points[i])

    return all_cues, all_outcomes


########################################################################################################################


def corpus2txt(corpus, output_file):

    """
    :param corpus:      a list of lists
    :param output_file: the path to a .txt file where the information in corpus will be written to
    """

    with open(output_file, "w") as f:
        for utterance in corpus:
            f.write(",".join(utterance))
            f.write("\n")


########################################################################################################################


def write_learning_events(corpus_file, output_folder, celex_dir, pos_dict,
                          separator='~', uni_phones=False, di_phones=False, tri_phones=True, syllable=False,
                          stress_marker=True, boundaries=False):
    """
    :param corpus_file:         a path pointing to .json object to be used as input corpus, consisting of two aligned
                                lists of lists, meaning that a second-order list in each first order list refers to a
                                same utterance; the first list contains utterances encoded as lists of tokens, the
                                second list contains utterances encoded as lists of lemmas and PoS tags
    :param output_folder:       the path to a folder where the output files for cues and outcomes will be written to
    :param celex_dir:           the path to the directory where the Celex dictionary is to be found (if no dictionary
                                is found at the given location, one is built on the fly
    :param pos_dict:            a dictionary mapping CHILDES PoS tags to corresponding Celex PoS tags
    :param separator:           a string indicating the character separating lemmas from PoS tags in the input corpus
    :param uni_phones:          a boolean indicating whether uni-phones are relevant phonetic cues
    :param di_phones:           a boolean indicating whether di-phones are relevant phonetic cues
    :param tri_phones:          a boolean indicating whether tri-phones are relevant phonetic cues
    :param syllable:            a boolean indicating whether syllables are relevant phonetic cues
    :param stress_marker:       a boolean indicating whether to discard or not the stress marker from the Celex phonetic
                                transcriptions
    """

    celex_dict = get_celex_dictionary(celex_dir, reduced=False)
    tokens2identifiers = tokens2ids(celex_dict)
    pos_dict = get_pos_mapping(pos_dict)
    corpus = json.load(open(corpus_file, 'r+'))

    # use the path of the input file to generate the path of the output file, adding encoding information to the
    # input filename; print to standard output a summary of all the encoding parameters
    input_filename, extension = os.path.splitext(corpus_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cue_file = os.path.join(output_folder, "cues.txt")
    outcome_file = os.path.join(output_folder, "outcomes.txt")

    # check whether the output file corresponding to the desired parameters already exist and stop if it does
    if os.path.isfile(cue_file) and os.path.isfile(outcome_file):
        print()
        print("The desired encoded version of the input corpus '%s' already exists at files '%s' and '%s'." %
              (os.path.basename(corpus_file), os.path.basename(cue_file), os.path.basename(outcome_file)))
        return cue_file, outcome_file
    else:

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Started encoding utterances from input corpus '%s'" % corpus_file)

        # get the corpus recoded into phonological cues and lexical outcomes
        cues, outcomes = encode_corpus(corpus, celex_dict, tokens2identifiers, pos_dict,
                                       separator=separator, uni_phones=uni_phones, di_phones=di_phones,
                                       tri_phones=tri_phones, syllable=syllable, stress_marker=stress_marker,
                                       boundaries=boundaries)
        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished encoding utterances from input corpus '%s'" % corpus_file)
        print()

        corpus2txt(cues, cue_file)
        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Written encoded cues to '%s'" % cue_file)
        print()

        corpus2txt(outcomes, outcome_file)
        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Written encoded outcomes to '%s'" % outcome_file)
        print()
