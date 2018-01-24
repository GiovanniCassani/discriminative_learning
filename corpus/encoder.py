__author__ = 'GCassani'

"""This function recodes each line of the input corpus into a set of phonological cues and a set of lexical outcomes"""

import os
import json
from time import strftime
from celex.get import get_celex_dictionary
from celex.utilities.dictionaries import tokens2ids
from corpus.encode.utilities import get_pos_mapping, encoding_features
from corpus.encode.corpus import encode_corpus


def corpus_encoder(corpus_name, celex_dir, pos_mapping, separator='~', reduced=True, outcomes='tokens',
                   uni_phones=True, di_phones=False, tri_phones=False, syllable=False, stress_marker=False):

    """
    :param corpus_name:     the path to a .json file containing transcripts of child-caregiver interactions extracted
                            from the CHILDES database. The json file consists of two lists of lists, of the same length,
                            both contain words but encoded differently. The first is a list of tokens, i.e. surface
                            forms as they appear in the transcriptions; the second is a list of lemmas, i.e. words in
                            their dictionary form, without any inflectional morphology, together with their
                            Part-of-Speech tags, joined by a specific character (which can be specified with the
                            parameter 'separator'.
    :param celex_dir:       a string indicating the path to the Celex directory containing files with phonological and
                            morphological information for words and lemmas ('epw.cd', 'epl.cd', 'emw.cd', eml.cd').
                            The function also checks whether this directory already contains the Celex dictionary: if
                            it is found, the dictionary is loaded and the function proceeds, otherwise the dictionary
                            is created.
    :param pos_mapping:     the path to a .txt file indicating the mapping between CHILDES and Celex PoS tags; it must
                            consist of two space-separated columns.
    :param separator:       the character that separates the word baseform from its PoS tag in the input corpus
    :param reduced:         a boolean specifying whether reduced phonological forms should be extracted from Celex
                            whenever possible (if set to True) or if standard phonological forms should be preserved
                            (if False)
    :param outcomes:        a string indicating which outcomes to use, whether 'tokens' (default) or 'lemmas'
    :param uni_phones:      a boolean indicating whether single phonemes are to be considered while encoding input
                            utterances
    :param di_phones:       a boolean indicating whether sequences of two phonemes are to be considered while
                            encoding input utterances
    :param tri_phones:      a boolean indicating whether sequences of three phonemes are to be considered while
                            encoding input utterances
    :param syllable:        a boolean indicating whether syllables are to be considered while encoding input
                            utterances
    :param stress_marker:   a boolean indicating whether stress markers from the phonological representations of Celex
                            need to be preserved or can be discarded
    :return out_file:   	the path to the file where the encoded version of the input file has been printed

    This function runs in linear time on the length of the input (if it takes 1 minute to process 1k utterances,
    it takes 2 minutes to process 2k utterances). It processes ~550k utterances in ~10 second on a 2x Intel Xeon 6-Core
    E5-2603v3 with 2x6 cores and 2x128 Gb of RAM.
    """

    # get the Celex dictionary; create a dictionary where token surface forms are keys, and values
    # are sets containing all the token IDs that match a given surface form; the get the vowel symbols from Celex and
    # the mapping from inflectional morphology codes to their meanings
    celex_dict = get_celex_dictionary(celex_dir, reduced=reduced)
    tokens2identifiers = tokens2ids(celex_dict)
    pos_dict = get_pos_mapping(pos_mapping)

    # use the path of the input file to generate the path of the output file, adding encoding information to the
    # input filename; print to standard output a summary of all the encoding parameters
    input_filename, extension = os.path.splitext(corpus_name)
    encoding_string = encoding_features(corpus_name, reduced=reduced, uni_phones=uni_phones, di_phones=di_phones,
                                        tri_phones=tri_phones, syllable=syllable, stress_marker=stress_marker,
                                        outcomes=outcomes)
    output_file = input_filename + "_" + encoding_string + '.json'

    # check whether the output file corresponding to the desired parameters already exist and stop if it does
    if os.path.isfile(output_file):
        print()
        print("The desired encoded version of the input corpus '%s' already exists at file '%s'." %
              (os.path.basename(corpus_name), os.path.basename(output_file)))
        return output_file
    else:

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Started encoding utterances from input corpus '%s'" % corpus_name)

        # get the corpus recoded into phonological cues and lexical outcomes, plus the percentage of utterances that
        # could not be recoded because one or more words do not have a corresponding entry in Celex
        encoded_corpus, missed = encode_corpus(corpus_name, celex_dict, tokens2identifiers, pos_dict,
                                               separator=separator, uni_phones=False, di_phones=True,
                                               tri_phones=False, syllable=False, stress_marker=True, outcomes=outcomes)
        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished encoding utterances from input corpus '%s'" % corpus_name)
        print()

        perc_covered = 100 - missed
        print()
        if os.path.exists(output_file):
            print("The file %s already exists." % output_file)
        else:
            json.dump(encoded_corpus, open(output_file, 'w'))
            print("The file %s has been created:" % output_file)
            print()
            print("%0.4f%% of the utterances could be entirely encoded." % perc_covered)
            print("The remaining %0.4f%% contain at least one word that could not be retrieved in CELEX and "
                  % missed)
            print("for which no phonological and morphological representation could be obtained.")

    return output_file
