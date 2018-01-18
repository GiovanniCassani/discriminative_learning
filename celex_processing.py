__author__ = 'GCassani'

import re
import os
import json
import argparse
from time import strftime
from collections import defaultdict


"""
This module creates a dictionary containing information from the CELEX database, to be used to recode transcribed speech
into phonologically and morphologically richer representations. It consists of the following functions, listed together
with a short description of what they do. Check the documentation of each function for details about input arguments
and output structures.

- inflection_dict               : return a dictionary mapping CELEX inflectional codes to their meanings
- vowels                        : return a set of ASCII symbols corresponding to English vowel sounds as encoded in
                                    CELEX
- get_celex_dictionary          : try to load the dictionary if it exists, creates it otherwise
- initialize_celex_dict         : initialize an empty dictionary, consisting of two dictionaries of dictionaries,
                                    'tokens' and 'lemmas'; the connection between the two parts is obtained by pairing a
                                    unique lemma ID to each token ID that is a token of a same lemma
- get_stressed_vowel            : extracts the stressed vowel from a CELEX phonological representation
- get_part_of_speech            : extracts the Part-of-Speech tag from a CELEX morphological analysis
- get_constituent_morphemes     : extracts a set of the morphemes making up a CELEX morphological representation
- print_celex_dict              : store the CELEX dictionary into a json file
- read_epw                      : updates the CELEX dictionary, using token identifiers as keys, each mapped to a
                                    dictionary containing information about its surface form, the lemma identifier of
                                    the corresponding lemma, the phonological form of the token and the stressed vowel
- read_epl                      : updates the CELEX dictionary, using lemma identifiers as keys, and mapping lemmas'
                                    phonological forms to their IDs
- read_emw                      : updates the CELEX dictionary, assigning the correct inflectional code to each token,
                                    if applicable (plurals, third person singular forms, etc...)
- read_eml                      : updates the CELEX dictionary, storing the surface form corresponding to each lemma ID,
                                    its flat morphological analysis and its PoS tag
- hardcode_missing_words        : augment the CELEX dictionary by adding hand-crafted representations for frequent words
                                    in CHILDES that are not found in CELEX
- create_celex_dictionary       : overarching function that calls the others to create a complete CELEX dictionary with
                                    all relevant information
- main                          : a function that is called from command line and runs create_celex_dictionary
"""


def inflection_dict():

    inflections = {'S': 'SINGULAR',
                   'P': 'PLURAL',
                   'i': 'INFINITIVE',
                   'p': 'PARTICIPLE',
                   'e': 'PRESENT',
                   'a': 'PAST',
                   '3': 'THIRDPERSON'}

    return inflections


########################################################################################################################


def vowels():

    celex_vowels = {'I', 'E', '{', 'V', 'Q', 'U', '@', 'i', '#', '$', 'u', '3',
                    '1', '2', '4', '5', '6', '7', '8', '9', 'c', 'q', 'O', '~'}

    return celex_vowels


########################################################################################################################


def get_celex_dictionary(celex_dict_file, reduced):

    """
    :param celex_dict_file: a string specifying the path to .json file where the dictionary should be located. If the
                            dictionary is found, it is loaded and returned, otherwise it is created
    :param reduced:         a boolean specifying whether reduce phonological form should be always preferred when
                            available
    :return celex_dict:     a Python dictionary containing information about phonology and morphology of words extracted
                            from the CELEX database (for further details, see the documentation of the
                            celex_processing.py module
    """

    # check whether the Celex dictionary already exists in the subdirectory Celex of the working directory and if it
    # does load it; if it doesn't, make it
    try:
        celex_dict = json.load(open(celex_dict_file, 'r'))
    except IOError:
        celex_dir = os.path.dirname(celex_dict_file)
        if not celex_dir:
            celex_dir = os.getcwd()
        celex_dict = create_celex_dictionary(celex_dir, reduced=reduced)

    return celex_dict


########################################################################################################################

def initialize_celex_dict():

    """
    :return celex_dict: this function makes a dictionary consisting of two dictionaries, 'tokens' and 'lemmas', which
                        in turn consists of dictionaries of dictionaries.
    """

    celex_dict = {'tokens': defaultdict(dict),
                  'lemmas': defaultdict(dict)}
    return celex_dict


########################################################################################################################


def get_stressed_vowel(phonetic_form, vowel_set):

    """
    :param phonetic_form:   a string containing the phonetic transcription of a token as extracted from the epw.cd file
                            from the CELEX database, using the DISC format and including stress markers. In this
                            encoding, the token is syllabified with syllables being divided by a dash ('-')
    :param vowel_set:          a set of vowel symbols used in the DISC encoding
    :return stressed_vowel: the vowel in the syllable bearing the stress marker (an apostrophe, " ' ")

    This function extracts the stressed vowel from the phonological representation of a token, as extracted from the
    CELEX database.
    """

    # get all the syllables that make up the full representation
    syllable = phonetic_form.split('-')

    # scan each syllable, check whether it contains the stress marker ("'"): if it does, get the vowel from the syllable
    # and return it. Each syllable only have one vowel, so no chance of picking the wrong one; moreover, no word has
    # more than one stressed syllable (primary stress, which is what we are interested in here). If a token doesn't
    # contain stressed vowels, return a dot ('.')
    for idx, syl in enumerate(syllable):
        if "'" in syl:
            for phoneme in syl:
                if phoneme in vowel_set:
                    stressed_vowel = phoneme
                    return stressed_vowel

    return '.'


########################################################################################################################


def get_part_of_speech(morphological_analyses):

    """
    :param morphological_analyses:  a string extracted from the eml.cd file from the CELEX database, which contains the
                                    morphological segmentation of a lemma, with brackets signaling the correct
                                    composition and a Part-of-Speech tag in brackets at the end of each morpheme. The
                                    outermost tag refers to the whole lemma.
    :return pos_tag:                a capital letter marking the lexical category of a lemma

    This function simply returns the PoS tag given the morphological analyses of a lemma extracted from the eml.cd file
    from the CELEX database.
    """

    # if no PoS tag is provided, return the label 'UNK'
    try:
        pos_tag = morphological_analyses[-2]
    except IndexError:
        pos_tag = 'UNK'

    return pos_tag


########################################################################################################################


def get_constituent_morphemes(morphological_analyses):

    """
    :param morphological_analyses:  a string extracted from the eml.cd file from the CELEX database, which contains the
                                    morphological segmentation of a lemma, with brackets signaling the correct
                                    composition and a Part-of-Speech tag in brackets at the end of each morpheme. The
                                    outermost tag refers to the whole lemma.
    :return morphemes_clean:        a list (order is preserved) of the morphemes that make up the lemma.

    This function gets a complete but flat morphological analyses from the structured one that is retrieved from the
    CELEX database: the goal is to have all the consituent morphemes, but the composition scheme is not important.
    """

    morphemes_clean = []

    # get rid of parantheses by getting all the strings of lowercase letters (PoS tags are capitalized, and we don't
    # want them here)
    morphemes = re.findall(r"\(([a-z]+)\)", morphological_analyses)

    # given that the morphological analyses includes the full composition scheme, morphemes may be repeated: only add a
    # morpheme to the output structure if the same morpheme isn't already included therein.
    for morpheme in morphemes:
        if morpheme not in morphemes_clean:
            morphemes_clean.append(morpheme)

    return morphemes_clean


########################################################################################################################


def print_celex_dict(celex_dict, outfile):

    """
    :param celex_dict:  the dictionary created using make_celex_dict(). It can or cannot have been updated using the
                        functions read_epw, read_emw, and read_eml: this function always tries to print information from
                        the dictionary but whenever it cannot find it, it substitutes the missing information with a
                        dash ('-')
    :param outfile:     the path to a file where the information in the celex dictionary is written in .json format.
    """

    with open(outfile, 'a+') as o_f:

        json.dump(celex_dict, o_f)


########################################################################################################################


def read_epw(epw_path, celex_dict, vowel_set, reduced=True):

    """
    :param epw_path:    the path to the epw.cd file from the CELEX database
    :param celex_dict:  a dictionary built using make_celex_dict (see make_celex_dict for the details)
    :param vowel_set:   a set containing the vowel characters used in the CELEX database to encode phonological
                        representations: CELEX contains 4 different encodings, here the DISC one is used and thus the
                        vowel set must contain all relevant vowels as encoded in DISC format
    :param reduced:     a boolean. If True, the function tries to get the reduced phonological versions every
                        time it can, to be closer to the spoken language. However, since not every token in CELEX has a
                        corresponding reduced version, when this try fails, the function falls back to the canonical
                        version of the token. If False, the standard phonological form is retrieved directly.
    :return celex_dict: the updated version of the celex dictionary, containing token identifiers as keys, each mapped
                        to a dictionary containing information about the token surface form, the lemma identifier of the
                        corresponding lemma, the phonological form of the token and the stressed vowel.

    ------------------------------------------------------------------------------
    |  EPW COLUMN KEYS                                                           |
    ------------------------------------------------------------------------------
    line[0]  : TokenID
    line[1]  : token surface form
    line[3]  : LemmaID
    line[4]  : number of pronunciations available
    line[5]  : status of pronunciation, primary vs secondary
    line[6]  : DISC phonetic encoding, with stress marker
    line[8]  : syllabified CELEX transcription with brackets
    ------------------------------------------------------------------------------
    |  NOT ALWAYS AVAILABLE                                                      |
    ------------------------------------------------------------------------------
    line[9]  : second pronunciation status, primary vs secondary
    line[10] : second pronunciation DISC phonetic encoding, with stress marker
    line[12] : second pronunciation syllabified CELEX transcription with brackets
    """

    # read the epw.cd file
    with open(epw_path, 'r+') as epw:
        for line in epw:
            records = line.strip().split('\\')

            # skip multiword expressions (every token containing whitespaces is a multiword expression)
            # expression)
            if ' ' in records[1]:
                continue

            else:
                # store the token identifier as key in the sub-dictionary 'tokens'
                # each token identifier is mapped to a dictionary containing several keys to which values are mapped,
                # including the token surface form ('surface'), the lemma identifier ('lemmaID'), the phonological form
                # ('phon'), and the stressed vowel ('stressed_vowel'). First assign values to the token surface form and
                # the lemma identifier.
                celex_dict['tokens'][records[0]]['surface'] = records[1]
                celex_dict['tokens'][records[0]]['lemmaID'] = records[3]

                # if reduced forms are allowed, try to fetch it and fall back onto the regular one if there isn't;
                # otherwise, go directly for the regular form
                if reduced:
                    try:
                        celex_dict['tokens'][records[0]]['phon'] = records[10]
                    except IndexError:
                        celex_dict['tokens'][records[0]]['phon'] = records[6]
                else:
                    celex_dict['tokens'][records[0]]['phon'] = records[6]

                # get the stressed vowel from the token phonological representation
                stressed_vowel = get_stressed_vowel(celex_dict['tokens'][records[0]]['phon'], vowel_set)
                celex_dict['tokens'][records[0]]['stressed_vowel'] = stressed_vowel

    return celex_dict


########################################################################################################################


def read_epl(epl_path, celex_dict):

    """
    :param epl_path:    the path to the emw.cd file from the CELEX database
    :param celex_dict:  a dictionary built using make_celex_dict and already updated using read_epw()
    :return celex_dict: the input dictionary, updated with information about the phonological form of each lemma
    """

    lemma_identifiers = set()
    for k in celex_dict['tokens']:
        lemma_identifiers.add(celex_dict['tokens'][k]['lemmaID'])

    with open(epl_path, 'r+') as epl:
        for line in epl:
            records = line.strip().split('\\')

            if records[0] in lemma_identifiers:
                celex_dict['lemmas'][records[0]]['lemma_phon'] = records[5]

    return celex_dict


########################################################################################################################


def read_emw(emw_path, celex_dict):

    """
    :param emw_path:    the path to the emw.cd file from the CELEX database
    :param celex_dict:  a dictionary built using make_celex_dict and already updated using read_epw() and read_epl()
    :return celex_dict: the input dictionary, updated with information about the inflectional morphology of each token

    ------------------------------------------------------------------------------
    |  EMW COLUMN KEYS                                                           |
    ------------------------------------------------------------------------------
    line[0]  : TokenID
    line[1]  : token surface form
    line[3]  : LemmaID
    line[4]  : flectional type
                  'S' -> singular
                  'P' -> plural
                  'b' -> positive
                  'c' -> comparative
                  's' -> superlative
                  'i' -> infinitive
                  'p' -> participle
                  'e' -> present
                  'a' -> past
                  '1' -> first person
                  '2' -> second person
                  '3' -> third person
                  'r' -> rare
                  'X' -> head (no N, V, Adj, Adv)
    line[5]  : flectional variant, which letters are removed or added to form the
                  inflected form [-(.) for deleted letters, +(.) for added letters,
                  @ for the stem]
    """

    # collect all token identifiers already stored in the celex dictionary
    token_ids = set()
    for k in celex_dict['tokens']:
        token_ids.add(k)

    # read the emw.cd file
    with open(emw_path, 'r+') as emw:
        for line in emw:
            records = line.strip().split('\\')

            # skip multi-word expressions (every token containing whitespaces is a multiword expression here)
            if ' ' in records[1]:
                continue
            else:
                # check that the token being considered already exists in the celex dictionary and then that its surface
                # form and lemma identifier are the same as those of the corresponding entry in the celex dictionary. If
                # all these conditions hold, get the inflection codes associated with the token and store them in the
                # celex dictionary
                if records[0] in token_ids:
                    if records[1] == celex_dict['tokens'][records[0]]['surface'] \
                            and records[3] == celex_dict['tokens'][records[0]]['lemmaID']:
                        celex_dict['tokens'][records[0]]['inflection'] = records[4]
                    else:
                        print("TokenID: " + records[0] +
                              " is in celex dict but the surface form or the lemmaID don't match.")
                else:
                    print("TokenID: " + records[0] + " couldn't be located in celex dict.")

    return celex_dict


########################################################################################################################


def read_eml(eml_path, celex_dict):

    """
    :param eml_path:    the path to the eml.cd file from the CELEX database
    :param celex_dict:  a dictionary built using make_celex_dict and already updated using read_epw(), read_epl(),
                        and read_emw()
    :return celex_dict: the input dictionary, updated with information about the lemma that corresponds to every token
                        and its Part-of-Speech


    ------------------------------------------------------------------------------
    |  EMW COLUMN KEYS                                                           |
    ------------------------------------------------------------------------------
    line[0]  : LemmaID
    line[1]  : lemma surface form
    line[3]  : morphological status
                  'C' -> morphologically complex
                  'M' -> monomorphemic
                  'Z' -> zero derivation
                  'F' -> contracted form
                  'I' -> irrelevant morphology
                  'O' -> obscure morphology
                  'R' -> may include a root
                  'U' -> undetermined
    line[21]  : structured segmentation with brackets and PoS tag of the first,
                  default parsing of the lemma:
                  e.g. revolution = ((revolt),(ution)[N|V.])[N]
                  where the final [N] tells revolution is a noun and the
                  bracketing tells that it consists of two morphomes, the
                  verb revolt and the affix ution, which is specified to be
                  a suffix to the verb.
    line[40, 59, +19...] contain structured segmentations with brackets and Pos
                         tag of alternative parsings of the lemma
    """

    # collect all lemma identifiers that are already present in the celex dictionary
    lemma_identifiers = set()
    for k in celex_dict['tokens']:
        lemma_identifiers.add(celex_dict['tokens'][k]['lemmaID'])

    # read the eml.cd file
    with open(eml_path, 'r+') as eml:
        for line in eml:
            records = line.strip().split('\\')

            # check if the lemma being considered exists in the celex dictionary
            # if it does, get its morphological analysis, and use it to derive the lemma PoS and its constituent
            # morphemes. Store these pieces of information in the celex dictionary
            if records[0] in lemma_identifiers:

                morphological_analyses = records[21]

                pos_tag = get_part_of_speech(morphological_analyses)
                morphemes = get_constituent_morphemes(morphological_analyses)

                celex_dict['lemmas'][records[0]]['surface'] = records[1]
                celex_dict['lemmas'][records[0]]['morph_analysis'] = morphemes
                celex_dict['lemmas'][records[0]]['pos'] = pos_tag

            # if the lemma being considered doesn't exist in the celex dictionary, keep running
            else:
                continue

    return celex_dict


########################################################################################################################


def hardcode_words(celex_dict, token_surface, token_id, lemma_id, token_phonetic, vowel, inflection, lemma_surface,
                   lemma_phonetic, morph, pos):

    """
    :param celex_dict:      a dictionary created using the function initialize_celex_dict from this module. The 
                            dictionary can be empty or already contain items
    :param token_surface:   a string containing the orthographic form of the token you want to hard-code
    :param token_id:        the numeric ID (encoded as string) of the token to be added. Make sure you choose an id
                            that doesn't already exist in CELEX
    :param lemma_id:        the numeric ID (encoded as string) of the lemma to which the token to be hard-coded  
                            corresponds. If the lemma already exists in CELEX, make sure you use the correct ID 
    :param token_phonetic:  a string containing the phonetic form of the token to be hard-coded - make sure you use the 
                            charset used in CELEX
    :param vowel:           a string containing the vowel bearing the stress in the token to be hard-coded
    :param inflection:      a string indicating the inflectional code(s) of the word to be hard-coded
    :param lemma_surface:   the orthographic form of the corresponding lemma
    :param lemma_phonetic:  the phonetic form of the corresponding lemma
    :param morph:           a list of strings indicating the morphological analysis of the lemma
    :param pos:             the Part-of-Speech tag of the lemma
    :return celex_dict:     the dictionary updated with the word to be hard-coded
    """

    celex_dict['tokens'][token_id]['surface'] = token_surface
    celex_dict['tokens'][token_id]['lemmaID'] = lemma_id
    celex_dict['tokens'][token_id]['phon'] = token_phonetic
    celex_dict['tokens'][token_id]['stressed_vowel'] = vowel
    celex_dict['tokens'][token_id]['inflection'] = inflection
    celex_dict['lemmas'][lemma_id]['surface'] = lemma_surface
    celex_dict['lemmas'][lemma_id]['morph_analysis'] = morph
    celex_dict['lemmas'][lemma_id]['lemma_phon'] = lemma_phonetic
    celex_dict['lemmas'][lemma_id]['pos'] = pos

    return celex_dict


########################################################################################################################


def create_celex_dictionary(celex_dir, reduced=True):

    """
    :param celex_dir:   a string specifying the path to the folder where the information extracted from the CELEX
                        database will be stored
    :param reduced:     a boolean specifying whether reduce phonological form should be always preferred when available
    :return celex_dict: a Python dictionary containing information about phonology and morphology about words from the
                        Celex database. For further details about which information is stored and how, check the
                        documentation of the functions in this module

    The function takes 6 seconds to run on a 2x Intel Xeon 6-Core E5-2603v3 with 2x6 cores and 2x128 Gb of RAM.
    """

    if not celex_dir.endswith("/"):
        celex_dir += "/"

    epw_path = celex_dir + 'epw.cd'
    epl_path = celex_dir + 'epl.cd'
    emw_path = celex_dir + 'emw.cd'
    eml_path = celex_dir + 'eml.cd'
    celex_vowels = vowels()
    celex_dict = initialize_celex_dict()

    celex_dict = read_epw(epw_path, celex_dict, celex_vowels, reduced=reduced)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": I finished processing the epw.cd file.")
    print()
    celex_dict = read_epl(epl_path, celex_dict)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": I finished processing the epl.cd file.")
    print()
    celex_dict = read_emw(emw_path, celex_dict)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": I finished processing the emw.cd file.")
    print()
    celex_dict = read_eml(eml_path, celex_dict)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": I finished processing the eml.cd file.")
    print()

    celex_dict = hardcode_words(celex_dict, "will'nt", '1000000', '51924', "'wIl-Ht", 'I', 'X', "won't",
                                "'w5nt", ['will', 'not'], 'V')
    celex_dict = hardcode_words(celex_dict, "mhm", '1000001', '1000001', "em", '_', 'X', "non_ling",
                                '_', ['non_ling'], 'C')
    celex_dict = hardcode_words(celex_dict, "lego", '1000002', '1000002', "'l3-go", 'e', 'S', "lego",
                                "'l3-go", ['lego'], 'N')
    celex_dict = hardcode_words(celex_dict, "colour", '1000003', '8425', "'kV-l@R", 'V', 'S', "colour",
                                "'kV-l@R", ['colour'], 'N')
    celex_dict = hardcode_words(celex_dict, "color", '1000004', '8425', "'kV-l@R", 'V', 'S', "colour",
                                "'kV-l@R", ['colour'], 'N')
    celex_dict = hardcode_words(celex_dict, "horsie", '1000005', '21619', "h$s", '$', 'S', "horse",
                                "'h$s", ['horse'], 'N')
    celex_dict = hardcode_words(celex_dict, "ssh", '1000006', '1000001', "C", '_', 'X', "non-ling",
                                '_', ['non-ling'], 'C')
    celex_dict = hardcode_words(celex_dict, "byebye", '1000007', '5863', "b2-'b2", '2', 'X', "bye-bye",
                                "b2-'b2", ['bye'], 'C')
    celex_dict = hardcode_words(celex_dict, "shooter", '1000008', '1000003', "'Su-t@R", 'u', 'S', "shooter",
                                "'Su-t@R", ['shoot', 'er'], 'N')
    celex_dict = hardcode_words(celex_dict, "mousie", '1000009', '29286', "'m2-sI", '2', 'S', "mouse",
                                "'m6s", ['mouse', 'y'], 'N')
    celex_dict = hardcode_words(celex_dict, "mummie", '1000010', '29460', "'mV-mI", 'V', 'S', "mummy",
                                "'mV-mI", ['mum', 'y'], 'N')
    celex_dict = hardcode_words(celex_dict, "favorite", '1000011', '16271', "'f1-vrIt", '1', 'b', "favourite",
                                "'f1-v@-rIt", ['favour', 'ite'], 'A')
    celex_dict = hardcode_words(celex_dict, "upside", '1000012', '1000004', "Vp-'s2d", '2', 'b', "upside",
                                "Vp-'s2d", ['upside'], 'B')
    celex_dict = hardcode_words(celex_dict, "carwash", '1000013', '1000005', "'k#R-wQS", '#', 'S', "carwash",
                                "'k#R-wQS", ['car', 'wash'], 'N')
    celex_dict = hardcode_words(celex_dict, "anymore", '1000014', '1000006', "E-nI-'m$R", '$', 'X', "anymore",
                                "E-nI-'m$R", ['any', 'more'], 'B')
    celex_dict = hardcode_words(celex_dict, "whee",  '1000015', '1000001', "wee", '_', 'X', "non-ling",
                                '_', ['non-ling'], 'C')
    celex_dict = hardcode_words(celex_dict, "carpark", '1000016', '1000007', "k#R-'p#k", '#', 'S', "carpark",
                                "k#R-'p#k", ['car', 'park'], 'N')
    celex_dict = hardcode_words(celex_dict, "lawnmower", '1000017', '1000008', "'l$n-m5-er", '$', 'S', "lawnmower",
                                "'l$n-m5-er", ['lawn', 'mow', 'er'], 'N')
    celex_dict = hardcode_words(celex_dict, "whoo", '1000018', '1000001', "hU", '_', 'X', "non-ling",
                                '_', ['non-ling'], 'C')
    celex_dict = hardcode_words(celex_dict, "doggie", '1000019', '13205', "'dQ-gI", 'Q', 'S', "doggy",
                                "'dQ-gI", ['dog', 'y'], 'N')
    celex_dict = hardcode_words(celex_dict, "hotdog", '1000020', '21690', "hQt-'dQg", 'Q', 'S', "hot dog",
                                "hQt-'dQg", ['hot', 'dog'], 'N')
    celex_dict = hardcode_words(celex_dict, "christmas", '1000021', '7479', "'krIs-m@s", 'I', 'X', "christmas",
                                "'krIs-m@s", ['christ', 'mas'], 'N')
    celex_dict = hardcode_words(celex_dict, "traveling", '1000022', '48164', "'tr{v-lIN", '{', 'pe', "travel",
                                "'tr{-vP", ['travel'], 'V')
    celex_dict = hardcode_words(celex_dict, "snowplow", '1000023', '43060', "'snO-pl8", 'O', 'X', "snowplough",
                                "'sn5-pl6", ['snow', 'plough'], 'N')
    celex_dict = hardcode_words(celex_dict, "colors", '1000024', '8425', "'kV-l@Rs", 'V', 'P', "colour",
                                "'kV-l@R", ['colour'], 'N')

    if reduced:
        celex_dict_file = "/".join([os.getcwd(), 'Celex/celex_dict_reduced.json'])
    else:
        celex_dict_file = "/".join([os.getcwd(), 'Celex/celex_dict.json'])
    print_celex_dict(celex_dict, celex_dict_file)

    return celex_dict


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Process arguments to create Celex dictionary.')

    parser.add_argument('-C', '--Celex_dir', required=True, dest='celex_dir',
                        help='Specify the directory containing the CELEX files.')
    parser.add_argument('-r', '--reduced', action='store_true', dest='reduced',
                        help='Specify if the function considers reduced phonological forms.')

    args = parser.parse_args()

    if not os.path.isdir(args.celex_dir):
        raise ValueError("The folder you provided does not exist. Please, provide the path to an existing folder.")
    else:
        create_celex_dictionary(args.celex_dir, reduced=args.reduced)


########################################################################################################################


if __name__ == '__main__':

    main()
