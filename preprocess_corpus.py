__author__ = 'GCassani'

import re
import os
import json
import argparse
import numpy as np
import celex_processing as clx
from collections import defaultdict
from time import strftime


"""
This module pre-process the input corpus, in the desired granularity, returning a two-layer representation for each
learning event, defined as a single unit of the desired granularity (be it a full utterance, a multi-word expression,
a prosodic-like chunk, or a single word): the first layer encodes the learning event into its constituent phonological
form; the second encodes the same learning event in terms of its lexical meaning. This module consists of the following
functions, listed together with a short description of what they do. Check the documentation of each function for
details about input arguments and output structures.

- get_token_identifier_mappings             : it swaps the ID-surface form mapping from the CELEX dictionary, returning
                                                a dictionary mapping surface forms to all IDs linked to it
- get_pos_category_mapping                  : it returns a dictionary mapping CHILDES PoS tags to coarser tags, as
                                                specified in a .txt file passed as input
- map_inflection                            : maps CELEX inflectional codes to the corresponding meaning, and returns a
                                                set of the meanings present in a given word
- get_syllables                             : returns a set containing all the syllables of which the input string - a
                                                phonological representation from CELEX of any granularity - consists
- get_phonetic_encoding                     : returns a string consisting of concatenation of the CELEX phonological
                                                forms for each word in the learning event (of whatever granularity)
- get_phonological_form                     : returns the CELEX phonological form for a single word, if it exists
- return_matching_phonology                 : makes sure that the correct CELEX phonological form is returned, checking
                                                that the lemma and the PoS tag of the corresponding token are consistent
                                                with the input
- concatenate_phonological_representations  : takes several individual phonological representations and concatenates
                                                them together with a specified symbol
- get_nphones                               : returns a list of the n-phones, i.e. sequences of phonemes of variable
                                                size, that make up the phonological representation of the learning event
- recode_stress                             : moves the stress marker from the beginning of the syllable where the
                                                stressed vowel falls, to immediately before the stressed vowel itself
- get_morphological_encoding                : returns a set of lexemes extracted from the tokens making up the learning
                                                event (lexical and, if desired, grammatical meanings)
- get_lemma_and_inflection                  : returns the lemma and the set of grammatical meanings from a single word,
                                                looking at the morphological representation in the CELEX dictionary
- return_matching_morphology                : makes sure that the correct CELEX morpohological meaning is returned,
                                                checking that the lemma and the PoS tag of the corresponding token are
                                                consistent with the input
- encoding_features                         : prints to screen a summary of the input features that specify that type of
                                                desired encoding (granularity of n-phones, presence of stress, presence
                                                of grammatical meanings)
- encode_corpus                             : the main function, encoding each learning event into a list of phonetic
                                                features and a set of lexical meanings, dumped in a json file consisting
                                                of two lists of lists
- main                                      : a function that calls encode_corpus when the module is called from command
                                                line

This function takes in a json file and returns a json file, consisting of two lists of lists, the first encoding
learning events (in the desired granularity) as lists of phonetic cues (of the desired granularity), the second encoding
the same learning events as set of meanings.
"""


def get_token_identifier_mappings(celex):

    """
    :param celex:               a dictionary containing information extracted from celex, see the documentation of the
                                celex_processing.py module
    :return tokens2identifiers: a dictionary mapping a token surface form from celex to all token ids linked to it

    The input dictionary uses token ids as keys for indicization issues but in this case a reverse mapping is needed.
    """

    # Each surface form is mapped to a set containing all the token identifiers to which the surface form is connected
    # in the celex database (the type of the values is set even when the token identifer is only one)
    tokens2identifiers = defaultdict(set)
    for k in celex['tokens']:
        token = celex['tokens'][k]['surface']
        if token in tokens2identifiers:
            tokens2identifiers[token].add(k)
        else:
            tokens2identifiers[token] = {k}

    return tokens2identifiers


########################################################################################################################


def get_pos_category_mapping(input_file):

    """
    :param input_file:  a .txt fil containing two elements per line separated by a white space. The first element of
                        each line is used as a key in a dictionary, and the second value in the line is used as the
                        corresponding value. If there duplicates among the first elements (the keys), the output
                        dictionary will contain the corresponding second element (the value) from the last line in
                        which the first element occurred.
    :return out_dict:   a dictionary containing elements from the input file arranged in the specified way
    """

    out_dict = {}

    with open(input_file, 'r') as r:
        for line in r:
            pos = line.rstrip("\n").split()
            out_dict[pos[0]] = pos[1]

    return out_dict


########################################################################################################################


def map_inflection(inflectional_code, inflectional_map):

    """
    :param inflectional_code:       a sequence of letters from the celex database, indicating the types of inflectional
                                    transformations that affected a token
    :param inflectional_map:        a dictionary mapping each letter to its meaning
    :return grammatical_lexemes:    a set containing all the grammatical lexemes (e.g., past, present, comparative, ...)
                                    indicated by the letters in the first argument
    """

    grammatical_lexemes = set()

    for char in inflectional_code:
        try:
            grammatical_lexemes.add(inflectional_map[char])
        except KeyError:
            continue

    return grammatical_lexemes


########################################################################################################################


def get_syllables(utterance):

    """
    :param utterance:           a string containing all the words from the utterance, with words separated by word
                                boundary markers ('+') and syllables being separated by syllable markers ('-')
    :return syllables:          a list of all the syllables that are present in the input words, preserving their order
    """

    utterance_syllables = []

    words = utterance.strip("+").split("+")
    for word in words:
        word = "+" + word + "+"
        utterance_syllables += word.split('-')

    return utterance_syllables

########################################################################################################################


def get_phonetic_encoding(word_list, celex, tokens2identifiers):

    """
    :param word_list:           a list of tuples, each containing three strings: first, the orthographic surface form of
                                a token extracted from CHILDES transcripts; second, the Part-of-Speech tag of the token;
                                third, the lemma corresponding to the token (e.g. the lemma 'sing' for the token 'sung')
    :param celex:               a dictionary containing information extracted from the CELEX database, encoded using the
                                celex_processing.py method (see documentation therein)
    :param tokens2identifiers:  a dictionary mapping tokens to all the token ids from CELEX that correspond to the given
                                surface form
    :return utterance:          a string containing all the input phonological forms, joined with word boundary markers
                                (plus, '+')
    *:return word, None:

    This function takes a list of words from an utterance of child-caregiver interaction and encode it in n-phones, i.e.
    sequences of phonemes, whose length can be specified via the parameter n (default is 2). Orthographic forms
    extracted from transcripts are mapped to phonological representations contained in the CELEX database. If a word
    from the input list cannot be retrieved from CELEX or is retrieved but its lexical category is different from the
    one of all lemmas retrieved in CELEX that correspond to the input token, this function returns the input token and
    None, to signal that no phonological representation could be retrieved from CELEX for that token (the function
    get_phonological_representations which is called inside the present function also prints a warning that specifies
    which problem was encountered).
    """

    phonological_representations = []

    # for each word in the input list, get its phonological representation from CELEX: if this step is successful, store
    # the phonological representation in a list (order is important!) and proceed; if this step fails, exit the function
    # and return the input token that couldn't be located in CHILDES and None
    for word in word_list:
        phonological_representation = get_phonological_form(word, celex, tokens2identifiers)
        if phonological_representation:
            phonological_representations.append(phonological_representation)
        else:
            return word, None

    # concatenate all words in the utterance using the word boundary symbol ('+') and also add it to the beginning
    # and end of the utterance itself
    utterance = concatenate_phonological_representations(phonological_representations)

    return utterance


########################################################################################################################


def get_phonological_form(word, celex, tokens2identifiers):

    """
    :param word:                a tuple consisting of three strings, the word form, its PoS tag, and the corresponding
                                lemma
    :param celex:               a dictionary containing information extracted from the CELEX database, encoded using the
                                celex_processing.py method (see documentation therein)
    :param tokens2identifiers:  a dictionary mapping tokens to all the token ids from CELEX that correspond to the given
                                surface form
    :return phonological_form:  the phonological representation of the token extracted from the CELEX database.
    *:return None:              when the input token cannot be found in CELEX or is found but its corresponding lemma
                                has a different PoS than the input token, the function returns None, to make it possible
                                to evaluate the outcome of this function before taking action.

    This function extracts the phonological representation of a token from the CELEX database, checking that the PoS of
    the token in CHILDES is consistent with the lemma corresponding to the token. This care is necessary because the
    same ortographic form can have different phonological representations depending on its PoS tag, e.g. object-N vs
    object-V, which have a different stress pattern. If the token is not included in CELEX or its PoS tag is different
    from that of all lemmas connected to the input token in CELEX, this function returns None, to allow the user to
    evaluate the function outcome before taking action.
    """

    surface_form, token_pos, lemma = word

    # get the CELEX tokenID or tokenIDs corresponding to a given surface form. If no tokenID is retrieved, token_ids is
    # an empty set
    token_ids = tokens2identifiers[surface_form]

    # check that token_ids contains at least one element, and then loop through each of them, get the corresponding
    # lemmaID and the Part-of-Speech associated with it. Check whether the PoS of the input word is consistent with the
    # PoS of the lemma: if they match, get the phonological representation associated with the tokenID being considered
    # and return it
    # If no lemma associated with the input surface form maps to the same PoS as the one of the input token, print a
    # warning message and return None
    if token_ids:
        phonological_form = return_matching_phonology(token_pos, lemma, token_ids, celex)
        return phonological_form

    # if the empty set is returned, meaning that no tokenID could be retrieved from CELEX, check whether it contains an
    # underscore or an apostrophe and split the surface form
    else:
        if "'" in surface_form or "_" in surface_form:
            phonological_forms = []
            surface_form = surface_form.replace("'", " '")
            components = re.findall(r"[a-zA-Z']+", surface_form)

            for component in components:
                # try to fetch the tokenIDs corresponding to each of the sub-units in which the surface string was
                # divided using underscores and apostrophes
                token_ids = tokens2identifiers[component]

                # if at least one tokenID is found, get the corresponding phonological form and append it to the list
                # of phonological representations for the current surface form, which consists of several subunits
                if token_ids:
                    phonological_forms.append(return_matching_phonology(token_pos, lemma, token_ids, celex))

                # otherwise, flag the surface form to warn that it's lacking from celex and return None, since the
                # complete phonological representation for the complex surface form cannot be entirely derived from
                # celex
                else:
                    return None

            # after all sub-units have been found in celex, concatenate them and return the full phonological
            # representation for the surface form as a string
            phonological_form = '-'.join(phonological_forms)
            return phonological_form

        # if the surface form was not found in celex and it cannot be broken up into different sub-units, add it to
        # the list of words for which no phonological representation can be found in celex and return None
        else:
            return None


########################################################################################################################


def return_matching_phonology(token_pos, lemma, token_ids, celex):

    """
    :param token_pos:           the Part-of-Speech tag of the surface form
    :param lemma:               a string indicating the lemma corresponding to the token for which the phonological
                                representation is required; it is required to ensure that the correct one is chosen for
                                the token being considered, in case of homography
    :param token_ids:           a set of unique identifiers from the celex dictionary, matching the surface form
    :param celex:               a dictionary containing information extracted from the CELEX database, encoded using the
                                celex_processing.py method (see documentation therein)
    :return phonological_form:  the string of phonemes extracted from Celex for the input surface form. If an entry is
                                found in CELEX that shares the same token_pos, the phonological form of that entry is
                                returned; otherwise, an entry matching the input surface form, regardless of token_pos,
                                is chosen at random and returned.
    """

    for token_id in token_ids:
        lemma_id = celex['tokens'][token_id]['lemmaID']
        lemma_pos = celex['lemmas'][lemma_id]['pos']
        target_lemma = celex['lemmas'][lemma_id]['surface']
        if lemma_pos == token_pos and lemma == target_lemma:
            # retrieve the corresponding phonological form in Celex
            phonological_form = celex['tokens'][token_id]['phon']
            return phonological_form
        elif lemma_pos == token_pos:
            phonological_form = celex['tokens'][token_id]['phon']
            return phonological_form

    # if a token appears in Celex but with a different PoS tag from the one in CHILDES, get the phonological
    # representation from the first entry from the set of matching tokens in Celex
    token_id = list(token_ids)[0]
    phonological_form = celex['tokens'][token_id]['phon']
    return phonological_form


########################################################################################################################


def concatenate_phonological_representations(phonological_representations):

    """
    :param phonological_representations:    a list of phonological forms retrieved from the Celex database
    :return utterance:                      a string containing all the input phonological forms, joined with word
                                            boundary markers (plus, '+')
    """

    # join each phonological representation with a word boundary marker ('+'), and also signal utterance boundaries
    # with the same symbol
    utterance = '+'.join(phonological_representations)
    utterance = '+' + utterance + '+'

    return utterance


########################################################################################################################

def get_nphones(utterance, n):

    """
    :param utterance:   a string, containing phonological representations extracted from the CELEX database
    :param n:           the length of outcome n-phones. If n=1, single characters are returned, if n=2, di-phones are
                        returned, i.e. sequences of two characters (+dog+, where + marks a word boundary, is encoded as
                        ['+d', 'do', 'og', 'g+']), and so on
    :return nphones:    a list of strings, containing all n-phones of the specified granularity used to encode the
                        input string
    """

    nphones = list()

    # if the granularity is higher, build n-phones and store them in a list
    # - len(utterance) - (n - 1) ensures you don't try to make an n-phone from the last character of the string (in the
    #   case of diphones), since there are no symbols after it (or from the second-to last in the case of triphones)
    # - n-phones are formed by picking every phoneme from the first and combining it with as many following phonemes as
    #   specified by the parameter n
    # - if stress needs to preserved, specified through the input parameter, then the stress marker (') is added to all
    #   nphones containing a stressed vowel, but the marker itself doesn't count as a symbol, so a diphone containing a
    #   stressed vowel actually contains 3 symbols, the two phonemes and the stress marker itself

    i = 0

    # loop through the input string considering every index except the last one
    while i < len(utterance):
        # initialize an empty string to build the nphone and a second index; then create a second variable indicating
        # the n-phone length that can be extended if the stress marker is found - the stress marker is not considered as
        # a phoneme but as something that modifies the stressed vowel and is part of it, e.g. +d'og+ is encoded as
        # ["+d", "d'o", "'og", "g+"] in diphones. The stress marker, though, makes "d'o" found in dog from "do" found in
        # "condo', where the o doesn't bear any stress
        nphone = ''
        j = 0
        flex_n = n
        while j < flex_n:
            try:
                nphone += utterance[i + j]
                # if the n-phone contains the stress marker, allow it to include a further character
                if utterance[i + j] == "'":
                    flex_n += 1
                j += 1
            except IndexError:
                i = len(utterance) + 1
                j = flex_n + 2

        if len(nphone) >= n:
            nphones.append(nphone)

        if i < len(utterance):
            # if the current symbol being processed is the stress marker, jump ahead of two symbols: this prevents the
            # function from storing two n-phones that only differ in the presence of the stress marker at the beginning,
            # e.g. 'US and US. US is the unstressed version of 'US and it is wrong to store them both, since only one
            # occurs
            if utterance[i] == "'":
                i += 2
            else:
                i += 1

    # make sure that no nphone of the wrong granularity was created by getting the stress marker wrong
    for nphone in nphones:
        table = str.maketrans(dict.fromkeys("'"))
        if len(nphone.translate(table)) < n:
            nphones.remove(nphone)
    return nphones


########################################################################################################################


def recode_stress(utterance, vowels):

    """
    :param utterance:           a string containing phonological representations extracted from Celex. Each word is
                                separated by a word boundary marker and a stress marker (" ' ") is placed at the
                                beginning of every stressed syllable
    :param vowels:              a set of the ASCII characters marking vowels in the phonological encoding that is being
                                used
    :return recoded_utterance:  a string containing the same phonological representations as the input utterance, but
                                where stress markers have been moved: instead of appearing at the beginning of each
                                stressed syllable they immediately precede each stressed vowel.
    """

    recoded_utterance = ''

    i = 0

    while i < len(utterance):
        # if the symbol being considered is not a stress marker, append it to the recoded utterance and move forward
        if utterance[i] != "'":
            recoded_utterance += utterance[i]
            i += 1
        # if it is, join phonemes until a vowel is reached and append this bag of phonemes to the recoded utterance,
        # then append a stress marker and the vowel bearing the stress. This way, the stress marker moves from the start
        # of the syllable, as it is encoded in Celex, to the position immediately preceding the vowel
        else:
            is_vowel = False
            bag_of_phonemes = ""
            j = 1
            while is_vowel == 0:
                # as long as the last processed symbol was not a vowel, check if the next is: if it is, append all
                # phonemes encountered between the stress marker and the vowel itself, the stress marker, and the vowel
                # to the recoded utterance. Then signal that a vowel was found and jump ahead of three steps to avoid
                # re-considering phonemes that have already been added to the recoded utterance
                try:
                    if utterance[i+j] in vowels:
                        recoded_utterance += bag_of_phonemes + "'" + utterance[i+j]
                        is_vowel = True
                        i += len(bag_of_phonemes)+2
                    # if it is not, add the next phoneme (a consonant) to the bag of phonemes and increment the index
                    # that allows to advance in the utterance, until a vowel is found
                    else:
                        bag_of_phonemes += utterance[i+j]
                        j += 1

                # in case there is a stress symbol but no vowel, return the original utterance stripped of the stress
                # symbol, since in this encoding only vowels can bear stress
                except IndexError:
                    if not utterance.startswith('+'):
                        utterance = '+' + utterance
                    if not utterance.endswith('+'):
                        utterance += '+'
                    table = str.maketrans(dict.fromkeys("'"))
                    return utterance.translate(table)

    # make sure that the utterance begins and ends with the word boundary symbol
    if not recoded_utterance.endswith('+'):
        recoded_utterance += '+'
    if not recoded_utterance.startswith('+'):
        recoded_utterance = '+' + recoded_utterance

    return recoded_utterance


########################################################################################################################


def get_morphological_encoding(word_list, celex, tokens2identifiers, inflection_map):

    """
    :param word_list:           a list of tuples, each consisting of three strings: first the word's baseform, then its
                                PoS tag, then the lemma corresponding to the token
    :param celex:               a dictionary containing information extracted from the CELEX database, encoded using the
                                celex_processing.py method (see documentation therein)
    :param tokens2identifiers:  a dictionary mapping tokens to all the token ids from CELEX that correspond to the given
                                surface form
    :param inflection_map:      a dictionary mapping celex codes for inflectional morphology to their explanations
    :return lexical_units:      a set of strings containing lexemes extracted from the input list of orthographic forms
    """

    lexical_nodes = set()

    # fetch the lemma and the inflectional morphology of the input token. The function get_lemma_and_inflection always
    # returns a tuple, where the first item is a string (either the input token itself or its corresponding lemma from
    # CELEX) and the second item can be either a set of inflectional morphemes or None, if no inflection could be
    # retrieved from CELEX for the input token. The function always add the first item (lemma or token) to the lexical
    # representation of the input list of words, and then check whether inflectional morphology exists for the input
    # token: if it does exist, all grammatical morphemes are added to the lexical representation of the input list of
    # words; if it doesn't nothing happens. When no inflection is found, the input token is treated as an unanalyzed
    # lexical unit and added to the lexical representation of the input list of words.

    for word in word_list:
        lemma, inflectional_meanings = get_lemma_and_inflection(word, celex, tokens2identifiers, inflection_map)
        for el in lemma:
            lexical_nodes.add(el)
        try:
            for i in inflectional_meanings:
                lexical_nodes.add(i)
        except TypeError:
            pass

    return lexical_nodes


########################################################################################################################


def get_lemma_and_inflection(word, celex, tokens2identifiers, inflection_map):

    """
    :param word:                a tuple consisting of a word's baseform, its PoS tag, and the corresponding lemma
    :param celex:               a dictionary containing information extracted from the CELEX database, encoded using the
                                celex_processing.py method (see documentation therein)
    :param tokens2identifiers:  a dictionary mapping tokens to all the token ids from CELEX that correspond to the given
                                surface form
    :param inflection_map:      a dictionary mapping celex codes for inflectional morphology to their explanations
    :return lemma:              a list containing the surface form(s) of the lemma corresponding to the input token,
                                extracted from the celex dictionary
    :return inflection:         the set of grammatical lexemes extracted using the inflectional codes
    *:return word,None:         when the input token cannot be found in CELEX or is found but its corresponding lemma
                                has a different PoS than the input token, the function returns the input word and None,
                                to make it possible to evaluate the outcome of this function before taking action.
                                Moreover, this choice allows to minimize problems of coverage: if something goes wrong
                                when retrieving the lemma or the inflection, the token can be nonetheless used as a
                                lexical unit on its own.

    This function takes a word from CHILDES transcript, looks in the CELEX database for the corresponding lemma and the
    inflectional changes that the lemma underwent. However, tha possibility exists that CELEX doesn't contain a lemma
    for all tokens found in CHILDES or that the token is coded with a PoS tag that is not included in CELEX. In both
    cases the function returns None, without making any guess.
    """

    surface_form, token_pos, lemma = word

    # get the set of token identifiers for the surface form being considered. If no tokenID is retrieved, token_ids is
    # an empty set
    token_ids = tokens2identifiers[surface_form]

    # check that token_ids contains at least one element, and then loop through each of them, get the corresponding
    # lemmaID and the Part-of-Speech associated with it. Check whether the PoS of the input word is consistent with the
    # PoS of the lemma: if they match, get the inflectional morphology associated with the tokenID being considered
    # and return it together with the lemma itself.
    # If no lemma associated with the input surface form maps to the same PoS as the one of the input token, print a
    # warning message and return the token and None
    if token_ids:
        lemma, inflection = return_matching_morphology(surface_form, token_pos, token_ids, celex,
                                                       inflection_map)
        return [lemma], inflection

    # if the empty set is returned, meaning that no tokenID could be retrieved from CELEX, check whether it contains an
    # underscore or an apostrophe and split the surface form
    else:
        if "'" in surface_form or "_" in surface_form:
            lemmas = []
            inflections = []
            surface_form = surface_form.replace("'", " '")
            components = re.findall(r"[a-zA-Z']+", surface_form)

            for component in components:
                # try to fetch the tokenIDs corresponding to each of the sub-units in which the surface string was
                # divided using underscores and apostrophes
                token_ids = tokens2identifiers[component]

                # if at least one tokenID is found, get the corresponding morphological analysis and append all lemmas
                # to the list of lemmas and all inflectional meanings to the list of inflections for the current surface
                # form, which consists of several subunits
                if token_ids:
                    lemma, inflection = return_matching_morphology(component, token_pos, token_ids, celex,
                                                                   inflection_map)
                    lemmas.append(lemma)
                    if inflection:
                        inflections.append(inflection)

                # otherwise, flag the surface form to warn that it's lacking from celex and return the surface form
                # together with its Part-of-Speech and None, since the complete morphological analysis for the complex
                # surface form cannot be entirely derived from celex and no inflectional meanings could be fetched
                else:
                    return ['|'.join([surface_form, token_pos])], None

            # after all sub-units have been found in celex, return a list containing all components' lemmas with the set
            # of inflectional meanings
            return lemmas, inflections

        # if the surface form was not found in celex and it cannot be broken up into different sub-units, add it to
        # the list of words for which no morphological analysis can be found in celex and return the surface form
        # together with its PoS tag and None, since no inflectional meaning could be retrieved either
        else:
            return ['|'.join([surface_form, token_pos])], None


########################################################################################################################


def return_matching_morphology(surface_form, token_pos, token_ids, celex, inflection_map):

    """
    :param surface_form:    a string representing a word form in its written form
    :param token_pos:       the Part-of-Speech tag of the input surface form, indicating the lexical category of the
                            input word as it is used in the input utterance
    :param token_ids:       the token identifiers from Celex that correspond to the input surface form. Each token
                            identifier may point to a different lemma, and the best matching lemma for the input surface
                            form needs to be found
    :param celex:           a dictionary obtained running the celex_processing.py module
    :param inflection_map:  a dictionary mapping celex codes for inflectional morphology to their explanations
    :return lemma:          a string representing the surface form of the lemma that matches the input surface form:
                            here, best means that the PoS tag of the input surface form and that of the output lemma
                            match (both are verbs, for example)
    :return inflection:     a set containing all the inflectional meanings contained in the input surface form (e.g.
                            3RD PERSON, PLURAL, PRESENT, ...
    :*return word_pos:      if no matching lemma was retrieved from Celex, the input surface form and its PoS tag are
                            returned, joined with a vertical bar ('|')
    :*return None:          Moreover, a None is returned to mark that no lemma and inflectional meanings were found
    """

    for token_id in token_ids:
        lemma_id = celex['tokens'][token_id]['lemmaID']
        lemma_pos = celex['lemmas'][lemma_id]['pos']

        if lemma_pos == token_pos:
            lemma = '|'.join([celex['lemmas'][lemma_id]['surface'], lemma_pos])
            inflection = map_inflection(celex['tokens'][token_id]['inflection'], inflection_map)
            return lemma, inflection

    word_pos = '|'.join([surface_form, token_pos])

    return word_pos, None


########################################################################################################################


def encoding_features(corpus_name, reduced=True, uni_phones=True, di_phones=False, tri_phones=False, syllable=False,
                      stress_marker=False, grammatical=False, log='', verbose=True):

    """
    :param corpus_name:         a string indicating the name of the corpus being processed
    :param reduced:             a boolean indicating whether reduced phonological forms are extracted from CELEX or not
    :param uni_phones:          a boolean indicating whether single phonemes are to be considered while encoding input
                                utterances
    :param di_phones:           a boolean indicating whether sequences of two phonemes are to be considered while
                                encoding input utterances
    :param tri_phones:          a boolean indicating whether sequences of three phonemes are to be considered while
                                encoding input utterances
    :param syllable:            a boolean indicating whether syllables are to be considered while encoding input
                                utterances
    :param stress_marker:       a boolean indicating whether stress markers from the phonological representations of
                                Celex need to be preserved or can be discarded
    :param grammatical:         a boolean specifying whether inflectional and grammatical meanings should be extracted
                                from input utterances (these include meanings such as PLURAL, PRESENT, THIRD PERSON, and
                                so on). If False, only lexical meanings are extracted from input utterances, using the
                                lemmas provided in the input corpus.
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
        encoding_string += 'w'

    if uni_phones:
        desired_cues.append('uniphones')
        encoding_string += 'u'
    if di_phones:
        desired_cues.append('diphones')
        encoding_string += 'd'
    if tri_phones:
        desired_cues.append('triphones')
        encoding_string += 't'
    if syllable:
        desired_cues.append('syllables')
        encoding_string += 's'

    if stress_marker:
        desired_cues.append('with stress marker')
        encoding_string += 'm'
    else:
        desired_cues.append('without stress marker')
        encoding_string += 'n'

    if grammatical:
        outcome_encoding = "lexical, grammatical, and inflectional meanings."
        encoding_string += 'g'
    else:
        outcome_encoding = "lexical meanings only."
        encoding_string += 'l'

    num_hash = 120
    desired_cues = ", ".join(desired_cues)
    padding_cues = " " * (num_hash - 15 - len(desired_cues))
    padding_outcomes = " " * (num_hash - 19 - len(outcome_encoding))
    padding_corpus = " " * (num_hash - 17 - len(corpus_name))
    if log:
        with open(log, "w+") as l:
            l.write("\n\n")
            l.write("#" * num_hash)
            l.write("\n")
            l.write("#####  CORPUS: " + corpus_name + padding_corpus + "##")
            l.write("\n")
            l.write("#####  CUES: " + desired_cues + padding_cues + "##")
            l.write("\n")
            l.write("#####  OUTCOMES: " + outcome_encoding + padding_outcomes + "##")
            l.write("\n")
            l.write("#" * num_hash)
            l.write("\n\n")
    else:
        if verbose:
            print()
            print("#" * num_hash)
            print("#####  CORPUS: " + corpus_name + padding_corpus + "##")
            print("#####  CUES: " + desired_cues + padding_cues + "##")
            print("#####  OUTCOMES: " + outcome_encoding + padding_outcomes + "##")
            print("#" * num_hash)
            print()

    return encoding_string


########################################################################################################################


def encode_item(item, celex_vowels, uni_phones=True, di_phones=False, tri_phones=False,
                syllable=False, stress_marker=False):

    """
    :param item:            a string 
    :param celex_vowels:    a set of strings indicating which symbols indicate vowels in Celex phonetic transcriptions
    :param uni_phones:      a boolean indicating whether single phonemes are to be considered while encoding input
                            utterances
    :param di_phones:       a boolean indicating whether sequences of two phonemes are to be considered while
                            encoding input utterances
    :param tri_phones:      a boolean indicating whether sequences of three phonemes are to be considered while
                            encoding input utterances
    :param syllable:        a boolean indicating whether syllables are to be considered while encoding input
                            utterances
    :param stress_marker:   a boolean indicating whether stress markers from the input phonological representation need
                            to be preserved or can be discarded
    :return nphones:        an iterable containing the desired phonetic cues from the input word
        """

    if stress_marker:
        item = recode_stress(item, celex_vowels)
    else:
        item = item.translate(None, "'")

    uniphones = []
    diphones = []
    triphones = []
    syllables = []

    if syllable:
        syllables = get_syllables(item)

    # check that syllables only contain one vowel, print otherwise to evaluate what to do:
    # loop through every character in the syllable, check if the character is a vowel and increment the
    # vowel count if it is; if the vowel count reaches 2, print the utterance and the syllable, and get
    # out of the loop
    # CAVEAT: this should not print anything, it's just a sanity check
    for s in syllables:
        v = 0
        for c in s:
            if c in celex_vowels:
                v += 1
            if v > 1:
                print(item, s)
                print()
                break

    # get rid of the syllable markers to extract n-phones, then collapse all cues in
    # a single list representing the phonetic layer of the input
    table = str.maketrans(dict.fromkeys("-"))
    item = item.translate(table)
    if uni_phones:
        uniphones = get_nphones(item, n=1)
    if di_phones:
        diphones = get_nphones(item, n=2)
    if tri_phones:
        triphones = get_nphones(item, n=3)
    nphones = uniphones + diphones + triphones + syllables

    return nphones


########################################################################################################################


def encode_corpus(corpus_name, celex_dict_file, separator='~', grammatical=False,
                  reduced=True, uni_phones=True, di_phones=False,
                  tri_phones=False, syllable=False, stress_marker=False):

    """
    :param corpus_name:     the path to a .json file containing transcripts of child-caregiver interactions extracted
                            from the CHILDES database. The json file consists of two lists of lists, of the same length,
                            both contain words but encoded differently. The first is a list of tokens, i.e. surface
                            forms as they appear in the transcriptions; the second is a list of lemmas, i.e. words in
                            their dictionary form, without any inflectional morphology, together with their
                            Part-of-Speech tags, joined by a specific character (which can be specified with the
                            parameter 'separator'.
    :param celex_dict_file: a string indicating the path to the CELEX directory file with information from the CELEX 
                            database ('epw.cd', 'epl.cd', 'emw.cd', eml.cd'). The function also checks whether this 
                            directory already contains the file 'celex_dict.json' which contains all the relevant
                            information from Celex that the function uses. If it is found, the dictionary is loaded and
                            the function proceeds, otherwise the dictionary is created.
    :param separator:       the character that separates the word baseform from its PoS tag in the input corpus
    :param reduced:         a boolean specifying whether reduced phonological forms should be extracted from Celex
                            whenever possible (if set to True) or if standard phonological forms should be preserved
                            (if False)
    :param grammatical:     a boolean specifying whether inflectional and grammatical meanings should be extracted
                            from input utterances (these include meanings such as PLURAL, PRESENT, THIRD PERSON, and
                            so on). If False, only lexical meanings are extracted from input utterances, using the
                            lemmas provided in the input corpus.
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

    # get CELEX vowels and the dictionary mapping inflectional morpohlogy codes to their meanings
    celex_vowels = clx.vowels()
    inflections = clx.inflection_dict()

    # get the directory where the corpus is stored: the encoded version of the corpus will be created
    # in the same folder; locate the pos_mapping file in the same folder and load its content
    corpus_dir = os.path.dirname(corpus_name) + "/"
    pos_mapping_file = corpus_dir + 'pos_mapping.txt'
    if not os.path.exists(pos_mapping_file):
        raise ValueError("The file 'pos_mapping.txt' could not be located in the same folder of the input corpus. "
                         "Please add the file to the folder, using the name 'pos_mapping.txt', following the required "
                         "structure, i.e. two space-separated columns containing strings.")
    else:
        pos_dict = get_pos_category_mapping(pos_mapping_file)

    # use the path of the input file to generate the path of the output file, adding encoding information to the
    # input filename; print to standard output a summary of all the encoding parameters
    input_filename, extension = os.path.splitext(corpus_name)
    encoding_string = encoding_features(corpus_name, reduced=reduced, uni_phones=uni_phones, di_phones=di_phones,
                                        tri_phones=tri_phones, syllable=syllable, stress_marker=stress_marker,
                                        grammatical=grammatical)
    output_file = input_filename + "_" + encoding_string + '.json'

    # get the Celex dictionary; create a dictionary where token surface forms are keys, and values
    # are sets containing all the token IDs that match a given surface form
    celex_dict = clx.get_celex_dictionary(celex_dict_file, reduced=reduced)
    tokens2identifiers = get_token_identifier_mappings(celex_dict)

    # count how many utterances contain a word whose phonological representation could not be located in Celex.
    missed_utterances = 0

    # check whether the output file corresponding to the desired parameters already exist and stop if it does
    if os.path.isfile(output_file):
        print()
        print("The desired encoded version of the input corpus '%s' already exists at file '%s'." %
              (os.path.basename(corpus_name), os.path.basename(output_file)))
        return output_file
    else:

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Started encoding utterances from input corpus '%s'" % corpus_name)

        # get a dictionary mapping utterance indices to the percentage of corpus that has been processed up to the
        # utterance itself
        corpus = json.load(open(corpus_name, 'r+'))
        total_utterances = len(corpus[0])
        check_points = {np.floor(total_utterances / float(100) * n): n for n in np.linspace(5, 100, 20)}

        encoded_corpus = [[], []]

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
            if words:

                utterance = get_phonetic_encoding(words, celex_dict, tokens2identifiers)

                # if a phonological representation could be found for all words in the utterance, proceed
                if isinstance(utterance, str):

                    table = str.maketrans(dict.fromkeys('"'))
                    utterance = utterance.translate(table)

                    n_phones = encode_item(utterance, celex_vowels, uni_phones=uni_phones, di_phones=di_phones,
                                           tri_phones=tri_phones, syllable=syllable, stress_marker=stress_marker)

                    # get the morphological representations for the words (when they don't have a morphological
                    # representation in CELEX, the token is simply used as an un-analyzed lexical unit.
                    if grammatical:
                        outcomes = get_morphological_encoding(words, celex_dict, tokens2identifiers, inflections)
                    else:
                        outcomes = set()
                        for j in range(len(corpus[0][i])):
                            lemma, pos_tag = corpus[1][i][j].split(separator)
                            if pos_tag in pos_dict:
                                new_tag = pos_dict[pos_tag]
                                outcomes.add('|'.join([lemma, new_tag]))

                    # append the phonetic representation of the current learning event to the list of phonetic
                    # representations for the whole corpus, and the lexical meanings of the current learning event to
                    # the list of lexical meanings for the whole corpus
                    encoded_corpus[0].append(n_phones)
                    encoded_corpus[1].append(list(outcomes))

                # if the phonological representation of a word from the utterance could not be retrieved from
                # CELEX, count the utterance as missed
                else:
                    missed_utterances += 1

            if i in check_points:
                print(strftime("%Y-%m-%d %H:%M:%S") +
                      ": %d%% of the input corpus has been processed and encoded in the desired way." % check_points[i])

        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished encoding utterances from input corpus '%s'" % corpus_name)
        print()

        perc_missed = missed_utterances / float(total_utterances) * 100
        perc_covered = 100 - perc_missed
        print()
        if os.path.exists(output_file):
            print("The file %s already exists." % output_file)
        else:
            json.dump(encoded_corpus, open(output_file, 'w'))
            print("The file %s has been created:" % output_file)
            print()
            print("%0.4f%% of the utterances could be entirely encoded." % perc_covered)
            print("The remaining %0.4f%% contain at least one word that could not be retrieved in CELEX and "
                  % perc_missed)
            print("for which no phonological and morphological representation could be obtained.")

    return output_file


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Process arguments to create Celex dictionary.')

    parser.add_argument("-i", "--input_file", required=True, dest="in_file",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-C", "--Celex_dict", required=True, dest="celex_dict_file",
                        help="Specify the path to the CELEX dictionary created with the modul celex_preprocessing.py.")
    parser.add_argument("-S", "--separator", dest="sep", default='~',
                        help="Specify the character separating lemma and PoS tag in the input corpus.")
    parser.add_argument("-u", "--uniphones", action="store_true", dest="uni",
                        help="Specify if uniphones need to be encoded.")
    parser.add_argument("-d", "--diphones", action="store_true", dest="di",
                        help="Specify if diphones need to be encoded.")
    parser.add_argument("-t", "--triphones", action="store_true", dest="tri",
                        help="Specify if triphones need to be encoded.")
    parser.add_argument("-s", "--syllables", action="store_true", dest="syl",
                        help="Specify if syllables need to be encoded.")
    parser.add_argument("-m", "--stress_marker", action="store_true", dest="stress",
                        help="Specify if stress need to be encoded.")
    parser.add_argument("-g", "--grammatical", action="store_true", dest="gramm",
                        help="Specify grammatical meanings need to be encoded at the outcome level")
    parser.add_argument("-r", "--reduced", action="store_true", dest="reduced",
                        help="Specify if reduced vowels are to be considered when extracting CELEX phonetic forms.")

    args = parser.parse_args()

    if not (args.uni or args.di or args.tri or args.syl):
        parser.error('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    if not os.path.exists(args.celex_dict_file) or not args.celex_dict_file.endswith(".json"):
        raise ValueError("There are problems with the CELEX dictionary file you provided: either the path does not "
                         "exist or the file extension is not .json. Provide a valid path to a .json file.")

    if not os.path.exists(args.in_file) or not args.in_file.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    encode_corpus(args.in_file, args.celex_dict_file, separator=args.sep, uni_phones=args.uni,
                  di_phones=args.di, tri_phones=args.tri, syllable=args.syl, stress_marker=args.stress,
                  grammatical=args.gramm, reduced=args.reduced)


########################################################################################################################


if __name__ == '__main__':

    main()
