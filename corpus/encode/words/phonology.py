__author__ = 'GCassani'

"""Get phonological information for words from the Celex dictionary and use it to encode words"""

import re


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

    return phonological_representations


########################################################################################################################


def get_phonological_form(word, celex_dict, tokens2identifiers):

    """
    :param word:                a tuple consisting of three strings, the word form, its PoS tag, and the corresponding
                                lemma
    :param celex_dict:          a dictionary containing information extracted from the CELEX database, encoded using the
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
        phonological_form = return_matching_phonology(token_pos, lemma, token_ids, celex_dict)
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
                    phonological_forms.append(return_matching_phonology(token_pos, lemma, token_ids, celex_dict))

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
                                            boundary markers ('+')
    """

    # join each phonological representation with a word boundary marker ('+'), and also signal utterance boundaries
    # with the same symbol
    utterance = '+'.join(phonological_representations)
    utterance = '+' + utterance + '+'

    return utterance
