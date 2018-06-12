__author__ = 'GCassani'

"""Function to """

import re
from celex.utilities.helpers import inflection_dict
from celex.utilities.dictionaries import map_inflection


def get_morphological_encoding(word_list, celex, tokens2identifiers):

    """
    :param word_list:           a list of tuples, each consisting of three strings: first the word's baseform, then its
                                PoS tag, then the lemma corresponding to the token
    :param celex:               a dictionary containing information extracted from the CELEX database, encoded using the
                                celex_processing.py method (see documentation therein)
    :param tokens2identifiers:  a dictionary mapping tokens to all the token ids from CELEX that correspond to the given
                                surface form
    :return lexical_units:      a set of strings containing lexemes extracted from the input list of orthographic forms
    """

    lexical_nodes = set()
    inflections = inflection_dict()

    # fetch the lemma and the inflectional morphology of the input token. The function get_lemma_and_inflection always
    # returns a tuple, where the first item is a string (either the input token itself or its corresponding lemma from
    # CELEX) and the second item can be either a set of inflectional morphemes or None, if no inflection could be
    # retrieved from CELEX for the input token. The function always add the first item (lemma or token) to the lexical
    # representation of the input list of words, and then check whether inflectional morphology exists for the input
    # token: if it does exist, all grammatical morphemes are added to the lexical representation of the input list of
    # words; if it doesn't nothing happens. When no inflection is found, the input token is treated as an unanalyzed
    # lexical unit and added to the lexical representation of the input list of words.

    for word in word_list:
        lemma, inflectional_meanings = get_lemma_and_inflection(word, celex, tokens2identifiers, inflections)
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
