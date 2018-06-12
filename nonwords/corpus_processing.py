__author__ = 'GCassani'

from collections import defaultdict
from celex.get import get_celex_dictionary
from celex.utilities.dictionaries import tokens2ids
from corpus.encode.item import encode_item
from corpus.encode.words.phonology import get_phonetic_encoding


def adjust_apostrophes(token, lemma):

    """
    :param token:           a string
    :return token, lemma:   the same string twice
    """

    # adjust apostrophes!
    if token == 'oclock':
        new_token = "o'clock"
        new_lemma = "o'clock"
    elif token == 'wheres_kitty':
        new_token = "where's_kitty"
        new_lemma = "where's_kitty"
    elif token == 'whats_her_name':
        new_token = "what's_her_name"
        new_lemma = "what's_her_name"
    elif token == 'thats_entertainment':
        new_token = "that's_entertainment"
        new_lemma = "that's_entertainment"
    elif token == 'childrens_museum':
        new_token = "children's_museum"
        new_lemma = "children's_museum"
    elif token == 'childrens_hospital':
        new_token = "children's_hospital"
        new_lemma = "children's_hospital"
    else:
        new_token = token
        new_lemma = lemma

    return new_token, new_lemma


########################################################################################################################


def code_derivational_morphology(pos):

    """
    :param pos:     the part of speech tag of the current word
    :return deriv:  the recoded derivational formology
    """

    if pos == 'JJS':
        deriv = 'SUPERLATIVE'
    elif pos == 'JJR':
        deriv = 'COMPARATIVE'
    elif pos == 'NNS':
        deriv = 'PL'
    elif pos == 'VBG':
        deriv = 'CONTINUOUS'
    elif pos == 'VBZ':
        deriv = 'PERSON3'
    elif pos == 'VBD' or pos == 'VBN':
        deriv = 'PAST'
    else:
        deriv = 'BASE'

    return deriv


########################################################################################################################


def write_mapping_file(mapping, output_file):

    """
    :param mapping:     a dictionary mapping sequences of token, lemma, pos1, pos2, derivational codes, compoundness,
                        and token:PoS information to the triphones of which the token consists
    :param output_file: the path to the file where each key:value pair from the mapping will be printed as
                        tab-separated columns
    """

    with open(output_file, 'w') as fw:
        for word in mapping:
            token, lemma, pos1, pos2, deriv, morpho, token_pos = word.split('|')
            triphones = mapping[word]
            fw.write('\t'.join([token, lemma, pos1, pos2, deriv, morpho, token_pos, triphones]))
            fw.write('\n')


########################################################################################################################


def write_output_corpus(new_corpus, output_file, lemmas2phonetics, minimalist=True):

    """
    :param new_corpus:          a list of lists, with each inner list being a sequence of 5-tuples
    :param output_file:         the path to the file where all elements in the 5-tuple will be printed as
                                pipe-separated strings, and all 5-tuples will be printed as comma separated strings
    :param lemmas2phonetics:    a dictionary mapping a lemma to all the different phonetic representations found for it
                                in Celex and the PoS tag corresponding to the phonetic realization
    :param minimalist:          a boolean. It specifies whether lemmas in the output file should be differentiated when
                                their phonetic realization changes depending on the part of speech: if minimalist is
                                True, lemmas are not differentiated (default), if it is False, lemmas are
                                differentiated by appending pos1 to the lemma, separated by a colon.
    """

    with open(output_file, 'w') as fw2:
        for utterance in new_corpus:
            if minimalist:
                fw2.write(','.join('|'.join(word) for word in utterance))
                fw2.write('\n')
            else:
                output_utterance = []
                for word in utterance:
                    token, lemma, pos1, pos2, triphones = word
                    out_lemma = lemma
                    if len(lemmas2phonetics[lemma]) > 1:
                        for phonetic_coding in lemmas2phonetics[lemma]:
                            pos_tags = lemmas2phonetics[lemma][phonetic_coding]
                            for pos_tag in pos_tags:
                                if pos_tag == pos1:
                                    out_lemma = ':'.join([lemma, pos1])
                                    break
                    output_utterance.append('|'.join([token, out_lemma, pos1, pos2, triphones]))
                fw2.write(','.join(output_utterance))
                fw2.write('\n')


########################################################################################################################


def map_phonology(corpus_file, mapping_file, output_file, celex_dir, compounds=True, reduced=False, minimalist=True):

    """
    :param corpus_file:         the path to a .txt file containing one utterance per line, with all words in the
                                utterance separated by a comma and each word being a tuple consisting of four
                                pipe-separated elements, token|lemma|PoS1|PoS2 where PoS1 is the coarse Celex tag and
                                PoS2 is the tag provided by the TreeTagger
    :param mapping_file:        the path to a .txt file where the output of the process will be written to
    :param output_file:         the path to a .txt file where the lines from the input will be rewritten as
                                comma-separated sequences of pipe-separated 5-tuples consisting of
                                token, lemma, pos1, pos2, 3phones
    :param celex_dir:           the directory where the Celex dictionary is to be found
    :param compounds:           a boolean. If true, all entries in Celex are considered; if False, entries which contain
                                spaces are discarded
    :param minimalist:          a boolean. It specifies whether lemmas in the output file should be differentiated when
                                their phonetic realization changes depending on the part of speech: if minimalist is
                                True, lemmas are not differentiated (default), if it is False, lemmas are differentiated
                                by appending pos1 to the lemma, separated by a colon
    :return mapping:            a dictionary mapping 4-tuples token|lemma|pos1|pos2 to the matching triphones
    """

    celex_dict = get_celex_dictionary(celex_dir, reduced=reduced, compounds=compounds)
    tokens2identifiers = tokens2ids(celex_dict)
    mapping = {}
    lemma2phon = defaultdict(dict)

    new_corpus = []

    with open(corpus_file, 'r') as fr:
        for line in fr:
            words = line.strip().split(',')
            new_line = []
            for word in words:
                if word:
                    try:
                        token, lemma, pos1, pos2 = word.split('|')
                    except ValueError:
                        token, lemma, pos1 = word.split('|')
                        pos2 = 'NN' if pos1 == 'N' else pos1

                    new_token, new_lemma = adjust_apostrophes(token, lemma)
                    new_token = new_token.replace('=', '_')
                    new_lemma = new_lemma.replace('=', '_')
                    token_phonological_form = get_phonetic_encoding([(new_token, pos1, new_lemma)],
                                                                    celex_dict, tokens2identifiers)
                    lemma_phonology = get_phonetic_encoding([(new_lemma, pos1, new_lemma)],
                                                            celex_dict, tokens2identifiers)
                    lemma_phonological_form = ''.join(lemma_phonology) if isinstance(lemma_phonology, list) else \
                        ''.join(token_phonological_form)

                    if isinstance(token_phonological_form, list):
                        triphones = encode_item(token_phonological_form[0], triphones=True, stress_marker=True,
                                                uniphones=False, diphones=False, syllables=False)
                        deriv = code_derivational_morphology(pos2)
                        output_token = token.replace('_', '=')
                        output_lemma = lemma.replace('_', '=')

                        morpho = 'COMPOUND' if '=' in output_token else 'MONO'
                        key = '|'.join([output_token, output_lemma, pos1, pos2, deriv, morpho,
                                        ':'.join([output_token, pos1])])
                        output_triphones = ';'.join(triphones)

                        mapping[key] = output_triphones
                        if lemma_phonological_form in lemma2phon[output_lemma]:
                            lemma2phon[output_lemma][lemma_phonological_form].add(pos1)
                        else:
                            lemma2phon[output_lemma][lemma_phonological_form] = {pos1}

                        new_line.append((output_token, ':'.join([output_lemma, pos1]), pos1, pos2, output_triphones))
            new_corpus.append(new_line)

    write_mapping_file(mapping, mapping_file)
    write_output_corpus(new_corpus, output_file, lemma2phon, minimalist=minimalist)

    return mapping

