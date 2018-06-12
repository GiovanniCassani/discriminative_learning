__author__ = 'GCassani'

"""Function to encode the full corpus"""

import json
import numpy as np
from time import strftime
from corpus.encode.item import encode_item
from corpus.encode.words.phonology import get_phonetic_encoding, concatenate_phonological_representations
# from corpus.encode.words.morphology import get_morphological_encoding


def encode_corpus(corpus_name, celex_dict, tokens2identifiers, pos_dict,
                  separator='~', uniphones=False, diphones=True, triphones=False, syllables=False,
                  stress_marker=True, outcomes='tokens', boundaries=False):

    """
    :param corpus_name:         the path indicating the .json file to be used as input corpus
    :param celex_dict:          the Celex dictionary to be used to recode the utterances into phonetic cues
    :param tokens2identifiers:  a dictionary mapping a token surface form from Celex to all token ids linked to it
    :param pos_dict:            a dictionary mapping CHILDES PoS tags to corresponding Celex PoS tags
    :param separator:           a string indicating the character separating lemmas from PoS tags in the input corpus
    :param uniphones:           a boolean indicating whether uni-phones are relevant phonetic cues
    :param diphones:            a boolean indicating whether di-phones are relevant phonetic cues
    :param triphones:           a boolean indicating whether tri-phones are relevant phonetic cues
    :param syllables:           a boolean indicating whether syllables are relevant phonetic cues
    :param stress_marker:       a boolean indicating whether to discard or not the stress marker from the Celex phonetic
                                transcriptions
    :param outcomes:            a string indicating which outcomes to use, whether 'tokens' (default) or 'lemmas'
    :param boundaries:          a boolean indicating whether to preserve or discard word boundaries
    :return encoded corpus:     the input corpus recoded as a list of lists, where each inner list is a learning event
                                and consist of two sub-lists, the first containing phonetic cues and the second
                                containing lexical outcomes
    :return perc_missed:        the percentage of learning events from the input corpus that could not be recoded
                                because the Celes dictionary didn't contain the necessary information
    """

    # get a dictionary mapping utterance indices to the percentage of corpus that has been processed up to the
    # utterance itself
    corpus = json.load(open(corpus_name, 'r+'))
    total = len(corpus[0])
    check_points = {np.floor(total / float(100) * n): n for n in np.linspace(5, 100, 20)}

    encoded_corpus = [[], []]
    missed = 0

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
        if 0 < len(words) <= 20:

            # get the phonetic encoding of the words in the current learning trial:
            # if they can all be encoded using Celex, a list is returned, other wise a tuple is
            phonological_representations = get_phonetic_encoding(words, celex_dict, tokens2identifiers)

            # if a phonological representation could be found for all words in the utterance, proceed
            if isinstance(phonological_representations, list):

                utterance = concatenate_phonological_representations(phonological_representations)
                table = str.maketrans(dict.fromkeys('"'))
                utterance = utterance.translate(table)

                n_phones = encode_item(utterance, stress_marker=stress_marker, boundaries=boundaries,
                                       uniphones=uniphones, diphones=diphones, triphones=triphones, syllables=syllables)

                outcomes_set = set()
                for word in words:
                    token, pos, lemma = word
                    if outcomes == 'tokens':
                        outcomes_set.add('|'.join([token, pos]))
                    elif outcomes == 'lemmas':
                        outcomes_set.add('|'.join([lemma, pos]))
                    else:
                        raise ValueError("Unrecognized specification concerning lexical outcomes. "
                                         "Please, choose either 'tokens' or 'lemmas'.")

                # append the phonetic representation of the current learning event to the list of phonetic
                # representations for the whole corpus, and the lexical meanings of the current learning event to
                # the list of lexical meanings for the whole corpus
                encoded_corpus[0].append(n_phones)
                encoded_corpus[1].append(list(outcomes_set))

            # if the phonological representation of a word from the utterance could not be retrieved from
            # CELEX, count the utterance as missed
            else:
                missed += 1

        if i in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": %d%% of the input corpus has been processed and encoded in the desired way." % check_points[i])

    perc_missed = missed / float(total) * 100

    return encoded_corpus, perc_missed
