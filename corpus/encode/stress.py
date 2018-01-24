__author__ = 'GCassani'

"""Function to recode the stress markers from Celex phonological representations"""


def recode_stress(phon_repr, vowels):

    """
    :param phon_repr:           a string containing phonological representations extracted from Celex. Each word is
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

    while i < len(phon_repr):
        # if the symbol being considered is not a stress marker, append it to the recoded utterance and move forward
        if phon_repr[i] != "'":
            recoded_utterance += phon_repr[i]
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
                    if phon_repr[i + j] in vowels:
                        recoded_utterance += bag_of_phonemes + "'" + phon_repr[i + j]
                        is_vowel = True
                        i += len(bag_of_phonemes)+2
                    # if it is not, add the next phoneme (a consonant) to the bag of phonemes and increment the index
                    # that allows to advance in the utterance, until a vowel is found
                    else:
                        bag_of_phonemes += phon_repr[i + j]
                        j += 1

                # in case there is a stress symbol but no vowel, return the original utterance stripped of the stress
                # symbol, since in this encoding only vowels can bear stress
                except IndexError:
                    if not phon_repr.startswith('+'):
                        phon_repr = '+' + phon_repr
                    if not phon_repr.endswith('+'):
                        phon_repr += '+'
                    table = str.maketrans(dict.fromkeys("'"))
                    return phon_repr.translate(table)

    # make sure that the utterance begins and ends with the word boundary symbol
    if not recoded_utterance.endswith('+'):
        recoded_utterance += '+'
    if not recoded_utterance.startswith('+'):
        recoded_utterance = '+' + recoded_utterance

    return recoded_utterance
