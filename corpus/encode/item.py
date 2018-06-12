__author__ = 'GCassani'

"""Functions to encode utterances in phonological cues"""

from corpus.encode.nphones import get_nphones
from corpus.encode.syllables import get_syllables
from corpus.encode.stress import recode_stress
from celex.utilities.helpers import vowels


def encode_item(item, uniphones=True, diphones=False, triphones=False, syllables=False,
                stress_marker=False, boundaries=True):

    """
    :param item:            a string
    :param uniphones:       a boolean indicating whether single phonemes are to be considered while encoding input
                            utterances
    :param diphones:        a boolean indicating whether sequences of two phonemes are to be considered while
                            encoding input utterances
    :param triphones:       a boolean indicating whether sequences of three phonemes are to be considered while
                            encoding input utterances
    :param syllables:       a boolean indicating whether syllables are to be considered while encoding input
                            utterances
    :param stress_marker:   a boolean indicating whether stress markers from the input phonological representation need
                            to be preserved or can be discarded
    :param boundaries:      a boolean indicating whether word boundaries should be considered or not in the output
    :return nphones:        an iterable containing the desired phonetic cues from the input word
        """

    celex_vowels = vowels()

    translation_table = dict.fromkeys(map(ord, "'"), None)
    if stress_marker:
        item = recode_stress(item, celex_vowels)
    else:
        item = item.translate(translation_table)

    uni, di, tri, syl = [[], [], [], []]

    if syllables:
        syl = get_syllables(item)

    # check that syllables only contain one vowel, print otherwise to evaluate what to do:
    # loop through every character in the syllable, check if the character is a vowel and increment the
    # vowel count if it is; if the vowel count reaches 2, print the utterance and the syllable, and get
    # out of the loop
    # CAVEAT: this should not print anything, it's just a sanity check
    for s in syl:
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
    if not boundaries:
        table = str.maketrans(dict.fromkeys("+"))
        item = item.translate(table)
        item = '+' + item + '+'

    if uniphones:
        uni = get_nphones(item, n=1)
    if diphones:
        di = get_nphones(item, n=2)
    if triphones:
        tri = get_nphones(item, n=3)
    nphones = uni + di + tri + syl

    return nphones
