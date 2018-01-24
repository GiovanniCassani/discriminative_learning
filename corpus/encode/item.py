__author__ = 'GCassani'

"""Functions to encode utterances in phonological cues"""

from corpus.encode.nphones import get_nphones
from corpus.encode.syllables import get_syllables
from corpus.encode.stress import recode_stress
from celex.utilities.helpers import vowels


def encode_item(item, uni_phones=True, di_phones=False, tri_phones=False,
                syllable=False, stress_marker=False):

    """
    :param item:            a string
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

    celex_vowels = vowels()

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
