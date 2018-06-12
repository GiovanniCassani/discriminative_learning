__author__ = 'GCassani'

"""Function to encode a phonological representation from Celex into the constituent syllables"""


def get_syllables(phon_repr, boundaries=True):

    """
    :param phon_repr:           a string containing where syllables are separated by hyphens ('-')
    :param boundaries:          a boolean indicating whether word boundaries are present or not
    :return syllables:          a list of all the syllables that are present in the input words, preserving their order
    """

    syllables = []

    words = phon_repr.strip("+").split("+")
    for word in words:
        word = "+" + word + "+" if boundaries else word
        syllables += word.split('-')

    return syllables
