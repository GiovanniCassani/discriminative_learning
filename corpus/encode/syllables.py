__author__ = 'GCassani'

"""Function to encode a phonological representation from Celex into the constituent syllables"""


def get_syllables(phon_repr):

    """
    :param phon_repr:           a string containing all the words from the utterance, with words separated by word
                                boundary markers ('+') and syllables being separated by syllable markers ('-')
    :return syllables:          a list of all the syllables that are present in the input words, preserving their order
    """

    utterance_syllables = []

    words = phon_repr.strip("+").split("+")
    for word in words:
        word = "+" + word + "+"
        utterance_syllables += word.split('-')

    return utterance_syllables
