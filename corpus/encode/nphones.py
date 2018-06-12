__author__ = 'GCassani'

"""Function to encode a Celex phonological representations into its constituents n-phones"""


def get_nphones(phon_repr, n):

    """
    :param phon_repr:   a string, containing phonological representations extracted from the CELEX database
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
    while i < len(phon_repr):
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
                nphone += phon_repr[i + j]
                # if the n-phone contains the stress marker, allow it to include a further character
                if phon_repr[i + j] == "'":
                    flex_n += 1
                j += 1
            except IndexError:
                i = len(phon_repr) + 1
                j = flex_n + 2

        if len(nphone) >= n:
            nphones.append(nphone)

        if i < len(phon_repr):
            # if the current symbol being processed is the stress marker, jump ahead of two symbols: this prevents the
            # function from storing two n-phones that only differ in the presence of the stress marker at the beginning,
            # e.g. 'US and US. US is the unstressed version of 'US and it is wrong to store them both, since only one
            # occurs
            if phon_repr[i] == "'":
                i += 2
            else:
                i += 1

    # make sure that no nphone of the wrong granularity was created by getting the stress marker wrong
    for nphone in nphones:
        table = str.maketrans(dict.fromkeys("'"))
        if len(nphone.translate(table)) < n:
            nphones.remove(nphone)
    return nphones
