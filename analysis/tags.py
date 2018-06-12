__author__ = 'GCassani'

"""Function to find which PoS categories are most represented across the top active lexical nodes given a test item"""

import random
import operator
from collections import Counter, defaultdict


def get_pos_tag(word):

    """
    :param word:        a string, with a vertical bar ('|') dividing the word-form from the PoS tag, i.e. a single
                        capital letter that marks the lexical category to which the word belong.
    :return pos:        a string indicating the PoS tag extracted from the input word. If no PoS tag can be retrieved
                        from the word, the function returns None
    """

    if not isinstance(word, str):
        word = word.decode('UTF-8')

    try:
        pos = word.split('|')[1]
        return pos
    # if the word being considered doesn't have a PoS tag, append None to the list of PoS tags
    except IndexError:
        return None


########################################################################################################################


def get_pos_activation(input_list, target_category):

    """
    :param input_list:      an iterable containing tuples. Each tuple consists of a string and a floating; the string,
                            in turn, consists of two parts, a word form and a PoS tag, joined by a vertical bar ('|')
    :param target_category: a string indicating a PoS tag
    :return category_alpha: a number indicating the total activation that the target category received from all the
                            elements in the input iterable. First, the PoS tag of each tuple is matched against the
                            target one: if they're the same, the second element of the tuple, the floating, is summed to
                            the category_alpha
    """

    category_alpha = 0

    for item in input_list:
        word = item[0].decode('UTF-8') if not isinstance(item[0], str) else item[0]
        category = word.split('|')[1]
        if category == target_category:
            category_alpha += item[1]

    return category_alpha


########################################################################################################################


def get_frequency_and_activation_for_each_pos(l):

    """
    :param l:       an iterable containing tuples. Each tuple consists of a string and a floating; the string, in
                    turn, consists of two parts, separated by a vertical bar ('|')
    :return freq:   a dictionary mapping all the PoS tags to their frequency in the input iterable
    :return act:    a dictionary mapping all the PoS tags to the sum of the activation values for words matching the
                    PoS tag across the words in the input iterable

    This function is used here to compute the frequency distribution of PoS tags in the k top active outcomes, provided
    in the input argument l, and the PoS activation across the same outcomes. The first dictionary maps PoS tags to
    their frequency counts, the second maps PoS tags to their total activation, i.e. the sum of activation values from
    all the words tagged with a same PoS.
    """

    freq = Counter([get_pos_tag(o[0]) for o in l])
    act = defaultdict(float)
    for pos in freq:
        act[pos] = get_pos_activation(l, pos)

    return freq, act


########################################################################################################################


def get_top_pos_from_dict(dict1, dict2):

    """
    :param dict1:       a dictionary mapping PoS tags to their frequency distribution across the k top active outcomes,
                        with k specified by the user. This dictionary is the first output of the function
                        pos_frequency_and_activation()
    :param dict2:       a dictionary mapping PoS tags to their total activation across the k top active outcomes, with
                        k specified by the user. The activation of each outcome tagged with a same PoS is summed to
                        obtain a total for each PoS. This dictionary is the second output of the function
                        pos_frequency_and_activation()
    :return top_key1:   the single key from dict1 with highest value. In case of ties, meaning that two or more keys
                        from dict1 have the same value, the values for the same keys from dict2 are used to resolve the
                        tie. The two dictionaries must have the same keys
    """

    # initialize an empty string variable to store the key with highest value in dict1 and a numeric variable to
    # keep track of the highest value so far in dict1
    top_key1 = ''
    highest_v = 0

    # loop through dict1 descending order
    for k, v in sorted(dict1.items(), key=operator.itemgetter(1), reverse=True):

        # if the value of the current key is higher than the value being stored, use the current value
        # instead and update value of the variable for the most common key1
        if v > highest_v:
            highest_v = v
            top_key1 = k

        # if there is a tie, meaning that the following key has the same valuet of the most frequent one being
        # currently stored, the corresponding values from dict2 are retrieved
        elif v == highest_v:
            most_common_pos_activation = dict2[top_key1]
            curr_pos_activation = dict2[k]

            # if the value from dict2 for the key1 being considered is higher than the value from dict2 for the key1
            # being stored as the most common, than update the information storing the current key1 as most
            # common. The value1 is the same, so no need to update it. If, on the contrary, the value from dict2 of the
            # key1 being considered is lower then the value from dict2 of the key1 being stored as the most common,
            # nothing needs to be updated
            if curr_pos_activation > most_common_pos_activation:
                top_key1 = k
            # if, however, the two key1 also tie in their value2, then pick one at random. Again, value1 is the same,
            # so no need to update it
            elif curr_pos_activation == most_common_pos_activation:
                top_key1 = random.choice([k, top_key1])

        # as soon as the value1 of a key1 is lower than the current value1 being stored, return the key1 being stored as
        # the most common
        else:
            return top_key1, highest_v

    # in the unlikely scenario in which all key1 have the same value1 and, when the function runs out of key1 to
    # evaluate, return the key1 that either has the highest value2 or the one that got selected at random if all PoS
    # tags also tied when value2 are considered
    return top_key1, highest_v
