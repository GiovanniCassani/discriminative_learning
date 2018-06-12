__author__ = 'GCassani'

"""Implement kNN algorithm for PoS tagging"""

import numpy as np
from collections import Counter


def get_nearest_indices(cosine_similarities, nn=1):

    """
    :param cosine_similarities: a NumPy 2-dimensional array corresponding to the column of interes from the matrix of
                                pairwise similarities
    :param nn:                  an integer indicating the number of nearest neighbours to consider (the function uses
                                nearest distances rather than neighbours: if two or more words are at the same closest
                                distance they're all consider - when nn=1, as in the default)
    :return nearest_indices:    a tuple whose first element contains the row indices from the input NumPy array
                                indicating the cells with the highest values in the column indicated by the input
                                parameter idx. The second element of the tuple is empty
    """

    # sort all the columns in the NumPy array independently and in descending order
    cosine_similarities_sorted = np.sort(cosine_similarities, axis=0)[::-1]

    # get the value corresponding to the closest similarity (if nn=1) or second closest (if nn=2), and so on;
    # if the vector is shorter then the chosen value for nn, the function simply takes the smallest value in the column,
    # which is the last one since the column is sorted in descending order
    try:
        t = sorted(set(cosine_similarities_sorted), reverse=True)[nn-1]
    except IndexError:
        t = sorted(set(cosine_similarities_sorted), reverse=True)[-1]

    # make sure that the threshold is positive, otherwise use 0
    t = t if t > 0 else 0

    # get the vector of row indices from the original, unsorted NumPy array that have a distance equal or higher than
    # the value of the desired number of neighbours (distances) set by nn
    nearest_indices = np.where(cosine_similarities >= t)

    return nearest_indices


########################################################################################################################


def get_nearest_neighbors(nearest_indices, words):

    """
    :param nearest_indices: an iterable containing the row indices from the input NumPy array indicating
                            the cells with the highest values in the column indicated by the input parameter idx.
    :param words:           a dictionary mapping numerical indices to word strings
    :return neighbors:      a set of strings containing those strings that match the indices in the input tuple
    """

    neighbors = set()
    for i in nearest_indices:
        neighbors.add(words[i])

    return list(neighbors)


########################################################################################################################


def tally_tags(l):

    """
    :param l:               an iterable of strings, consisting of a word form and a PoS tag separated by a
                            pipe symbol ("|")
    :return tallied_tags:   a sorted list of tuples, each containing a string as first element (a PoS tag) and a
                            frequency count as second element, indicating the frequency count of the PoS tag among the
                            nearest neighbors provided in the input iterable
    """

    pos_tags = list()
    for i in l:
        # isolate the PoS tag
        tag = i.split("|")[1]
        pos_tags.append(tag)

    # count frequencies of occurrence for each tag in the list of neighbours and return the resulting list of tuples
    tallied_tags = Counter(pos_tags).most_common()
    return tallied_tags


########################################################################################################################


def categorize(tags_distribution):

    """
    :param tags_distribution:   a dictionary mapping a pos tag to its frequency among the nearest neighbors of the test
                                item
    :return predicted:          a string indicating the predicted PoS tag given the tallied tags and the nearest
                                neighbours together with the frequency information contained in the training matrix
    """

    # Resolve ties by picking the PoS tag of the nearest neighbour that occurred more frequently in the corpus
    # if frequency is enough to break the tie, pick randomly a tag from the most frequent words
    if len(tags_distribution) == 1:
        predicted = list(tags_distribution)[0][0]
    else:
        max_freq = 0
        most_frequent = []
        for tag, freq in tags_distribution:
            if freq > max_freq:
                max_freq = freq
                most_frequent = [tag]
            elif freq == max_freq:
                most_frequent.append(tag)

        if len(most_frequent) > 1:
            i = int(np.random.randint(0, high=len(most_frequent), size=1))
            predicted = most_frequent[i]
        else:
            predicted = most_frequent[0]

    return predicted
