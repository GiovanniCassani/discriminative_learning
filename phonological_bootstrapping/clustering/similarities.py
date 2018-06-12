__author__ = 'GCassani'

"""Functions to compute pairwise similarities between column or row vectors in a cue-outcome matrix"""

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import phonological_bootstrapping.clustering.kNN as knn


def pairwise_corr(matrix, target='columns', plot_path=''):

    """
    :param matrix:          a NumPy 2d array
    :param target:          a string indicating whether to compute pairwise correlations for rows or for columns
                            (default to rows)
    :param plot_path:       the path to the file where the plot will be saved (default to the empty string, meaning
                            that the plot is shown)
    :return correlations:   a NumPy array containing pairwise correlations between vectors along the specified
                            dimensions
    """

    rowvar = False if target=='columns' else True
    correlations = np.corrcoef(matrix, rowvar=rowvar)
    plt.matshow(correlations)

    if plot_path:
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()
        plt.close()

    return correlations


########################################################################################################################


def pairwise_cos(matrix, target='columns', plot_path=''):

    """
    :param matrix:              a NumPy 2d array

    :param target:              a string indicating whether to compute pairwise correlations for rows or for columns
                                (default to rows)
    :param plot_path:           the path to the file where the plot will be saved (default to the empty string, meaning
                                that the plot is shown)
    :return cos_similarities:   a NumPy 2d array with pairwise cosine similarities along the desired dimension
    :return outcomes2ids:       a dictionary mapping outcomes from the input dictionary to the corresponding columns of
                                the output matrix
    """

    matrix = np.rot90(matrix) if target=='columns' else matrix
    cos_similarities = cosine_similarity(matrix)
    cos_similarities[np.diag_indices_from(cos_similarities)] = 0
    plt.matshow(cos_similarities)

    if plot_path:
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()
        plt.close()

    return cos_similarities


########################################################################################################################


def neighborhood(ids, similarity_matrix, nn=10):

    """
    :param ids:                 a dictionary mapping strings to indices, either row or columns
    :param similarity_matrix:   a square matrix where each cell indicates the similarity score between two items
                                (indices from this matrix map to indices in ids), with the main diagonal set to 0
    :return categorization:     a dictionary mapping each string in ids to its column index in the similarity_matrix,
                                to the category most represented in the nearest neighbors, and to the frequency
                                distribution of such categories

    For each word in ids, the set of nearest neighbors in the similarity matrix is retrieved, making sure that
    neighbors are only considered if they also are in ids: this is achieved by setting all rows for indices not in ids
    to 0.
    """

    # map each numerical index to the corresponding string
    inverted_ids = {v:k for k, v in ids.items()}

    categorization = defaultdict(dict)

    # retrieve nearest neighbors only for words in the input dictionary
    for el in ids:
        ii = ids[el]
        correct = el.split('|')[1]
        nearest_indices = knn.get_nearest_indices(similarity_matrix[:, ii], nn=nn)
        nearest_neighbors = knn.get_nearest_neighbors(nearest_indices[0], inverted_ids)
        tags_distribution = knn.tally_tags(nearest_neighbors)
        predicted_pos = knn.categorize(tags_distribution)
        categorization[el] = {'id': ii,
                              'correct': correct,
                              'predicted':predicted_pos,
                              'tags': tags_distribution}

    return categorization


########################################################################################################################


def clustering_precision(categorization):

    """
    :param categorization:  the output of the function neighborhood(), i.e. a dictionary mapping a word to its column id
                            in the matrix used to estimate clustering, the correct pos tag, the predicted pos tag given
                            its nearest neighbors, and the distribution of tags over the neighbours
    :return accuracy:       the proportion of words which were categorized correctly
    :return baseline_acc:   the accuracy that would be achieved by simply choosing the most frequent PoS tag across the
                            target words
    :return h:              the entropy of the distribution of predicted tags
    :return baseline_h:     the entropy of the distribution of correct tags
    """

    total = len(categorization)
    correct = 0

    baseline_tags = []
    predicted_tags = []

    for word in categorization:
        baseline_tags.append(categorization[word]['correct'])
        predicted_tags.append(categorization[word]['predicted'])
        if categorization[word]['correct'] == categorization[word]['predicted']:
            correct += 1

    accuracy = correct / total
    baseline_distr = Counter(baseline_tags)
    predicted_distr = Counter(predicted_tags)
    baseline_acc = baseline_distr.most_common()[0][1] / sum(baseline_distr.values())
    h = entropy(list(predicted_distr.values()), base=len(baseline_tags))
    baseline_h = entropy(list(baseline_distr.values()), base=len(baseline_tags))

    return accuracy, baseline_acc, h, baseline_h


########################################################################################################################


def mat2df(associations, cols2ids, rows2ids):

    """
    :param associations:    a 2d NumPy array
    :param cols2ids:        a dictionary mapping strings to column indices
    :param rows2ids:        a dictionary mapping strings to row indices
    :return df:             a Pandas dataframe containing the input array with rows named using the strings in rows2ids
                            and columns named using the strings in cols2ids
    """

    col_indices = [k for k, v in sorted(cols2ids.items(), key=operator.itemgetter(1))]
    row_indices = [k for k, v in sorted(rows2ids.items(), key=operator.itemgetter(1))]
    pos_tags = [ii.split('|')[1] for ii in row_indices]
    df = pd.DataFrame(associations, index=row_indices, columns=col_indices)
    df['Target'] = pd.Series(pos_tags, index=df.index)

    return df
