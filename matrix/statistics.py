__author__ = 'GCassani'

"""Functions to process activation matrices and compute several statistics"""

import numpy as np


def norm(weight_matrix, indices, axis=0, p=1):

    """
    :param weight_matrix:   a NumPy matrix
    :param indices:         a vector of numerical indices indicating which rows or columns to consider
    :param axis:            0 for column-wise vector norms, 1 for row-wise vector norms
    :param p:               1 to get the absolute length of the vector; 2 to get its Euclidean length
    :return vector_norms:   the p-norm of the vectors, computed according to the specification of p

    The function computes the vector norms from the input matrix, according to the order specified by the parameter p,
    along the dimension specified by the parameter axis. Indices is an array-like structure that can operate on rows or
    columns, depending on the value passed to the argument axis:
    - if axis=0, i.e. norms are computed for column vectors, indices is interpreted as indicating the rows to be
        considered in the computation of the column vector norms. Vector norms are computed for all the columns, but
        only considering the rows whose indices are specified in the input vector.
    - if axis=1, i.e. norms are computed for row vectors, indices is interpreted as indicating the columns to be
        considered in the computation of the row vector norms. Vector norms are computed for all the rows, but only
        considering the columns whose indices are specified in the input vector.

    """

    if axis:
        vector_norms = np.linalg.norm(weight_matrix[:, indices], ord=p, axis=axis)
    else:
        vector_norms = np.linalg.norm(weight_matrix[indices, :], ord=p, axis=axis)

    return vector_norms


########################################################################################################################


def median_absolute_deviation(weight_matrix, indices, axis=0):

    """
    :param weight_matrix:   a NumPy matrix
    :param indices:         a vector of numerical indices indicating which rows or columns to consider
    :param axis:            0 for column-wise MADs, 1 for row-wise MADs
    :return med_abs_dev:    an array of MAD values computed over the specified dimension for the specified subset of
                            rows/columns

    The function computes the Median Absolute Deviations (MADs) from the input matrix, along the dimension specified by
    the parameter axis. Indices is an array-like structure that can operate on rows or columns, depending on the value
    passed to the argument axis:
    - if axis=0, i.e. MADs are computed for column vectors, indices is interpreted as indicating the rows to be
        considered in the computation of the column vector MADs. MADs are computed for all column vectors, but only
        considering the rows whose indices are specified in the input vector.
    - if axis=1, i.e. MADs are computed for row vectors, indices is interpreted as indicating the columns to be
        considered in the computation of the row vector MADs. MADs are computed for all the rows, but only considering
        the columns whose indices are specified in the input vector.
    """

    if axis == 0:
        median = np.median(weight_matrix[indices, :], axis=axis)
        med_abs_dev = np.median(np.absolute(weight_matrix[indices, :] - median), axis=axis)
    else:
        median = np.median(weight_matrix[:, indices], axis=axis)
        med_abs_dev = np.median(np.absolute(weight_matrix[:, indices] - np.vstack(median)), axis=axis)

    return med_abs_dev


########################################################################################################################


def activations(weight_matrix, indices, axis=0):

    """
    :param weight_matrix:   a NumPy matrix
    :param indices:         a vector of numerical indices indicating which rows or columns to consider
    :param axis:            0 for column-wise summed activations, 1 for row-wise summed activations
    :return alphas:         a vector of activation values computed over the specified dimension for the specified subset
                            of rows/columns

    The function computes the summed activation values from the input matrix, along the dimension specified by
    the parameter axis. Indices is an array-like structure that can operate on rows or columns, depending on the value
    passed to the argument axis:
    - if axis=0, i.e. activation values are computed over column vectors, indices is interpreted as indicating the rows
        to be considered in the computation of the column activation values. Activation values are computed for all
        column vectors, but only considering the rows whose indices are specified in the input vector.
    - if axis=1, i.e. activation values are computed over row vectors, indices is interpreted as indicating the columns
        to be considered in the computation of the row activation values. Activation values are computed for all
        row vectors, but only considering the columns whose indices are specified in the input vector.
    """
    if axis == 0:
        alphas = np.sum(weight_matrix[indices, :], axis=axis)
    else:
        alphas = np.sum(weight_matrix[:, indices], axis=axis)

    return alphas
