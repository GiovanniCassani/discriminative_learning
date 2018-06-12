__author__ = 'GCassani'

"""Functions to visualize activation matrices and rank plots"""

import matplotlib.pyplot as plt


def plot_matrix(weight_matrix, figname='Figure title', output_path=''):

    """
    :param weight_matrix:   a NumPy array
    :param figname:         a string indicating the plot title - default is 'Figure title'
    :param output_path:     a string indicating where to save the plot. If no path is provided (default), the plot is
                            shown in the current window
    """

    fig = plt.figure()
    im = plt.imshow(weight_matrix, aspect='auto', interpolation='nearest')
    fig.colorbar(im)
    plt.xlabel('Outcomes')
    plt.ylabel('Cues')
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')

    plt.title(figname)

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
