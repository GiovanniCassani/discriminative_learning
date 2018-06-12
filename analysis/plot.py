__author__ = 'GCassani'

"""Function to scatter plot data points and get a visual sense of correlation across pairs of measures derived
   for cues and outcomes from cue-outcome association matrices"""

import os
import operator
import numpy as np
import matplotlib.pyplot as plt


def scatter(measures, summary, plot_path, name):

    """
    :param measures:
    :param summary:
    :param plot_path:
    :param name:
    :return:
    """

    sorted_measures = sorted(measures.items(), key=operator.itemgetter(1))
    relevant_subplots = []
    all_subplots = []
    for measure1, id1 in sorted_measures:
        for measure2, id2 in sorted_measures:
            subplot_id = (id2, id1, measure2, measure1)
            all_subplots.append(subplot_id)
            if id1 != id2 and (id1, id2, measure1, measure2) not in relevant_subplots:
                relevant_subplots.append(subplot_id)

    f_outcome_corr, axarr = plt.subplots(len(measures), len(measures))
    f_outcome_corr.suptitle('Outcomes: correlations')

    for subplot in all_subplots:
        r, c, y_name, x_name = subplot
        if subplot in relevant_subplots:
            x = summary[measures[x_name]]
            y = summary[measures[y_name]]
            axarr[r, c].scatter(x, y)
            xlow = x.min() - x.min() / float(10)
            xhigh = x.max() + x.max() / float(10)
            axarr[r, c].set_xlim([xlow, xhigh])
            ylow = y.min() - y.min() / float(10)
            yhigh = y.max() + y.max() / float(10)
            axarr[r, c].set_ylim([ylow, yhigh])
            if r == len(measures) - 1:
                axarr[r, c].set_xlabel(x_name)
            else:
                axarr[r, c].set_xlabel('')
            if c == 0:
                axarr[r, c].set_ylabel(y_name)
            else:
                axarr[r, c].set_ylabel('')
            axarr[r, c].set_xticklabels([])
            axarr[r, c].set_yticklabels([])
        else:
            axarr[r, c].axis('off')

    f_outcome_corr.savefig(os.path.join(plot_path, name))
    plt.close(f_outcome_corr)


########################################################################################################################


def plot_ranks(l, xname='IV', yname='DV', figname='Figure title', output_path=''):

    """
    :param l:               a list of tuples, where the second element of each tuple is numerical
    :param xname:           a string indicating the label of the x axis - default is IV (Independent Variable)
    :param yname:           a string indicating the label of the y axis - default is DV (Dependent Variable)
    :param figname:         a string indicating the plot title - default is 'Figure title'
    :param output_path:     a string indicating where to save the plot. If no path is provided (default), the plot is
                            shown in the current window
    """

    x_vec = np.linspace(1, len(l), len(l))
    y_vec = np.zeros((len(l)))
    for i in range(len(l)):
        y_vec[i] = l[i][1]

    fig = plt.figure()
    plt.scatter(x_vec, y_vec, alpha=0.75)
    axes = plt.gca()
    axes.set_xlim([0, len(l) + 1])
    ymin = y_vec.min() - y_vec.min() / float(10)
    ymax = y_vec.max() + y_vec.max() / float(10)
    axes.set_ylim([ymin, ymax])
    plt.title(figname)
    plt.xlabel(xname)
    plt.ylabel(yname)

    plt.tick_params(
        axis='both',
        which='minor',
        bottom='off',
        top='off',
        left='off',
        right='off')

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
