__author__ = 'GCassani'

"""Function to perform a single run of the clustering simulation using a kNN approach"""

import os
import json
from time import strftime
from matrix.matrix import load
from rescorla_wagner.ndl import ndl
from corpus.encoder import corpus_encoder
from celex.get import get_celex_dictionary
from analysis.discrimination import find_discriminated
import phonological_bootstrapping.clustering.similarities as sim
from phonological_bootstrapping.clustering.log import make_log_file


def cluster_words(corpus, output_folder, celex_folder, pos_mapping, distance='cosine', reduced=False, outcomes='tokens',
                  uniphones=False, diphones=False, triphones=True, syllables=False, stress_marker=True,
                  boundaries=True,  at=5, nn=25, a=0.01, b=0.01, longitudinal=False):

    """
    :param corpus:          the corpus to be used for training the model
    :param output_folder:   the folder where the logfile of the clustering experiment will be saved
    :param celex_folder:    the folder containing the data from the Celex database
    :param pos_mapping:     the path to the file mapping CHILDES pos tags to Celex tags
    :param distance:        a string (either 'correlation' or 'cosine' indicating which distance metric to use
    :param reduced:         a boolean indicating whether to use reduced or full phonetic transcriptions from Celex
    :param outcomes:        a string (either 'tokens' or 'lemmas') indicating which outcomes to consider for learning
    :param uniphones:       a boolean indicating whether to consider uniphones as cues
    :param diphones:        a boolean indicating whether to consider diphones as cues
    :param triphones:       a boolean indicating whether to consider triphones as cues
    :param syllables:       a boolean indicating whether to consider syllables as cues
    :param stress_marker:   a boolean indicating whether to consider or discard stress information
    :param boundaries:      a boolean indicating whether to consider or discard word boundaries
    :param at:              an integer indicating how many outcomes to compute to compute discrimination's precision
    :param nn:              an integer indicating how many nearest neighbors to consider when evaluating clustering
    :param a:               the alpha parameter from the Rescorla Wagner model
    :param b:               the beta parameter from the Rescorla Wagner model
    :param longitudinal:    a boolean indicating whether to adopt a longitudinal design or not
    :return accuracies:     a dictionary mapping time indices to the clustering accuracy obtained at that time point
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_folder = os.path.join(output_folder, 'plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    encoded_corpus = corpus_encoder(corpus, celex_folder, pos_mapping, separator='~', stress_marker=stress_marker,
                                    reduced=reduced, uniphones=uniphones, diphones=diphones, triphones=triphones,
                                    syllables=syllables, outcomes=outcomes, boundaries=boundaries)

    corpus_dir = os.path.dirname(encoded_corpus)

    file_paths = ndl(encoded_corpus, alpha=a, beta=b, lam=1, longitudinal=longitudinal)

    celex_dict = get_celex_dictionary(celex_folder, reduced=reduced)

    accuracies = {}
    for idx, file_path in file_paths.items():

        logfile = make_log_file(corpus, output_folder, 'json', dist=distance, nn=nn, at=at, time=idx,
                                outcomes=outcomes, reduced=reduced, stress_marker=stress_marker, boundaries=boundaries,
                                syllables=syllables, uniphones=uniphones, diphones=diphones, triphones=triphones)
        plotfile = make_log_file(corpus, plot_folder, 'pdf', dist=distance, nn=nn, at=at, time=idx, outcomes=outcomes,
                                reduced=reduced, stress_marker=stress_marker, boundaries=boundaries,
                                syllables=syllables, uniphones=uniphones, diphones=diphones, triphones=triphones)

        if os.path.exists(logfile):
            print()
            print("The file %s already exists, statistics for the corresponding "
                  "parametrization are loaded from it" % logfile)
            clusters = json.load(open(logfile, "r"))

        else:
            print()
            matrix, cues2ids, outcomes2ids = load(file_path)

            # get the column ids of all perfectly discriminated outcomes at the current time point
            # perfectly discriminated outcomes are considered to be those whose jaccard coefficient
            # between true phonetic cues and most active phonetic cues for the outcome is 1
            discriminated_file = os.path.join(corpus_dir, '.'.join(['discriminatedOutcomes', str(int(idx)), 'json']))
            if not os.path.exists(discriminated_file):
                discriminated = find_discriminated(matrix, cues2ids, outcomes2ids, celex_dict,
                                                   stress_marker=stress_marker, uniphones=uniphones,
                                                   diphones=diphones, triphones=triphones,
                                                   syllables=syllables, boundaries=boundaries, at=at)
                json.dump(discriminated, open(discriminated_file, 'w'))
            else:
                discriminated = json.load(open(discriminated_file, 'r'))

            print()
            print(strftime("%Y-%m-%d %H:%M:%S") + ": Start clustering, using %s as weight matrix..."
                  % (os.path.basename(file_path)))
            if distance == 'cosine':
                similarities, discriminated = sim.pairwise_cos(matrix, discriminated, plot_path=plotfile)
            else:
                similarities, discriminated = sim.pairwise_corr(matrix, discriminated, plot_path=plotfile)

            df = sim.sim2df(similarities, discriminated)
            similarities_file = os.path.join(corpus_dir, '.'.join(['similarities', distance, str(int(idx)), 'csv']))
            df.to_csv(similarities_file, sep='\t')

            clusters = sim.neighborhood(discriminated, similarities, nn=nn)
            json.dump(clusters, open(logfile, 'w'))
            print(strftime("%Y-%m-%d %H:%M:%S") + ": ...completed test phase.")

        accuracy, baseline_acc, h, baseline_h = sim.clustering_precision(clusters)
        accuracies[idx] = {'accuracy': accuracy,
                           'baseline_acc': baseline_acc,
                           'entropy': h,
                           'baseline_entr': baseline_h}

    return accuracies
