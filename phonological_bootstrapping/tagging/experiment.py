__author__ = 'GCassani'

"""Function to perform the phonological bootstrapping experiment from scratch, given an input corpus and
   the specifications about how to process the corpus, about which cues and outcomes to use, how to build the test set, 
   how to perform categorization, and finally the parameters for the ndl algorithm"""

import os
import json
from time import strftime
from matrix.matrix import load
from rescorla_wagner.ndl import ndl
from corpus.encoder import corpus_encoder
from celex.get import get_celex_dictionary
from analysis.discrimination import find_discriminated
from phonological_bootstrapping.tagging.log import make_log_file
from phonological_bootstrapping.tagging.categorize import categorize
from phonological_bootstrapping.helpers import compute_summary_statistics


def tag_words(input_file, test_set, celex_dir, pos_mapping, output_folder,
              method='freq', evaluation='count', k=50, flush=0, threshold=0,
              separator='~', reduced=False, outcomes='tokens', boundaries=True,
              uniphones=True, diphones=False, triphones=False, syllable=False, stress_marker=False,
              alpha=0.01, beta=0.01, lam=1.0, longitudinal=False, at=5):
    """
    :param input_file:          a .json file containing transcripts of child-caregiver interactions extracted from the
                                CHILDES database. The json file consists of two lists of lists, of the same length,
                                both contain utterances but encoded differently. The first encodes each utterance as a
                                list of tokens; the second encodes each utterance as a list of lemmas and
                                Part-of-Speech tags, joined by a vertical bar ('|')
    :param test_set:            a dictionary mapping the file name to:
                                - 'test_set['filename']: the basename of the file
                                - 'test_set['items']: the set of phonological forms to be categorized, complete of the
                                   target PoS tag (phonological form and PoS tag are separated by a vertical bar ('|')
    :param celex_dir:           a string specifying the path to the Celex directory
    :param pos_mapping:         a .txt file mapping CHILDES PoS tags to CELEX tags
    :param output_folder:       the path to the folder where the logfiles will be saved
    :param method:              a string indicating the way in which the function looks at top active outcomes; two
                                options are available:
                                - 'freq' makes the function compute the distribution of PoS tags over the k top active
                                    nodes (see the explanation of the parameter k) and rank PoS tags according to their
                                    frequency among the top active cues
                                - 'sum' makes the function compute the sum of activation from all outcomes belonging to
                                    a given PoS tag within the k top active outcomes given the input cues, and rank PoS
                                     tags according to their total activation among the top active cues
    :param evaluation:          a string indicating how to compare baseline activations to item-triggered ones; two
                                options are available:
                                - 'count', simply tag the test item with the PoS tag that either was more frequent or
                                    had highest summed activation within the top active outcomes; frequency or
                                    activation are returned and can be correlated to reaction times
                                - 'distr', compare the frequency counts or summed activations generated by a specific
                                    test item to the frequency counts or summed activations at baseline and tag the
                                    test item with the PoS tag receiving highest support by the change in the
                                    distribution of frequencies or summed activations (a statistic is returned,
                                    Chi-squared for frequency distributions and t-test for summed activations, whose
                                    value can be correlated to reaction times)
    :param k:                   an integer specifying how many elements to consider from the baseline activations and
                                the activations triggered by a specific test item. By default, the top 50 outcomes are
                                considered, and compared according to the chosen combination of method and eval
    :param flush:               specify whether (and how many) top active outcome at baseline to flush away from
                                subsequent computations. It may be the case that whatever the input cues, the same high
                                frequency outcomes come out as being the most active. It may then make sense to not
                                consider them when evaluating the distribution of lexical categories over the most
                                active outcomes given an input item
    :param threshold:           the minimum activation of an outcome to be considered in the list of top activated
                                neighbors, default is 0 and shouldn't be lowered, but can be increased.
    :param separator:           the character that separates the word baseform from its PoS tag in the input corpus
    :param reduced:             a boolean specifying whether reduced phonological forms should be extracted from Celex
                                whenever possible (if set to True) or if standard phonological forms should be
                                preserved (if False)
    :param outcomes:            a string indicating which outcomes to use, whether 'tokens' (default) or 'lemmas'
    :param boundaries:          a boolean specifying whether word boundaries are to be considered when training on full
                                utterances
    :param uniphones:          a boolean indicating whether single phonemes are to be considered while encoding input
                                utterances
    :param diphones:           a boolean indicating whether sequences of two phonemes are to be considered while
                                encoding input utterances
    :param triphones:          a boolean indicating whether sequences of three phonemes are to be considered while
                                encoding input utterances
    :param syllable:            a boolean indicating whether syllables are to be considered while encoding input
                                utterances
    :param stress_marker:       a boolean indicating whether stress markers from the phonological representations of
                                Celex need to be preserved or can be discarded
    :param alpha:               a number indicating cue salience. For simplicity, we assume that every cue has the same
                                salience, so changing the value of this parameter does not affect the relative strength
                                of of cue-outcome associations but only their absolute magnitude
    :param beta:                a number indicating the learning rate for positive and negative situations. Again, we
                                make the simplifying assumption that our simulated learners are equally affected by
                                positive and negative feedback. Changing the beta value can have a significant impact
                                on the learning outcome, but 0.1 is a standard choice for this model. If the number of
                                learning trials or the number of different cues in a learning trial are very large,
                                both beta and alpha need to be lowered considerably
    :param lam:                 maximum amount of association that an outcome can receive from all the cues. It simply
                                acts as a scaling factor, so changing its value has the same effects of changing alpha
    :param longitudinal:        a boolean specifying whether to work in a longitudinal setting or not
    :param at:                  the number of top active outcomes to consider to compute precision
    :return accuracies:         a dictionary mapping the categorization accuracy on the PoS tagging experiment to each
                                time index (1 if the longitudinal parameter is set to False, 10 if it's set to True)
    :return entropies:          a dictionary mapping the normalized entropy of the distribution of the PoS tags
                                assigned by the model to each time index (1 if the longitudinal parameter is set to
                                False, 10 if it's set to True)
    :return most_frequents:     a dictionary mapping the PoS tag that was applied the most by the model to each time
                                index (1 if the longitudinal parameter is set to False, 10 if it's set to True)
    :return frequencies:        a dictionary mapping the frequency count of the most frequent PoS tag applied by the
                                model, to each time index (1 if the longitudinal parameter is set to False, 10 if it's
                                set to True)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    encoded_corpus = corpus_encoder(input_file, celex_dir, pos_mapping, separator=separator, uniphones=uniphones,
                                    diphones=diphones, triphones=triphones, syllables=syllable,
                                    stress_marker=stress_marker, reduced=reduced, outcomes=outcomes,
                                    boundaries=boundaries)

    file_paths = ndl(encoded_corpus, alpha=alpha, beta=beta, lam=lam, longitudinal=longitudinal)

    # for each test item, compute the items from the matrix of weights that are most activated given the cues in the
    # item, get the PoS tag that is most present among the most active lexical nodes and check whether the predicted
    # PoS tag matches the gold-standard one provided along the test item. Return a global score indicating the accuracy
    # on the test set
    accuracies = {}
    entropies = {}
    most_frequents = {}
    frequencies = {}
    log_dicts = {}

    celex_dict = get_celex_dictionary(celex_dir, reduced=reduced)

    for idx, file_path in file_paths.items():

        logfile = make_log_file(input_file, test_set['filename'], output_folder, method, evaluation, flush, k, at, idx,
                                reduced=reduced, uniphones=uniphones, diphones=diphones, triphones=triphones,
                                syllables=syllable, stress_marker=stress_marker, outcomes=outcomes,
                                boundaries=boundaries)

        if os.path.exists(logfile):
            print()
            print("The file %s already exists, statistics for the corresponding parametrization are loaded from it" %
                  logfile)
            log_dict = json.load(open(logfile, "r"))

        else:
            print()
            matrix, cues2ids, outcomes2ids = load(file_path)

            # get the column ids of all perfectly discriminated outcomes at the current time point
            # perfectly discriminated outcomes are considered to be those:
            # - whose jaccard coefficient between true phonetic cues and most active phonetic cued for the outcome is 1
            # - and that appear in the top active outcomes given the cues they consist of
            corpus_folder = os.path.dirname(encoded_corpus)
            discriminated_file = os.path.join(corpus_folder, '.'.join(['discriminatedOutcomes', str(int(idx)), 'json']))
            if not os.path.exists(discriminated_file):
                discriminated = find_discriminated(matrix, cues2ids, outcomes2ids, celex_dict,
                                                   stress_marker=stress_marker, uniphones=uniphones,
                                                   diphones=diphones, triphones=triphones,
                                                   syllables=syllable, boundaries=boundaries, at=at)
            else:
                discriminated = json.load(open(discriminated_file, 'r'))

            print()
            print(strftime("%Y-%m-%d %H:%M:%S") + ": Start test phase, using %s as weight matrix and %s as test set..."
                  % (os.path.basename(file_path), os.path.basename(test_set['filename'])))

            log_dict = categorize(test_set['items'], matrix, cues2ids, discriminated,
                                  method=method, evaluation=evaluation, flush=flush, k=k, threshold=threshold,
                                  stress_marker=stress_marker, syllables=syllable, uniphones=uniphones,
                                  diphones=diphones, triphones=triphones, boundaries=boundaries)
            json.dump(log_dict, open(logfile, 'w'))

            print(strftime("%Y-%m-%d %H:%M:%S") + ": ...completed test phase.")

        f1, h, pos, freq = compute_summary_statistics(log_dict)

        accuracies[idx] = f1
        entropies[idx] = h
        most_frequents[idx] = pos
        frequencies[idx] = freq
        log_dicts[idx] = log_dict

        print("Accuracy: %0.5f" % f1)
        print()

    return log_dicts, accuracies, entropies, most_frequents, frequencies