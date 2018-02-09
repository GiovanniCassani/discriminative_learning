__author__ = 'GCassani'

"""Function to perform the phonological bootstrapping experiment from scratch, given an input corpus and
   the specifications about how to process the corpus, about which cues and outcomes to use, how to build the test set, 
   how to perform categorization, and finally the parameters for the ndl algorithm"""

import os
import json
from time import strftime
from collections import defaultdict
from rescorla_wagner.ndl import ndl
from corpus.encoder import corpus_encoder
from phonetic_bootstrapping.experiment.log import make_log_file
from phonetic_bootstrapping.experiment.categorize import categorize
from phonetic_bootstrapping.experiment.helpers import compute_summary_statistics


def phonetic_bootstrapping(input_file, test_set, celex_dir, pos_mapping,
                           method='freq', evaluation='count', k=100, flush=0,
                           separator='~', reduced=False, outcomes='tokens',
                           uni_phones=True, di_phones=False, tri_phones=False, syllable=False, stress_marker=False,
                           alpha=0.01, beta=0.01, lam=1.0, longitudinal=False):
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
                                the activations triggered by a specific test item. By default, the top 100 outcomes are
                                considered, and compared according to the chosen combination of method and eval
    :param flush:               specify whether (and how many) top active outcome at baseline to flush away from
                                subsequent computations. It may be the case that whatever the input cues, the same high
                                frequency outcomes come out as being the most active. It may then make sense to not
                                consider them when evaluating the distribution of lexical categories over the most
                                active outcomes given an input item
    :param separator:           the character that separates the word baseform from its PoS tag in the input corpus
    :param reduced:             a boolean specifying whether reduced phonological forms should be extracted from Celex
                                whenever possible (if set to True) or if standard phonological forms should be
                                preserved (if False)
    :param outcomes:            a string indicating which outcomes to use, whether 'tokens' (default) or 'lemmas'
    :param uni_phones:          a boolean indicating whether single phonemes are to be considered while encoding input
                                utterances
    :param di_phones:           a boolean indicating whether sequences of two phonemes are to be considered while
                                encoding input utterances
    :param tri_phones:          a boolean indicating whether sequences of three phonemes are to be considered while
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

    encoded_corpus = corpus_encoder(input_file, celex_dir, pos_mapping, separator=separator, uni_phones=uni_phones,
                                    di_phones=di_phones, tri_phones=tri_phones, syllable=syllable,
                                    stress_marker=stress_marker, reduced=reduced, outcomes=outcomes)

    weight_matrices, cues2ids, outcomes2ids = ndl(encoded_corpus, alpha=alpha, beta=beta,
                                                  lam=lam, longitudinal=longitudinal)

    output_folder = os.path.dirname(encoded_corpus)

    print()
    print("#####" * 20)
    print()

    # for each test item, compute the items from the matrix of weights that are most activated given the cues in the
    # item, get the PoS tag that is most present among the most active lexical nodes and check whether the predicted
    # PoS tag matches the gold-standard one provided along the test item. Return a global score indicating the accuracy
    # on the test set
    accuracies = {}
    entropies = {}
    most_frequents = {}
    frequencies = {}
    log_dict = defaultdict(dict)
    for idx in weight_matrices:

        logfile = make_log_file(input_file, test_set['filename'], output_folder, method, evaluation, flush, k, idx,
                                reduced=reduced, uni_phones=uni_phones, di_phones=di_phones, tri_phones=tri_phones,
                                syllable=syllable, stress_marker=stress_marker, outcomes=outcomes)

        if os.path.exists(logfile):
            print("The file %s already exists, statistics for the corresponding parametrization are loaded from it" %
                  logfile)
            log_dict = json.load(open(logfile, "r"))

        else:

            print(strftime("%Y-%m-%d %H:%M:%S") + ": Start test phase, using %s as test set..."
                  % os.path.basename(test_set['filename']))

            log_dict = categorize(test_set['items'], logfile, weight_matrices[idx], cues2ids, outcomes2ids,
                                  method=method, evaluation=evaluation, stress_marker=stress_marker, syllable=syllable,
                                  uni_phones=uni_phones, di_phones=di_phones, tri_phones=tri_phones, flush=flush, k=k)

            print(strftime("%Y-%m-%d %H:%M:%S") + ": ...completed test phase.")

        f1, h, pos, freq = compute_summary_statistics(log_dict)

        accuracies[idx] = f1
        entropies[idx] = h
        most_frequents[idx] = pos
        frequencies[idx] = freq

        print("Accuracy: %0.5f" % f1)

    return log_dict, accuracies, entropies, most_frequents, frequencies
