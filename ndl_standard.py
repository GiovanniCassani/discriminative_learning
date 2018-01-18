__author__ = 'GCassani'

import os
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from time import strftime

"""
This module uses the Rescorla-Wagner update rule to incrementally update a matrix of cue-outcome associations. This is
implemented in a standard Python implementation that organizes the matrix as a dictionary of dictionaries;

- compute_total_activation          : computes the total sum for the activation of an outcome given a set of input cues
- update_outcome_weights            : updates the cue-outcome associations for a given outcome, considering the
                                        prediction error resulting from the predicted total activation
- discriminative_learner            : the perceptron that learns the cue-outcome association matrix (this implementation
                                        uses standard Python dictionaries)
- main                              : a function that runs the standard implementation of the discriminative learner 
                                        when the module is called from command line

"""


def file_exists(output_files, cue_files, outcome_files):

    """
    :param output_files:    a Python dictionary mapping numerical indices to file paths pointing to cue-outcome
                            association matrices
    :param cue_files:       a Python dictionary mapping numerical indices to file paths pointing to cue indices
    :param outcome_files:   a Python dictionary mapping numerical indices to file paths pointing to outcome indices
    :return:                a boolean, True only if all paths in the input dictionaries point to an existing file, False
                            otherwise, i.e. even if only one file from the input dictionary doesn't exist
    """

    for k in output_files:
        if not os.path.isfile(output_files[k]) or not os.path.isfile(cue_files[k]) or \
                not os.path.isfile(outcome_files[k]):
            return False

    return True


########################################################################################################################


def ndl_load(output_files, cue_files, outcome_files, weight_matrices, cues, outcomes):

    """
        :param output_files:        a Python dictionary mapping numerical indices to strings pointing to cue-outcome
                                    association matrices
        :param cue_files:           a Python dictionary mapping numerical indices to file paths pointing to cue indices
        :param outcome_files:       a Python dictionary mapping numerical indices to file paths pointing to outcome
                                    indices
        :param weight_matrices:     a Python dictionary
        :param cues:                a Python dictionary
        :param outcomes:            a Python dictionary
        :return weight_matrices:    a Python dictionary mapping numerical indices to large dictionaries of dictionaries
                                    containing cue-outcome associations computed using the Rescorla-Wagner model of
                                    learning
        :return cues:               a set containing all the cues from the input corpus
        :return outcomes:           a set containing all the outcomes from the input corpus
        """

    for k in output_files:
        weight_matrices = json.load(open(output_files[k], "r"))
        cues = json.load(open(cue_files[k], "r"))
        outcomes = json.load(open(outcome_files[k], "r"))

    return weight_matrices, cues, outcomes


########################################################################################################################


def compute_total_activation(d, outcome, trial_cues, lam):

    """
    :param d:           a Python dictionary of dictionaries, where first and second level keys are strings and values
                        are numbers
    :param outcome:     a string indicating one of the first level keys
    :param trial_cues:  a dictionary mapping strings to integers: the strings are second level keys from the input
                        dictionary, while the integers tell about how many times each string occurs in a learning trial
    :param lam:         a number indicating the amount of total activation that each first-level key can bear
    :return v_total:    the sum of all the values in the input dictionary that match the input first level keys and
                        second level keys: it represents the amount of activation for the first level key, the outcome,
                        given all the second level keys, i.e. the cues, from the learning trial
    """

    v_total = 0
    for cue, count in trial_cues.items():
        try:
            v_total += d[outcome][cue] * count
        except KeyError:
            d[outcome][cue] = 0

    if v_total > lam:
        print(strftime("%Y-%m-%d %H:%M:%S"))
        print(trial_cues, outcome)
        print(v_total)
        print('Something went wrong! The total amount of activation for the current learning instance exceeded '
              'the chosen value for the lambda parameter: the amount of activation was set to the lambda value.')
        v_total = lam

    return v_total


########################################################################################################################


def update_outcome_weights(d, outcome, trial_cues, trial_outcomes, ab, lam):
    """
    :param d:               a Python dictionary of dictionaries, where first and second level keys are strings and
                            values are numbers
    :param outcome:         a string indicating one of the first level keys
    :param trial_cues:      a dictionary mapping strings to integers: the strings are second level keys from the input
                            dictionary, while the integers tell about how many times each string occurs in a learning
                            trial
    :param trial_outcomes:  a set of strings, to which the input outcome is compared for inclusion
    :param ab:              a number indicating the learning rate
    :param lam:             a number indicating the amount of total activation that each first-level key can bear
    :return matrix:         the input Python dictionary, where values have been updated given the learning triggered by
                            the learning trial consisting of cues and outcomes; learning is computed according to the
                            Rescorla-Wagner model
    """

    v_total = compute_total_activation(d, outcome, trial_cues, lam)

    # compute association change for each cue-outcome relation for every cue in the current trial and update
    # the association score accordingly. If a cue is not present in the current learning trial, the change
    # in for each cue-outcome association involving the absent cue is 0 and we don't compute it. On the
    # contrary, change in association from present cues to all outcomes, present and absent, are computed
    if outcome not in trial_outcomes:
        lam = 0
    delta_v = ab * (lam - v_total)

    for cue in trial_cues:
        try:
            d[outcome][cue] += delta_v
        except KeyError:
            d[outcome][cue] = delta_v

        if d[outcome][cue] > 1:
            print(strftime("%Y-%m-%d %H:%M:%S"))
            print(outcome, cue, d[outcome][cue])
            raise ValueError('An outcome can only sustain an activation of %d and it looks like the ' +
                             'activation from a single cue is higher than this threshold. Something ' +
                             'went wrong, try lowering the learning rate, alpha, or the cue salience ' +
                             'parameter, beta.')

    return d


########################################################################################################################


def discriminative_learner(input_file, alpha=0.01, beta=0.01, lam=1.0, longitudinal=False):
    """
    :param input_file:      the path to a a .json file consisting of two lists of lists, the first containing phonetic
                            cues and the second containing outcome meanings; each list contains as many lists as there
                            are learning events in the input corpus (be them full utterances, single words, or any
                            intermediate representation derived from transcribed Child-directed speech). The first list
                            from the list of cue representations matches the first list from the list of meaning
                            representations, both encoding the two layers of the first learning event in the corpus.
    :param alpha:           cue salience. For simplicity, we assume that every cue has the same salience, so
                            changing the value of this parameter does not affect the relative strength of
                            of cue-outcome associations but only their absolute magnitude.
    :param beta:            learning rate. Again, we make the simplifying assumption that our simulated learners are
                            equally affected by positive and negative feedback. Changing the beta value can have a
                            significant impact on learning outcome, but 0.1 is a standard choice for this model.
    :param lam:             maximum amount of association that an outcome can receive from all the cues. It simply
                            acts as a scaling factor, so changing its value has the same effects of changing alpha.
    :param longitudinal:
    :return weight_matrix:  a dictionary of dictionaries containing cue-outcome associations, estimated using the
                            Rescorla-Wagner model of learning
    :return cues:           a set containing all the cues
    :return outcome:        a set containing all the outcomes

    This function implements Naive Discriminative Learning (NDL, see Baayen, Milin, Durdevic, Hendrix, Marelli (2011)
    for a detailed description of the model and its theoretical background. This learner uses the Rescorla-Wagner
    equations to update cue-outcome associations: it is a simple associative network with no hidden layer, that
    incrementally updates associations between cues and outcome.
    """

    weight_matrices = {}
    cues = {}
    outcomes = {}

    output_files = {}
    cue_files = {}
    outcome_files = {}

    if longitudinal:
        indices = np.linspace(5, 100, 20)
    else:
        indices = [100]

    # derive the filename for the output file(s)
    file_path = os.path.splitext(input_file)[0]
    for idx in indices:
        output_files[idx] = file_path + '_associationsDict_' + str(idx) + '.json'
        cue_files[idx] = file_path + '_cues_' + str(idx) + '.json'
        outcome_files[idx] = file_path + '_outcomes_' + str(idx) + '.json'

    exist = file_exists(output_files, cue_files, outcome_files)

    if exist:
        print("The matrix of weights for the desired model has already been computed and has been loaded.")
        weight_matrices, cues, outcomes = ndl_load(output_files, cue_files, outcome_files,
                                                   weight_matrices, cues, outcomes)

    else:

        weight_matrix = defaultdict(dict)
        curr_outcomes = set()
        curr_cues = set()

        ab = alpha * beta

        corpus = json.load(open(input_file, 'r+'))
        total_utterances = len(corpus[0])
        check_points = {np.floor(total_utterances / float(100) * n): n for n in np.linspace(5, 100, 20)}

        for i in range(len(corpus[0])):

            trial_cues = Counter(corpus[0][i])
            curr_cues.update(trial_cues.keys())
            trial_outcomes = set(corpus[1][i])
            curr_outcomes.update(trial_outcomes)

            # compute total activation for cues in the current trial, separately for each outcome:
            # this step sums all weights going from cues in the current trial to a single outcome and uses the total
            # weight to predict whether the outcome is going to be present or absent in the current learning trial.
            # If a cue-outcome association is new, its weight is 0.
            for outcome in curr_outcomes:
                weight_matrix = update_outcome_weights(weight_matrix, outcome, trial_cues, trial_outcomes, ab, lam)

            if i in check_points:
                print(strftime("%Y-%m-%d %H:%M:%S") + ": %d%% of the input corpus has been processed." %
                      check_points[i])
                weight_matrices[check_points[i]] = weight_matrix
                cues[check_points[i]] = curr_cues
                outcomes[check_points[i]] = curr_outcomes

        # save the association dictionary to a .json file
        for k in weight_matrices:
            json.dump(weight_matrices[k], open(output_files[k], 'w+'))
            json.dump(cues[k], open(cue_files[k], 'w+'))
            json.dump(outcomes[k], open(outcome_files[k], 'w+'))

    return weight_matrices, cues, outcomes


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Process arguments to create Celex dictionary.')

    parser.add_argument("-i", "--input_file", required=True, dest="in_file",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-L", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to work in a longitudinal design or not (default: False).")
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01,
                        help="Specify the value of the alpha parameter.")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01,
                        help="Specify the value of the beta parameter.")
    parser.add_argument("-l", "--lambda", dest="lam", default=1.0,
                        help="Specify the value of the lambda parameter.")

    args = parser.parse_args()

    if not os.path.exists(args.in_file) or not args.in_file.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    discriminative_learner(args.in_file, alpha=float(args.alpha), beta=float(args.beta), lam=float(args.lam),
                           longitudinal=args.longitudinal)

########################################################################################################################


if __name__ == '__main__':

    main()
