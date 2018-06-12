__author__ = 'GCassani'

"""Functions to perform the error analysis from a .json logfile resulting from the PoS tagging experiment."""

import json
import numpy as np
import pandas as pd
import itertools as it
from time import strftime
from collections import Counter, defaultdict
from phonological_bootstrapping.tagging.log import make_log_file


def get_errors(log_dict):

    """
    :param log_dict:    the output dictionary resulting from the categorization experiment (the file is the output of
                        the function categorize from the module phonological_bootstrapping.experiment). It consists of a
                        dictionary mapping each test item to several fields, ['predicted'] being necessary for this
                        function. Each test item is assumed to be a string consisting of two parts divided by a
                        vertical bar ('|'): the word as first substring, the tag as second substring.
    :return errors:     a dictionary mapping strings to integers. Strings are of the form 'PoS1_as_PoS2' where PoS1 is
                        the correct target PoS and PoS2 is the predicted PoS. The integer is how often words from a PoS
                        were tagged in a certain way by the model.
    """

    errors = Counter()

    for item in log_dict:
        correct = item.split("|")[1]
        predicted = log_dict[item]['predicted']
        name = "_as_".join([correct, predicted])
        errors[name] += 1

    return errors


########################################################################################################################


def compute_error_analysis(folder, test_sets, corpora, boundaries, outcomes, cues, stress_marker, reduced_vowels,
                           methods, evaluations, k, flush, at, time_indices):

    """
    :param folder:                      the path to the folder where log files are to be found
    :param test_sets:                   an iterable containing the base-names of the test files to be considered
    :param corpora:                     an iterable containing the base-names of the corpora used to learn cue-outcome
                                        associations used by the models to categorize test items
    :param boundaries:                  an iterable containing booleans specifying whether to consider word boundaries
                                        or not during training
    :param outcomes:                    an iterable containing strings indicating which lexical outcomes to use,
                                        whether 'lemmas' or 'tokens'
    :param cues:                        an iterable containing strings indicating which cues to be considered
                                        (at least one among 'uniphones', 'diphones', 'triphones', and 'syllables')
    :param stress_marker:               an iterable containing booleans specifying whether or not to consider stress
    :param reduced_vowels:              an iterable containing booleans specifying whether or not use reduced phonetic
                                        transcriptions whenever possible
    :param methods:                     an iterable containing strings specifying which method to use to process top
                                        active lexical nodes given a test item (at least one among 'sum' and 'freq')
    :param evaluations:                 an iterable containing strings specifying which evaluation to use when
                                        comparing activation values triggered by a test item to baseline activation
                                        values (at least one among 'count' and 'distr')
    :param k:                           an iterable containing integers specifying how many top active lexical nodes to
                                        consider to pick the most represented PoS category
    :param flush:                       an iterable containing integers specifying how many lexical nodes to flush away,
                                        to get rid of nodes with very high baseline activations that would be among the
                                        top active lexical nodes for all the test items
    :param at:                          the number of top active outcomes considered to compute precision of
                                        discrimination
    :param time_indices:                an iterable containing the relevant time integers to consider (one in a simple
                                        design, several in a longitudinal design)
    :return categorization_outcomes:    a dictionary mapping the path of a log_file to the categorization and
                                        mis-categorization patterns detected in the dictionary contained in the
                                        log_file, themselves contained in a dictionary where keys are of the form
                                        'PoS1_as_PoS2' with PoS1 being the correct target PoS and PoS2 the predicted
                                        PoS. The integer is how often words from PoS1 were tagged as PoS2y by the model.
    :return categorization_columns:     a set containing all the keys of the error dictionaries for all the log_files
                                        found in categorization_outcomes
    """

    categorization_outcomes = defaultdict(dict)
    categorization_columns = set()
    parametrizations = it.product(test_sets, boundaries, corpora, outcomes, cues, stress_marker, reduced_vowels,
                                  methods, evaluations, k, flush, time_indices)
    for parametrization in parametrizations:
        test_set, boundary, corpus, outcome, cue, marker, r, method, evaluation, k_val, f_val, time = parametrization
        uniphones = True if cue == 'uniphones' else False
        diphones = True if cue == 'diphones' else False
        triphones = True if cue == 'triphones' else False
        syllables = True if cue == 'syllables' else False

        log_file = make_log_file(corpus, test_set, folder, method, evaluation, f_val, k_val, at, time,
                                 reduced=r, stress_marker=marker, boundaries=boundary, outcomes=outcome,
                                 uniphones=uniphones, diphones=diphones, triphones=triphones, syllables=syllables)

        # get the errors error analysis for the current log file
        try:
            log_dict = json.load(open(log_file, "r"))
            print(strftime("%Y-%m-%d %H:%M:%S") + ": Started error analysis on file %s..." % log_file)
            errors = get_errors(log_dict)
            categorization_outcomes[log_file] = errors
            categorization_columns = categorization_columns.union(set(errors.keys()))
            print(strftime("%Y-%m-%d %H:%M:%S") + ": ...completed error analysis on file %s." % log_file)
            print()

        except IOError:
            print("The file %s was not found" % log_file)

    return categorization_outcomes, categorization_columns


########################################################################################################################


def update_error_analysis_dataset(categorization_outcomes, categorization_columns):

    """
    :param categorization_outcomes: a dictionary mapping the path of a log_file to the categorization and
                                    mis-categorization patterns detected in the dictionary contained in the log_file,
                                    themselves contained in a dictionary where keys are of the form 'PoS1_as_PoS2'
                                    with PoS1 being the correct target PoS and PoS2 the predicted PoS. The integer is
                                    how often words from a PoS were tagged in a certain way by the model.
    :param categorization_columns:  a set containing all the keys of the error dictionaries for all the log_files found
                                    in categorization_outcomes
    :return df:                     the data frame, with as many rows as there are log_files in categorization_outcomes
                                    and columns indicating model specifications and classification and
                                    mis-classification patterns detected in the error analysis
    """

    # initialize a data set with all the required rows and columns
    dataset_columns = ["Test_set", "Corpus", "Boundaries", "Outcomes", "Cues", "Stress", "Vowels",
                       "Method", "Evaluation", "K", "F", "At", "Time"]
    dataset_columns.extend(sorted(categorization_columns))
    df = pd.DataFrame(index=np.arange(0, len(categorization_outcomes)),
                      columns=dataset_columns)

    # update the output df
    row_id = 0
    for log_file in categorization_outcomes:

        corpus, test_set, model, method, evaluation, time, k, f, prec = log_file.split(".")[1:-1]
        time, k, f, at = (time[1:], k[1:], f[1:], prec[2:])
        vowels = 'full' if 'f' in model else 'reduced' if 'r' in model else 'unk'
        sm = 'stress' if 'm' in model else 'no-stress' if 'n' in model else 'unk'
        outcome = 'lemmas' if 'l' in model else 'tokens' if 'k' in model else 'unk'
        boundary = 'yes' if 'b' in model else 'no' if 'c' in model else 'unk'

        cues = []
        if 's' in model:
            cues.append('syllables')
        if 't' in model:
            cues.append('triphones')
        if 'd' in model:
            cues.append('diphones')
        if 'u' in model:
            cues.append('uniphones')
        cue = ", ".join(cues)

        # indicate model specifications
        dataset_row = {"Test_set": test_set, "Corpus": corpus, "Boundaries": boundary, "Cues": cue, "Outcomes": outcome,
                       "Stress": sm, "Vowels": vowels, "Method": method, "Evaluation": evaluation, "K": k, "F": f,
                       "Time": time, "At": at}

        # add classification and mis-classification patterns detected for the model being considered
        # if a certain pattern is not present in the error analysis, assign 0 to the corresponding column
        for key in categorization_columns:
            try:
                dataset_row[key] = categorization_outcomes[log_file][key]
            except KeyError:
                dataset_row[key] = 0

        df.loc[row_id] = pd.Series(dataset_row)
        row_id += 1

    return df
