__author__ = 'GCassani'

import os
import argparse
import glob
import preprocess_corpus as prc
import numpy as np
from time import strftime
from phonetic_bootstrapping import phonetic_bootstrapping
from collections import defaultdict


def split_and_decode(record):

    """
    :param record:  a list of at least eight elements 
    :return model:  a string combining the information in the list, after applying a series of decoding steps 
    """

    filename = record[0]
    reduced = "w" if record[1] == '0' or record[1] == "full" else "r"
    method = record[2]
    evaluation = record[3]
    cues = "t" if record[4] == "triphones" else "s"
    stress = "m" if record[5] == '0' or record[5] == 'nostress' else "n"
    training = record[6]
    k = "k" + str(record[7])
    f = "f" + str(record[8])
    model = "_".join(["logfile", filename, "aggregate", training, "".join([reduced, cues, stress, "l"]),
                      method, evaluation, k, f]) + ".txt"

    return model


########################################################################################################################


def check_missing(table, folder):

    """
    :param table:                   the path to a .txt file containing summary information about the experiments 
                                    (columns are tab separated)
    :param folder:                  the path to a folder
    :return in_table_not_in_folder: the set of models that is found in the input table but not in the input folder
    :return in_folder_not_in_table: the set of models that is found in the input folder but not in the input table
    """

    in_table = list()
    in_folder = list()
    curr_folder = os.getcwd()

    with open(table, "r") as t:
        next(t)
        for line in t:
            record = line.strip().split("\t")
            in_table.append(split_and_decode(record))

    if isinstance(folder, str):
        os.chdir(folder)
        in_folder.extend(glob.glob("logfile_*"))

    elif isinstance(folder, list):
        for f in folder:
            os.chdir(f)
            in_folder.extend(glob.glob("logfile_*"))
            os.chdir(curr_folder)

    else:
        raise ValueError ("Unrecognized object: provide a single folder as a string or a list of strings.")

    in_folder_not_in_table = set(in_folder) - set(in_table)
    in_table_not_in_folder = set(in_table) - set(in_folder)

    return in_table_not_in_folder, in_folder_not_in_table


########################################################################################################################


def clean(entries, where):

    """
    :param entries: a set of strings, indicating the things to eliminate
    :param where:   the path to a table or to a directory, indicating where to eliminate models. If the path to a table
                    is passed, then a new temporary table is created where all lines that are not found in the input 
                    entries are printed. The original table is deleted, and the temporary one gets renamed as the 
                    original one.
    """

    if os.path.isfile(where):
        new_file = where+"~"
        with open(where, "r") as f:
            for line in f:
                record = line.strip().split("\t")
                model = split_and_decode(record)
                if model not in entries:
                    with open(new_file, "a") as n:
                        n.write(line)

        os.remove(where)
        os.rename(new_file, where)

    elif os.path.isdir(where):
        os.chdir(where)
        for entry in entries:
            try:
                os.remove(entry)
            except OSError:
                pass

    else:
        raise ValueError("Unrecognized object! The argument 'where' can either be a directory or a .txt file.")


########################################################################################################################


def grid_search(test_set, method, evaluation, cues, training, reduced_vowels, stress, k, f, longitudinal):

    """
    :param test_set:
    :param method:
    :param evaluation:
    :param cues:
    :param training:
    :param reduced_vowels:
    :param stress:
    :param k:
    :param f:
    :param longitudinal
    :return summary_table:
    """

    summary_table = defaultdict(dict)

    if not isinstance(k, int):
        k = int(k)

    if not isinstance(f, int):
        f = int(f)

    utterance_corpus = "/home/cassani/phoneticBootstrapping/corpus/utteranceTraining/aggregate_utterances.json"
    word_corpus = "/home/cassani/phoneticBootstrapping/corpus/wordTraining/aggregate_words.json"

    if training == 'utterances':
        input_corpus = utterance_corpus
        a = 0.00001
        b = 0.00001
    else:
        input_corpus = word_corpus
        a = 0.001
        b = 0.001

    triphones = False
    syllables = False
    if cues == 'triphones':
        triphones = True
    elif cues == 'syllables':
        syllables = True
    else:
        raise ValueError("Unrecognized cue specification, choose between 'triphones' and 'syllables'.")

    reduced = False
    celex = "/home/cassani/phoneticBootstrapping/Celex/celex_dict.json"
    vowels = "full"
    if reduced_vowels:
        reduced = True
        celex = "/home/cassani/phoneticBootstrapping/Celex/celex_dict_reduced.json"
        vowels = "reduced"

    stress_marker = False
    sm = "nostress"
    if stress:
        stress_marker = True
        sm = "stress"

    encoding_string = prc.encoding_features(input_corpus, reduced=reduced,
                                            uni_phones=False, di_phones=False,
                                            tri_phones=triphones,
                                            syllable=syllables,
                                            stress_marker=stress_marker,
                                            grammatical=False, verbose=False)

    test_basename = os.path.splitext(os.path.basename(test_set))[0]
    corpus_dir = os.path.dirname(input_corpus)
    filename = os.path.splitext(os.path.basename(input_corpus))[0]
    log_file = "/".join([corpus_dir,
                         ".".join(["_".join(['logfile', test_basename, filename,
                                             encoding_string, method, evaluation,
                                             ''.join(['k', str(k)]),
                                             ''.join(['f', str(f)])]), 'txt'])])
    if not os.path.exists(log_file):
        acc, entr, most_freq, freq = phonetic_bootstrapping(input_corpus, test_set,
                                                            log_file, celex,
                                                            method=method,
                                                            evaluation=evaluation,
                                                            k=k,
                                                            flush=f,
                                                            reduced=reduced,
                                                            stress_marker=stress_marker,
                                                            grammatical=False,
                                                            uni_phones=False,
                                                            di_phones=False,
                                                            tri_phones=triphones,
                                                            syllable=syllables,
                                                            alpha=a, beta=b,
                                                            lam=1.0,
                                                            longitudinal=longitudinal)

        summary_table[test_set][vowels][method][evaluation][cues][sm][training][k][f]["accuracies"] = acc
        summary_table[test_set][vowels][method][evaluation][cues][sm][training][k][f]["entropies"] = entr
        summary_table[test_set][vowels][method][evaluation][cues][sm][training][k][f]["most"] = most_freq
        summary_table[test_set][vowels][method][evaluation][cues][sm][training][k][f]["frequencies"] = freq

        print(acc)
        print(entr)
        print(most_freq)
        print(freq)
        print(strftime("%Y-%m-%d %H:%M:%S"))

    else:
        print("Model %s was already evaluated" % log_file)

    return summary_table


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Run a grid search to explore all possible parameters.")

    parser.add_argument("-i", "--input_file", required=True, dest="in_file",
                        help="Specify the test set (encoded as a .txt file, one word per line, encoded as word|pos).")
    parser.add_argument("-o", "--output_file", required=True, dest="out_file",
                        help="Specify the path of the output file.")
    parser.add_argument("-m", "--method", dest="method", default="sum",
                        help="Specify whether to look at frequency ('freq') or total activation ('sum') of PoS tags.")
    parser.add_argument("-e", "--evaluation", dest="evaluation", default="count",
                        help="Specify whether to consider counts ('count') or compare distributions ('distr').")
    parser.add_argument("-c", "--cues", dest="cues", default="triphones",
                        help="Specify whether to consider triphones ('triphones') or syllables ('syllables').")
    parser.add_argument("-t", "--training", dest="training", default="words",
                        help="Specify whether to train on single words ('words') or on full utterances ('utterances').")
    parser.add_argument("-r", "--reduced", action="store_true", dest="reduced",
                        help="Specify whether to use reduced unstressed vowels or not (default: False).")
    parser.add_argument("-s", "--stress", action="store_true", dest="stress",
                        help="Specify whether the model is sensitive to stress (default: False).")
    parser.add_argument("-k", "--neighbours", dest="neighbours", default=20,
                        help="Specify the number of nearest neighbours to consider (default: 20).")
    parser.add_argument("-f", "--flush", dest="flush", default=0,
                        help="Specify how many elements to flush away considering baseline activation (default: 0).")
    parser.add_argument("-L", "--longitudinal", action="store_true", dest="longitudinal",
                        help="Specify whether to use a longitudinal design (default: False).")

    args = parser.parse_args()

    summary_table = grid_search(args.in_file, args.method, args.evaluation, args.cues, args.training, args.reduced,
                                args.stress, args.neighbours, args.flush, args.longitudinal)
    if args.longitudinal:
        with open(args.out_file, "a+") as w:
            indices = np.linspace(5,100,20)
            a = ["Accuracy_" + str(int(i)) for i in indices]
            e = ["Entropy_" + str(int(i)) for i in indices]
            m = ["Most_frequent_" + str(int(i)) for i in indices]
            f = ["Frequency_" + str(int(i)) for i in indices]
            w.write("\t".join(["Test_set", "Reduced", "Method", "Evaluation", "Cues", "Stress", "Training", "K", "F",
                               "\t".join(a), "\t".join(e), "\t".join(m), "\t".join(f)]))
            w.write("\n")
    else:
        with open(args.out_file, "a+") as w:
            w.write("\t".join(["Test_set", "Reduced", "Method", "Evaluation", "Cues", "Stress", "Training", "K", "F",
                               "Accuracy", "Entropy", "Most_frequent", "Frequency"]))
            w.write("\n")

    for test_set in summary_table:
        for vowel in summary_table[test_set]:
            for method in summary_table[test_set][vowel]:
                for evaluation in summary_table[test_set][vowel][method]:
                    for cue in summary_table[test_set][vowel][method][evaluation]:
                        for sm in summary_table[test_set][vowel][method][evaluation][cue]:
                            for training in summary_table[test_set][vowel][method][evaluation][cue][sm]:
                                for k in summary_table[test_set][vowel][method][evaluation][cue][sm][training]:
                                    for f in summary_table[test_set][vowel][method][evaluation][cue][sm][training]:
                                        accuracies = list(map(str,
                                                              summary_table[test_set][vowel][method][evaluation][cue][
                                                                  sm][training][f]["accuracies"]))
                                        entropies = list(map(str,
                                                              summary_table[test_set][vowel][method][evaluation][cue][
                                                                  sm][training][f]["entropies"]))
                                        most_frequents = list(map(str,
                                                              summary_table[test_set][vowel][method][evaluation][cue][
                                                                  sm][training][f]["most"]))
                                        frequencies = list(map(str,
                                                              summary_table[test_set][vowel][method][evaluation][cue][
                                                                  sm][training][f]["frequencies"]))

                                        with open(args.out_file, "a+") as w:
                                            w.write("\t".join([test_set, vowel, method, evaluation, cue, sm, training,
                                                               str(k), str(f), "\t".join(accuracies),
                                                               "\t".join(entropies), "\t".join(most_frequents),
                                                               "\t".join(frequencies)]))
                                            w.write("\n")


########################################################################################################################


if __name__ == '__main__':

    main()
