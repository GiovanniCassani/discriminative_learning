__author__ = 'GCassani'

"""Helper functions to monitor the grid search experiment and check which models might not have been processed"""

import os
import glob


def split_and_decode(record):

    """
    :param record:  a list of at least eight elements, corresponding to the information stored in a line of the file
                    containing summary information about the grid search experiment
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
    :param table:                   the path to a .txt file containing summary information about the grid search
                                    (columns are tab separated)
    :param folder:                  the path to a folder containing processed data using different parametrizations
                                    specified in the grid search experiment
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
        raise ValueError("Unrecognized object: provide a single folder as a string or a list of strings.")

    in_folder_not_in_table = set(in_folder) - set(in_table)
    in_table_not_in_folder = set(in_table) - set(in_folder)

    return in_table_not_in_folder, in_folder_not_in_table


########################################################################################################################


def clean(entries, where):

    """
    :param entries: a set of strings, indicating the entries to eliminate
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
