__author__ = 'GCassani'

"""Function to analyze the activation matrices learned by the naïve discriminative learner
   (can be called from command line)"""

import os
import argparse
from phonetic_bootstrapping.analysis.inspect import inspect_the_matrix


def check_input_arguments(args, parser):

    """
    :param args:    the result of an ArgumentParser structure
    :param parser:  a parser object, created using argparse.ArgumentParser
    """

    # check corpus file
    if not os.path.exists(args.input_corpus) or not args.input_corpus.endswith(".json"):
        raise ValueError("There are problems with the corpus file you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    # check association matrix
    if not os.path.exists(args.associations) or not args.associations.endswith(".npy"):
        raise ValueError("There are problems with the association matrix file you provided: either the path does not "
                         "exist or the file extension is not .npy. Provide a valid path to a .npy file.")

    # check that at least one phonetic encoding is specified
    if not (args.uni or args.di or args.tri or args.syl):
        parser.error('No specified phonetic encoding! Provide at least one of the following options: -u, -d, -t, -s')

    # check CELEX folder
    if not os.path.exists(args.celex_dir):
        raise ValueError("The provided folder does not exist. Provide the path to an existing folder.")

    # check if plot folder exists, and make it in case it does not
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description='Analyze the structure of an association matrix derived with the NDL.')

    parser.add_argument('-I', '--input_corpus', required=True, dest='input_corpus',
                        help='Give the path to the input corpus (.json),'
                             'encoded in phonetic cues and lexical outcomes.')
    parser.add_argument('-A', '--association_matrix', required=True, dest='associations',
                        help='Give the path to the .npy cue-outcome associations matrix, computed using the NDL.')
    parser.add_argument('-C', "--Celex_dir", required=True, dest="celex_dir",
                        help="Specify the path to the folder containing Celex files or dictionary.")
    parser.add_argument('-P', "--plot_path", required=True, dest="plot_path",
                        help="Specify the path to the folder where plots will be stored.")
    parser.add_argument('-r', '--reduced', action='store_true', dest='reduced',
                        help='Specify if reduced phonological forms are to be considered.')
    parser.add_argument("-u", "--uniphones", action="store_true", dest="uni",
                        help="Specify if uniphones need to be encoded.")
    parser.add_argument("-d", "--diphones", action="store_true", dest="di",
                        help="Specify if diphones need to be encoded.")
    parser.add_argument("-t", "--triphones", action="store_true", dest="tri",
                        help="Specify if triphones need to be encoded.")
    parser.add_argument("-s", "--syllables", action="store_true", dest="syl",
                        help="Specify if syllables need to be encoded.")
    parser.add_argument("-m", "--stress_marker", action="store_true", dest="stress",
                        help="Specify if stress need to be encoded.")

    args = parser.parse_args()
    check_input_arguments(args, parser)

    # compute pairwise correlations between frequency, MAD, activation, and Jaccard coefficients for the outcomes
    # in the input corpus, then print them to standard output

    inspect_the_matrix(args.input_corpus, args.associations, args.celex_dir,
                       args.plot_path, args.uni, args.di, args.tri, args.syl, args.stress)


########################################################################################################################


if __name__ == '__main__':

    main()
