__author__ = 'GCassani'

"""Function to chunk the corpus to the desired granularity (can be called from command line)"""

import os
import argparse
from corpus.chunker import chunk_corpus


def main():

    parser = argparse.ArgumentParser(description='Recode utterances from the corpus.')

    parser.add_argument("-I", "--input_corpus", required=True, dest="input_corpus",
                        help="Specify the corpus to be used as input (encoded as .json).")
    parser.add_argument("-s", "--size", dest="size", default='utterances',
                        help="Specify granularity of the resulting encoding.")

    args = parser.parse_args()

    encoding_options = ['utterances', 'prosodic', 'multiword', 'words']

    if args.size not in encoding_options:
        raise ValueError("The encoding option you selected is not available. Choose among 'utterances', 'prosodic',"
                         "'multiword', or 'words'. Check the documentation for information about each encoding.")

    if not os.path.exists(args.input_corpus) or not args.input_corpus.endswith(".json"):
        raise ValueError("There are problems with the input corpus you provided: either the path does not exist or"
                         "the file extension is not .json. Provide a valid path to a .json file.")

    chunk_corpus(args.input_corpus, args.size)


########################################################################################################################


if __name__ == '__main__':

    main()
