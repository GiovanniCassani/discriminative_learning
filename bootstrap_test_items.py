__author__ = 'GCassani'

"""Function to get the test items (can be called from command line)"""

import argparse
from grid_search.bootstrap_test_sets import bootstrap_test_sets


def main():

    parser = argparse.ArgumentParser(description='Get a list of test items from the input corpus and Celex.')

    parser.add_argument('-T', '--test_set', required=True, dest='test_set',
                        help='Give the path to a .txt file containing the test set from which to sample.')
    parser.add_argument('-O', '--output_folder', required=True, dest='output_folder',
                        help='Specify the path of the folder where sampled test sets will be written to.')
    parser.add_argument('-n', dest='n', default=36,
                        help='Specify how many samples to draw from the test set')
    parser.add_argument('-k',  dest="k", default=1000,
                        help="Specify how many words from each category to sample.")

    args = parser.parse_args()

    bootstrap_test_sets(args.test_set, args.output_folder, int(args.n), int(args.k))


########################################################################################################################


if __name__ == '__main__':

    main()
