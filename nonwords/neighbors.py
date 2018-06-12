__author__ = 'GCassani'

import helpers as help
from collections import defaultdict, Counter


def map_neighbours_to_pos(neighbours_file, wordform2pos, ids2words):

    """
    :param neighbours_file: the file with the neighbours retrieved for each test item: each item is on a different
                            column, with neighbours retrieved for it in the same column
    :param wordform2pos:    a dictionary mapping wordforms to the set of pos tags with which the wordform was found in
                            the input corpus
    :param ids2words:       a dictionary mapping column indices to test items
    :return neighbours:     a dictionary mapping each test item to the list of pos tags of the words retrieved as
                            neighbours. in case a neighbour is ambiguous with respect to the category, all categories
                            with which it was retrieved are considered
    """

    neighbours = defaultdict(list)
    with open(neighbours_file, "r") as f:
        for line in f:
            words = line.strip().split('\t')
            for col_id, word in enumerate(words):
                if not word in ids2words.values():
                    if ':' in word:
                        neighbours[ids2words[col_id]].append(word.split(':')[1])
                    else:
                        target_tags = wordform2pos[word]
                        for tag in target_tags:
                            neighbours[ids2words[col_id]].append(tag)

    return neighbours


########################################################################################################################


def get_tag_distribution(neighbours):

    """
    :param neighbours:          a dictionary mapping words to the list of pos tags of the neighbors
    :return neighbours_distr:   a dictionary mapping each test word to each category which appears in the neighborhood
                                and its frequency of occurrence
    :return pos_set:            a set containing the unique pos tags found in the neighbours list
    """

    neighbours_distr = defaultdict(dict)
    pos_set = set()
    for word in neighbours:
        neighbours_list = neighbours[word]
        distr = Counter(neighbours_list)
        for pos in distr:
            neighbours_distr[word][pos] = distr[pos]
            pos_set.add(pos)

    return neighbours_distr, pos_set


########################################################################################################################


def write_neighbor_summary(output_file, neighbours_distr, pos_set, table_format="long", cond="minimalist"):

    """
    :param output_file:         the path where the summary will be written to
    :param neighbours_distr:    a dictionary mapping words to the pos tags of its neighboring vectors and their
                                frequency count in the neighbor list
    :param pos_set:             an iterable of strings, indicating the pos tags encountered across the neighbors
    :param cond:                a string indicating the input used for the experiment
    :param table_format:        a string indicating how to print data to table, either 'long' or 'wide'. In the long
                                format, five columns are created, first the nonword followed by its intended pos tag,
                                then the condition, then the category, then the frequency. In the wide format, each
                                category is a different column, with each nonword-category cell indicating how many
                                neighbors tagged with the category were found in the neighborhood of the nonword.
                                An extra column indicates the condition.
    """

    pos_tags = sorted(pos_set)

    with open(output_file, "w") as f:

        if table_format == "wide":
            f.write("\t".join(["Nonword", "Target", "Condition", "\t".join(pos_tags)]))
            f.write('\n')
            for word in neighbours_distr:
                baseform, target = word.split("|")
                counts = []
                for tag in pos_tags:
                    try:
                        counts.append(str(neighbours_distr[word][tag]))
                    except KeyError:
                        counts.append("0")
                f.write('\t'.join([baseform, target, cond, '\t'.join(counts)]))
                f.write('\n')

        elif table_format == "long":
            f.write("\t".join(["Nonword", "Target", "Condition", "Category", "Count"]))
            f.write('\n')
            for word in neighbours_distr:
                baseform, target = word.split("|")
                for tag in pos_tags:
                    try:
                        f.write('\t'.join([baseform, target, cond, tag, str(neighbours_distr[word][tag])]))
                    except KeyError:
                        f.write('\t'.join([baseform, target, cond, tag, '0']))
                    f.write('\n')

        else:
            raise ValueError("unrecognized format %s!" % table_format)


########################################################################################################################


def nn_analysis(neighbours_file, tokens_file, output_file, cond="minimalist", table_format="long"):

    """
    :param neighbours_file:     the path to the file containing neighbours for the target words
    :param tokens_file:         the path to the file containing summary information about tokens, lemmas, pos tags, and
                                triphones
    :param output_file:         the path where the summary will be written to
    :param cond:                a string indicating the input used for the experiment
    :param table_format:        a string indicating how to print data to table, either 'long' or 'wide'. In the long
                                format, five columns are created, first the nonword followed by its intended pos tag,
                                then the condition, then the category, then the frequency. In the wide format, each
                                category is a different column, with each nonword-category cell indicating how many
                                neighbors tagged with the category were found in the neighborhood of the nonword.
                                An extra column indicates the condition.
    """

    ids2words = help.map_indices_to_test_words(neighbours_file)
    words2pos = help.map_words_to_pos(tokens_file)
    neighbors2pos = map_neighbours_to_pos(neighbours_file, words2pos, ids2words)
    neighbors_distr, pos_set = get_tag_distribution(neighbors2pos)
    write_neighbor_summary(output_file, neighbors_distr, pos_set, cond=cond, table_format=table_format)
