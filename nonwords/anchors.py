__author__ = 'GCassani'

import helpers as help
from collections import defaultdict


def map_anchors_to_idx(anchor_file):

    """
    :param anchor_file:     the path to the file containing the ordered list of anchor words, with their category
    :return ids2anchors:    a dictionary mapping numerical indices to anchor words
    """

    ids2anchors = {}
    with open(anchor_file, "r") as f:
        for idx, line in enumerate(f):
            if line.strip().islower():
                ids2anchors[idx] = line.strip()

    return ids2anchors


########################################################################################################################


def map_nonwords_to_anchors(correlation_file, ids2anchors):

    """
    :param correlation_file:    the path to the file storing correlations for each nonword
    :param ids2anchors:         a dictionary mapping row indices to anchor words
    :return nonwords2anchors:   a dictionary mapping each nonword to all anchors and the corresponding pairwise
                                correlation between the nonword and anchor semantic vector
    """

    ids2nonwords = help.map_indices_to_test_words(correlation_file)

    nonwords2anchors = defaultdict(dict)
    with open(correlation_file, "r") as f:
        for row_id, line in enumerate(f):
            words = line.strip().split('\t')
            for col_id, corr in enumerate(words):
                if corr not in ids2nonwords.values():
                    nonwords2anchors[ids2nonwords[col_id]][ids2anchors[row_id]] = float(corr)

    return nonwords2anchors


########################################################################################################################


def get_most_correlated_anchor(nonwords2anchors):

    """
    :param nonwords2anchors:    a dictionary mapping each nonword to all anchors and the corresponding pairwise
                                correlation between the nonword and anchor semantic vector
    :return most_correlated:    a dictionary mapping each nonword to the closest nouns and the highest noun correlation,
                                and to the closest verb and the highest verb correlation
    """

    most_correlated = defaultdict(dict)

    for nonword in nonwords2anchors:
        best_noun, best_verb = ["none", "none"]
        highest_noun, highest_verb = [0, 0]
        for anchor in nonwords2anchors[nonword]:
            base, pos = anchor.split(':')
            corr = nonwords2anchors[nonword][anchor]
            if pos == "N":
                if corr > highest_noun:
                    highest_noun = corr
                    best_noun = anchor
            else:
                if corr > highest_noun:
                    highest_verb = corr
                    best_verb = anchor
        most_correlated[nonword] = {"closest noun": best_noun,
                                    "closest verb": best_verb,
                                    "corr noun": highest_noun,
                                    "corr verb": highest_verb}

    return most_correlated


########################################################################################################################


def write_correlations(nonwords2anchors, output_file, ids2anchors, cond="minimalist", table_format="long"):

    """
    :param nonwords2anchors:    a dictionary mapping each nonword to all anchors and the corresponding pairwise
                                correlation between the nonword and anchor semantic vector
    :param output_file:         the path where the summary will be written to
    :param ids2anchors:         a dictionary mapping numerical indices to anchor words
    :param cond:                a string indicating the input used for the experiment
    :param table_format:        a string indicating how to print data to table, either 'long' or 'wide'. In the long
                                format, six columns are created, first the nonword followed by its intended pos tag,
                                then the condition, then the anchor word followed by its pos tag, then the correlation
                                between the nonword and the anchor. In the wide format, each anchor word is a different
                                column, with each nonword-anchor cell indicating the correlation between the generated
                                semantic vector for the nonword and the semantic vector for the anchor word. An extra
                                column indicates the condition.
    """

    # make sure to only use anchor words (and no headers)
    anchors = sorted(ids2anchors.values())

    with open(output_file, "w") as f:

        if table_format == "long":
            f.write('\t'.join(["Nonword", "Target", "Condition", "Anchor", "Category", "Correlation"]))
            f.write('\n')
            for nonword in nonwords2anchors:
                baseform, tag = nonword.split("|")
                for anchor in sorted(nonwords2anchors[nonword]):
                    word, pos = anchor.split(":")
                    corr = str(nonwords2anchors[nonword][anchor])
                    f.write('\t'.join([baseform, tag, cond, word, pos, corr]))
                    f.write('\n')

        elif table_format == "wide":
            f.write("\t".join(["Nonword", "Target", "Condition", "\t".join(anchors)]))
            f.write('\n')
            for nonword in nonwords2anchors:
                baseform, tag = nonword.split("|")
                correlations = []
                for anchor in sorted(nonwords2anchors[nonword]):
                    correlations.append(str(nonwords2anchors[nonword][anchor]))
                f.write('\t'.join([baseform, tag, cond, "\t".join(correlations)]))
                f.write('\n')

        else:
            raise ValueError("unrecognized format %s!" % table_format)


########################################################################################################################


def anchor_analysis(anchors_file, correlations_file, output_file, table_format="long", cond="minimalist"):

    """
    :param anchors_file:        the path to the file containing the ordered list of anchor words
    :param correlations_file:   the path containing the correlations between each nonword and all anchor words
    :param output_file:         the path to the file where the output is going to be written to
    :param cond:                a string indicating the input used for the experiment
    :param table_format:        a string indicating how to print data to table, either 'long' or 'wide'. In the long
                                format, six columns are created, first the nonword followed by its intended pos tag,
                                then the condition, then the anchor word followed by its pos tag, then the correlation
                                between the nonword and the anchor. In the wide format, each anchor word is a different
                                column, with each nonword-anchor cell indicating the correlation between the generated
                                semantic vector for the nonword and the semantic vector for the anchor word. An extra
                                column indicates the condition.
    """

    ids2anchors = map_anchors_to_idx(anchors_file)
    nonwords2anchors = map_nonwords_to_anchors(correlations_file, ids2anchors)
    write_correlations(nonwords2anchors, output_file, ids2anchors, table_format=table_format, cond=cond)
