__author__ = 'GCassani'

import helpers as help
from collections import defaultdict


def map_affix_to_idx(affix_file):

    """
    :param affix_file:      the path to the file containing the ordered list of affixes
    :return ids2affixes:    a dictionary mapping numerical indices to affixes
    """

    ids2affixes = {}
    with open(affix_file, "r") as f:
        for idx, line in enumerate(f):
            if line.strip().isupper():
                ids2affixes[idx] = line.strip()

    return ids2affixes


########################################################################################################################


def map_nonwords_to_affix(correlation_file, ids2affixes):

    """
    :param correlation_file:    the path to the file storing correlations for each nonword
    :param ids2affixes:         a dictionary mapping row indices to affixes
    :return nonwords2affixes:   a dictionary mapping each nonword to all anchors and the corresponding pairwise
                                correlation between the nonword and anchor semantic vector
    """

    ids2nonwords = help.map_indices_to_test_words(correlation_file)

    nonwords2affixes = defaultdict(dict)
    with open(correlation_file, "r") as f:
        for row_id, line in enumerate(f):
            words = line.strip().split('\t')
            for col_id, corr in enumerate(words):
                if not corr in ids2nonwords.values():
                    nonwords2affixes[ids2nonwords[col_id]][ids2affixes[row_id]] = float(corr)

    return nonwords2affixes


########################################################################################################################


def write_correlations(nonwords2affixes, ids2affixes, output_file, table_format="long", cond="minimalist"):

    """
    :param nonwords2affixes:    a dictionary mapping each nonword to all anchors and the corresponding pairwise
                                correlation between the nonword and anchor semantic vector
    :param ids2affixes:         a dictionary mapping numerical indices to affixes
    :param output_file:         the path to the file where the output is going to be written to
    :param cond:                a string indicating the input used for the experiment
    :param table_format:        a string indicating how to print data to table, either 'long' or 'wide'. In the long
                                format, five columns are created, first the nonword, then the condition, then the
                                affix, then the correlation. In the wide format, each affix is a different column,
                                with each nonword-affix cell indicates the correlation between the semantic vector for
                                the nonword and the semantic vector for the affix. An extra column indicates the
                                condition.
    """

    inflections = sorted(set(ids2affixes.values()))

    with open(output_file, "w") as f:

        if table_format == 'long':
            f.write('\t'.join(["Nonword", "Target", "Condition", "Affix", "Correlation"]))
            f.write('\n')
            for nonword in nonwords2affixes:
                baseform, tag = nonword.split("|")
                for affix in inflections:
                    corr = str(nonwords2affixes[nonword][affix])
                    f.write('\t'.join([baseform, tag, cond, affix, corr]))
                    f.write('\n')

        elif table_format == 'wide':
            f.write('\t'.join(["Nonword", "Target", "Condition", "\t".join(inflections)]))
            f.write('\n')
            for nonword in nonwords2affixes:
                baseform, tag = nonword.split("|")
                correlations = []
                for affix in inflections:
                    correlations.append(str(nonwords2affixes[nonword][affix]))
                f.write('\t'.join([baseform, tag, cond, '\t'.join(correlations)]))
                f.write('\n')

        else:
            raise ValueError("unrecognized format %s!" % table_format)


########################################################################################################################


def affix_analysis(affix_file, correlations_file, output_file, table_format="long", cond="minimalist"):

    """
    :param affix_file:          the path to the file containing the ordered list of affixes
    :param correlations_file:   the path containing the correlations between each nonword and all affixes
    :param output_file:         the path to the file where the output is going to be written to
    :param cond:                a string indicating the input used for the experiment
    :param table_format:        a string indicating how to print data to table, either 'long' or 'wide'. In the long
                                format, five columns are created, first the nonword, then the condition, then the
                                affix, then the correlation. In the wide format, each affix is a different column,
                                with each nonword-affix cell indicateing the correlation between the semantic vector
                                for the nonword and the semantic vector for the affix. An extra column indicates the
                                condition.
    """

    ids2affixes = map_affix_to_idx(affix_file)
    nonwords2affixes = map_nonwords_to_affix(correlations_file, ids2affixes)
    write_correlations(nonwords2affixes, ids2affixes, output_file, table_format=table_format, cond=cond)
