__author__ = 'GCassani'


def store_apostrophe_mapping(in_file):

    """
    :param in_file:     a text file with two space separated columns: to the left original forms with apostrophes, to
                        the right the corresponding new form
    :return mapping:    a dictionary mapping old forms to corresponding new ones
    """

    mapping = {}
    with open(in_file, 'r') as f:
        for line in f:
            old, new = line.strip().split(' ')
            mapping[old] = new

    return mapping


########################################################################################################################


def substitute_apostrophes(source_file, dest_file, mapping):

    """
    :param source_file: the file containing the corpus with the apostrophes to be removed
    :param dest_file:   the file where to write the new version of corpus
    :param mapping:     the dictionary mapping old forms to corresponding new forms
    """

    with open(dest_file, 'w') as fw:
        with open(source_file, 'r') as fr:
            for line in fr:
                new_line = line.strip()
                substitutions = {s for s in set(mapping.keys()) if s in new_line}
                if substitutions:
                    for substitution in substitutions:
                        new = mapping[substitution]
                        new_line = new_line.replace(substitution, new)
                fw.write(new_line)
                fw.write('\n')
