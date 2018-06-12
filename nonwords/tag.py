__author__ = 'GCassani'


def prepare_for_treetagger(source_file, dest_file, idx):

    """
    :param source_file: a text file containg the corpus with each word being represented as a triple consisting of
                        the token, the lemma, the coarse PoS (separated by a vertical bar. words are comma separated.
    :param dest_file:   the path to the file where utterances will be printed as sequences of words, space-separated
    :param idx:         the index in the triple to be printed: if 0, tokens; if 1, lemmas; if 2 PoS tags
    """

    with open(dest_file, 'w') as fw:
        with open(source_file, 'r') as rw:
            for line in rw:
                words = line.strip().split(',')
                new_line = []
                for word in words:
                    to_add = word.split('|')[idx]
                    new_line.append(to_add)
                fw.write(' '.join(new_line))
                fw.write('\n')


########################################################################################################################


def map_treetagged(treetagged_corpus, source_file, destination_file):

    """
    :param treetagged_corpus:   a corpus tagged using the tree tagger: it consists of 1 word per line, with the token,
                                the PoS tag, and the lemma tab-separated
    :param source_file:         the original corpus with comma-separated words represented as triples of token, lemma,
                                PoS
    :param destination_file:    the path where the new version of the corpus will be written to. Words will be
                                comma-separated quadruple consisting of token, lemma, original PoS, and PoS tag
                                assigned by the tree-tagger
    """

    tokens2tags = []
    with open(treetagged_corpus, 'r') as fr1:
        for line in fr1:
            token, pos, lemma = line.strip().split('\t')
            tokens2tags.append('|'.join([token, pos]))

    counter = 0
    with open(destination_file, 'w') as fw:
        with open(source_file, 'r') as fr2:
            for line in fr2:
                new_line = []
                words = line.strip().split(',')
                for word in words:
                    token1, lemma, pos = word.split('|')
                    token2, treetag = tokens2tags[counter].split('|')
                    if token1 == token2:
                        new_outcome = '|'.join([token1, lemma, pos, treetag])
                        new_line.append(new_outcome)
                        counter += 1
                    else:
                        raise ValueError("MISALIGNMENT! Token %s from utterance %s was going to be mapped to token %s" %
                                         (token1, line.strip(), token2))
                fw.write(','.join(new_line))
                fw.write('\n')
