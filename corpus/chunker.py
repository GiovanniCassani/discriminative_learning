__author__ = 'GCassani'

import os
import json

from corpus.chunk.words import single_words

"""Function to recode an input corpus into a list of units (single words, multiword expressions, prosodic chunks,
   or full utterances depending on which option is specified"""


def chunk_corpus(corpus, granularity):

    """
    :param corpus:      the corpus to chunk, consisting of two lists of lists, the first containing lists of tokens
                        (a list=an utterance), the second containing lists of lemmas and part-of-speech tags,
                        corresponding to the tokens in the adjacent list.
    :param granularity: the granularity of the desired chunks (single words, prosodic chunks, multi-word expressions,
                        or whole utterances (the corpus is returned as is, assuming it already consists of an utterance
                        per line
    :return:            the corpus chunked to the desired granularity and the path of the file where it has been saved
    """

    chunked_corpus = [[], []]
    if granularity == 'words':
        chunked_corpus = single_words(corpus)
    elif granularity == 'prosodic':
        pass
    elif granularity == 'multiword':
        pass
    else:
        chunked_corpus = json.load(open(corpus, 'r+'))

    basename, ext = os.path.splitext(corpus)
    output_file = "".join(["_".join([basename, granularity]), ext])

    json.dump(chunked_corpus, open(output_file, 'w'))

    return chunked_corpus, output_file
