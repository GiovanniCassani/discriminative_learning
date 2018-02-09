__author__ = 'GCassani'

"""Function to build a toy scenario for testing phonetic bootstrapping functions (can be called from command line)"""

import os
import json
from celex.utilities.add_words import hardcode_words
from celex.get import initialize_celex_dict, print_celex_dict


celex_dict = initialize_celex_dict()

celex_dict = hardcode_words(celex_dict, "I", '01', '01', "'2", '2', 'X', "i", "'2", ['i'], 'O')
celex_dict = hardcode_words(celex_dict, "want", '02', '02', "'wQnt", 'Q', 'X', "want", "'wQnt", ['want'], 'V')
celex_dict = hardcode_words(celex_dict, "an", '03', '03', "'{n", '{', 'X', "a", "'{n", ['a'], 'D')
celex_dict = hardcode_words(celex_dict, "ice-cream", '04', '04', "2s-'krim", 'i', 'X', "ice-cream",
                                "2s-'krim", ['ice', 'cream'], 'N')
celex_dict = hardcode_words(celex_dict, "you", '05', '05', "'ju", 'u', 'X', "you", "'ju", ['you'], 'O')
celex_dict = hardcode_words(celex_dict, "report", '06', '06', "rI-'p$t", '$', 'X', "report", "rI-'p$t",
                                ['report'], 'V')
celex_dict = hardcode_words(celex_dict, "report", '07', '07', "'rI-p$t", 'I', 'X', "report", "'rI-p$t",
                                ['report'], 'N')
celex_dict = hardcode_words(celex_dict, "the", '08', '08', "'Di", 'i', 'X', "the", "'Di", ['the'], 'D')
celex_dict = hardcode_words(celex_dict, "crime", '09', '09', "'kr2m", '2', 'X', "crime", "'kr2m", ['crime'], 'N')
celex_dict = hardcode_words(celex_dict, "that", '10', '10', "'D{t", '{', 'X', "that", "'D{t", ['that'], 'D')
celex_dict = hardcode_words(celex_dict, "idea", '11', '11', "2-'d7", '7', 'X', "idea", "2-'d7", ['idea'], 'N')
celex_dict = hardcode_words(celex_dict, "is", '12', '12', "'Iz", 'I', 'X', "be", "'bi", ['be'], 'V')
celex_dict = hardcode_words(celex_dict, "gold", '13', '13', "'g5ld", '5', 'X', "gold", "'g5ld", ['gold'], 'A')
celex_dict = hardcode_words(celex_dict, "gold", '14', '14', "'g5ld", '5', 'X', "gold", "'g5ld", ['gold'], 'N')
celex_dict = hardcode_words(celex_dict, "are", '15', '12', "'#R", '#', 'X', "be", "'bi", ['be'], 'V')
celex_dict = hardcode_words(celex_dict, "like", '16', '15', "'l2k", '2', 'X', "like", "'l2k", ['like'], 'V')
celex_dict = hardcode_words(celex_dict, "like", '17', '16', "'l2k", '2', 'X', "like", "'l2k", ['like'], 'P')
celex_dict = hardcode_words(celex_dict, "old", '18', '17', "'5ld", '5', 'X', "old", "'5ld", ['old'], 'A')

corpus_utterances = [[['i', 'want', 'an', 'ice-cream'],
                      ['you', 'report', 'the', 'crime'],
                      ['that', 'idea', 'is', 'gold']],
                     [['i|pro', 'want|v', 'a|det', 'ice-cream|n'],
                      ['you|pro', 'report|v', 'the|det', 'crime|n'],
                      ['that|det', 'idea|n', 'be|v', 'gold|adj']]]

toy_data_folder = "/home/cassani/phoneticBootstrapping/toyData/"
if not os.path.exists(toy_data_folder):
    os.makedirs(toy_data_folder)
os.chdir(toy_data_folder)

print_celex_dict(celex_dict, "toyCelex.json")
with open("toyCorpus.json", 'a+') as o_f:
    json.dump(corpus_utterances, o_f)

# new-ambiguous:
# - like~V
# - like~P
# - gold~N
#
# new-unambiguous:
# - report~N
# - old~A
#
# known-ambiguous:
# - gold~A
#
# known-unambiguous:
# - I~O
# - want-V
# - a~D
# - ice-cream~N
# - you~O
# - report~V
# - the~D
# - crime~N
# - that~D
# - idea~N
# - be~V
