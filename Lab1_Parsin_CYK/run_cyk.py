from cyk import CYK

# consider drawing it graphically with graphviz or networkx

from typing import List, Dict

print("Running CYK algorithm on the input sentence...")


pos = [["DET"], ["NOUN"], ["VERB"], ["PREP"], ["DET"], ["NOUN"]]
sentences: List[List[str]] = [
    ["The", "cat", "sat", "on", "the", "couch"],
    ["Time", "flies", "like", "an", "arrow", ";", "fruit", "flies", "like", "banana"],
]
sentence = sentences[0]
# defining the grammar rules
G = [
    # S is axiom
    ("S", ("NP", "VP")),
    # non terminal rules
    ("NP", ("DET", "NOUN")),
    ("PP", ("PREP", "NP")),
    ("VP", ("VERB", "PP")),
    # terminal (if using pos directly)
    ("DET", ("DET",)),
    ("VERB", ("VERB",)),
    ("NOUN", ("NOUN",)),
    ("PREP", ("PREP",)),
]
cyk = CYK(G)
trees = cyk(pos, sentence)
print(
    f"CYK algorithm completed. We have {len(trees)} parse trees. Here are the recognized parse trees:"
)
if trees:
    for i, tree in enumerate(trees, start=1):
        print(f"Tree {i}:")
        print(tree)
        print("====================")
else:
    print("Sentence not recognized")
