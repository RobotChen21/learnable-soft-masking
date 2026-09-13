import numpy as np
from .spanningtrees.graph import Graph
from .spanningtrees.kbest import KBest


def draw_tree(score_matrix, length):
    score_matrix = -score_matrix  # API finds minimum trees.
    result = []
    for scores, dialogue_length in zip(score_matrix, length):
        scores = scores[:, :dialogue_length + 1][:dialogue_length + 1, :]
        scores = np.transpose(scores)
        scores = np.nan_to_num(scores, nan=-float("inf"))
        graph = Graph.build(scores)
        trees = []
        tree_scores = []
        for index, tree in enumerate(KBest(graph).kbest()):
            current_score = sum(
                scores[head][tail]
                for tail, head in enumerate(tree.to_array().tolist())
                if tail != 0
            )
            if index == 1:
                break
            if not tree_scores or abs(tree_scores[-1] - current_score) < 0.1:
                trees.append(tree.to_array().tolist())
            tree_scores.append(current_score)
        result.append(trees)
    return result
