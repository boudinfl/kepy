# -*- coding: utf-8 -*-

"""Word-based ILP keyphrase extraction methods.

"""

import re
import networkx as nx
import pulp

from itertools import combinations, permutations
from collections import defaultdict
from bisect import insort

from kepy import utils


class WordBasedILPKeyphraseExtractor(utils.LoadFile):
    """Word-based ILP keyphrase extraction model. 

    """

    # def __init__(self, input_file, use_stems=False):
    #     super(WordBasedILPKeyphraseExtractor, self).__init__(input_file, 
    #                                                          use_stems)


    def random_walk_word_scoring(self):
        """Compute a random walk ranking on the words using the power method.

        """
        G = nx.Graph()

        # loop through the sentences to build the graph
        for i, sentence in enumerate(self.sentences):
            nodes = set([])
            for words, offset in sentence.candidates:
                for w in words:
                    nodes.add(w)

            # add the missing nodes to the graph
            for node in nodes:
                if not node in G:
                    G.add_node(node)
            
            # add the edges to the graph
            for n1, n2 in combinations(nodes, 2):
                if not G.has_edge(n1, n2):
                    G.add_edge(n1, n2, weight=0)
                G[n1][n2]['weight'] += 1.0

        # return the random walk scores
        return self.normalize(nx.pagerank_scipy(G))


    def rank_candidates_with_sum(self,
                                 scores, 
                                 use_norm=False, 
                                 remove_redundancy=True):
        """Rank the candidate keyphrases using the sum of their word scores.

        Args:
            scores (dict): the word scores.
            use_norm (bool): whether candidate scores should be normalized by 
              their length, defaults to False.
            remove_redundancy (bool): whether redundant candidates should be 
              removed from the returned list, defaults to True.

        """
        scored_candidates = []
        for i, sentence in enumerate(self.sentences):
            for candidate, offset in sentence.candidates:
                score = sum([scores[u] for u in candidate])
                if use_norm:
                    score /= float(len(candidate))
                insort(scored_candidates, (score, candidate))
        scored_candidates.reverse()

        # remove redundant keyphrases
        if remove_redundancy:
            tmp_candidates = []
            for score, candidate in scored_candidates:
                is_redundant = False
                for prev_score, prev_candidate in tmp_candidates:
                    if len(set(candidate).difference(set(prev_candidate))) == 0:
                        is_redundant = True
                if not is_redundant:
                    insort(tmp_candidates, (score, candidate))

            scored_candidates = tmp_candidates
            scored_candidates.reverse()

        return scored_candidates


    def normalize(self, scores):
        """Normalize scores."""
        min_v = min(scores.values())
        max_v = max(scores.values())
        for w in scores:
            scores[w] = (scores[w]-min_v)/(max_v - min_v)
        return scores


    def rank_candidates_with_ilp(self,
                                 scores,
                                 regularization=0.05,
                                 L=10):
        """Rank the candidate keyphrases using the ILP model.

        """

        # initialize container shortcuts
        w = scores
        words = scores.keys()
        m = len(words)
        
        candidates = {}
        # f = defaultdict(int)
        for sentence in self.sentences:
            for candidate, offset in sentence.candidates:
                candidates[' '.join(candidate)] = candidate
                # f[' '.join(candidate)] += 1

        candidates = candidates.values()
        n = len(candidates)
        
        # mu = sum([len(candidates[j]) for j in range(n)])
        # mu /= float(len(candidates))
        mu = 1.0
        p = [(len(candidates[j])-mu) for j in range(n)]

        # formulation of the ILP problem 
        prob = pulp.LpProblem(self.input_file, pulp.LpMaximize)

        # initialize the word binary variables
        x = pulp.LpVariable.dicts(name='x', 
                                  indexs=range(m), 
                                  lowBound=0, 
                                  upBound=1,
                                  cat='Integer')

        # initialize the keyphrase binary variables
        c = pulp.LpVariable.dicts(name='c', 
                                  indexs=range(n), 
                                  lowBound=0, 
                                  upBound=1,
                                  cat='Integer')

        # OBJECTIVE FUNCTION
        prob += sum([w[words[i]]*x[i] for i in range(m)])\
                - regularization*sum([p[j]*c[j] for j in range(n)])

        # CONSTRAINT FOR KEYPHRASE SET SIZE
        prob += sum([ c[j] for j in range(n) ]) <= L

        # INTEGRITY CONSTRAINTS
        for i in range(m):
            for j in range(n):
                if words[i] in candidates[j]:
                    prob += c[j] <= x[i]

        for i in range(m):
            prob += sum( [c[j] for j in range(n) \
                    if words[i] in candidates[j]] ) >= x[i]

        # solving the ilp problem
        prob.solve(pulp.GLPK(msg = 0))

        # retreive the optimal subset of keyphrases
        solution = set([j for j in range(n) if c[j].varValue == 1])

        # # returns the (objective function value, solution) tuple
        return (pulp.value(prob.objective), [candidates[j] for j in solution])


