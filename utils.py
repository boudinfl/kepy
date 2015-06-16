# -*- coding: utf-8 -*-

"""Base structures and functions for the kepy module.

"""

import os
import re
import codecs

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def str2tuple(s, sep=u'/'):
    """Decomposing a string into a word, POS tuple.

    Args:
        s (str): the string to split into a word, POS tuple.
        sep (str): the separator used for splitting the string, defaults to /

    Returns:
        (token, POS) (str, str) tuple

    """
    m = re.match(u"^(.+)"+sep+u"(.+)$", s)
    return (m.group(1), m.group(2).upper())


class Sentence:
    """The sentence data structure.

    Args: 
        words (list of str): the list of word tokens.
        POS (list of str): the list of word Part-Of-Speeches.
    """
    def __init__(self, words, POS):

        self.words = words
        """ tokens as a list. """

        self.POS = POS
        """ Part-Of-Speeches as a list. """

        self.stems = []
        """ stems as a list. """

        self.candidates = []
        """ keyphrase candidates of the sentence. """

        self.length = len(words)
        """ length of the sentence. """


class LoadFile(object):
    """Objects which inherit from this class have read file functions.

    """

    def __init__(self, input_file, use_stems=False):
        """
        Args:
            input_file (str): the path of the input file.
            use_stems (bool): whether stems should be used instead of words,
              defaults to False.

        """
        self.input_file = input_file
        self.sentences = []
        self.use_stems = use_stems


    def read_document(self):
        """Read the input file.

        Load the input file and populate the sentence list. Input file is 
        expected to be in one tokenized, POStagged sentence per line format.
        """
        with codecs.open(self.input_file, 'r', 'utf-8') as f:
            for line in f:
                tokens = line.strip().lower().split(' ')
                if tokens:
                    tokens = [str2tuple(u) for u in tokens]

                    self.sentences.append(
                        Sentence(words=[u[0] for u in tokens], 
                                 POS=[u[1] for u in tokens])
                        )

                    if self.use_stems:
                        self.sentences[-1].stems = \
                          [SnowballStemmer('porter').stem(u[0]) for u in tokens]


    def select_candidates(self):
        """Extract the keyphrase candidates from the sentences. Keyphrases 
           candidates are the longest sequences of Nouns and Adjectives.

        """
        offset = 0
        for i, sentence in enumerate(self.sentences):
            candidate = []
            for j in range(sentence.length):

                # adds offset to the current_candidate
                if sentence.POS[j][:2] in ['JJ', 'NN']:
                    candidate.append(j)
                    if j < (sentence.length - 1):
                        continue

                # if there is a candidate in the container
                if candidate:

                    # test for special characters
                    if [u for u in candidate \
                        if not re.search(u'(?u)^[a-zA-Z0-9\s\-\.\/]+$',
                                         sentence.words[u]) ]:
                        break

                    # test for stopwords
                    if [u for u in candidate \
                        if sentence.words[u] in stopwords.words('english')]:
                        break

                    # test for length
                    if not [u for u in candidate if len(sentence.words[u]) > 2]:
                        break

                    # test for adjectives only
                    if not [u for u in candidate if sentence.POS[u][:2]=='NN']:
                        break

                    if self.use_stems:
                        self.sentences[i].candidates.append(
                            ([sentence.stems[u] for u in candidate],
                             offset+j))
                    else:
                        self.sentences[i].candidates.append(
                            ([sentence.words[u] for u in candidate],
                             offset+j))

                # flush the current candidate
                candidate = []

            offset += sentence.length

                    

















