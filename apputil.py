from collections import defaultdict
import numpy as np


class MarkovText(object):
    """Generate text from a corpus using the Markov property."""
    def __init__(self, corpus):
        """Store the corpus. The term dictionary is built later."""
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self):
        """Build a dictionary of each token and every token that follows it."""
        tokens = self.corpus.split()
        term_dict = defaultdict(list)
        for current, next_word in zip(tokens[:-1], tokens[1:]):
            term_dict[current].append(next_word)
        # zip stops one word early, so the last token never becomes a
        # key. Add it with an empty list so every token is a key.
        if tokens:
            term_dict.setdefault(tokens[-1], [])

        self.term_dict = dict(term_dict)
        return self.term_dict

    def _random_term(self):
        """Pick a random starting term from the dictionary."""
        return np.random.choice(list(self.term_dict.keys()))

    def _next_term(self, current):
        """Pick the next term at random from the words that follow it."""
        followers = self.term_dict.get(current)
        # a word with no followers (e.g., the last word of the corpus,
        # if it only appears once) is a dead end, so start fresh
        if not followers:
            return self._random_term()
        return np.random.choice(followers)

    def generate(self, seed_term=None, term_count=15):
        """Generate `term_count` terms of text, starting at `seed_term`."""
        if self.term_dict is None:
            self.get_term_dict()
        if seed_term is None:
            current = self._random_term()
        elif seed_term in self.term_dict:
            current = seed_term
        else:
            raise ValueError(f"{seed_term!r} is not in the corpus.")

        terms = [current]
        while len(terms) < term_count:
            current = self._next_term(current)
            terms.append(current)
        return " ".join(terms)
