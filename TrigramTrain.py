# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.

Modified by Almir Aljic & Alexander Jakobsen in December 2019. Original version, named "BigramTrainer", has been extended and thus
re-named to "TrigramTrainer".
"""


class TrigramTrainer:
    """
    This class constructs a trigram language model from a corpus.
    """

    def __init__(self):
        """
        Constructor. Initializes the necessary data structures.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # The unigram counts.
        self.unigram_count = defaultdict(int)

        # The bigram counts.
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The trigram counts.
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # The identifier of the previous word processed.
        self.last_index = -1

        # The identifier of the word processed 2 iterations ago.
        self.sub_two_index = -1

        # Number of unique words in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

    def process_files(self, file_path):
        """
        Processes the file at the given path.
        """
        with codecs.open(file_path, 'r', 'utf-8') as text_file:
            text = text_file.read()

        try:
            self.tokens = nltk.word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)

        for token in self.tokens:
            self.process_token(token)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram,
        bigram and trigram counts.

        :param token: The current word to be processed.
        """
        current_index = len(self.index)
        self.total_words += 1

        if token not in self.index:
            self.index[token] = current_index
            self.word[current_index] = token

        self.unigram_count[token] += 1

        if self.last_index != -1:
            last_word = self.word[self.last_index]
            self.bigram_count[last_word][token] += 1

        if self.sub_two_index != -1:
            sub_two_word = self.word[self.sub_two_index]
            last_word = self.word[self.last_index]
            self.trigram_count[sub_two_word][last_word][token] += 1

        self.sub_two_index, self.last_index = self.last_index, self.index[token]
        self.unique_words = len(self.index)

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        output_rows = []
        bigram_entries = []
        trigram_entries = []

        header = f"{self.unique_words} {self.total_words}"
        output_rows.append(header)

        vocabulary_size = len(self.index)

        for word_id in range(len(self.word)):
            current_word = self.word[word_id]
            unigram_freq = self.unigram_count[current_word]
            output_rows.append(f"{word_id} {current_word} {unigram_freq}")

            for next_word, bigram_count in self.bigram_count[current_word].items():
                # Laplace smoothing for bigrams
                bigram_prob = math.log((bigram_count + 1) / (unigram_freq + vocabulary_size))
                bigram_entries.append(f"{self.index[current_word]} {self.index[next_word]} {bigram_prob:.15f}")

                for following_word, trigram_count in self.trigram_count[current_word][next_word].items():
                    bigram_freq = self.bigram_count[current_word][next_word]
                    # Laplace smoothing for trigrams
                    trigram_prob = math.log((trigram_count + 1) / (bigram_freq + vocabulary_size))
                    trigram_entries.append(f"{self.index[current_word]} {self.index[next_word]} {self.index[following_word]} {trigram_prob:.15f}")

        output_rows.extend(bigram_entries)
        output_rows.append("-2")  # End of bigrams marker
        output_rows.extend(trigram_entries)
        output_rows.append("-1")  # End of file marker

        return output_rows



def main():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str, required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer()
    trigram_trainer.process_files(arguments.file)

    stats = trigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8') as f:
            for row in stats:
                f.write(row + '\n')
    else:
        for row in stats:
            print(row)


if __name__ == "__main__":
    main()
