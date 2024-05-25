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
        n = len(self.index)
        self.total_words += 1

        if token not in self.index:
            self.index[token] = n
            self.word[n] = token

        self.unigram_count[token] += 1

        if self.last_index != -1:
            prev_word = self.word[self.last_index]
            self.bigram_count[prev_word][token] += 1

        if self.sub_two_index != -1:
            sub_two_word = self.word[self.sub_two_index]
            prev_word = self.word[self.last_index]
            self.trigram_count[sub_two_word][prev_word][token] += 1

        self.sub_two_index = self.last_index
        self.last_index = self.index[token]
        self.unique_words = len(self.index)

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []
        bigram_rows = []
        trigram_rows = []

        first_row = f"{self.unique_words} {self.total_words}"
        rows_to_print.append(first_row)

        for i in range(len(self.word)):
            word = self.word[i]
            frequency_of_word = self.unigram_count[word]
            rows_to_print.append(f"{i} {word} {frequency_of_word}")

            for second_word, bigram_occurrences in self.bigram_count[word].items():
                p = f"{math.log(bigram_occurrences / frequency_of_word):.15f}"
                bigram_rows.append(f"{self.index[word]} {self.index[second_word]} {p}")

                for third_word, trigram_occurrences in self.trigram_count[word][second_word].items():
                    p = f"{math.log(trigram_occurrences / bigram_occurrences):.15f}"
                    trigram_rows.append(f"{self.index[word]} {self.index[second_word]} {self.index[third_word]} {p}")

        rows_to_print.extend(bigram_rows)
        rows_to_print.append("-2")  # Signifies end of bigrams
        rows_to_print.extend(trigram_rows)
        rows_to_print.append("-1")  # Signifies end of file

        return rows_to_print


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
