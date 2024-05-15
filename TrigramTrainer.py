#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import faulthandler
faulthandler.enable()
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs


class TrigramTrainer(object):
    """
    This class constructs a trigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file f.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = text_file.read().encode('utf-8').decode() #Maintain capitalization
        try :
            self.tokens = nltk.word_tokenize(text) 
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram and trigram counts.

        :param token: The current word to be processed.
        """
        # Increment total word count
        self.total_words += 1

        # If the token is not in the index, add it and increment the unique words count
        if token not in self.index:
            self.index[token] = self.unique_words
            self.word[self.unique_words] = token
            self.unique_words += 1

        # Increment the unigram count for this token, [key as uniquenumber, value as count]
        self.unigram_count[self.index[token]] += 1

        # If there were previous words, increment the bigram and trigram counts
        if self.last_index_1 != -1 and self.last_index_2 != -1:
            # Increment the bigram count
            self.bigram_count[(self.last_index_2, self.last_index_1)] += 1
            # Increment the trigram count
            self.trigram_count[(self.last_index_2, self.last_index_1, self.index[token])] += 1

        # Update the last index for the next token
        self.last_index_2 = self.last_index_1
        self.last_index_1 = self.index[token]


    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []

        # First line: vocabulary size and total words count
        first_row = f"{self.unique_words} {self.total_words}"
        rows_to_print.append(first_row)

        # Unigram counts
        for word_id, word in self.word.items():
            frequency_of_word = self.unigram_count[word_id]
            rows_to_print.append(f"{word_id} {word} {frequency_of_word}")

        # Bigram probabilities
        for (first_word_id, second_word_id), bigram_occurrences in self.bigram_count.items():
            p = math.log(bigram_occurrences / self.unigram_count[first_word_id])
            rows_to_print.append(f"{first_word_id} {second_word_id} {p:.15f}")

        # Trigram probabilities
        for (first_word_id, second_word_id, third_word_id), trigram_occurrences in self.trigram_count.items():
            bigram_occurrences = self.bigram_count[(first_word_id, second_word_id)]
            p = math.log(trigram_occurrences / bigram_occurrences)
            rows_to_print.append(f"{first_word_id} {second_word_id} {third_word_id} {p:.15f}")

        # End of bigrams and end of file markers
        rows_to_print.extend(["-2", "-1"])

        return rows_to_print

    def __init__(self):
        """
        Constructor. Processes the file f and builds a language model
        from it.

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # Arrays holding the unigram, bigram and trigram counts.
        self.unigram_count = defaultdict(int)
        self.bigram_count = defaultdict(lambda: defaultdict(int))
        self.trigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index_1 = -1

        # The identifier of the second previous word processed.
        self.last_index_2 = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer()

    trigram_trainer.process_files(arguments.file)

    stats = trigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


if __name__ == "__main__":
    main()
