import math
import argparse
import codecs
from collections import defaultdict
import random

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            # Open the language model file in 'r' mode with utf-8 encoding
            with codecs.open(filename, 'r', 'utf-8') as f:
                # Read the first line containing the number of unique words and total words
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # Loop through each line representing unique words and their counts
                for i in range(self.unique_words):
                    # Read index, word, and count from each line
                    index, word, count = f.readline().strip().split(' ')
                    index, count = int(index), int(count)
                    # Store index, word, and count in respective data structures
                    self.index[word] = index
                    self.word[index] = word
                    self.unigram_count[index] = count

                # Loop through each line representing bigram probabilities
                for line in f:
                    # Split each line into parts based on space delimiter
                    parts = line.strip().split(' ')
                    # Check if it's the end of file marker
                    if parts[0] == '-1':
                        # End of file, break the loop
                        break
                    # Extract index1, index2, and log probability from parts
                    index1, index2, logprob = int(parts[0]), int(parts[1]), float(parts[2])
                    # Compute and store the actual probability (exponentiate the log probability)
                    self.bigram_prob[index1][index2] = math.exp(logprob)
                return True  # Successfully processed the entire file
        except IOError:
            # Handle file not found error
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False  # Failed to process the file

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 
        ##
        if w not in self.index:
            print("Not in vocab, this is a random start.")
            w = random.choice(list(self.index.keys()))

        current_index = self.index[w]
        for i in range(n):
            if current_index not in self.bigram_prob or not self.bigram_prob[current_index]:
                # If the current word has no matching words in bigram or it is not in the model,
                # pick a random word from the vocabulary
                current_index = random.choice(list(self.index.values()))
            else:
                # Create a list of possible next words weighted by their bigram probabilities
                next_words = list(self.bigram_prob[current_index].keys())
                probabilities = []
                for nw in next_words:
                    probabilities.append(math.exp(self.bigram_prob[current_index][nw]))

                # To account for exception in question
                if all(prob == 0 for prob in probabilities):
                    current_index = random.choice(list(self.index.values()))
                
                # Choose a new word based on the bigram probabilities
                else:
                    current_index = random.choices(next_words, weights=probabilities, k=1)[0]
    
            print(self.word[current_index], end=' ')

def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()

