import argparse
import codecs
from collections import defaultdict
from operator import itemgetter
import sys
import nltk

class WordPredictor:
    """
    This class is designed to predict words using a pre-built language model.
    """
    def __init__(self, filename, stats_file=None):

        # Maps words to their unique identifiers.
        self.index = {}

        # Maps identifiers back to words.
        self.word = {}

        # Dictionary holding the count of unigrams.
        self.unigram_count = {}

        # Dictionary for storing bigram probabilities.
        self.bigram_prob = defaultdict(dict)

        # Dictionary for storing trigram probabilities using nested default dictionaries.
        nested_dict = lambda: defaultdict(nested_dict)
        self.trigram_prob = nested_dict()

        # Stores the number of distinct words in the training data.
        self.unique_words = 0

        # Stores the total number of words in the training data.
        self.total_words = 0

        # List to store user-entered words.
        self.words = []

        # Number of words to suggest to the user. Should be less than 10.
        self.num_words_to_recommend = 3  # Also known as the prediction window size.

        if not self.load_model(filename):
            # If the model cannot be read (e.g., file not found).
            print("Error: Could not read the model file. Please check the filepath.")
            sys.exit()

        if stats_file:
            self.display_stats(stats_file)
            sys.exit()

        self.greet_user()

    def load_model(self, filename):
        """
        Reads the language model file and populates the relevant data structures.

        :param filename: The name of the language model file.
        :return: True if the file is successfully processed, False otherwise.
        """
        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                for i in range(self.unique_words):
                    _, word, frequency = map(str, f.readline().strip().split(' '))
                    self.word[i], self.index[word], self.unigram_count[word] = word, i, int(frequency)
                # Reading bigram probabilities.
                for line in f:
                    if line.strip() == "-2":
                        break
                    i, j, prob = map(str, line.strip().split(' '))
                    first_word, second_word = self.word[int(i)], self.word[int(j)]
                    self.bigram_prob[first_word][second_word] = float(prob)
                # Reading trigram probabilities.
                for line in f:
                    if line.strip() == "-1":
                        break
                    i, j, k, p = map(str, line.strip().split(' '))
                    first_word, second_word, third_word = self.word[int(i)], self.word[int(j)], self.word[int(k)]
                    self.trigram_prob[first_word][second_word][third_word] = float(p)

                return True
        except IOError:
            print("Error: Cannot locate the model file {}".format(filename))
            return False

    def greet_user(self):
        print("--Word Predictor--")
        user_input = ""
        while user_input != "quit":
            print("Enter 'type' to start typing.")
            print("Enter 'quit' to exit.")
            user_input = input("Your choice: ")
            if user_input == "type":
                self.run_typing()
            else:
                if user_input != "quit":
                    print("\nPlease enter 'type' to start typing or 'quit' to exit the program.")

    def run_typing(self):
        while True:
            if self.process_typing():
                print("\nExiting typing mode.")
                break
        self.words = []  # Reset word list

    def display_console(self, words, new_word):
        """
        Outputs the current sentence to the console, displaying the typed words and the new word being formed.
        """
        print("\n")

        # Construct the full sentence with the new word being typed.
        full_sentence = ""
        for word in words:
            if word in [".", ",", "!", "?"]:
                # Attach punctuation directly to the previous word without a space.
                full_sentence = full_sentence.rstrip() + word + " "
            else:
                # Add words with a trailing space.
                full_sentence += word + " "
        
        # Add the new word being typed with an underscore.
        full_sentence += new_word + "_"
        
        # Display the constructed sentence.
        print(full_sentence)


    def fetch_n_grams(self, prev_word=None, two_words_back=None, user_input=""):
        """
        Returns bigram probabilities based on the previous word or
        trigram probabilities based on the previous two words.
        If no previous words are given, unigram counts are returned.

        Filters the results based on user_input.
        """
        if prev_word and two_words_back:
            w = self.trigram_prob.get(two_words_back, "empty")
            if w != "empty":
                w = w.get(prev_word, "empty")
                if w != "empty":
                    words_and_probs = sorted(w.items(), key=itemgetter(1), reverse=True) # Sort by descending probability
                    return [w for w, p in words_and_probs if w.startswith(user_input)] # Filter words by user_input
        elif prev_word:
            w = self.bigram_prob.get(prev_word, "empty")
            if w != "empty":
                words_and_probs = sorted(w.items(), key=itemgetter(1), reverse=True) # Sort by descending probability
                return [w for w, p in words_and_probs if w.startswith(user_input)] # Filter words by user_input
        else:
            words_and_probs = sorted(self.unigram_count.items(), key=itemgetter(1), reverse=True) # Sort by descending probability
            return [w for w, p in words_and_probs if w.startswith(user_input)] # Filter words by user_input
        return []

    def suggest_words(self, prev_word=None, two_words_back=None, user_input="", possible_words=None):
        """
        Suggests possible words using self.fetch_n_grams(). If possible_words is provided, it filters the list to remove words that don't match user_input.
        """
        if possible_words:
            return [w for w in possible_words if w.startswith(user_input)]
        return self.fetch_n_grams(prev_word, two_words_back, user_input)

    def get_letter_input(self, possible_choices):
        """
        Prompts the user to input letters or choose from recommended words.

        :param possible_choices: List of ["1", "2", ...] recommended words the user can choose from.
        """
        while True:
            letter = input("Enter a character or choose a recommended word: ")
            if len(letter) == 1 or letter in ["quit", "reset", " "] + possible_choices:
                return letter
            print("\nPlease enter a single character. You can also type 'quit' to exit, 'reset' to start over, or a space to finish typing your word.")


    def process_typing(self):
        letter = ""
        new_word = ""

        check_unigram = True  # Flag to check unigrams if possible_words is empty.

        while letter != " ":
            self.display_console(self.words, new_word)
            
            if len(self.words) == 0 and new_word == "":
                letter = self.get_letter_input([])
                if letter == "quit":
                    return True
                if letter == "reset":
                    return False
                if letter != " ":
                    new_word += letter
                continue

            if len(self.words) == 0:
                # After the first letter, get start-of-sentence probs (bigrams)
                possible_words = self.suggest_words(prev_word=".", user_input=new_word)
            elif len(self.words) == 1:
                possible_words = self.suggest_words(prev_word=str(self.words[-1]), user_input=new_word) # Get bigram probs
            else:
                possible_words = self.suggest_words(prev_word=str(self.words[-1]),
                                                    two_words_back=str(self.words[-2]),
                                                    user_input=new_word) # Get trigram probs
                possible_words += self.suggest_words(prev_word=str(self.words[-1]), user_input=new_word)

            words_to_recommend = []
            i = 0
            while len(words_to_recommend) < self.num_words_to_recommend and len(possible_words) != 0:
                # Ensure unique words are recommended
                try:
                    word = possible_words[i]
                    if word in words_to_recommend:
                        i += 1
                        continue
                    else:
                        words_to_recommend.append(word)
                except IndexError:
                    break
    
            # Check unigrams if not enough bigrams/trigrams found.
            if len(words_to_recommend) < self.num_words_to_recommend and check_unigram:
                possible_words += self.suggest_words(user_input=new_word)
                check_unigram = False
                continue

            for i, word in enumerate(words_to_recommend):
                print(f"{i + 1}: {word}")

            possible_choices = [str(i + 1) for i in range(len(words_to_recommend))]
            letter = self.get_letter_input(possible_choices)

            if letter == "quit":
                return True
            if letter == "reset":
                return False

            if letter in possible_choices:
                index = int(letter) - 1
                word = words_to_recommend[index]
                self.words.append(word)
                self.display_console(self.words, "")
                return False

            if letter != " ":
                new_word += letter

                if len(self.words) == 0:
                    possible_words = self.suggest_words(prev_word=".", user_input=new_word)
                elif len(self.words) == 1:
                    possible_words = self.suggest_words(prev_word=str(self.words[-1]), user_input=new_word)
                else:
                    possible_words = self.suggest_words(prev_word=str(self.words[-1]),
                                                        two_words_back=str(self.words[-2]),
                                                        user_input=new_word)
                    possible_words += self.suggest_words(prev_word=str(self.words[-1]), user_input=new_word)

        self.words.append(new_word)
        return False



    def display_stats(self, filename):
        """
        Reads the language model file and prints various statistics:
        (i) Keystrokes without prediction
        (ii) Keystrokes with prediction
        (iii) Keystrokes saved
        (iv) Percentage of keystrokes saved when using the model on the input file.

        :param filename: The name of the language model file or text file.
        :return: True if the file is successfully processed, False otherwise.
        """
        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                input_text = f.read()
        except UnicodeDecodeError:
            try:
                with codecs.open(filename, 'r', 'latin-1') as f:
                    input_text = f.read()
            except UnicodeDecodeError:
                print("Error: Cannot decode the input file {}".format(filename))
                return False

        total_keystrokes = len(input_text.replace(" ", ""))
        keystrokes_saved = 0
        words = nltk.word_tokenize(input_text)
        typed_words = []

        for word in words:
            typed_word = ""
            for char in word:
                typed_word += char
                if self.suggest_words(prev_word=typed_words[-1] if typed_words else None, user_input=typed_word):
                    break
            typed_words.append(typed_word)
            keystrokes_saved += len(word) - len(typed_word)

        keystrokes_saved_percentage = (keystrokes_saved / total_keystrokes) * 100
        keystrokes_with_pred = total_keystrokes - keystrokes_saved
        print("\nTotal keystrokes without prediction: {}".format(total_keystrokes))
        print("Total keystrokes with prediction: {}".format(keystrokes_with_pred))
        print("Keystrokes saved: {}".format(keystrokes_saved))
        print("Percentage of keystrokes saved: {:.2f}%".format(keystrokes_saved_percentage))

        return True


def main():
    parser = argparse.ArgumentParser(description="Predict words based on entered text.")
    parser.add_argument('--file', '-f', type=str,  required=True, help='file containing the language model')
    parser.add_argument("--stats", metavar="file", help="display statistics about the language model")
    args = parser.parse_args()
    wp = WordPredictor(args.file, args.stats)

if __name__ == "__main__":
    main()
