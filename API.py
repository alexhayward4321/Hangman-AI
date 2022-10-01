# %%
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:39:58 2021

@author: alexh
"""
import pickle
import json
import re
import requests
import random
import string
import secrets
import time
import pandas as pd
import collections
import numpy as np
from datetime import datetime
try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

# I added this bit

from sklearn.model_selection import train_test_split

#######################


# %%

class HangmanAPI(object):
    def __init__(self):

        # Attributes included in original base solution
        self.guessed_letters = []
        full_dictionary_location = "C:\\Users\\alexh\\OneDrive\\Documents\\Coding\\Python\\Personal Projects\\Hangman\\dictionaries\\words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted =\
            collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []

        # ADDED ATTRIBUTES
        # For my self-made start_game function to run
        self.tries_remains = 7  # or 6
        self.training_dictionary, self.validation_dictionary =\
            train_test_split(self.full_dictionary, random_state=0)

        # Variables to prevent unnecessary and expensive repetition within guess function
        self.prev_word = ""
        # counter object with counts of letters that appear in regex matches
        # to the letters already correctly guessed
        self.letter_preference = collections.Counter()
        self.whole_game_regexes = []

        # Reads in dataframe used to keep track of common regular expressions
        self.regex_df_path = "C:\\Users\\alexh\\OneDrive\\Documents\\Coding\\Python\\Personal Projects\\Hangman\\regexes.pkl"
        self.regex_df = self.read_regex_df(self.regex_df_path)

        #######################

    def guess(self, word):
        """
        Method that takes the hangman word and returns a letter to guess
        """
        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_", ".")

        previous_word = self.prev_word
        self.prev_word = clean_word

        # Variables for conditional to select appropriate algorithm
        num_guessed_letters = len(clean_word.replace(".", ""))
        proportion_guessed_letters = num_guessed_letters / len(clean_word)
        len_current_dictionary = len(self.current_dictionary)

        if (len_current_dictionary < self.weights[0] and
            proportion_guessed_letters < self.weights[1])\
                or len(clean_word) <= 3:

            guess_letter = self.algorithm1(clean_word)
            return guess_letter

        else:
            guess_letter = self.algorithm2(clean_word,
                                           previous_word == clean_word)
            return guess_letter

    # Baseline method provided

    def algorithm1(self, clean_word):

        new_dictionary = []
        for dict_word in self.current_dictionary:
            if len(dict_word) != len(clean_word):
                continue
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)

        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        # count occurrence of all characters in possible word matches
        c = collections.Counter("".join(new_dictionary)).most_common()

        # Choose the guess letter with the highest count, if all letters have been guessed,
        # use the total training dictionary distribution once more
        guess_letter = self.__counter_to_guess_letter(c)

        return guess_letter

    def algorithm2(self, clean_word, repeat):
        """
        Executes a more sophisticated second algorithm as discussed in the
        README.md. Method relies on producing all possible regular
        expressions from correctly guessed letters, performs biased count
        of matching words in dictionary that fill the gaps and returns
        highest counting unguessed letter.
        """

        # checks to see if the hangman word has changed since we last saw it, if not, we don't
        # have to recalculate all of the regular expression matches in the dictionary
        if not repeat:
            # Produce a series of regular expressions based on the hangman word
            regexes = self.produce_regexes(clean_word)

            # counter object which tabulates letters with strongest matches
            letter_preference = collections.Counter()

            for expression in regexes:
                # add regular expression to list of regular expressions for the whole game
                self.whole_game_regexes.append(expression)
                # Check reference dataframe to see if counter has been pre-calculated
                if expression in self.regex_df.index:
                    counter = self.regex_df.loc[expression,
                                                "letter_counter"]

                # If not, calculate the counter yourself
                else:
                    # Find total number of letters appearing in all dictionary words
                    # fitting a regular expression
                    counter = self.get_counter(expression)
                    # Update list with regular expressions to later add to the DataFrame
                    self.new_regexes["Regex"].append(expression)
                    self.new_regexes["letter_counter"].append(counter)
                    self.new_regexes["counter"].append(0)

                # Figure out the weighting to apply to the collections.Counter
                # object for the expression based on the number of known letters in the
                # regular expression

                # Find weighting
                num_letters = sum(c.isalpha() for c in expression)
                num_letters = 6 if num_letters > 6 else num_letters
                weighting = self.weights[num_letters + 1]

                # Take into account weighting
                temp_counter = collections.Counter()
                for key, value in counter.items():
                    temp_counter[key] = value * weighting
                letter_preference.update(temp_counter)

            self.letter_preference = letter_preference.most_common()

        # Choose the guess letter with the highest count, if all letters have been guessed,
        # use the total training dictionary distribution once more
        guess_letter = self.__counter_to_guess_letter(self.letter_preference)

        return guess_letter

    def produce_regexes(self, clean_word):
        """
        Calculates all possible sub-regular expressions from a parent regular expression.
        (E.g. for word 'le...a ' all possible regular expressions would be:
        '^le..', '^le.', '.a$', '...a$', '..a$', '^le...a$', 'e...a$', '^le...', 'le...a '
        catching unwanted letters in words that match the regular expression,
        """

        len_word = len(clean_word)
        regexes = []
        for i in range(len_word):
            for j in range(i, len_word + 1):
                regex = clean_word[i:j]
                if "." in regex:
                    if i == 0:
                        if j == len_word:
                            regexes.append("^" + clean_word[i:j] + "$")
                        else:
                            regexes.append("^" + clean_word[i:j])
                    if i > 0:
                        if j == len_word:
                            regexes.append(clean_word[i:j] + "$")
                        else:
                            regexes.append(clean_word[i:j])

        # Remove regular expressions that only have one letter or less (they're not useful)
        regexes = [item for item in regexes if len(item.replace(".", "")) > 1]
        # Removing potential duplicates
        regexes = list(set(regexes))
        return regexes

    # Returns collections.Counter object for total number of letters appearing in all
    # dictionary words fitting a regular expression. Only guesses letters that occupy
    # the position the "." would take.

    def get_counter(self, expression):

        # Find positions of unknown letters in the expression
        cleaned_expression = expression.replace("^", "").replace("$", "")
        indexes = [i for i, ltr in enumerate(cleaned_expression) if ltr == "."]

        # Checking if the word matches the regular expression and taking
        # the letter values in and immediately around where the regular
        # expression matches
        all_matching_letters = []
        for dict_word in self.full_dictionary:
            re_obj = re.search(expression, dict_word)
            if re_obj:
                a, b = re_obj.span()
                matching_letters = [dict_word[a + idx] for idx in indexes]
                all_matching_letters.append("".join(matching_letters))

        LETTERS = "".join(all_matching_letters)
        return collections.Counter(LETTERS)

    def __counter_to_guess_letter(self, counter):
        guess_letter = '!'

        for letter, instance_count in counter:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break

        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break

        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def read_regex_df(self, regex_df_path):
        regex_df = pd.read_pickle(regex_df_path)
        clean_regex_df_index = regex_df.index.drop_duplicates()
        regex_df = regex_df.loc[clean_regex_df_index]
        return regex_df

    def start_game(self, weights, verbose=True, practice=1):

        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []

        # I added this bit
        self.current_dictionary = self.training_dictionary

        self.tries_remains = 7
        self.whole_game_regexes = []
        self.weights = weights

        # Dictionary of newly encountered regex to store for future speed
        self.new_regexes = {"Regex": [],
                            "letter_counter": [], "counter": []}

        # Chooses a word from the validation dictionary and creates a hangman equivalent
        # to show the user
        word = random.choice(self.validation_dictionary)
        hangman_word = "_ " * len(word)

        if verbose:
            print("Successfully start a new game! # of tries remaining: {0}.\
                  Word: {1}.".format(self.tries_remains, word))

        while self.tries_remains > 0:
            # get guessed letter from guess method
            guess_letter = self.guess(hangman_word)

            # append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)

            if verbose:
                print("Guessing letter: {0}".format(guess_letter))

            # Check if your guessed letter is in the word, if not then decrement tries remaining
            if guess_letter in word:
                indexes = [i for i, ltr in enumerate(
                    word) if ltr == guess_letter]
                hangman_word_list = list(hangman_word)
                for idx in indexes:
                    hangman_word_list[idx * 2] = guess_letter
                hangman_word = "".join(hangman_word_list)
                if verbose:
                    print(f"Correct guess!   Updated word: {hangman_word}\
                          # tries remaining: {self.tries_remains}")

                self.wrong_guess = False

            else:
                self.tries_remains -= 1
                self.wrong_guess = True
                if verbose:
                    print(f"Incorrect guess! Updated word: {hangman_word}\
                          # tries remaining: {self.tries_remains}")

            game_end = False
            if self.tries_remains >= 0 and "_" not in hangman_word:
                if verbose:
                    print("Successfully finished game!")

                game_end = True
                game_return_val = True
            elif self.tries_remains == 0:
                if verbose:
                    print(
                        f"Failed game. You ran out of lives. \n The word you had to guess: {word}")
                game_end = True
                game_return_val = False
            if game_end:
                # Update counts of appearances of regular expressions

                # Removing duplicates
                duplicates_removed = set(self.whole_game_regexes)
                for regex in duplicates_removed:
                    if self.regex_df.loc[regex, "counter"] == 0:
                        self.regex_df.loc[regex, "counter"] = 1
                    else:
                        self.regex_df.loc[regex,
                                          "counter"] = self.regex_df.loc[regex, "counter"] + 1

                # Write DataFrame of all known pickles to external file for safekeeping
                self.regex_df.to_pickle(self.regex_df_path)

                return game_return_val

        return None


# =============================================================================
#     Below are newly defined functions created in order to try and find the optimal weights for the above guess method
# =============================================================================

    def cost_function(self, weights):
        """
        A cost function based on the accuracy of my guess method for a particular set of 'weights' - which are key values that come up in that method that could be varied to produce different results. The cost_function is literally (1 - accuracy) with accuracy measured as fraction of correctly guessed words within the permitted tries_remains. Unfortunately even after 100 repeats for the same weights I found the cost calculated varied over a range of 0.08, which wasn't good enough for my gradient descent method below.
        """

        success_counter = 0
        for i in range(100):

            if self.start_game(weights, verbose=False):
                success_counter += 1
                print(f"Game {i}: SUCCESS")
            else:
                print(f"Game {i}: FAILURE")
        return 1 - success_counter / 100

    def gradient_descent(self, learning_rate):
        """ 
            This very rudimentary gradient descent function is something I designed because I didn't really quite know what machine learning tools from what package to use to apply to this problem. Despite my best efforts to reduce the cost function would just take too long, and if the repeats were made shorter then there would be too much error in the accuracy.
        """

        # Initialising starting weights and costs,
        #weights = np.random.randint(0, 1000, size=8)
        #weights = input("Enter your eight selected weights: ")
        weights = [50, 0.3, 1, 2, 4, 8, 12, 20]
        old_cost = 1
        consecutive_failures = 0

        print_time()

        # Storage for the different weights and learning costs
        cost_to_weights = dict()

        while consecutive_failures < 5:

            """PICK UP HERE"""

            for i, weight in enumerate(weights):
                temp = weights.copy()
                temp[i] = weight * (1 - learning_rate)
                cost1 = self.cost_function(temp)
                cost_to_weights[cost1] = temp
                print(temp)
                print(cost1)

                print_time()

                temp = weights.copy()
                temp[i] = weight * (1 + learning_rate)
                cost2 = self.cost_function(temp)
                cost_to_weights[cost2] = temp
                print(temp)
                print(cost2)

                print_time()

            if min(cost_to_weights) + 0.025 < old_cost:
                print("IMPROVING WEIGHTS")
                old_cost = min(cost_to_weights)
                weights = cost_to_weights[old_cost]
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            learning_rate *= 0.9

        return weights

    def trees(self):
        """Generates random weights, then finds the accuracy of the model for these random weights. Plan was then to use ML algorithm to figure out relationship between the weights as input and the accuracy as a label. Unfortunately, there were similar problems with time and accuracy of the cost function to be realistic to implement on my laptop."""
        cost_to_weights = dict()
        i = 0
        while i < 100:
            weights = []
            weights.append(np.random.lognormal(6, 3))
            weights.append(np.random.randint(0, 7))
            for i in range(2, 8):
                weights.append(np.random.randint(0, 100))
            cost = self.cost_function(weights)
            cost_to_weights[cost] = weights
            print(weights, ": ", cost)
        return cost_to_weights


# %%
api = HangmanAPI()

# %%

api.start_game([500, 0.35, 1, 2, 4, 8, 12, 20])

# %%

print_time()

print(api.cost_function([50, 0.3, 1, 2, 4, 8, 12, 20]))

print_time()

# %%


final_weights = api.gradient_descent(0.3)

# I used 0.1 initially, then thought that would take a really, REALLY long time, then I switched it to 0.5 just so I could get a relatively quick answer so I could see if it what I had written would actually work. Then I moved to 0.3, that's actually a reasonable number because 1-0.3 = 0.7 and 1+0.3 = 1.3 and 0.7/1.3 ~ 1/2

# final weights of first attempt [5333, 1694, 6156, 2324, 2424, 4986, 3315, 4017]

#

# %%

""" used during debugging to reset the dataframe storing the regular expressions"""

# def reset_DataFrames():

#     df = pd.DataFrame(columns=["Regex", "letter_counter", "counter"])
#     df = df.set_index("Regex")
#     df.to_pickle("C:/Users/alexh/OneDrive/Documents/Coding/Python/Personal Projects/Hangman/regexes.pkl")

# reset_DataFrames()

# %%

# So I can check the regular expression file


with open("C:/Users/alexh/OneDrive/Documents/Coding/Python/Personal Projects/Hangman/regexes.pkl", 'rb') as pickle_file:
    content = pickle.load(pickle_file)


# %%

"""
    Under this weighting: [50, 0.3, 1, 2, 4, 8, 12, 20]
    the cost function with 100 repeats gave the following accuracies:
        [0.42, 0.5, 0.43, 0.49]
    Even with 100 repeats there is still huge variation within the calculated accuracy, making my gradient descent function very flawed, unfortunately




"""

# %%


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Time =", current_time)


# %%
print(api.cost_function([500, 3, 1, 2, 4, 8, 16, 32]))

# %%

final_weights = api.gradient_descent(0.3)