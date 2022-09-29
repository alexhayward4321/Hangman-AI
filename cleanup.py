#%%
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
        full_dictionary_location = "C:/Users/alexh/OneDrive/Documents/Coding/Python/Careers/Trexquant application/dictionaries/words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted =\
            collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []

        # ADDED ATTRIBUTES
        # For my self-made start_game function to run
        self.tries_remains = 7
        self.training_dictionary, self.validation_dictionary =\
            train_test_split(self.full_dictionary, random_state=0)

        # Variables to prevent unnecessary and expensive repetition within guess function
        self.first_time = True
        self.wrong_guess = False
        self.total_count = collections.Counter()
        self.whole_game_regexes = []

        # Reads in dataframe used to keep track of common regular expressions
        self.regex_df_loc = "C:/Users/alexh/OneDrive/Documents/Coding/Python/Careers/Trexquant application/regexes.pkl"
        self.regex_df = pd.read_pickle(self.regex_df_loc)
        clean_regex_df_index = self.regex_df.index.drop_duplicates()
        self.regex_df = self.regex_df.loc[clean_regex_df_index]

        #######################

    def guess(self, word, weights):
        """
        Example method works quite well until the word you need to guess doesn't exist in
        the training dictionary and you start guessing letters from a very small set of 
        words that fit your regular expression from the training dictionary. A better method is
        needed that takes into account common letter combinations in the English language, but
        so long as you have a large number of words you can take letters to guess from then the
        example method can still be leveraged advantageously.
        """

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_", ".")

        # find length of passed word
        len_word = len(clean_word)

        # Calculating a check for the conditional to see if we want to use the example method
        num_guessed_letters = len(clean_word.replace(".", ""))
        proportion_guessed_letters = num_guessed_letters / len_word

        # Another check for the conditional to see if we want to use the example method
        len_current_dictionary = len(self.current_dictionary)

        # Use the example method under the following conditions
        if (len_current_dictionary > weights[0] and proportion_guessed_letters < weights[1])\
                or len_word <= 3:

            # Initialize new possible words dictionary to empty
            new_dictionary = []

            # iterate through all of the words in the previous plausible dictionary
            for dict_word in self.current_dictionary:
                # continue if the word is not of the appropriate length
                if len(dict_word) != len_word:
                    continue

                # if dictionary word is a possible match then add it to the current dictionary
                if re.match(clean_word, dict_word):
                    new_dictionary.append(dict_word)

            # overwrite old possible words dictionary with updated version
            self.current_dictionary = new_dictionary

            # count occurrence of all characters in possible word matches
            full_dict_string = "".join(new_dictionary)

            c = collections.Counter(full_dict_string)
            sorted_letter_count = c.most_common()

            guess_letter = '!'

            # return most frequently occurring letter in all possible words that hasn't been guessed yet
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break

            # if no word matches in training dictionary, default back to ordering of full dictionary
            if guess_letter == '!':
                sorted_letter_count = self.full_dictionary_common_letter_sorted
                for letter, instance_count in sorted_letter_count:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break

            return guess_letter

        else:

            """
                This method relies upon looking at the known letters surrounding an unknown letter, tabulating a number of regular expressions for different combinations of those letters and weighting how significantly the unguessed letters appear in each regular expression based on the different lengths of those regular expressions sizes as well as the number of other unknown letters in the environment. 
          """

            # Checks if there are only three or fewer gaps remaining, in which case we
            # won't just consider regular expressions as described above but all
            # expressions formed surrounding unknown letters. (E.g. for word "a.lia." we
            # consider "a.l" as a regular expression) to prevent catching unwanted letters
            # in words that match the regular expression, we'll only guess letters that
            # occupy the position the "." would take. This in theory would be the optimal
            # approach anyway, given the correct weighting to regular expressions of
            # different lengths, but it is quite computationally expensive and time
            # consuming.

            # and clean_word.count(".") <= 3:
            if not self.wrong_guess or self.first_time:
                self.first_time = False
                """
                    THIS COULD BE ANOTHER WEIGHTING TO EXPERIMENT WITH ^
                    Could also consider either adding it as a permanent feature
                    or removing it altogether to see what effect that has.
                """
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

                # Checks to see if a particular conditional is entered
                entered = False

                # Remove regular expressions that only have one letter or less
                temp1 = [item for item in regexes if len(
                    item.replace(".", "")) > 1]
                # Removing potential duplicates
                temp2 = set(temp1)
                regexes = list(temp2)
                breakpoint()

                alt_total_count = collections.Counter()
                alt_pandas_list = []

                for expression in regexes:

                    # Keeps track of all regular expressions that appear in our game, so that we can
                    # later count their appearances over different words to see which ones are
                    # most important to keep in our reference data frame
                    self.whole_game_regexes.append(expression)

                    # Check if we already have the letter counts for a particular regex
                    # If so, get those values, if not, then calculate them yourself

                    if expression in self.regex_df.index:

                        alt_counter = self.regex_df.loc[expression,
                                                        "letter_counter"]

                    else:

                        entered = True

                        cleaned_expression = expression.replace(
                            "^", "").replace("$", "")
                        # Find positions of unknown letters in the expression
                        indexes = [i for i, ltr in enumerate(
                            cleaned_expression) if ltr == "."]

                        # All letters that correspond with "." in words which match with a
                        # regular expression
                        all_matching_letters = []

                        for word1 in self.training_dictionary:

                            # some smaller regular expressions are going to match just
                            # everything to add tens of thousands more words to our new
                            # dictionary doesn't really give us more information

                            # Checking if the word matches the regular expression and taking
                            # the letter values in and immediately around where the regular
                            # expression matches
                            re_obj = re.search(expression, word1)
                            if re_obj:
                                a, b = re_obj.span()
                                matching_letters = [word1[a + idx]
                                                    for idx in indexes]
                                all_matching_letters.append(
                                    "".join(matching_letters))

                        LETTERS = "".join(all_matching_letters)
                        alt_counter = collections.Counter(LETTERS)

                        # Update list with regular expressions to later add to the DataFrame
                        alt_pandas_list.append({"Regex": expression,
                                                "letter_counter": alt_counter, "counter": 0})

                    # Figure out the weighting we are going to apply to the collections.Counter
                    # object for the expression based on the number of known letters in the
                    # Regular expression (whether the object was looked up or calculated)

                    # Find weighting
                    num_letters = sum(c.isalpha() for c in expression)
                    num_letters = 6 if num_letters > 6 else num_letters
                    weighting = weights[num_letters + 1]

                    # Take into account weighting
                    temp_counter = collections.Counter()
                    for key, value in alt_counter.items():
                        temp_counter[key] = value * weighting
                    alt_total_count.update(temp_counter)

                    self.total_count = alt_total_count.most_common()

                # Check if we saw some new regular expressions and counted up the relevant letter
                # appearances in the dictionary, if so, add these new regular expressions to our
                # data frame
                if entered:
                    df = pd.DataFrame(alt_pandas_list)
                    df = df.set_index("Regex")
                    self.regex_df =\
                        pd.concat([self.regex_df, df])

            # Choose the guess letter with the highest count, if all letters have been guessed,
            # use the total training dictionary distribution once more

            guess_letter = '!'

            for letter, instance_count in self.total_count:
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

    def start_game(self, weights, verbose=True, practice=1):

        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []

        # I added this bit
        self.first_time = True
        self.current_dictionary = self.training_dictionary

        self.tries_remains = 7
        self.whole_game_regexes = []

        # Chooses a word from the validation dictionary and creates a hangman equivalent
        # to show the user
        word = random.choice(self.validation_dictionary)
        hangman_word = "_ " * len(word)

        if verbose:
            print("Successfully start a new game! # of tries remaining: {0}.\
                  Word: {1}.".format(self.tries_remains, word))

        while self.tries_remains > 0:
            # get guessed letter from guess method
            guess_letter = self.guess(hangman_word, weights)

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
                self.regex_df.to_pickle(self.regex_df_loc)

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
#     df.to_pickle("C:/Users/alexh/OneDrive/Documents/Coding/Python/Careers/Trexquant/regexes.pkl")

# reset_DataFrames()

# %%

# So I can check the regular expression file


with open("C:/Users/alexh/OneDrive/Documents/Coding/Python/Careers/Trexquant/regexes.pkl", 'rb') as pickle_file:
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



#%%

print(api.cost_function([500, 3, 1, 2, 4, 8, 16, 32]))

#%%

final_weights = api.gradient_descent(0.3)