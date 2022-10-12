# %%
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:39:58 2021

@author: alexh
"""

import pickle
from importlib import reload
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
import AI as ai
reload(ai)


class HangmanGameAPI(object):
    """
    A hangman game object that can be played either by a human or an AI
    """

    def __init__(self):

        self.guessed_letters = []
        full_dictionary_location = "C:\\Users\\alexh\\OneDrive\\Documents\\Coding\\Python\\Personal Projects\\Hangman\\dictionaries\\words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.tries_remains = 7  # or 6

    def human_guess(self):
        letter = input("Please enter the letter you would like to guess: ")
        while not letter.isalpha() or len(letter) != 1 or letter in self.guessed_letters:
            if not letter.isalpha():
                letter = input("Invalid input, please enter a single letter,\
                    no special characters permitted: ")
            elif len(letter) != 1:
                letter = input("Invalid input, please enter a single letter: ")
            else:
                letter = input(
                    "That letter has been guessed already, please try another one: ")
        return letter

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def start_game(self, verbose=True, practice=1, AI=False, test=False,
                   regex_store=False, max_word_length=100, my_word=False):

        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.tries_remains = 7

        # Chooses a word from the validation dictionary and creates a hangman equivalent
        # to show the user
        word = 'a' * 101
        if max_word_length > 100:
            max_word_length = 100

        if AI:
            self.training_dictionary, self.validation_dictionary =\
                train_test_split(self.full_dictionary, random_state=0)

        while len(word) > max_word_length:
            if my_word:
                word = input(
                    "Player, specify the word you wish your adversary to guess: ")
                if len(word) > max_word_length:
                    print(
                        f"Word is longer than max word length of {max_word_length} characters specified")
            else:
                if AI:
                    word = random.choice(self.validation_dictionary)
                else:
                    word = random.choice(self.full_dictionary)
        hangman_word = "_ " * len(word)

        if AI and test:
            print(f"This is the word you want to guess: {word}")

        if verbose:
            print("Successfully start a new game! # of tries remaining: {0}.\
                  Word: {1}.".format(self.tries_remains, hangman_word))

        if not AI:
            while self.tries_remains > 0:
                # get guessed letter from guess method

                guess_letter = self.human_guess()

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

                else:
                    self.tries_remains -= 1
                    if verbose:
                        print(f"Incorrect guess! Updated word: {hangman_word}\
                            # tries remaining: {self.tries_remains}")

                if self.tries_remains >= 0 and "_" not in hangman_word:
                    if verbose:
                        print("Successfully finished game!")
                        break
                elif self.tries_remains == 0:
                    if verbose:
                        print('\n'.join(("Failed game. You ran out of lives.",
                              f"The word you needed to guess: {word}")))
        else:

            AI = ai.HangmanAI(self.training_dictionary,
                              regex_store=regex_store)

            while self.tries_remains > 0:
                # get guessed letter from guess method
                guess_letter = AI.guess(
                    hangman_word, self.guessed_letters)

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

                else:
                    self.tries_remains -= 1
                    if verbose:
                        print(f"Incorrect guess! Updated word: {hangman_word}\
                            # tries remaining: {self.tries_remains}")

                if self.tries_remains >= 0 and "_" not in hangman_word:
                    if verbose:
                        print("Successfully finished game!")
                        break
                elif self.tries_remains == 0:
                    if verbose:
                        print('\n'.join(("Failed game. You ran out of lives.",
                              f"The word you needed to guess: {word}")))

            AI.conclude()

        return None

if __name__ == "__main__":
    game = HangmanGameAPI()
    game.start_game(test=True, my_word=True, AI=True,
                    regex_store=True)

# %%
