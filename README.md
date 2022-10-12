
# Hangman AI


## Purpose 

The primary purpose of this project was to design an AI algorithm to play the game of hangman for you, far better than a human ever could. 

## The Traditional Game

The traditional game of hangman is played with two or more players. One person thinks of a word for the other player(s) to try to guess and writes down a number of underscores on a page, whiteboard or other writing surface indicating the number of letters in the word. The other player(s) successively guess(es) letters they suspect make up the word. There are a limited number of times they can incorrectly guess a letter before the game is over and the player who came up with the word wins.

In the traditional game, towards the start when there is little to no information due to their being few or no letters making up the word, guesses typically come from common letters that appear in the English language. As the game progresses and more letters are guessed, the new letters are guessed based on player's impressions of what words would fit the pattern indicated by the letters already guessed correctly or what letters would commonly go with subsets of the guessed letters.

The game is called hangman because to symbolise the number of guesses decreasing, the person who came up with the word incrementally draws the parts that form a gallows and a person about to be hung from it. In retrospect, this is quite morbid for a children's game.

## Play the game yourself

Despite the primary purpose of the game being the design of an AI, a human - playable version of the game has also been implemented. At the moment it lacks a nice interface, but if you run the game.py file and modify the .start_game() method at the bottom with the following parameters:
```
game.start_game(my_word=True)
```
you can play the traditional human adversarial version of the game as explained above. If you don't want to play with anyone else, you can also run that method without any parameters and have a random word generated for you to guess. If you want to have the AI guess a word of your design, you can pass AI=True as a parameter.


## The AI solution

Several different strategies have been explored, some of which have been later combined to create the AI solution, which interact or are preferentially chosen depending on the state of the game. 

The most basic is to simply count how many times each letter appears in the training dictionary of words, and pick letters in order of which letter appears most frequently. Extremely simple, and not that accurate, it has an accuracy of x%. The most basic adjustment to that would be to match the words that you count in the dictionary in length to the secret word and then count which letters appear most frequently. 

A more sophisticated algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary. This is the algorithm implemented in the `AI.algorithm1()` method. This strategy is successful approximately 18% of the time. This benchmark strategy was the inspiration I received to start the rest of this project. We can do better. 



## My Strategy:

`algorithm1` works quite well until the word you need to guess doesn't exist in the training dictionary and you start guessing letters from an increasingly small set of words that fit your regular expression from that dictionary but bear no relation to the word you are trying to guess. A better method would take into account common letter combinations in the English language - let us call this next method `algorithm2` Nevertheless, `algorithm1` is still useful under the conditions: 
- you have a large number of words that match the length and regular expression of the unknown word 
- There are still many unknown letters 

The biggest motivation for using this method is relative computational speed

   


`algorithm2` first looks at the letters in the unknown word, and produces all possible unique regular expressions that can be made with that word using the known letters in the word. Example:

Word guessed so far: 'l e _ _ _ a ' all possible regular expressions would be:

>      '^le..', '^le.', '.a$', '...a$', '..a$', '^le...a$', 'e...a$', '^le...', 'le...a '
##### (Might change in future for optimisation)

      
For each regular expression, it then tabulates all of the letters from words in the dictionary that could fill the gaps in the regular expression that correspond to the gaps in the word and counts them. It then multiplies all of those letter counts by a weighting that increases according to the number of known letters there are in the regular expression. 

This was done because if you had guessed a word up to the point of "c o u n _", you would rather have the algorithm guess "t" due to words like "counting", "accounting" or "viscount" matching the maximum length regular expression "coun." rather than a letter like "a" simply because typically when you find the letter "n" in the dictionary it is normally followed by "a" and the simple sheer number of n's in the dictionary tips the scales of what letter to guess.
   
Some efforts were made to optimise the weights to use for each number of letters in a regular expression. These were experimental and unfortunately weren't too successful.

    
### Weighting refinement attempts:
    
When I first implemented my algorithm, there were a few values I needed to arbitrarily pick - or make an educated guess about -  that could be adjusted to bring about an optimal solution. These include the weights for different lengths of regular expression as well as the point of transition between the `algorithm1` and `algorithm2`. I chose to apply the idea of a cost function and gradient descent from a ML / optimisation course.

While minimising computational time was a key objective, maximising overall accuracy of the AI was the primary goal (within reason), therefore the cost function centred around accuracy. 

Cost function: 1 minus the fractional accuracy of 50 games
    
The gradient descent algorithm functioned simply as follows:
   1. Start with arbitrary set of weights I picked that 'seemed about right' 
   1. For each weight in weights:
      1. Adjust value by a small fraction either up or down
      1. Compute cost function
      1. Compute gradient between former and latter accuracy
      1. Adjust weight in direction of increasing accuracy by learning parameter magnitude

The second method, one I called "trees", was planned to be that I generate a very large number of random weights within a reasonable range of what I would expect each weight to fall in, calculate the accuracy of all of those sets of weights, then use an algorithm like a random forest (hence "trees") to predict what the best mapping of weights to accuracy would be.
    
The big problem with both of these methods was time. I am in the process of figuring out ways to reduce computational complexity now.

Unfortunately for my weights optimisation strategy, for 50 repeats of the hangman game, there was too much variation in the accuracy to perform a rigorous gradient descent. In the end a cost function with 100 repeats and a high learning rate was run to some small success, but did not run through many repeats. It was ultimately hard to tell whether the 'optimised' weights performed better at all.

### Reducing computational complexity

Originally, each time a letter is guessed by the AI, several regular expressions were checked across a dictionary of 250000 words. This was slow. Below are some ways I reduced computational time:

- If a letter is incorrectly guessed, the regular expressions searched through the dictionary and therefore the counts of letters in the matching words have not changed. Therefore, no need to repeat the algorithm, move on to the next most likely letter.

[IDEAS TO BE IMPLEMENTED LATER]
- After matching the expressions to dictionary words for one guess, the words that match the next regular expressions must be a subset of those previous. By keeping track of the matches from the previous guess, the dictionary of words that needs to be searched can be reduced.
- Along a similar vein, there will exist regular expressions that form a subset of previous regular expressions even within a guess. By matching the 'largest' regular expressions first and keeping track of which regular expressions are subsets of others, less dictionary searching can be done 

Note, these last two ideas increase the memory required since new arrays are created to keep track of previous words.

[NEED TO EDIT THE BELOW]

### Pandas table of regular regular expressions

I had heard that pandas dataframe slicing was extremely efficient, and hypothesised that a number of regular expressions would frequently reoccur in the English Language. A table mapping regular expressions to their collections.Counter objects could be stored in a permanent DataFrame in an external file and expanded with new regular expressions each time a game is played. In the end, this did not prove worth it (too many regular expressions).
   

