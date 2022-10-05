
README FILE UNDER REVIEW - WILL BE CHANGED SOON


## The Game

In the game of Hangman, a secret word is selected at random from a list. The game API then returns row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
or (2) the user has made a specified number of incorrect guesses (in this case, 6).

## Introduction

In this scenario I have split a dictionary of words into a training set and a test set. 

within the solution I have devised for the game, several different strategies have been employed which interact or are preferentially chosen depending on the state of the game. 

The most basic is to simply count how many times each letter appears in the training dictionary of words, and pick letters in order of which letter appears most frequently. Extremely simple, and not that accurate, it has an accuracy of x%. The most basic adjustment to that would be to match the words that you count in the dictionary in length to the secret word and then count which letters appear most frequently. 

A more sophisticated algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary. This strategy is successful approximately 18% of the time. This benchmark strategy was the inspiration I received to start the rest of this project. We can do better.


## My Strategy:

   The benchmark method works quite well until the word you need to guess doesn't exist in the training dictionary and you start guessing letters from an increasingly small set of words that fit your regular expression from that dictionary but bear no relation to the word you are trying to guess. A better method is needed that takes into account common letter combinations in the English language. However, so long as you have a large number of words that match the length and regular expression of the unknown word and there are still many unknown letters then the example method can still be leveraged advantageously over the two alternative methods of taking letters from the letter distribution of the full dictionary and the method that takes into account common letter combinations.
    
   The method that takes into account common letter combinations first looks at the letters in the unknown word, and produces all possible unique regular expressions that can be made with that word using the known letters in the word (more details line 126 below) . For each regular expression, it then tabulates all of the letters from words in the dictionary that could fill the gaps in the regular expression that correspond to the gaps in the word and counts them. It then multiplies all of those letter counts by a weighting that increases according to the number of known letters there are in the regular expression. This was done because if you had guessed a word up to the point of "c o u n _", you would rather have the algorithm guess "t" due to words like "counting", "accounting" or "viscount" matching the maximum length regular expression "coun." rather than a letter like "a" simply because typically when you find the letter "n" in the dictionary it is normally followed by "a" and the simple sheer number of n's in the dictionary tips the scales of what letter to guess.
    
   Some efforts were made to optimise the weights to use for each number of letters in a regular expression, I'll put them below in another section which provides more detail of the problem solving procedure I undertook (I separated it because it is less critical information).
    
    
    

    
    
### Weighting:
    
   When I first implemented my algorithm, I realised that there were a few numbers that I had to quite arbitrarily pick - and actually just made an educated guess about, that could and indeed should be adjusted to bring about an optimal solution. The places where these numbers appear in the code are lines 68 and lines 221-230 - they include not only the weights for different lengths of regular expression, but also the point of transition between the benchmark algorithm provided and the algorithm I devised. I have explored machine learning before, but fairly long ago and not in great detail, so it was not immediately clear to me what standard procedure I could follow to try to optimise the weights, or whether there would be some function in some package that would easily solve it for me. 
    
   In any case, I attempted to implement some of my own functions based off of my current understanding of machine learning. I wanted to adjust the weights to improve accuracy, so clearly my performance measure would be accuracy, and the inputs would be the weights. The algorithm would be a supervised learning task as the data has a label, but I spotted a couple of ways to go about it. To be able to implement both of these, the first task was to create a cost function, which I initially made to be 1 minus the fractional accuracy of 50 games. 
    
   The first method, which I called "gradient descent", is indeed based on what I understand of gradient descent. I would start with an arbitrary set of weights I picked that 'seemed about right' iterate across each weight in turn, adjust their value by a small fraction either up or down, then measure the accuracy of that method and pick the direction that resulted in the lower accuracy. There were two ways to do this, either you could look at a weight, see which accuracy is better up or down, and then adjust that weight on the spot; or you could wait until you have passed through the entire list of weights having adjusted them all up and down individually (keeping the others constant), and then made the decision (this is how it is coded currently). The latter clearly assumes that the weightings are significantly codependent.
    
   The second method, one I called "trees", was planned to be that I generate a very large number of random weights within a reasonable range of what I would expect each weight to fall in, calculate the accuracy of all of those sets of weights, then use an algorithm like a random forest (hence "trees") to predict what the best mapping of weights to accuracy would be.
    
   Unfortunately, the big problem with both of these methods was time. The time to play even a single game was slow because it has to check regular expressions across an entire dictionary of 250000 words several times. I tried to make the function more efficient in two principle ways. The first was simply to not have the function recalculate all of the counts for every regular expression when an incorrect letter was guessed (line 132). This is because if you incorrectly guess a letter, the regular expressions you get from the incomplete word that is returned are identical to the ones you just calculated, you just have to move on to the next most common letter you determined before as your guessed letter. The second way is that I had heard that pandas dataframe slicing was extremely efficient, and hypothesised that a number of regular expressions would frequently reoccur in the English Language. By storing those common regular expressions in a permanent DataFrame in an external file, I could build a table mapping regular expressions to their collections.Counter items that I could read in every time I wanted to run a hangman game. A final small thing I considered to speed up the algorithm was to reduce the proportion of the dictionary I would read through, but ultimately I ran out of time and didn't want to compromise accuracy for speed, especially given the performance metric we seem to be measuring the algorithm by is accuracy alone.
   
   What was even worse, is that I discovered that for 50 repeats of the hangman game, there was still simply too much variation in the cost method return value to realistically be able to implement well either of these methods. In the end I changed it to 100 repeats, and only ran the gradient descent method with a high learning rate to some small success, but it only had time to go through two full iterations while assuming the weights were not co-dependent. In the end it is hard to tell whether the weights I came up with (and rounded to nicer numbers) are really much better than the ones I started with.
    
   This problem of speed was so immovable that I attempted for a time to learn about AWS to run my ML algorithm with more computing power. Unfortunately I don't have much experience working with cloud computing or computer networks so as soon as I started seeing I needed to use unfamilar tools like PuTTY I thought I would rather spend time trying to improve what I had than take the risk of trying to learn something that I may not even be able to learn in time to use (at the time my algorithm was still below 50% accuracy).

