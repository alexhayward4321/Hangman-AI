
Need to separate current API into a game one and an AI one. 
Game one must have an AI and a human mode
Need easy communication between AI object and game object



Ideas
- Would it be possible to have the AI do one guess for you while you're 
  playing in human mode 
- Might want to think about how you want to do reinitialisation before
the start of a new game
- Historical accuracy of human vs AI
- Login modes



## Optimisations

### Producing regular expressions: the difference between '^le..' and '^le.'.

In my current regular expression producing algorithm, I use the re.search function...