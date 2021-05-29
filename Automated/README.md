# Automated Process

main.py does the following:

Loads the list of subreddits from config. For each subreddit, it:
  1. Looks in ../Data/subreddit_{subreddit} to see if there is a file called full.csv. 
      1. If yes, it proceeds to the next step. 
      2. If not, it scrapes all submissions on that subreddit between the start and end date information from config.txt , saves the result as the dataframe full.csv, and proceeds to the next step.
  1. Checks whether changepoint_preprocessed.pkl is in ../Data/subreddit_{subreddit}. If not, it creates it and then proceeds to the next step. This file contains the submission data with tokenized and stopword removed titles.
  1. Uses the information in changepoint_preprocessed.pkl to do a Bayesian analysis of changepoints. 
  2. For each word of interest (any word that is in the top ten most popular words used in titles on any given day), it:
      3. Saves an image 'ChangePoint_{word}.png' depicting the location of the changepoint for that word. 
  4. Trains a neural net on the subreddit data that takes time and title information as input and predicts whether a post will have score greater than the median. 
  5. Returns the neural net, a list of its accuracies (accuracy of the dummy classifier that guesses all posts are less than or equal to the median), the tokenizer the model uses to convert strings into lists of numbers, and a copy of the subreddit's dataframe with additional columns added. 
  6. Saves the neural net in ../Data/subreddit_{subreddit} as 'NN_model.keras'
