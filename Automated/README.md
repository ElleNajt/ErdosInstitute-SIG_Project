# Automated Process

main.py does the following:

1. Loads the list of subreddits from config. For each subreddit it:
  a.  Looks in ../Data/subreddit_{subreddit} to see if there is a file called full.csv.

  If yes, it proceeds to the next step.

  If not, it scrapes all submissions on that subreddit between the start and end date information from config.txt , saves the result as the dataframe full.csv, and proceeds to the next step.

  b. Checks whether changepoint_preprocessed.pkl is in ../Data/subreddit_{subreddit}. If not, it creates it and then proceeds to the next step. This file contains the submission data with tokenized and stopword removed titles.

  c. It uses the information in changepoint_preprocessed.pkl to do a Bayesian analysis of changepoints. For more information, see [TODO]. 
