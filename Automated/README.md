# Automated Process

main.py does the following:

1. Loads the list of subreddits from config. For each subreddit it:
    1. Looks in ../Data/subreddit_{subreddit} to see if there is a file called full.csv. 
        1. If yes, it proceeds to the next step. 
        2. If not, it scrapes all submissions on that subreddit between the start and end date information from config.txt , saves the result as the dataframe full.csv, and proceeds to the next step.
    3. Checks whether changepoint_preprocessed.pkl is in ../Data/subreddit_{subreddit}. If not, it creates it and then proceeds to the next step. This file contains the submission data with tokenized and stopword removed titles.
    4. It uses the information in changepoint_preprocessed.pkl to do a Bayesian analysis of changepoints. 
