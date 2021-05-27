import os
import sys
import datetime as dt


from DataGathering import RedditScraper
from ChangePointAnalysis import ChangePointAnalysis

with open("config.txt") as file:
    exec(file.read())

# Scrape:

for subreddit in subreddits:
    if not os.path.exists(f"../Data/subreddit_{subreddit}/full.pkl"):
        print("Did not find scraped data, scraping.")
        RedditScraper.scrape_data(subreddits = [subreddit], start = dt.datetime(2020, 1,1), end = dt.datetime(2020, 2,18))

# Compute changepoint data

print('Computing the changepoints:')
ChangePointAnalysis.changepointanalysis(subreddits)

# Compute classifier stuff
