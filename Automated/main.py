import os
import sys
import datetime as dt


from DataGathering import RedditScraper
from ChangePointAnalysis import ChangePointAnalysis
from NeuralNets import CreateNeuralNets

with open("config.txt") as file:
    contents = file.read()
    exec(contents)

start = dt.datetime(start_year, start_month, start_day)
end =  dt.datetime(end_year, end_month, end_day)

print(start)
# Scrape:

for subreddit in subreddits:
    if not os.path.exists(f"../Data/subreddit_{subreddit}/full.pkl"):
        print("Did not find scraped data, scraping.")

        RedditScraper.scrape_data(subreddits = [subreddit], start = start, end = end)


# Compute changepoint data
print('Computing the changepoints:')
ChangePointAnalysis.changepointanalysis(subreddits)


# Compute classifier stuff
print('Training the neural nets:')
nn_results = CreateNeuralNets.buildnets(subreddits)
