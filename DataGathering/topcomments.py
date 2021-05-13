import pandas as pd
import praw
import re
import datetime as dt
import requests
import json
import sys
import time
import os
from bs4 import BeautifulSoup
from datetime import timezone, datetime
import numpy as np
from matplotlib import pyplot

'''

This downloads the comment forest for a list of IDs
'''
with open("API.env") as file:
    exec(file.read())

reddit = praw.Reddit(
  client_id = client_id,
  client_secret = client_secret,
  user_agent = user_agent
)

wsb = pd.read_csv('wsb_cleaned.csv')

upper_threshold = 950 # wsb.num_comments.max()
threshold = 900

ids = wsb.loc[ (threshold <= wsb.num_comments) & (wsb.num_comments <= upper_threshold)].id

print(f"Retrieving comment forests from {len(ids)} submissions, at threshold {threshold}")

comments = []

for i, submission_id in enumerate(ids):
    #print(f"Retrieving from entry {i}, with id {submission_id}")
    submission = reddit.submission(id = submission_id)
    submission.comments.replace_more(limit=None)

    for comment in submission.comments.list():
        comments.append(vars(comment))

comments_df = pd.DataFrame(comments)

comments_df.to_pickle(f"Data/topcomments_more_than_{threshold}_less_than_{upper_treshold}.pkl")

#comments_df.to_csv(f"comments_forests_more_than_{threshold}.csv")
