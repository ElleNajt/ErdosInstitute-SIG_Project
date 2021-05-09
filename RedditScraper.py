# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:26:41 2021

@author: lnajt
"""

import pandas as pd           
import praw                   
import re                     
import datetime as dt
import requests
import json
import sys
import time

'''

## acknowledgements

https://stackoverflow.com/questions/48358837/pulling-reddit-comments-using-python-praw-and-creating-a-dataframe-with-the-resu
https://www.reddit.com/r/redditdev/comments/2e2q2l/praw_downvote_count_always_zero/
https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91

For navigating pushshift: https://github.com/Watchful1/Sketchpad/blob/master/postDownloader.py

'''


with open("API.env") as file:
    exec(file.read()) # this loads the ids and secrets...

reddit = praw.Reddit(
  client_id = client_id,
  client_secret = client_secret,
  user_agent = user_agent
)


subreddit_df = pd.DataFrame()

'''
Some helper functions for the reddit API.
'''

def extract_num_rewards(awardings_data):
    return sum( x["count"] for x in awardings_data)


def extract_data(submission, comments = False):
    postlist = []

    # extracts top level comments

    if comments:
        submission.comments.replace_more(limit=0)
        for comment in submission.comments: 
            post = {} # put this here
            
            post = vars(comment)


            postlist.append(post)

    content = vars(submission)
    content["author"] = str(content["author"])
    content["total_awards"] = extract_num_rewards(content["all_awardings"])
    return content

'''
Code for getting all submissions between certain date time
'''

def convert_to_utc(date):
    return int(date.replace(tzinfo=dt.timezone.utc).timestamp())

def get_all_submissions(start_time, end_time, subreddit):
    end = end_time
    df = pd.DataFrame()
    while end > start_time:
        time.sleep(1) # Requests are rate limited
        print(f"Target time: {start_time}, current end point {end}, remaining {end - start_time}")

        url = f"https://api.pushshift.io/reddit/submission/search/?after={start_time}&before={end}&sort_type=created_utc&sort=desc&subreddit={subreddit}&limit=1000"

        data = requests.get(url)
        #print(url)
        
        if data.headers['Content-Type'] == 'application/json; charset=UTF-8':
            data_json = data.json()
            if len(data_json['data']) == 0:
                # break if there is no returned data
                break

            temp_df = pd.DataFrame(data_json['data'])
            end = min(temp_df.created_utc) 
            df = df.append(temp_df, ignore_index = True)
        else:
            print(data.headers['Content-Type']) 
            
    # Get the current score from praw
    print(f"pushshift found {len(df)} submissions.")
    print("Getting the updated values.")

    # Based on this: https://www.reddit.com/r/redditdev/comments/aoe4pk/praw_getting_multiple_submissions_using_by_id/
    
    if len(df) == 0:
        # Pushshift ingestion went down, this creates gaps in the data: https://www.reddit.com/r/pushshift/comments/n38roy/missing_data/
        # can see this at 
        #     start = dt.datetime(2021, 2,5)
        # end = dt.datetime(2021, 2,7)
        # The daily post on that day is missing: https://www.reddit.com/r/wallstreetbets/search/?q=What%20Are%20Your%20Moves%20Tomorrow%2C%20February%2007%2C%202021&restrict_sr=1
        # idk what that means, maybe its just a coincidence or maybe the subreddit went private then ( wikipedia says they went private for a few hours late Jan)
        return pd.DataFrame() 
        # return an empty data frame, otherwise pulling
        # the id column gives an error 
    
    ids2 = [i if i.startswith('t3_') else f't3_{i}' for i in list(df.id)]

    
    # Because there are some days that pushshift disagrees with praw, I changed the code to just get the submissions 
    # ids from pushshift, and then to download the data from PRAW.
    praw_submissions = []
    for submission in reddit.info(ids2): # Makes a single call to the PRAW API, much faster than doing them one by one.
        praw_submissions.append(extract_data(submission))
    
    praw_df = pd.DataFrame(praw_submissions)
    print(f"PRAW found {len(praw_df)} submissions.")
    return praw_df

new_start = True #False for picking up where it was left off
if new_start == True:
    start = dt.datetime(2021, 1,1)
    end = dt.datetime(2021, 5,8)
    delta = dt.timedelta(days=1)
    window_left = start
    
    window_right = start + delta

while window_right < end:
    # Decided to go day by day to avoid losing data due to crashing partway through
    print(f"Processing {window_left} to {window_right}")
    
    start_time = convert_to_utc(window_left)
    end_time = convert_to_utc(window_right)
    df = get_all_submissions(start_time, end_time, "wallstreetbets")
    df.to_pickle(f"Data/wsb_start{convert_to_utc(window_left)}.pkl")
    subreddit_df = subreddit_df.append(df, ignore_index = True)
    window_left += delta
    window_right += delta

subreddit_df.to_pickle("Data/2021wsb.pkl")
