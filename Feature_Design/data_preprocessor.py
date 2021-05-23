# -*- coding: utf-8 -*-
"""
Created on Sat May 22 08:37:45 2021

@author: lnajt
"""
import pandas as pd
import numpy as np
from datetime import datetime as dt

def preprocess(df):
    # Removes posts that have been removed/deleted
    # Removes posts with na title or selftext
    # Also removes the daily discussion threads.

    df = df.loc[df.author != "None"]

    df=df.loc[(((df.removed_by_category.isnull()))) & ((df.is_self==True) &  ~(df["title"].str.contains("Thread|thread|Sunday Live Chat|consolidation zone|Containment Zone|Daily Discussion|Daily discussion|Saturday Chat|What Are Your Moves Tomorrow|What Are Your Moves Today|MEGATHREAD",na=False)))]
    
    df = df.dropna(subset = ['title', 'selftext'])
    
    
    # Creates datetune features:
    df["created_datetime_utc"] = df["created_utc"].apply(dt.utcfromtimestamp)
    df['weektime'] = df['created_datetime_utc'].apply( lambda x : x.weekday()*24 + x.hour)
    df['time_of_day'] = df['created_datetime_utc'].apply( lambda x : x.hour)
    
    
    # Creates columns counting awards of various types -- this expects the data
    # to come from the csvs created earlier.
    df['all_awardings'] = df.all_awardings.apply(eval)

    award_types = set()
    for awardings in df.all_awardings:
        for award in awardings:
            award_types.add(award['name'])
            
    def count_award(award, awardings):
        counter = 0
        for given_award in awardings:
            if given_award['name'] == award:
                counter += given_award['count']
        return counter
    
    for award in award_types:
        df["award_" + award] = df.all_awardings.apply(lambda awardings : count_award(award,awardings))
        
    award_cols = [ "award_" + award for award in award_types]
    
    #fix up and downvotes:
    # score = ups - downs
    # ratio = ups/downs

    df["ups"] = np.where(2 * df["upvote_ratio"] - 1 != 0, df["score"] * df["upvote_ratio"] / ( 2 * df["upvote_ratio"] - 1), np.nan) 
    # put nan when value cant be solved for, because the ratio is .5, i.e. the score is zero.
    df["ups"] = np.where( df["score"] == 0, np.nan, df["ups"])
    # because of rounding in calculating the upvote ratio we can end up with situations with score = 0 and upvote ratio = .48
    # so we go through and filter to set those to also be nan.
    
    df["downs"] = df["ups"] - df["score"]
    return df, award_cols