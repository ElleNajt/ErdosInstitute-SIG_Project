# -*- coding: utf-8 -*-
"""
Created on Sat May 22 08:37:45 2021

@author: lnajt
"""
import pandas as pd

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
    
    return df