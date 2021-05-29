import numpy as np
import pandas as pd
import praw
import re
import nltk

import gensim.models
import datetime
import os
import numpy as np

import itertools

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

regex = re.compile('[^a-zA-Z ]')

def tokenize(text, stopwords_temp):
    # given a body of text, this splits into sentences, then processes each word in the sentence to remove
    # non alphabetical characters... (? bad idea, what about users with numbers in their name)
    # returns it as a list of lists of words, the format desired by gensims word2vec

    sentences = []
    if type(text) == str:
        for sentence in nltk.tokenize.sent_tokenize(text):
            processed = [regex.sub('', word.lower()) for word in sentence.split(' ') ]
            processed = [word for word in processed if word not in set( ['' ])]
            processed = [word for word in processed if word not in stopwords_temp]
            sentences.append(processed)
    return sentences


def preprocess(df):

    df["date"] = df["created_utc"].apply(lambda x : datetime.datetime.utcfromtimestamp(x).date() )
    stopwords_temp = set(stopwords.words("english"))
    df['tokenized_title'] = df.title.apply(lambda x : tokenize(x, stopwords_temp))
    #df['tokenized_selftext'] = wsb.selftext.apply(lambda x : tokenize(x, stopwords_temp))
    sub_df = pd.DataFrame( df[['tokenized_title', 'author', 'ups', 'id', 'date']])

    return sub_df

def load_and_preprocess(dataframe_path):
    preprocessed_location = dataframe_path + "changepoint_preprocessed.csv"
    if os.path.exists(preprocessed_location):
        return pd.read_csv(preprocessed_location)
    else:
        print("Did not find preprocessed version, preprocessing. (This will only be done once.)")
        #df = pd.read_pickle(dataframe_path + "full.pkl")
        df = pd.read_csv(dataframe_path + "full.csv")
        preprocessed = preprocess(df)
        #preprocessed.to_pickle(preprocessed_location)
        preprocessed.to_csv(preprocessed_location)
    return preprocessed
