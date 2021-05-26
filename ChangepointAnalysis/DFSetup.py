from edahelper import *
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

import pandas as pd
import praw
import re
import nltk

import gensim.models
import datetime
import networkx as nx
import xgboost as xgb

import numpy as np
import seaborn as sns


import sklearn
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import SpectralClustering

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

regex = re.compile('[^a-zA-Z ]')

#@numba.jit # unfortunately this doesn't jit easily :(
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

wsb = pd.read_pickle("../Data/subreddit_WallStreetBets/otherdata/wsb_cleaned.pkl")
