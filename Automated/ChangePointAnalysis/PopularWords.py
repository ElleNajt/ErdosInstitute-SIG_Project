import numpy as np
import pandas as pd
import praw
import re
import nltk

import os
import gensim

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import pandas as pd

regex = re.compile('[^a-zA-Z ]')



def popular_words(df, num_words_per_day):

    corpus = []
    for tokenized in df['tokenized_title']:
        corpus += tokenized

    dictionary = gensim.corpora.dictionary.Dictionary(documents = corpus)

    items = list(dictionary.items())
    items.sort( key = lambda x : dictionary.cfs[x[0]], reverse = True)
    return [x[1] for x in items[:num_words_per_day]]

def popular_words_unioned_each_date(df, num_words_per_day):
    if "date" not in df.columns:
        return "Dataframe has no date column."

    dates = list(set(df.date))
    pop_words_daily = [ popular_words_date(df, x, num_words_per_day ) for x in dates]
    import itertools
    pop_words = list(set(itertools.chain.from_iterable(pop_words_daily)))
    return pop_words


def popular_words_date(wsb, date, numwords):
    day_df = wsb[ wsb.date == date]
    return popular_words( day_df, numwords)

def frequency_dictionary(wsb, date, numwords):
    day_df = wsb[ wsb.date == date]
    corpus = []
    for tokenized in day_df['tokenized_title']:
        corpus += tokenized

    dictionary = gensim.corpora.dictionary.Dictionary(documents = corpus)

    return dictionary.cfs
