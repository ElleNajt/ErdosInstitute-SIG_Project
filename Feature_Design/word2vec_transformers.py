# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:23:28 2021

@author: lnajt
"""

import pandas as pd
import itertools
from scipy.stats import powerlaw
import nltk
import re


import gensim.models
#import numba 
import numpy as np

import sklearn 

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


regex = re.compile('[^a-zA-Z ]')

def tokenize(text):
    # given a body of text, this splits into sentences, then processes each word in the sentence to remove
    # non alphabetical characters... (? bad idea, what about users with numbers in their name)
    # returns it as a list of lists of words, the format desired by gensims word2vec
    
    sentences = []
    if type(text) == str:
        for sentence in nltk.tokenize.sent_tokenize(text):
            processed = [regex.sub('', word.lower()) for word in sentence.split(' ') ]
            processed = [word for word in processed if word not in set( ['' ])]
            sentences.append(processed)
    return sentences

def train_w2v(tokenized_text):

    corpus = []
    for tokenized in tokenized_text:
        corpus += tokenized

    model = gensim.models.Word2Vec(sentences = corpus,  min_count=10, vector_size=300, epochs = 4)
    
    return model


class tokenizer(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        return None

    def fit(self, X = None, y = None):
        return self
        
    def transform(self, X = None, y = None, cols = None, in_place = False):
        # takes in the column to tokenize
        return X.apply(tokenize)

    def in_place(self, X = None, cols = None):
        for col in cols:
            X['tokenized_' + col] = X[col].apply(tokenize)
        return None
    
class conceptcluster(TransformerMixin, BaseEstimator):
    
    
    def cluster_function(self):
        
        
        return 0
    
    def __init__(self, minclustersize = 10, maxclustersize = 100, roughclustersize = 200, verbose = True):
        """

        """

        self.minclustersize = minclustersize
        self.maxclustersize = maxclustersize
        self.cluster_dictionary = None
        self.corpus_df = None
        self.w2vmodel = None
        self.clusteringmodel = None
        
        self.roughclustersize = roughclustersize
        self.verbose = verbose

        return None
    
    def fit(self, X = None, y = None):
        # Takes in a dataframe X
        

        corpus=X['tokenized_title'].append(X ['tokenized_selftext'])
        self.w2vmodel = train_w2v(corpus) 

        self.corpus_df = pd.DataFrame(self.w2vmodel.wv.key_to_index.keys(), columns = ['word' ])
        self.corpus_df['vector'] = self.corpus_df['word'].apply(lambda x : self.w2vmodel.wv[x])

    
    
        
        X = np.vstack(self.corpus_df['vector'].to_numpy())
        
        if self.clusteringmodel == None:
            # we leave the option of manually setting it.
            self.clusteringmodel = KMeans(n_clusters= int(len(self.corpus_df)/self.roughclustersize),verbose= self.verbose )
        
        
        self.clusteringmodel.fit(X)
        self.corpus_df ['prediction'] = self.clusteringmodel.predict(X)
        
        self.corpus_df  = self.corpus_df.set_index('word')
        self.cluster_dictionary = dict(self.corpus_df['prediction'])
        
        self.cluster_names =  list( set(self.cluster_dictionary.values()))
        
        return self
      
        
    def cluster_counter(self, tokenization, value):
        counter = 0
        for sent in tokenization:
            for token in sent:
                if token in self.cluster_dictionary:
                    if self.cluster_dictionary[token] == value:
                        counter += 1
        return counter


    def transform(self, X):


        concept_features_df = pd.DataFrame()

        for value in self.cluster_names:
            concept_features_df["title_cluster_" + str(value) + "_counts"] = X['tokenized_title'].apply(lambda x : self.cluster_counter(x, value))
            concept_features_df["selftext_cluster_" + str(value) + "_counts"] = X['tokenized_selftext'].apply(lambda x : self.cluster_counter(x, value))
                
        return concept_features_df
    
    def cluster_of_word(self, word):
        if self.corpus_df is None:
            print("Need to fit this first!")
            return None
        
        return list(self.corpus_df [self.corpus_df.prediction == self.corpus_df.loc[word].prediction].index)
    
