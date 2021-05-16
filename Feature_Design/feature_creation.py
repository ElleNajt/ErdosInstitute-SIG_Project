# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:00:28 2021

@author: lnajt

This contains functions that create the word embeddings and author 
influence features from the training set.

"""

import pandas as pd
import itertools
from scipy.stats import powerlaw

import re
import nltk


#import networkx as nx
import gensim.models
#import numba 
import numpy as np

import sklearn 


class author_influence:
    
    def __init__(self):
        self.influence_df = None
        self.global_mean = None
        
    def convert_to_mean(self, params):
        return powerlaw.moment(1, *params)


    
    def fit(self, kind, submissions_df):
        # A more systematic way to do this would be to use empirical bayes shrinkage
        # I'm still pretty unsure about what the right model for upvotes is anyway...
        # some kind of heavy tailed counting data parametric family...
        
        
        if kind in ['mean', '50%', 'max']:
            self.global_params = submissions_df.ups.describe()[kind]
            
            
            submissions_df = submissions_df.loc[submissions_df.author != "None"]
            author_df = submissions_df[['author', 'ups']].groupby('author').agg('describe')
            self.influence = author_df[ ('ups', kind)]
        
        if kind == 'power_law':
            # this is very slow
            self.global_params = self.convert_to_mean( powerlaw.fit (submissions_df.ups))
            
            
            submissions_df = submissions_df.loc[submissions_df.author != "None"]
            author_df = submissions_df[['author', 'ups']].groupby('author').agg( [lambda x: tuple(x), 'count', 'mean'])
            
            author_df = author_df.loc[  author_df[('ups', 'count')] >= 10 ] 
            author_df['popularity_parameters'] = author_df[('ups', '<lambda_0>')].apply(powerlaw.fit)
            author_df['popularity_aggregate'] = author_df['popularity_parameters'].apply(self.convert_to_mean)
            self.influence_df = author_df['popularity_aggregate']
            
            

        # TODO: We can use beta_shrinkage for the upvote ratio term
        
        return self
        

    def transform(self, submissions_df):
        merged = submissions_df.merge( self.influence_df, how = 'left', left_on = 'author', right_index = True)
        # this will leave NAs for authors in submission df that are not in self.influence_df
        # so we replace those Nas with self.global_mean
        
        #merged.popularity_aggregate = merged.popularity_aggregate.fillna(self.global_mean)

        merged = merged.fillna( {"popularity_aggregate" : self.global_params})
        
        return merged
    
    
    
    
#########Word2Vec stuff:
    
    
def similarity(vec_1, vec_2):
    return sklearn.metrics.pairwise.cosine_similarity([vec_1], [vec_2])[0]

def make_similarity_col(df, given_index):
    given_vector = df['avg_vector'][given_index] 
    df['similarity'] = df['avg_vector'].apply( lambda x : similarity(x, given_vector))
    
def sims(args, model):
    for word, sim in model.wv.most_similar(**args, topn = 10):
        print( f"{word} - similarity {sim}")

######################




class tokenization:
    
    def __init__(self):
        
        self.regex = re.compile('[^a-zA-Z ]')

        return self
    
    def tokenize(self, text):
    # given a body of text, this splits into sentences, then processes each word in the sentence to remove
    # non alphabetical characters... (? bad idea, what about users with numbers in their name)
    # returns it as a list of lists of words, the format desired by gensims word2vec

        sentences = []
        if type(text) == str:
            for sentence in nltk.tokenize.sent_tokenize(text):
                processed = [self.regex.sub('', word.lower()) for word in sentence.split(' ') ]
                processed = [word for word in processed if word not in set( ['' ])]
                sentences.append(processed)
        return sentences

    
    
    def fit_transform(self, df):
    
        df['tokenized_title'] = df.title.apply(self.tokenize)
        df['tokenized_selftext'] = df.selftext.apply(self.tokenize)

        return df
    
class word2vec:
    
    
    def __init__(self, model_kind = gensim.models.Word2Vec):
        # other options  gensim.models.FastText
        
        self.model_kind = model_kind
        self.model = None
        
        
    def average_vector(self, text):
        present_keys = [x for x in text if x in self.model.wv.key_to_index ]
        if not present_keys:
            return np.zeros( self.model.wv.vector_size)
        return sum( [self.model.wv[x] for x in present_keys] ) /len(present_keys)

    def average_vector_paragraph(self, text):
        if text == []:
            return np.zeros( self.model.wv.vector_size)
        return sum( self.average_vector(sentence)  for sentence in text )


    def fit(self, tokenized_text):
        corpus = []
        for tokenized in tokenized_text:
            corpus += tokenized


        self.model = self.model_kind(sentences = corpus,  min_count=10, vector_size=300, epochs = 4)

        return self
    
    def transform(self, tokenized_text):
        
        avg_vectors = tokenized_text.apply(self.average_vector_paragraph)
        X = np.vstack(avg_vectors.to_numpy())
        return X
