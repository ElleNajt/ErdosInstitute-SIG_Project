# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:32:23 2021

@author: lnajt
"""


import pandas as pd
import itertools
from scipy.stats import powerlaw

#import re
#import nltk


#import networkx as nx
#import gensim.models
#import numba 
import numpy as np

import sklearn 

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class author_influence(TransformerMixin, BaseEstimator):
    
    
    def __init__(self, kind = 'mean', prior = None):
        """
        
        self.kind can be anything that .agg() accepts
        'power_law'
        or 'total'
        """
        
        self.influence_df = None
        self.global_params = None
        self.kind = kind
        self.prior = prior
        
    def convert_to_mean(self, params):
        return powerlaw.moment(1, *params)

    def fit(self, X = None, y = None):
        df = X
        
        if self.kind in ['sum', 'count',]:
            self.global_params = 0 # for someone who has never posted...

            author_df = df[['author', 'ups']].groupby('author').agg(self.kind)
            self.influence_df = author_df[ 'ups' ].to_frame('popularity_aggregate')
            
        if self.kind in ['mean', 'median',  'std', 'max']:
            
            self.global_params = getattr(df.ups, self.kind)()

            author_df = df[['author', 'ups']].groupby('author').agg(self.kind )
            self.influence_df = author_df[ 'ups'].to_frame('popularity_aggregate')

        if self.kind == 'beta_shrinkage_upvoteratio':

            author_df = df[['author', 'ups', 'downs']].groupby('author').agg('sum')

            author_df = author_df.dropna()
            
            mu = df.upvote_ratio.mean()
            self.global_params = mu
            sigma = df.upvote_ratio.std()
            v = mu* (1 - mu)/sigma - 1
            
            if self.prior == 'empirical_bayes':
                alpha = df.ups.sum()
                beta = df.downs.sum()   
            else:
                alpha = self.prior[0]
                beta = self.prior[1]
            self.global_params = alpha/ ( alpha + beta)
            
            self.global_params_beta = (alpha, beta)
            
            author_df['beta_mean'] = (alpha + author_df["ups"]) / ( alpha + beta + author_df["downs"] + author_df["ups"])
            self.author_df = author_df # for inspection
            self.influence_df = author_df['beta_mean'].to_frame('popularity_aggregate')
        if self.kind == 'power_law':
            # this is very slow
            
            
            self.global_params = self.convert_to_mean( powerlaw.fit (df.ups))
            
            author_df = df[['author', 'ups']].groupby('author').agg( [lambda x: tuple(x), 'count'])
 
            author_df['popularity_parameters'] = author_df[('ups', '<lambda_0>')].apply(powerlaw.fit)
            author_df['popularity_aggregate'] = author_df['popularity_parameters'].apply(self.convert_to_mean)
            self.influence_df = author_df['popularity_aggregate'].to_frame('popularity_aggregate')
            
        self.influence_df.rename( columns = { "popularity_aggregate" : "auth_agg" + self.kind  }, inplace = True)
        
        return self
        

    def transform(self, X):
        df = X
        #self.influence_df.rename( columns = { "popularity_aggregate" : col_name} )
        
        merged = df.merge( self.influence_df, how = 'left', left_on = 'author', right_index = True)
        # this will leave NAs for authors in submission df that are not in self.influence_df
        # so we replace those Nas with self.global_mean
        
        merged["auth_agg" + self.kind ] = merged["auth_agg" + self.kind ].fillna(self.global_params)
  
        merged.fillna( {"auth_agg" + self.kind : self.global_params}, inplace = True)
        
        return merged[["auth_agg" + self.kind ]]
    
    