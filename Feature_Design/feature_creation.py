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