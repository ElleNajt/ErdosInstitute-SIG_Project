import numpy as np
import itertools
import matplotlib.pylab as pl

import pandas as pd
import praw
import re
import nltk

import gensim.models
import datetime
import networkx as nx


import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import seaborn as sns
import pickle

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import SpectralClustering
import json


### Functions:


def get_timeseries(df, keyword, statistic):
    filter =  pd.Series(df.tokenized_title.apply(lambda x : contains_word(x, keyword)))


    agged_with_keyword = df[filter][['date', 'ups']].groupby("date").agg(statistic)

    total = df[['date', 'ups']].groupby("date").agg(statistic)

    merged = agged_with_keyword.merge(total, left_index = True, right_index = True, how= 'outer')
    merged['ups_x'] = merged.ups_x.fillna(1) # to avoid p = 0

    agged = merged['ups_x']/(merged['ups_y'])


    agged.dropna()
    timeseries_df = pd.DataFrame(agged)

    timeseries_df = timeseries_df[timeseries_df[0] != 0]

    return timeseries_df


def post_process(trace, timeseries_df, keyword, plot = False):

    tau_ints = [ int(x) for x in trace['tau']]

    t_dict = pd.Series(tau_ints).value_counts()
    for x in range(len(timeseries_df)):
        if x not in t_dict.keys():
            t_dict[x] = 0

    timeseries_df['tau'] = np.zeros(len(timeseries_df))
    timeseries_df['tau'] = [t_dict[x] for x in range(len(timeseries_df))]

    timeseries_df['tau'] = timeseries_df['tau'] / timeseries_df['tau'].sum()
    timeseries_df.rename( columns = { 0 : keyword},  inplace = True)

    change_point_guess = np.mean(trace['delta'])

    print(f"Change Point Guess {change_point_guess}")

    if plot == True:
        timeseries_df.plot()
        plt.show()

    tau_map_arg = timeseries_df.tau.argmax()


    return (timeseries_df, trace, tau_map_arg, timeseries_df.iloc[tau_map_arg].name, change_point_guess)

def bayesian_optional_change_point(df, keyword, statistic = 'sum', plot = True, method = "Auto",
                                   cp_alpha = 1, cp_beta = 1,
                                   steps = 5000, tune = 500):

    timeseries_df = get_timeseries(df, keyword, statistic)

    observations = list(timeseries_df[0])
    num_observation = len(observations)


    with pm.Model() as model:
        mu = np.mean(observations)
        var = np.var(observations)
        alpha = ((1 - mu) / var - 1 / mu) * (mu**2)
        beta = alpha * (1 / mu - 1)
        #https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance


        alpha_1 = pm.Exponential("alpha_1", 1/alpha)
        beta_1 = pm.Exponential("beta_1", 1/beta)
        alpha_2 = pm.Exponential("alpha_2", 1/alpha)
        beta_2 = pm.Exponential("beta_2", 1/beta)

        tau = pm.Deterministic("tau", (num_observation - 1) * pm.Beta("tau_beta", 2,2))
        # We want to concentrate the change point towards the middle -- having a changepoint near the
        # ends of the time interval might not really make sense. Unless it does.
        # is there a more principled way to pick the parameters here? maybe all these variables could be part of
        # one giant hierarchical model?


        change_point_beta = pm.Beta("change_point_beta", alpha = cp_alpha, beta = cp_beta)

        exists_change_point = pm.Bernoulli("delta", p = change_point_beta)

        idx = np.arange(num_observation) # Index

        alpha_2_ = pm.math.switch( tt.eq(exists_change_point,1), alpha_2, alpha_1)
        beta_2_ = pm.math.switch( tt.eq(exists_change_point,1), beta_2, beta_1)

        alpha_ = pm.math.switch( idx < tau, alpha_1, alpha_2_)
        beta_ = pm.math.switch( idx < tau, beta_1, beta_2_)

        observation = pm.Beta("obs", alpha = alpha_, beta = beta_, observed=observations)
        # You can't have observations with value zero

        if method == 'Metropolis':
            step = pm.Metropolis()
            trace = pm.sample(steps, tune=tune, step = step)
        if method == 'Auto':
            trace = pm.sample(steps, tune=tune)


    return post_process(trace, timeseries_df, keyword, plot)


def bayesian_change_point(df, keyword, statistic = 'sum', plot = True, method = None, steps = 50, tune = 50):
    timeseries_df = get_timeseries(df, keyword, statistic)

    observations = list(timeseries_df[0])
    num_observation = len(observations)

    print("running chain")

    with pm.Model() as model:
        mu = np.mean(observations)
        var = np.var(observations)
        alpha = 1 #((1 - mu) / var - 1 / mu) * (mu**2)
        beta = 1 #alpha * (1 / mu - 1)
        #https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance

        alpha_1 = pm.Exponential("alpha_1", 1/alpha)
        beta_1 = pm.Exponential("beta_1", 1/beta)
        alpha_2 = pm.Exponential("alpha_2", 1/alpha)
        beta_2 = pm.Exponential("beta_2", 1/beta)

        tau = pm.Uniform("tau", lower=0, upper= num_observation - 1)

        idx = np.arange(num_observation) # Index

        alpha_ = pm.math.switch( idx < tau, alpha_1, alpha_2)
        beta_ = pm.math.switch( idx < tau, beta_1, beta_2)
        observation = pm.Beta("obs", alpha = alpha_, beta = beta_, observed=observations)
        # You can't have observations with value zero
        if method == 'Metropolis':
            step = pm.Metropolis()
            trace = pm.sample(steps, tune=tune, step = step)
        if method == 'Auto':
            trace = pm.sample(steps, tune=tune)

    return post_process(trace, timeseries_df, plot)

def entropy(histogram):
    total = 0
    for p in histogram:
        if p != 0:
            total += -1 * p * np.log(p)
    return total

def changepoint_entropy(df, keyword, stastic = 'sum'):
    histogram = bayesian_change_point(df, keyword = keyword, statistic = statistic, plot = False)[0]['tau']
    return entropy(histogram)

def contains_word(tokenized, target):
    for sentence in tokenized:
        if target in sentence:
            return True
    return False

def beta_mean(alpha, beta):
    # returns the average mean of the beta with parameters given by the posterior samples
    return (alpha/ ( alpha + beta)).mean()
