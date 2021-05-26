from edahelper import *
import numpy as np
import matplotlib.pylab as pl

import pandas as pd
import praw
import re
import nltk

import gensim.models
import datetime
#import networkx as nx
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

wsb = pd.read_pickle("cultureshifts_preprocessed.pkl")

### Functions:

def bayesian_change_point(wsb, keyword, statistic = 'sum', plot = True):
    wsb['contains_keyword'] =  wsb.tokenized_title.apply(lambda x : contains_word(x, keyword))

    agged_withgme = wsb[wsb.contains_keyword][['date', 'ups']].groupby("date").agg(statistic)

    agged_withoutgme = wsb[~wsb.contains_keyword][['date', 'ups']].groupby("date").agg(statistic)
    merged = agged_withgme.merge(agged_withoutgme, left_index = True, right_index = True, how= 'left')
    #merged['ups_x'] = merged.ups_x.fillna(0)
    agged = merged['ups_x']/(merged['ups_x'] + merged['ups_y'])


    agged.dropna()
    agged_df = pd.DataFrame(agged)

    agged_df = agged_df[agged_df[0] != 0]


    observations = list(agged_df[0])
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

        tau = pm.Uniform("tau", lower=0, upper= num_observation - 1)

        idx = np.arange(num_observation) # Index

        alpha_ = pm.math.switch( idx < tau, alpha_1, alpha_2)
        beta_ = pm.math.switch( idx < tau, beta_1, beta_2)
        observation = pm.Beta("obs", alpha = alpha_, beta = beta_, observed=observations)
        # You can't have observations with value zero
        #step = pm.Metropolis()
        trace = pm.sample(1000, tune=200)#, step = step)

    tau_ints = [ int(x) for x in trace['tau']]

    t_dict = pd.Series(tau_ints).value_counts()
    for x in range(len(agged_df)):
        if x not in t_dict.keys():
            t_dict[x] = 0

    agged_df['tau'] = np.zeros(len(agged_df))
    agged_df['tau'] = [t_dict[x] for x in range(len(agged_df))]

    agged_df['tau'] = agged_df['tau'] / agged_df['tau'].sum()

    if plot == True:
        agged_df.plot()

    tau_map_arg = agged_df.tau.argmax()


    return (agged_df, trace, tau_map_arg, agged_df.tau.iloc[tau_map_arg])

def entropy(histogram):
    total = 0
    for p in histogram:
        if p != 0:
            total += -1 * p * np.log(p)
    return total

def changepoint_entropy(wsb, keyword, stastic = 'sum'):
    histogram = bayesian_change_point(wsb, keyword = keyword, statistic = statistic, plot = False)[0]['tau']
    return entropy(histogram)

pw = pd.read_pickle("pop_words_10.pkl")
pop_words = list(set(itertools.chain.from_iterable(pw)))

data = {}
entropies = {}

up_to = 1

for word in pop_words[:1]:
    data[word] = bayesian_change_point(wsb, keyword = word, statistic = 'sum', plot = False)
    agged_df = data[word][0]
    agged_df.plot()
    plt.save_fig(f"ChangePoint{word}.png")
    entropies[word] = entropy(data[word][0]['tau'])
    print("Word: ", word, "Entropy: ", entropies[word])

entropies = pd.DataFrame(entropies)
entropies.to_csv("Entropies.csv")

with open('data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
