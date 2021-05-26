from edahelper import *
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

def bayesian_change_point(wsb, keyword, statistic = 'sum', plot = True, method = None, steps = 50, tune = 50):
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

    print("running chain")

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
        if method == 'Metropolis':
            step = pm.Metropolis()
            trace = pm.sample(steps, tune=tune, step = step)
        if method == 'Auto':
            trace = pm.sample(steps, tune=tune)
    tau_ints = [ int(x) for x in trace['tau']]

    t_dict = pd.Series(tau_ints).value_counts()
    for x in range(len(agged_df)):
        if x not in t_dict.keys():
            t_dict[x] = 0

    agged_df['tau'] = np.zeros(len(agged_df))
    agged_df['tau'] = [t_dict[x] for x in range(len(agged_df))]

    agged_df['tau'] = agged_df['tau'] / agged_df['tau'].sum()

    agged_df.rename( columns = { 0 : keyword },  inplace = True)

    if plot == True:
        agged_df.plot()

    tau_map_arg = agged_df.tau.argmax()


    return (agged_df, trace, tau_map_arg, agged_df.iloc[tau_map_arg].name)

def entropy(histogram):
    total = 0
    for p in histogram:
        if p != 0:
            total += -1 * p * np.log(p)
    return total

def changepoint_entropy(wsb, keyword, stastic = 'sum'):
    histogram = bayesian_change_point(wsb, keyword = keyword, statistic = statistic, plot = False)[0]['tau']
    return entropy(histogram)

def contains_word(tokenized, target):
    for sentence in tokenized:
        if target in sentence:
            return True
    return False




wsb = pd.read_pickle("cultureshifts_preprocessed.pkl")


print("finished reading the data in")

pop_words = ['gme', 'buy', 'amc', 'hold', 'moon', 'im', 'stock', 'robinhood', 'like', 'get', 'go', 'dont', 'going', 'today', 'short', 'us', 'new', 'lets', 'market', 'next', 'wsb', 'k', 'shares', 'time', 'stocks', 'sell', 'holding', 'nok', 'money', 'one', 'still', 'trading', 'bb', 'guys', 'buying', 'bought', 'make', 'help', 'know', 'good', 'calls', 'fuck', 'right', 'back', 'got', 'squeeze', 'day', 'need', 'retards', 'fucking', 'gamestop', 'think', 'week', 'tomorrow', 'anyone', 'people', 'cant', 'yolo', 'line', 'first', 'dip', 'options', 'big', 'hands', 'whats', 'puts', 'see', 'much', 'price', 'want', 'stop', 'apes', 'rh', 'keep', 'take', 'doge', 'selling', 'hedge', 'made', 'boys', 'way', 'diamond', 'dd', 'tendies', 'please', 'share', 'let', 'last', 'pltr', 'long', 'app', 'tsla', 'even', 'call', 'put', 'funds', 'would', 'everyone', 'silver', 'dogecoin']

entropies = {}
mus = {}
mu_diff = {}
tau_map = {}
tau_std = {}

up_to = None

method = "Metropolis"
steps = 20000
tune = 5000


for word in pop_words[:up_to]:
    print('working on', word)
    data = bayesian_change_point(wsb, keyword = word, statistic = 'sum', plot = False, method = method, steps = steps, tune = tune)
    agged_df = data[0]
    agged_df.plot()
    plt.savefig(f"Data_{method}/ChangePoint_{word}.png")
    entropies[word] = entropy(data[0]['tau'])

    trace = data[1]
    alpha_1 = trace['alpha_1']
    beta_1 = trace['beta_1']
    alpha_2 = trace['alpha_2']
    beta_2 = trace['beta_2']
    mu_1 = (alpha_1/ ( alpha_1 + beta_1)).mean()
    mu_2 = (alpha_2/ ( alpha_2 + beta_2)).mean()

    mus[word] = (mu_1, mu_2)
    mu_diff[word] = mu_2 - mu_1
    tau_map[word] = data[3]
    tau_std[word] = np.std(data[1]['tau'])

    print("Word: ", word, "Entropy: ", entropies[word], "Mu's: ", mus[word], "Diff: ", mu_diff[word])

    with open(f'Data_{method}/{word}_data.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)




with open(f'Data_{method}/mus.txt', 'w') as convert_file:
     convert_file.write(json.dumps(mus))

with open(f'Data_{method}/mu_diffs.txt', 'w') as convert_file:
     convert_file.write(json.dumps(mu_diff))

with open(f'Data_{method}/entropies.txt', 'w') as convert_file:
     convert_file.write(json.dumps(entropies))


with open(f'Data_{method}/tau_map.txt', 'w') as convert_file:
     convert_file.write(json.dumps(tau_map))


with open(f'Data_{method}/tau_std.txt', 'w') as convert_file:
     convert_file.write(json.dumps(tau_std))
