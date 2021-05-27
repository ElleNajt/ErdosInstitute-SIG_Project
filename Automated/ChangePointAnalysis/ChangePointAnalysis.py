
import pandas as pd
import importlib

import seaborn as sns
import re
import praw
from matplotlib import pyplot as plt
import numpy as np
import pickle

import sys
import os

#os.chdir("HelperFunctions")
from ChangePointAnalysis import BayesianMethods, Preprocess, PopularWords
#os.chdir("../")


def compute_changepoints(subreddit = "Jokes"):
    starting_dir = os.getcwd()
    subreddit_path = f"../Data/subreddit_{subreddit}/"

    df = Preprocess.load_and_preprocess(subreddit_path)
    df = pd.DataFrame( df)
    results_df = pd.DataFrame()

    with open("mcmc_config.txt") as file:
        exec(file.read())

    pop_words = PopularWords.popular_words_unioned_each_date(df, daily_words)

    print(pop_words, "pop_words")

    DATA_FOLDER = f"../Data/subreddit_{subreddit}/Changepoints/{method}_{steps}Draws_{tune}Tune/"
    if not os.path.isdir(DATA_FOLDER):
        print("making directory", DATA_FOLDER)
        os.makedirs(DATA_FOLDER)
        print('made')
    print(os.getcwd())
    os.chdir(DATA_FOLDER)

    for word in pop_words[:up_to]:


        print('working on', word)
        data = BayesianMethods.bayesian_optional_change_point(df, keyword = word, statistic = 'count', plot = False, method = method, steps = steps, tune = tune)
        # for readability, this should be packaged as a dictionary
        timeseries_df = data[0]
        timeseries_df.plot()
        plt.ylim(0,1)
        plt.xticks(rotation=90)
        plt.savefig(f"ChangePoint_{word}.png")

        trace = data[1]
        mu_1 = BayesianMethods.beta_mean( trace['alpha_1'],  trace['beta_1']  )
        mu_2 = BayesianMethods.beta_mean( trace['alpha_2'],  trace['beta_2']  )

        results_word = { "change_point_confidence" : data[-1],  "mus": (mu_1, mu_2), "mu_diff" : mu_2 - mu_1, "tau_map" : str(data[3]) , "tau_std" : np.std(data[1]['tau']) , "entropy" : BayesianMethods.entropy(data[0]['tau'])}
        results_word["change_point_guess" ] = data[4]
        for x in results_word.keys():
            results_word[x] = [results_word[x]]
        print(results_word)

        results_row = pd.DataFrame( results_word, index = [word])

        print('appending')
        results_df = results_df.append(results_row)


        with open(f'{word}_data.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    results_df.to_csv("results.csv")
    os.chdir(starting_dir)

def changepointanalysis(subreddits = ["Jokes", "WallStreetBets", "WritingPrompts",  "TraditionalCurses", "TwoSentenceHorror"]):
    for subreddit in subreddits:
        print("working on ", subreddit)
        compute_changepoints(subreddit)

#changepointanalysis()
