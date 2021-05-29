# Erdos Institute SIG Project
Bootcamp Project For [Erdos Institute](https://www.erdosinstitute.org/).

This is inspired by a project suggestion provided to the Erdos institute by Susquehanna International Group (SIG).

# Project description:
Our project is a data analysis pipeline with the following steps:

1. Given a subreddit and a time period, it downloads all submissions made to that subreddit during that time period.
2. It then finds words that were popular on any day during that period, and then uses pymc3 to find changepoints in the time series of the frequency of posts that contain them. This gives insight into culture shifts.
3. It then trains a neural net to classify whether a post achieves more than the median number of posts. 
4. That neural net can then be used to workshop a post or to understand which posts may become popular. 

# Folders:

## Automated:

This runs the entire process, from data collection to the neural net training. Instructions inside.

## Guided Tour:

This walks the user through the steps of our data processing pipeline.

## Sandbox:

This stores some unfinished experiments, including some interesting results from clustering word embeddings.

# Acknowledgements:

We are indebted to the following sources:

1. https://minimaxir.com/2017/06/reddit-deep-learning/
2. Probabilistic Programming & Bayesian Methods for Hackers by Cameron Davidson-Pilon

We would like to thank our mentors and teachers from the Erdös Institute, especially Lauren Fink, Lindsay Warrenburg and Matt Osborne. We would also like to thank Roman Holowinsky for organizing the Erdös institute, and his tireless efforts for its cause.
