# Erdos Institute SIG Project
Bootcamp Project For Erdos Institute

Our project is a data analysis pipeline with the following steps:

1. Given a subreddit and a time period, it downloads all submissions made to that subreddit during that time period.
2. It then finds words that were popular on any day during that period, and then uses pymc3 to find changepoints in the time series of the frequency of posts that contain them. This gives insight into culture shifts.
3. It then trains a neural net to classify whether a post achieves more than the median number of posts. Because shifts in culture influence what becomes successful, this neural net uses the change points we discovered to guide its training.
4. That neural net can then be used to workshop a post, or understand how the success of a post might vary depending on when it was submitted.

# Folders:

1. Automated:

This runs the entire process, from data collection to the neural net training.

2. Guided Tour:

This walks the user through the steps of our data processing pipeline.

3. Sandbox:

This stores some unfinished experiments, including some interesting results from clustering word embeddings.

# Acknowledgements:

We are indebted to the following sources:

1. https://minimaxir.com/2017/06/reddit-deep-learning/
2. Probabilistic Programming & Bayesian Methods for Hackers by Cameron Davidson-Pilon