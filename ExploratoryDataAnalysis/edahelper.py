import pandas as pd
import seaborn as sns
import re
import praw
from matplotlib import pyplot as plt
import numpy as np




def sigmoid(x):
    return 1/ ( 1 + np.exp(-x))

def logit(x):
    if x != 0:
        return x/(1 - x)
    else:
        return -999999999 # ...
