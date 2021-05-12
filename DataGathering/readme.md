This folder stores the scripts used to gather data. 

## Processing pipeline:
1.  RedditScraper.py works: First, this uses pushshifts search to get all submission ids between the chosen dates. Then, it passes those id's to the praw API to download the data. Because it uses reddit.info(), it is able to download submission data for many posts using a single API call; this trick was taken from here: https://www.reddit.com/r/redditdev/comments/aoe4pk/praw_getting_multiple_submissions_using_by_id/ .
2. At the end of RedditScraper, we have a dataframe where each column is the dictionary of the variables of the submission object of every submission found by pushshift's search API.
3. CleaningRedditData.ipynb is then used to remove uninteresting fields, and convert others to usable formats, for instance by extracting text from the html code of a text submission using beautifulsoup, or by converting timestamps into readable UTC datetimes.

## API Keys:

To use this, you need an "API.env" file formatted like:

```
reddit_client_id = "your reddit API id"
reddit_client_secret = "your reddit API secret key" 
user_agent = "your user agent"
```

in the same folder as RedditScraper.py .

.env files are in the gitignore.


## Data:
The result of running this pipeline on /r/wallstreetbets between Jan 1 2021 and May 7 2021 is stored here: https://drive.google.com/file/d/1fcrXvG7tF-Fzv8xMREQz4N7a_Am2E9QB/view

The pickled dataframes where made with a python 3.8.6 build. You also need to have praw version 7.2.0 for the dataframes to unpickle without an error.

There is also a csv version here: https://drive.google.com/file/d/1-hiZVtO-nEwi92F9FbOwTbkvOFNLC3Tj/view?usp=sharing

## Caveats:

1. Pushshift's ingest went down for various periods of time, and hasn't been backfilled. This leads to some gaps in the data that are apparent one plots a time series. See here for discussion: https://www.reddit.com/r/pushshift/comments/n38roy/missing_data/
2. Some of the field names are counter intuitive, e.g. score actual refers to the number of upvotes: https://praw.readthedocs.io/en/latest/code_overview/models/submission.html
3. Compared to the kaggle data here https://www.kaggle.com/gpreda/reddit-wallstreetsbets-posts/code , this dataset is missing about 9000 entries, but also has 500k+ entries from the overlapping time period that the kaggle data set does not have. See the end of the cleaning file for a comparison of the two.


## Comparison with kaggle:

There is a kaggle data set of scraped wsb data here:  https://www.kaggle.com/gpreda/reddit-wallstreetsbets-posts

The key difference is that this appears ( by looking at the authors github https://github.com/gabrielpreda/reddit_extract_content )  to have been built by scraping reddit.new on a regular basis. This means that while it doesn't have the gaps that the pushshift data has, it is missing many entries. The file CompareToKaggle shows some of the differences between the two data sets.
