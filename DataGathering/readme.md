This folder stores the scripts used to gather data. We did not upload them to github because they are very large.

RedditScraper.py works as follows : first, it uses pushshifts search to get all submission ids between the chosen dates. Then, it passes those id's to the praw API to download the data. Because it uses reddit.info(), it is able to download submission data for many posts using a single API call; this trick was taken from here: https://www.reddit.com/r/redditdev/comments/aoe4pk/praw_getting_multiple_submissions_using_by_id/ .

The wsb data from pushshift's search data, with fields updated using praw, since Jan 1 2021 is stored here: https://drive.google.com/file/d/1fcrXvG7tF-Fzv8xMREQz4N7a_Am2E9QB/view

Some caveats about this data:

1. Pushshift's ingest went down for various periods of time, and hasn't been backfilled. This leads to some gaps in the data that are apparent one plots a time series. See here for discussion: https://www.reddit.com/r/pushshift/comments/n38roy/missing_data/
2. Some of the field names are counter intuitive, e.g. score actual refers to the number of upvotes: https://praw.readthedocs.io/en/latest/code_overview/models/submission.html
3. Compared to the kaggle data here https://www.kaggle.com/gpreda/reddit-wallstreetsbets-posts/code , this dataset is missing about 9000 entries, but also has 500k+ entries from the overlapping time period that the kaggle data set does not have. See the end of the cleaning file for a comparison of the two.

