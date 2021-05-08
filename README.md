# ErdosInstitute-SIG_Project
Bootcamp Project For Erdos Institute


# Files:

RedditData.ipynb = Notebook for scraping reddit data.
RedditScraper.py = Python script that scrapes reddit data. 

The pickled dataframes where made with a python 3.8.6 build, with pandas version (?). (NB: Some of the files here were made on 3.6, this is only temporary.)

You also need to have praw version 7.2.0 for the dataframes to unpickle correctly.


# API Keys:

To use this, you need an "API.env" file formatted like:

```
reddit_client_id = "your reddit API id"
reddit_client_secret = "your reddit API secret key" 
user_agent = "your user agent"
```

.env files are in the gitignore.
