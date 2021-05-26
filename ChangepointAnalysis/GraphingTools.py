

def plot_single_word_prop(wsb, statistic  = "sum", keyword = "gme"):

    wsb['contains_keyword'] =  wsb.tokenized_title.apply(lambda x : contains_word(x, keyword))

    agged_withgme = wsb[wsb.contains_keyword][['date', 'ups']].groupby("date").agg(statistic)
    agged_withoutgme = wsb[~wsb.contains_keyword][['date', 'ups']].groupby("date").agg(statistic)
    agged = agged_withgme/(agged_withgme + agged_withoutgme)

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.lineplot(ax=ax, data = agged, x = "date", y = "ups", label = "filtered")


    agged = wsb[['date', 'ups']].groupby("date").agg(statistic)
    agged = agged / agged.ups.max()
    sns.lineplot(ax=ax, data = agged, x = "date", y = "ups", label = 'total', alpha = .1)
    plt.legend()




def plot_word_prop(wsb, keywords ,statistic  = "sum",):

    fig, ax = plt.subplots(figsize=(15, 9))
    for keyword in keywords:

        wsb['contains_keyword'] =  wsb.tokenized_title.apply(lambda x : contains_word(x, keyword))

        agged_withgme = wsb[wsb.contains_keyword][['date', 'ups']].groupby("date").agg(statistic)
        agged_withoutgme = wsb[~wsb.contains_keyword][['date', 'ups']].groupby("date").agg(statistic)
        agged = agged_withgme/(agged_withgme + agged_withoutgme)

        sns.lineplot(ax=ax, data = agged, x = "date", y = "ups", label = keyword)

    agged = wsb[['date', 'ups']].groupby("date").agg(statistic)
    agged = agged / (2 * agged.ups.max())
    sns.lineplot(ax=ax, data = agged, x = "date", y = "ups", label = 'total', alpha = .5, color = "black")

    plt.legend()
