def popular_words(wsb, date, numwords):
    day_df = wsb[ wsb.date == date]
    corpus = []
    for tokenized in day_df['tokenized_title']:
        corpus += tokenized

    dictionary = gensim.corpora.dictionary.Dictionary(documents = corpus)
    #dictionary.filter_n_most_frequent(len(dictionary) - numwords)


    items = list(dictionary.items())
    items.sort( key = lambda x : dictionary.cfs[x[0]], reverse = True)
    return [x[1] for x in items[:numwords]]

def frequency_dictionary(wsb, date, numwords):
    day_df = wsb[ wsb.date == date]
    corpus = []
    for tokenized in day_df['tokenized_title']:
        corpus += tokenized

    dictionary = gensim.corpora.dictionary.Dictionary(documents = corpus)

    return dictionary.cfs


pop_words_10 = [ popular_words(wsb, x, 200 ) for x in dates]
import itertools
pop_words = list(set(itertools.chain.from_iterable(pop_words_10)))
pop_words
