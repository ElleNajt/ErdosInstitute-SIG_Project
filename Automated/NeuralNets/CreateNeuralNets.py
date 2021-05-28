##set default values for all constants
##these can all be changed when calling PostClassificationModel

embeddings_path = '../Data/glove.6B.50d.txt'




max_features = 40000 #number of words to put in dictionary
maxlen=40 #number of words of title (or title+selftext combination) you will use
batch_size = 32 #batch size for training NN
epochs = 20 #number of epochs for training NN
meta_embedding_dims = 64 #dimension of the embedding for the time information
dense_layer_size = 256 #size of the final dense layer in the NN
text_cols_used=['title'] #which text columns to use
exclude_removed = True #exclude removed and deleted posts from the data set
use_year = False #whether to include the year in the calculation
split = 0.25 #percent of training set to use for validation
test_split = 0.2 #percent of data set to use for testing
optimization_quantity = ['val_main_out_accuracy','max'] #we want to maximize the accuracy on the validation set
early_stopping_patience = 5 #how soon to stop if accuracy is not improving
model_loss='binary_crossentropy'  #loss function used in model
model_optimizer='adam' #optimizer used in model
model_metrics=['accuracy'] #metric used to gauge model performance
model_loss_weights=[1, 0.2] #first weight is for main loss, second is for auxiliary loss (adjusting the word embedding)
custom_seed = 123




import numpy as np
import os
import csv
from random import random, sample, seed
import pandas as pd
from datetime import datetime
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.models import Input, Model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, concatenate, Activation
from keras.layers.core import Masking, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping


def buildnets(subreddits):
    for subreddit in subreddits:
        data_path = f'../Data/subreddit_{subreddit}'
        model, accuracies = PostClassificationModel(data_path = data_path, use_year = True)

        #save model
        model.save( f'../Data/subreddit_{subreddit}/NN_model.keras')

#returns 1 if ups>threshold, 0 otherwise
def GoodPost(ups,threshold=1):
    if ups>threshold:
        return 1
    return 0

#take dfog, restrict to only posts with selftext, exclude posts that were removed or deleted (if exclude_removed is True), and drop columns in drop_na_cols with NaN entries
def DataSetup(dfog, exclude_removed=True, drop_na_cols=['title']):
    if exclude_removed:
        tempdf=dfog.loc[(((dfog.removed_by_category.isnull()))) & ((dfog.is_self==True))]
        #tempdf=dfog.loc[(((dfog.removed_by_category.isnull()))) & ((dfog.is_self==True) & ~(dfog["title"].str.contains("Thread|thread|Sunday Live Chat|consolidation zone|Containment Zone|Daily Discussion|Daily discussion|Saturday Chat|What Are Your Moves Tomorrow|What Are Your Moves Today|MEGATHREAD",na=False)))]
    else:
        tempdf=dfog.loc[dfog.is_self==True]
        #tempdf=dfog.loc[((dfog.is_self==True) & ~(dfog["title"].str.contains("Thread|thread|Sunday Live Chat|consolidation zone|Containment Zone|Daily Discussion|Daily discussion|Saturday Chat|What Are Your Moves Tomorrow|What Are Your Moves Today|MEGATHREAD",na=False)))]
    tempdf=tempdf.dropna(subset = drop_na_cols)
    tempdf['utc']=tempdf.created_utc.apply(lambda x : datetime.utcfromtimestamp(x))
    return tempdf





#data_path should either be a string (the location of the subreddit data) or a pandas dataframe
#embeddings_path should be a string, the location of the embeddings file
#returns a pair: the model and a list [accuracy to beat, accuracy on validation set, accuracy on test set]
def PostClassificationModel(data_path, embeddings_path = embeddings_path, custom_seed=custom_seed,
                            max_features = max_features, maxlen = maxlen, batch_size = batch_size,
                            epochs = epochs, meta_embedding_dims = meta_embedding_dims,
                            dense_layer_size = dense_layer_size, text_cols_used = text_cols_used,
                            exclude_removed = exclude_removed, use_year = use_year, split = split,
                            test_split = test_split, optimization_quantity = optimization_quantity,
                            early_stopping_patience = early_stopping_patience, model_loss = model_loss,
                            model_optimizer = model_optimizer, model_metrics = model_metrics,
                            model_loss_weights = model_loss_weights):
    print("Starting Post Classification Model.")
    #if data_path is a string, read in the corresponding file as df. Otherwise we assume it's a pandas dataframe
    if type(data_path) == str:
        df = pd.read_pickle(data_path + "/full.pkl")
    else:
        df = data_path.copy()

    #change_point_information = pd.read_csv(data_path + "/Changepoints/results.csv")


    #drop irrelevant data points
    df=DataSetup(df, exclude_removed=exclude_removed, drop_na_cols=text_cols_used)

    #extract data from the dataframe
    #find the median number of upvotes
    ups_median=np.median(df.ups)
    #titles is either the titles or the titles + selftext, depending on the number of text_cols_used
    if len(text_cols_used)>1:
        titles = np.array(df.title + " " + df.selftext)
    else:
        titles = np.array(df[text_cols_used[0]])
    hours = np.array(df.utc.apply(lambda x : x.hour), dtype=int)
    minutes = np.array(df.utc.apply(lambda x : x.minute), dtype=int)
    dayofweeks = np.array(df.utc.apply(lambda x : x.weekday()), dtype=int)
    dayofyears = np.array(df.utc.apply(lambda x : x.timetuple().tm_yday), dtype=int)
    is_top_submission = np.array(df.ups.apply(lambda x : GoodPost(x,ups_median)), dtype=int)
    #zero-index dayofyears
    dayofyears_tf=dayofyears-1

    #if applicable, get year information and zero-index it
    if use_year:
        years = np.array(df.utc.apply(lambda x : x.year), dtype=int)
        years = years - min(years)


    #process text
    word_tokenizer = Tokenizer(max_features)
    word_tokenizer.fit_on_texts(titles)
    titles_tf = word_tokenizer.texts_to_sequences(titles)
    titles_tf = sequence.pad_sequences(titles_tf, maxlen=maxlen)


    #set up pre-trained embeddings
    embedding_vectors = {}
    with open(embeddings_path, 'r',encoding='latin-1') as f:
        for line in f:
            #print(line)
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            word = line_split[0]
            embedding_vectors[word] = vec

    embedding_dims = len(embedding_vectors['the'])

    weights_matrix = np.zeros((max_features + 1, embedding_dims))

    for word, i in word_tokenizer.word_index.items():

        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None and i <= max_features:
            weights_matrix[i] = embedding_vector





    #returns the compiled model
    def create_model(use_year=use_year, maxlen=maxlen, max_features=max_features,
                     embedding_dims=embedding_dims,meta_embedding_dims=meta_embedding_dims,
                     model_loss=model_loss,model_optimizer=model_optimizer,
                     model_metrics=model_metrics, model_loss_weights=model_loss_weights):
        #title (and/or selftext) layers
        titles_input = Input(shape=(maxlen,), name='titles_input')
        titles_embedding = Embedding(max_features + 1, embedding_dims, weights=[weights_matrix])(titles_input)
        titles_pooling = GlobalAveragePooling1D()(titles_embedding)

        #auxiliary output to regularize text
        aux_output = Dense(1, activation='sigmoid', name='aux_out')(titles_pooling)


        #set up time layers
        hours_input = Input(shape=(1,), name='hours_input')
        hours_embedding = Embedding(24, meta_embedding_dims)(hours_input)
        hours_reshape = Reshape((meta_embedding_dims,))(hours_embedding)

        dayofweeks_input = Input(shape=(1,), name='dayofweeks_input')
        dayofweeks_embedding = Embedding(7, meta_embedding_dims)(dayofweeks_input)
        dayofweeks_reshape = Reshape((meta_embedding_dims,))(dayofweeks_embedding)

        minutes_input = Input(shape=(1,), name='minutes_input')
        minutes_embedding = Embedding(60, meta_embedding_dims)(minutes_input)
        minutes_reshape = Reshape((meta_embedding_dims,))(minutes_embedding)

        dayofyears_input = Input(shape=(1,), name='dayofyears_input')
        dayofyears_embedding = Embedding(366, meta_embedding_dims)(dayofyears_input)
        dayofyears_reshape = Reshape((meta_embedding_dims,))(dayofyears_embedding)

        #if we are using the year, add those layers as well
        if use_year:
            years_input=Input(shape=(1,), name='years_input')
            years_reshape=years_input
            #create the appropriate merged layer
            merged = concatenate([titles_pooling, hours_reshape, dayofweeks_reshape, minutes_reshape, dayofyears_reshape, years_reshape])
        else:
            merged = concatenate([titles_pooling, hours_reshape, dayofweeks_reshape, minutes_reshape, dayofyears_reshape])


        hidden_1 = Dense(dense_layer_size, activation='relu')(merged)
        hidden_1 = BatchNormalization()(hidden_1)

        main_output = Dense(1, activation='sigmoid', name='main_out')(hidden_1)

        #compile the model
        if use_year:
            model = Model(inputs=[titles_input,
                                  hours_input,
                                  dayofweeks_input,
                                  minutes_input,
                                  dayofyears_input,
                                  years_input], outputs=[main_output, aux_output])
        else:
            model = Model(inputs=[titles_input,
                                  hours_input,
                                  dayofweeks_input,
                                  minutes_input,
                                  dayofyears_input], outputs=[main_output, aux_output])

        model.compile(loss=model_loss,
                      optimizer=model_optimizer,
                      metrics=model_metrics,
                      loss_weights=model_loss_weights)
        return model

    model = create_model(use_year=use_year, maxlen=maxlen, max_features=max_features,
                     embedding_dims=embedding_dims,meta_embedding_dims=meta_embedding_dims,
                     model_loss=model_loss,model_optimizer=model_optimizer,
                     model_metrics=model_metrics, model_loss_weights=model_loss_weights)


    ###train, validation, test split
    # returns randomized indices with no repeats
    num_rows=len(titles_tf)
    seed(custom_seed)
    idx = sample(range(num_rows), num_rows)

    #reindex the lists according to idx
    titles_tf = titles_tf[idx, :]
    hours = hours[idx]
    dayofweeks = dayofweeks[idx]
    minutes = minutes[idx]
    dayofyears_tf = dayofyears_tf[idx]
    is_top_submission = is_top_submission[idx]
    if use_year:
        years=years[idx]

    #find the end of the training set
    train_end = int((1-test_split)*num_rows)

    #create training and test sets
    titles_train = titles_tf[:train_end, :]
    titles_test = titles_tf[train_end:, :]
    hours_train = hours[:train_end]
    hours_test = hours[train_end:]
    dayofweeks_train = dayofweeks[:train_end]
    dayofweeks_test = dayofweeks[train_end:]
    minutes_train = minutes[:train_end]
    minutes_test = minutes[train_end:]
    dayofyears_train = dayofyears[:train_end]
    dayofyears_test = dayofyears[train_end:]
    is_top_submission_train = is_top_submission[:train_end]
    is_top_submission_test = is_top_submission[train_end:]

    if use_year:
        years = years[idx]
        years_train = years[:train_end]
        years_test = years[train_end:]
        training_data = [titles_train, hours_train, dayofweeks_train, minutes_train, dayofyears_train, years_train]
    else:
        training_data = [titles_train, hours_train, dayofweeks_train, minutes_train, dayofyears_train]



    #set up early stopping
    earlyStopping = EarlyStopping(monitor=optimization_quantity[0], min_delta=0, patience=early_stopping_patience, verbose=0, mode=optimization_quantity[1], restore_best_weights=True)




    history = model.fit(training_data, [is_top_submission_train, is_top_submission_train], batch_size=batch_size,epochs=epochs,validation_split=split, callbacks=[earlyStopping])


    ####print results:
    #print baseline accuracy
    print("Using the dummy classifier (assuming all posts are less than or equal to the median), the accuracy is: ")
    acc_to_beat=1-np.mean(is_top_submission[:(int(titles_tf.shape[0] * split))])
    print(acc_to_beat)
    #print best validation accuracy
    print("The accuracy of the model on the validation set is: ")
    acc_on_val = max(history.history["val_main_out_accuracy"])
    print(acc_on_val)
    #get the accuracy on the test set
    print("The accuracy of the model on the test set is: ")
    if use_year:
        acc_on_test = model.evaluate([titles_test, hours_test, dayofweeks_test, minutes_test, dayofyears_test,
                                      years_test], [is_top_submission_test, is_top_submission_test], verbose=0)[3]
    else:
        acc_on_test = model.evaluate([titles_test, hours_test, dayofweeks_test, minutes_test, dayofyears_test],
                                     [is_top_submission_test, is_top_submission_test], verbose=0)[3]
    print(acc_on_test)

    #return the model and the accuracy information
    return model, [acc_to_beat, acc_on_val, acc_on_test]
