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
use_year = True #whether to include the year in the calculation
split = 0.25 #percent of training set to use for validation
test_size = 0.2 #percent of data set to use for testing
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

from sklearn.model_selection import train_test_split

def buildnets(subreddits):
    for subreddit in subreddits:
        data_path = f'../Data/subreddit_{subreddit}'
        model, accuracies, word_tokenizer, cleaned_df = PostClassificationModel(data_path = data_path, use_year = True)

        #save model
        model.save( f'../Data/subreddit_{subreddit}/NN_model.keras')


#For predicting time series of post popularity.
def encode_text(text, word_tokenizer, maxlen=maxlen):
    encoded = word_tokenizer.texts_to_sequences([text])
    return sequence.pad_sequences(encoded, maxlen=maxlen)

def timeseries(df, text, model, word_tokenizer):
    #get the minimum year appearing in the data set
    min_year = np.array(df.utc.apply(lambda x : x.year), dtype=int)

    df['date'] = df.utc.apply( lambda x : x.date())
    all_dates_utcs = df.date.unique()

    #all_dates_utc = [datetime.datetime.(x[0]+min_year,1,1) + datetime.timedelta(x[1]) for x in all_dates]
    encoded_text = encode_text(text,word_tokenizer)

    # Fixing a specific time for input on each day
    input_hour = np.array([12])
    input_minute = np.array([0])

    predict_list = []
    for d in all_dates_utcs:
        input_dayofweek = np.array([d.weekday()])
        input_dayofyear = np.array([d.timetuple().tm_yday-1])
        input_year = np.array([d.year-min_year])
        predict_list.append(model.predict([encoded_text, input_hour, input_dayofweek, input_minute, input_dayofyear, input_year])[0][0][0])
    plt.ylim(0,1)
    ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.xlabel("Date")
    plt.ylabel("Probability of Success")
    plt.scatter(all_dates_utcs, predict_list)

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



def make_embedding_matrix(word_tokenizer, embeddings_path):

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

    return weights_matrix, embedding_dims


#data_path should either be a string (the location of the subreddit data) or a pandas dataframe
#embeddings_path should be a string, the location of the embeddings file
#returns a pair: the model and a list [accuracy to beat, accuracy on validation set, accuracy on test set]
def PostClassificationModel(data_path, embeddings_path = embeddings_path, custom_seed=custom_seed,
                            max_features = max_features, maxlen = maxlen, batch_size = batch_size,
                            epochs = epochs, meta_embedding_dims = meta_embedding_dims,
                            dense_layer_size = dense_layer_size, text_cols_used = text_cols_used,
                            exclude_removed = exclude_removed, use_year = use_year, split = split,
                            test_size = test_size, optimization_quantity = optimization_quantity,
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

    print("size-2", len(df))
    #drop irrelevant data points

    df=DataSetup(df, exclude_removed=exclude_removed, drop_na_cols=text_cols_used)
    print("size-1", len(df))
    time_features = ['minute','hour','dayofweek', 'dayofyear' ]
    binary_features = ['year']

    for new_col in ['hour', 'minute', ]:
            df[new_col] = np.array(df.utc.apply(lambda x : getattr(x, new_col)), dtype=int)
    df['dayofweek'] = np.array(df.utc.apply(lambda x : x.weekday()), dtype=int)
    df['dayofyear'] = np.array(df.utc.apply(lambda x : x.timetuple().tm_yday), dtype=int)  - 1
    df['year'] = np.array(df.utc.apply(lambda x : x.year), dtype=int)
    df['year'] = df['year'] - df['year'].min()

    #extract data from the dataframe
    #find the median number of upvotes
    ups_median=np.median(df.ups)
    df['is_top_submission'] = np.array(df.ups.apply(lambda x : GoodPost(x,ups_median)), dtype=int)

    #titles is either the titles or the titles + selftext

    for col in text_cols_used:
        df[col] = " " +  df[col] + " "
    df["training_text"] = df[text_cols_used].sum(axis = 1)


    #process text
    word_tokenizer = Tokenizer(max_features)
    word_tokenizer.fit_on_texts(df["training_text"])
    text_seq = word_tokenizer.texts_to_sequences(df["training_text"])
    text_padded = pd.DataFrame(sequence.pad_sequences(text_seq, maxlen=maxlen), index = df.index)
    text_padded_cols = text_padded.columns
    features = list(text_padded_cols) + time_features + binary_features
    other_cols = time_features + binary_features
    print('features', features)
    print("size0", len(df))
    df = pd.concat([df, pd.DataFrame(text_padded)], axis = 1)
    #print('after padding', 472 in df.index)
    #print(df.columns)
    #print(text_padded)
    #set up pre-trained embeddings

    weights_matrix, embedding_dims = make_embedding_matrix(word_tokenizer, embeddings_path)


    def create_one_dimensional_layer(name, dimensions, meta_embedding_dims):
        input_layer = Input(shape=(1,), name=name)
        embedding = Embedding(dimensions, meta_embedding_dims)(input_layer)
        reshape = Reshape((meta_embedding_dims,))(embedding)

        return input_layer, reshape


    #returns the compiled model
    def create_model(use_year=use_year, maxlen=maxlen, max_features=max_features,
                     embedding_dims=embedding_dims,meta_embedding_dims=meta_embedding_dims,
                     model_loss=model_loss,model_optimizer=model_optimizer,
                     model_metrics=model_metrics, model_loss_weights=model_loss_weights):
        #title (and/or selftext) layers
        text_input = Input(shape=(maxlen,), name='text_input')
        text_embedding = Embedding(max_features + 1, embedding_dims, weights=[weights_matrix])(text_input)
        text_pooling = GlobalAveragePooling1D()(text_embedding)

        #auxiliary output to regularize text
        aux_output = Dense(1, activation='sigmoid', name='aux_out')(text_pooling)


        #set up time layers
        time_features = [('minute', 60),  ( 'hour', 24), ('dayofweek', 7), ('dayofyear', 366)]
        time_inputs = []
        time_reshapes = []
        for name, dimension in time_features:
            input_layer, reshape = create_one_dimensional_layer(name, dimension, meta_embedding_dims)
            time_reshapes.append(reshape)
            time_inputs.append(input_layer)


        binary_features  = ['year']
        binary_reshapes = []
        binary_inputs = []
        for feature in binary_features:
            layer = Input(shape=(1,), name=feature)
            binary_reshapes.append(layer)
            binary_inputs.append(layer)


        merged = concatenate([text_pooling] + time_reshapes + binary_reshapes)

        hidden_1 = Dense(dense_layer_size, activation='relu')(merged)
        hidden_1 = BatchNormalization()(hidden_1)

        main_output = Dense(1, activation='sigmoid', name='main_out')(hidden_1)

        input_layers = [text_input] + time_inputs + binary_inputs
        model = Model( inputs = input_layers, outputs = [main_output, aux_output])

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


    # permute the rows because of how keras takes the validation set from the end
    print("size1", len(df))
    df = df.sample(frac = 1, random_state = custom_seed)

    features_df = df[features]
    target = 'is_top_submission'
    target_df = df[target]
    print("size2", len(df))
    train_X,  test_X, train_y, test_y = train_test_split(features_df, target_df, test_size=test_size, random_state=custom_seed)
    print(len(train_X), len(train_y), len(test_X), len(test_y))
    #set up early stopping
    earlyStopping = EarlyStopping(monitor=optimization_quantity[0], min_delta=0, patience=early_stopping_patience, verbose=0, mode=optimization_quantity[1], restore_best_weights=True)

    def convert_df_for_keras(df, text_padded_cols, other_cols):
        text_columns = []
        for col in text_padded_cols:
            text_columns.append( np.array( df[col] ))
        text_array = np.stack(text_columns, axis = -1)

        inputs = [text_array]
        for col in other_cols:
            inputs.append( np.array( df[col]))
        return inputs
    #print(train_X)
    #print("converting")
    X_train = convert_df_for_keras(train_X, list(text_padded_cols), other_cols)
    print(len(X_train))

    history = model.fit(X_train, [train_y, train_y], batch_size=batch_size,epochs=epochs,validation_split=split, callbacks=[earlyStopping])


    ####print results:
    #print baseline accuracy
    print("Using the dummy classifier (assuming all posts are less than or equal to the median), the accuracy is: ")
    #from sklearn.dummy import DummyClassifier
    #dc = DummyClassifier()
    #dc.fit( features_df, target_df)
    acc_to_beat=1-np.mean(target_df)
    print("talk more about this")

    print(acc_to_beat)
    #print best validation accuracy
    print("The accuracy of the model on the validation set is: ")
    acc_on_val = max(history.history["val_main_out_accuracy"])
    print(acc_on_val)
    #get the accuracy on the test set
    print("The accuracy of the model on the test set is: ")

    acc_on_test = model.evaluate(convert_df_for_keras(test_X, list(text_padded_cols), other_cols), [test_y, test_y], verbose=0)[3]

    print(acc_on_test)

    #return the model and the accuracy information
    return model, [acc_to_beat, acc_on_val, acc_on_test], word_tokenizer, df
