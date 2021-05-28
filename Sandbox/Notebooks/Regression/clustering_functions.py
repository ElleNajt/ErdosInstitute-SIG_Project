#given the corpus_df, runs clustering. 
#Returns new dataframe, a copy of corpus_df with added clustering column (called 'prediction')
def Clust(corpus_df, rough_cluster_size=200, num_pc=0.95,random_state=42, inflation=0):
    
    newdf=corpus_df.copy()
    X = np.vstack(newdf['vector'].to_numpy())
    pca=PCA(n_components = num_pc)
    pca.fit(X)
    cluster_model = KMeans(n_clusters= int(len(newdf)/rough_cluster_size),verbose=0, random_state=random_state)
    trans=pca.transform(X)
    cluster_model.fit(trans)
    newdf['prediction'] = cluster_model.predict(trans) + inflation
    return newdf


#iteratively run Clust until all clusters are below max_cluster_size
def MaxClust(corpus_df, rough_cluster_size=200, num_pc=0.95,random_state=42, inflation=0, max_cluster_size=1000):
    #runs clust, adding prediction column to corpus_df
    newdf=Clust(corpus_df, rough_cluster_size, num_pc,random_state, inflation)
    print(newdf.prediction.value_counts())
    #set inflation (the smallest cluster number available for the future)
    inf=max(newdf.prediction) + 1
    #for each cluster
    for clust in set(newdf.prediction):
        #if there are more than max_cluster_size elements in the cluster:
        if newdf.prediction.value_counts()[clust]>max_cluster_size:
            print(clust)
            #do MaxClust on it, using and then updating inf
            tempdf=MaxClust(newdf.loc[newdf.prediction==clust],rough_cluster_size=rough_cluster_size, 
                     num_pc=num_pc,random_state=random_state, inflation=inf,
                     max_cluster_size=max_cluster_size)
            inf=max(newdf.prediction)+1
            #for each row of newdf that coincides with a row of tempdf, change the value
            for index in tempdf.index.values:
                newdf.at[index,'prediction']=tempdf.prediction[index]
            
    return newdf

#given a word and a corpus dataframe with 'prediction' column
#return all words in that cluster
def GetRelated(df,word):
    return list(df.loc[df.prediction==list(df.loc[df['word']==word].prediction)[0]][0])


#given a paragraph (tokenized, a list of lists of strings), check if it has any of the words in the cluster
def HasCluster(text_list, cluster_list):
    #return any([w in text_list for w in cluster_list])
    return any([any([w in text_list[i] for i in range(len(text_list))]) for w in cluster_list])

#given a series, create a new series telling whether each entry contains something in the same cluster as the given word
def ColHasCluster(col,cluster_df,word):
    cluster_list=GetRelated(cluster_df,word)
    return col.apply( lambda x : HasCluster(x,cluster_list))

#given a dataframe, sample the means many times and plot
#returns the average of these sample means
def SampleMeanPlot(df_subset, bins=1000, sample_size=100, num_samples=10000):
    values=[np.mean(df_subset.sample(sample_size).ups) for i in range(num_samples)]
    plt.figure(figsize = (8,8))
    plt.hist(values, bins=bins)
    
    plt.show()
    return np.mean(values)

#given corpus_df with 'prediction' column and a cluster number, return the list of words in that cluster
def WordsInCluster(clust_df,cluster):
    return list(clust_df.loc[clust_df.prediction==cluster]['word'])

#return all posts in the given cluster
def PostsInCluster(df,col_name,clust_df,cluster):
    print(cluster)
    cluster_list=WordsInCluster(clust_df,cluster)
    temp=df.loc[df[col_name].apply(lambda x : HasCluster(x,cluster_list))]
    print(len(temp))
    return temp

#given a dataframe df, a corpus dataframe with prediction column clust_df, 
#and the name of a column in df (tokenized_title or tokenized_selftext, or the sum),
#return a dataframe with rows indexed by the clusters, containing useful statistics about the clusters:
#the posts in the cluster, the number of such posts, the mean, median, and variance
def ClusterStats(df, col_name,clust_df):
    cluster_set=set(clust_df.prediction.values)
    cluster_stats=pd.DataFrame(index=cluster_set)
    cluster_stats['cluster_name']=cluster_stats.index
    print(cluster_stats)
    cluster_stats['posts']=cluster_stats.cluster_name.apply(lambda x : np.array(PostsInCluster(df,col_name,clust_df,x).ups))
    cluster_stats['num_posts']=cluster_stats.posts.apply(lambda x : len(x))
    cluster_stats['mean']=cluster_stats.posts.apply(lambda x : np.mean(x))
    cluster_stats['median']=cluster_stats.posts.apply(lambda x : np.median(x))
    cluster_stats['var']=cluster_stats.posts.apply(lambda x : np.var(x))
    return cluster_stats