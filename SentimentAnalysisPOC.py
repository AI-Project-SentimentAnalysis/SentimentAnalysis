from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas
import numpy as np
import random
import matplotlib.pyplot as plt
from SentimentAnalysis.useful_components import TwitterDataSet


'''
This function returns a dataframe object with the full data set in it. 
'''


def build_stanford_set():
    columns = ["polarity", "unique_id", "date", "query", "user_handle", "tweet_text"]
    filepath = 'StanfordDataSet/testdata.manual.2009.06.14.csv'
    df = pandas.read_csv(filepath, engine='python', names=columns)
    return df


def build_full_data_set():
    # Define a filename.

    filename = "DataSet/manifest.txt"
    # Open the file as f.
    # The function readlines() reads the file.
    with open(filename) as f:
        csv_file_list = f.read().splitlines()
    columns = ["polarity", "unique_id", "date", "query", "user_handle", "tweet_text"]
    segment_list = list()
    for file in csv_file_list:
        df = pandas.read_csv('DataSet/' + file, names=columns)
        segment_list.append(df)
    # concatenate the dataframes, so we hav e one complete dataset
    data_set = pandas.concat(segment_list)
    return data_set


# build list of tweets and their polarity ratings.
"""
The data is a CSV with emoticons removed. Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is co
"""


def sentiment_analysis():
    data_set = TwitterDataSet()
    # Load the DataSet
    data_frame = data_set.data_frame
    # only need polarity and tweet text, so drop other columns

    # Drop columns from the DataFrame that are not needed, we only need the
    # tweet text and the polarities
    pruned_data = data_frame.drop(columns=["unique_id", "date", "query", "user_handle"])
    tweets = data_frame['tweet_text']
    polarities = pandas.Series(data_frame["polarity"])

    # print(pruned_data)

    # extract features
    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(
        tweets
    )
    print(polarities.shape)
    print(features.shape)
    features_nd = features.nonzero()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        polarities,
        train_size=0.90)

    # initialize model
    log_model = LogisticRegression()
    # train model
    log_model = log_model.fit(X=X_train, y=y_train)
    # label the evaluation set
    y_pred = log_model.predict(X_test)

    # check accuracy by eyeballing it
    data = tweets.tolist()

    '''
    X_test = X_test.toarray()
    j = random.randint(0, len(X_test) - 7)
    for i in range(j, j + 7):
        print(y_pred[0])
        ind = features_nd.tolist().index(X_test[i].tolist())
        print(data[ind].strip())
    '''
    print(accuracy_score(y_test, y_pred))

sentiment_analysis()
