import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

'''
The data is a CSV with emoticons removed. Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is co
"""
'''
'''
The format of each line in the csv files is as follows:
"polarity", "unique_id", "date", "query", "user_handle", "tweet_text"
'''


class TwitterDataSet:
    data_frame = pd.DataFrame()

    def __init__(self, TfIdfTransform_Bool=False, train_proportion=.90):
        self.data_frame = build_full_data_set()

        self.vectorizer = CountVectorizer(#max_df=.9,
                                          analyzer='word',
                                          lowercase=False,
                                          )
        self.tfidfTransformer = TfidfTransformer(use_idf=False)
        self.X_train, self.X_test, self.y_train, self.y_test = vectorize_words(dataSetObj=self,
                                                                               tf_bool=TfIdfTransform_Bool,
                                                                               train_proportion=train_proportion)

    def get_dataframe_format_of_data_set(self):
        return self.data_frame

    def get_test_train_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def map_text_to_polarity(self):
        tweet_to_polarity_map = dict()
        df = self.data_frame
        for ind in df.index:
            key = df['tweet_text'][ind]
            value = df['polarity'][ind]
            tweet_to_polarity_map[key] = value


def vectorize_words(dataSetObj, tf_bool=False, train_proportion=.95):
    data_frame = dataSetObj.data_frame
    tweets = pd.Series(data_frame['tweet_text'])
    polarities = pd.Series(data_frame["polarity"])
    # extract features
    vectorizer = dataSetObj.vectorizer

    feature_count = vectorizer.fit_transform(
        tweets
    )
    # find term frequencies
    if tf_bool:
        feature_count_tfidf = dataSetObj.tfidfTransformer.fit_transform(feature_count)
        X_train, X_test, y_train, y_test = train_test_split(
            feature_count_tfidf,
            polarities,
            train_size=train_proportion)
    else:
        # split data set for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            feature_count,
            polarities,
            train_size=train_proportion)

    return X_train, X_test, y_train, y_test


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
        df = pd.read_csv('DataSet/' + file, names=columns)
        segment_list.append(df)
    # concatenate the dataframes, so we hav e one complete dataset
    data_set = pd.concat(segment_list)
    return data_set
