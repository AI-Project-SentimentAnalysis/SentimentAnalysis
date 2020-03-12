import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import WordPunctTokenizer

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
    data_frame = None

    def __init__(self, TfIdfTransform_Bool=False, train_proportion=.90, clean_tweets_bool=False):
        self.data_frame = build_full_data_set()
        self.tweets = pd.Series(self.data_frame['tweet_text'])
        self.polarities = pd.Series(self.data_frame["polarity"])
        if clean_tweets_bool:
            df = clean_data_set(self.tweets)
            self.data_frame = pd.DataFrame(df)

        if TfIdfTransform_Bool:
            self.vectorizer = TfidfVectorizer(analyzer='word',
                                              lowercase=False)
        else:
            self.vectorizer = CountVectorizer(  # max_df=.9,
                analyzer='word',
                lowercase=False,
            )
        self.tfidfTransformer = TfidfTransformer(use_idf=False)
        self.train_tweets = []
        self.test_tweets = []
        self.X_train, self.X_test, self.y_train, self.y_test = vectorize_words(dataSetObj=self,
                                                                               train_proportion=train_proportion)

    def get_dataframe_format_of_data_set(self):
        return self.data_frame

    def get_num_features(self):
        return self.vectorizer.vocabulary_.__len__()

    def get_test_train_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def map_text_to_polarity(self):
        tweet_to_polarity_map = dict()
        df = self.data_frame
        #for column_label, content in df.items():

        for ind in df.index:
            tweet = df.at[ind, 'tweet_text']

            value = df.at[ind, 'polarity']
            tweet_to_polarity_map[tweet] = value
        return tweet_to_polarity_map


def vectorize_words(dataSetObj, train_proportion=.95):
    data_frame = dataSetObj.data_frame
    tweets = dataSetObj.tweets
    polarities = dataSetObj.polarities
    # extract features
    vectorizer = dataSetObj.vectorizer

    # split data prior to vectorizing
    dataSetObj.train_tweets, dataSetObj.test_tweets, y_train, y_test = train_test_split(
        tweets,
        polarities,
        train_size=train_proportion)

    # Vectorize the tweets
    X_train = vectorizer.fit_transform(dataSetObj.train_tweets)
    X_test = vectorizer.transform(dataSetObj.test_tweets)

    # find term frequencies

    '''
    if tf_bool:
        feature_tfidf = dataSetObj.tfidfTransformer.fit_transform(feature_matrix)
        X_train, X_test, y_train, y_test = train_test_split(
            feature_tfidf,
            polarities,
            train_size=train_proportion)
    else:
        # split data set for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            polarities,
            train_size=train_proportion)
    '''

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
    data_set = pd.concat(segment_list, ignore_index=True)
    return data_set


def build_stanford_set():
    columns = ["polarity", "unique_id", "date", "query", "user_handle", "tweet_text"]
    filepath = 'StanfordDataSet/testdata.manual.2009.06.14.csv'
    df = pd.read_csv(filepath, engine='python', names=columns)
    return df


tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


def clean_data_set(tweets_series):
    index = 0
    for item in tweets_series.array:
        if index < 10:
            print(str(index) + " " + str(item))
        clean_tweet = tweet_cleaner(item)
        index += 1

    index = 0
    for index, item in tweets_series.items():
        if index < 10:
            print(str(index) + " " + str(item))
        index += 1

    return tweets_series


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()







