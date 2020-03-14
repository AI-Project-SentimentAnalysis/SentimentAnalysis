import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import WordPunctTokenizer
from os import path
from joblib import dump, load



def save_persistant_model(vec_path, model_path, classifier, twitter_obj):
    print("building dict")
    pol_map = twitter_obj.get_map_for_test_tweets_only()
    print("Dict built")

    trained_model = TrainedModel(classifier, twitter_obj.test_tweets,
                                 twitter_obj.X_test, twitter_obj.y_test, pol_map)


    # Now use libdump to save the model and vectorizor persistantly
    print("dumping")
    dump(twitter_obj.vectorizer, vec_path, compress=9)

    dump(trained_model, model_path, compress=9)
    print("finished Dumping")




class TrainedModel:
    def __init__(self, classifier, test_tweets, X_test, y_test, tweet_to_polarity_map):
        self.cfr = classifier

        self.test_tweets = test_tweets
        self.X_test = X_test
        self.y_test = y_test
        self.get_polarity = tweet_to_polarity_map


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

        if clean_tweets_bool:
            if path.exists('CleanedDataSet.csv'):
                self.data_frame = build_cleaned_set()
            else:
                self.data_frame = build_full_data_set()
                self.data_frame = clean_data_set(self.data_frame)
        else:
            self.data_frame = build_full_data_set()
        self.tweets = pd.Series(self.data_frame['tweet_text'])
        self.polarities = pd.Series(self.data_frame["polarity"])
        if TfIdfTransform_Bool or not TfIdfTransform_Bool:
            self.vectorizer = TfidfVectorizer(analyzer='word',
                                              lowercase=False,
                                              ngram_range=(1, 2))
        #else:
        #    self.vectorizer = CountVectorizer(  # max_df=.9,
        #        analyzer='word',
        #        lowercase=False,
        #        ngram_range=(1, 2)
        #    )
        self.tfidfTransformer = TfidfTransformer(use_idf=False)
        self.train_tweets = None
        self.test_tweets = None
        self.get_test_tweet_polarity = None
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
        # for column_label, content in df.items():

        for ind in df.index:
            tweet = df.at[ind, 'tweet_text']

            value = df.at[ind, 'polarity']
            tweet_to_polarity_map[tweet] = value
        return tweet_to_polarity_map

    def get_map_for_test_tweets_only(self):
        if self.get_test_tweet_polarity:
            return self.get_test_tweet_polarity
        tweets = self.test_tweets.array
        pol = self.y_test.array
        get_pol = dict()
        for i in range(0, len(tweets)):
            get_pol[tweets[i]] = pol[i]

        self.get_test_tweet_polarity = get_pol
        return self.get_test_tweet_polarity



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


def build_cleaned_set():
    columns = ["polarity", "unique_id", "date", "query", "user_handle", "tweet_text"]
    filepath = 'CleanedDataSet.csv'
    df = pd.read_csv(filepath, engine='python', names=columns)
    return df


tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


def clean_data_set(df):
    for ind in df.index:
        tweet = df.at[ind, 'tweet_text']
        clean = tweet_cleaner(tweet)
        df.at[ind, 'tweet_text'] = clean

    return df


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    # letters_only = re.sub("[^a-zA-Z]", " ", clean)
    # lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    # words = tok.tokenize(lower_case)
    # return (" ".join(words)).strip()
    return clean


if __name__ == '__main__':
    data = TwitterDataSet()
    new_data = clean_data_set(data.data_frame)
    new_data.to_csv('CleanedDataSet.csv', index=False)
