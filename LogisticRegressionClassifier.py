from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas
from useful_components import TwitterDataSet, save_persistant_model


def build_stanford_set():
    columns = ["polarity", "unique_id", "date", "query", "user_handle", "tweet_text"]
    filepath = 'StanfordDataSet/testdata.manual.2009.06.14.csv'
    df = pandas.read_csv(filepath, engine='python', names=columns)
    return df


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
    '''
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
    '''
    data_set = TwitterDataSet()
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    # initialize model

    classifier = LogisticRegression()
    # train model
    classifier.fit(X=X_train, y=y_train)
    # label the evaluation set
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    vec_path = 'LogRegression_persistent_model/vectorizer.joblib'
    model_path = 'LogRegression_persistent_model/model.joblib'
    save_persistant_model(vec_path, model_path, classifier, data_set)

    return classifier


sentiment_analysis()
