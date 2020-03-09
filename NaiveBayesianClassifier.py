from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from SentimentAnalysis.useful_components import TwitterDataSet



def Bayesian_Sentiment_Analysis():
    data_set = TwitterDataSet()
    # Load the DataSet
    data_frame = data_set.data_frame

    tweets = pd.Series(data_frame['tweet_text'])
    polarities = pd.Series(data_frame["polarity"])
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

    #split data set for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        polarities,
        train_size=0.95)

    clf = MultinomialNB()
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))




Bayesian_Sentiment_Analysis()


