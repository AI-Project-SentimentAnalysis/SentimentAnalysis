from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from useful_components import TwitterDataSet


def Bayesian_Sentiment_Analysis():
    data_set = TwitterDataSet()
    # Load the DataSet

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    MultinomialNaiveBayesianClassifier = MultinomialNB()
    #train model
    MultinomialNaiveBayesianClassifier.fit(X_train, y_train)
    #predict values of Xtest
    y_pred = MultinomialNaiveBayesianClassifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    return MultinomialNaiveBayesianClassifier


Bayesian_Sentiment_Analysis()
