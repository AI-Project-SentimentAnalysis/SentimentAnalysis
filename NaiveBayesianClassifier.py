import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from useful_components import TwitterDataSet, save_persistant_model


def Bayesian_Sentiment_Analysis():
    data_set = TwitterDataSet(clean_tweets_bool=False)
    # Load the DataSet

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    MultinomialNaiveBayesianClassifier = MultinomialNB()
    # train model
    MultinomialNaiveBayesianClassifier.fit(X_train, y_train)
    # predict values of Xtest
    y_pred = MultinomialNaiveBayesianClassifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    print(MultinomialNaiveBayesianClassifier.score(X_test, y_test))



    vec_path = "NB_persistent_model/vectorizer.joblib"
    model_path = "NB_persistent_model/trained_NB_model.joblib"
    save_persistant_model(vec_path, model_path, MultinomialNaiveBayesianClassifier, data_set)


    return MultinomialNaiveBayesianClassifier, data_set


if __name__ == '__main__':
    classifier, twitter_data_obj = Bayesian_Sentiment_Analysis()



