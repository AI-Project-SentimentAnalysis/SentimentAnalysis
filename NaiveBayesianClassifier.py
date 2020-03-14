from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load

from useful_components import TwitterDataSet, TrainedModel


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

    # which data should we save persistently?
    '''
    X_test(vectorized version of tweets, ready to plugin to get accuracy score,
    test_tweets (the test tweets themselves, to be used for demonstrating the classifier
    y_test (the correct polarities for the training set, ready to be used for calculating accuracy)
    vectorizer (this has the word-space-representation used for the model. Can be used to turn the demo 
                sample into the properly vectorized version expected by the classifier.
    classifier (obvs)
     dict-tweets-to-polarity (makes it easier to pull random sample of tweets and check their polarities)
    Good way to do this, that won't throw off the team: class to wrap this data up neatly. 
    '''
    print("building dict")
    pol_map = data_set.get_map_for_test_tweets_only()
    print("Dict built")

    trained_model = TrainedModel(MultinomialNaiveBayesianClassifier,data_set.vectorizer, data_set.test_tweets,
                                 data_set.X_test, data_set.y_test, pol_map)

    #to check size
    dump(data_set.vectorizer, "vectorizer.joblib")
    print("dumping")
    dump(trained_model,"trained_NB_model.joblib", compress=9)
    print("finished Dumping")

    return MultinomialNaiveBayesianClassifier, data_set


if __name__ == '__main__':
    classifier, twitter_data_obj = Bayesian_Sentiment_Analysis()
    tweet_to_polarity_map = twitter_data_obj.map_text_to_polarity()
    tweets_to_test = twitter_data_obj.test_tweets.sample(20).array

    tweets_series = pd.Series(tweets_to_test, name="Tweet Text")
    # tweets_to_test = twitter_data_obj.test_tweets.head(10).array

    # vectorize the tweets
    vec = twitter_data_obj.vectorizer
    tweet_vecs = vec.transform(tweets_to_test).toarray()

    # get predictions
    preds = classifier.predict(tweet_vecs)
    # build series for predictions
    pred_series = pd.Series(preds, name="model-predicted polarity")

    # build series for actual polarity ratings
    pols = list()
    for item in tweets_to_test:
        pols.append(tweet_to_polarity_map[item])
    pol_series = pd.Series(pols, name="Actual Polarity")

    X_test = twitter_data_obj.X_test
    y_test = twitter_data_obj.y_test
    print(classifier.score(X_test, y_test))



    # build dataFrame
    columns = [tweets_series.name, pred_series.name, pol_series.name]
    print(columns)
    results = pd.concat([tweets_series, pred_series, pol_series], ignore_index=True, axis=1)
    results.columns = columns

    print(results)
    results.to_csv('bayesian_results.csv')

