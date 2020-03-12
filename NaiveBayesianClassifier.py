from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from useful_components import TwitterDataSet


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

    return MultinomialNaiveBayesianClassifier, data_set


if __name__ == '__main__':
    classifier, twitter_data_obj = Bayesian_Sentiment_Analysis()
    tweet_to_polarity_map = twitter_data_obj.map_text_to_polarity()
    tweets_to_test = twitter_data_obj.test_tweets.sample(10).array

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

    # build dataFrame
    columns = [tweets_series.name, pred_series.name, pol_series.name]
    print(columns)
    results = pd.concat([tweets_series, pred_series, pol_series], ignore_index=True, axis=1)
    results = pd.DataFrame(results, columns=columns)

    print(results)
    results.to_csv('bayesian_results.csv')

'''
print("Tweet\t\t\tModels Polarity Label\t\tActual Polarity Label")
for i in range(0, 9):
    tw = tweets_to_test[i]
    p = preds[i]
    actual = tweet_to_polarity_map[tw]
    print(tw + "\t" + str(p) + "\t\t" + str(actual))

'''
