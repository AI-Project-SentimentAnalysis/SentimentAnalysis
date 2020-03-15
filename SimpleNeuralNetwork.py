#Cameron Eng
#Kelsey Larson
#Program was pair programmed

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from useful_components import TwitterDataSet

def neural_network():
    data_set = TwitterDataSet(TfIdfTransform_Bool=True)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,5),activation='tanh', max_iter=1000, random_state=1)
    #Note: Default iteration size of 200 will yeild an accruracy of ~80%

    clf.fit(X_train,y_train)

    y_predict = clf.predict(X_test)

    print(accuracy_score(y_test,y_predict))
    return clf, data_set

if __name__ == '__main__':
    classifier, twitter_data_obj = neural_network();
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
    results.columns = columns

    print(results)