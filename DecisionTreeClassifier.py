from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
import pandas as pd
from math import sqrt
from useful_components import TwitterDataSet


def decision_tree_classifier():
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
    feature_count = vectorizer.fit_transform(
        tweets
    )
    n_features = vectorizer.vocabulary_.__len__()
    print(n_features)


    # find term frequencies
    feature_count_tfidf = TfidfTransformer(use_idf=False).fit_transform(feature_count)
    print(feature_count_tfidf.shape)
    print(polarities.shape)




    # split data set for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        feature_count_tfidf,
        polarities,
        train_size=0.90)

    max_feat = int(sqrt(n_features))
    classifier = tree.DecisionTreeClassifier(max_features=max_feat,
                                             max_depth=100, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))


decision_tree_classifier()
