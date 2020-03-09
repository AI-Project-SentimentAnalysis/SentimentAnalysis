from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from useful_components import TwitterDataSet
from sklearn.svm import LinearSVC


def support_vec_classifier():
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
    print(polarities.shape)
    print(feature_count.shape)

    # find term frequencies
    feature_count_tfidf = TfidfTransformer(use_idf=False).fit_transform(feature_count)
    print(feature_count_tfidf.shape)
    print(polarities.shape)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        feature_count_tfidf,
        polarities,
        train_size=0.95)

    classifier = LinearSVC(max_iter=1000, C=.5, loss='hinge', intercept_scaling=10)
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    return model


support_vec_classifier()
