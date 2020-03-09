from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from useful_components import TwitterDataSet
from sklearn import linear_model


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

    classifier = linear_model.SGDClassifier(loss='hinge', # hinge is what makes it default to Support Vector Machine
                                            max_iter=1000, tol=1e-3)
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))


support_vec_classifier()
