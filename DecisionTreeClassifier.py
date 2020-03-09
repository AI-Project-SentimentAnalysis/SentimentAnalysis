from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
from SentimentAnalysis.useful_components import TwitterDataSet



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
    features = vectorizer.fit_transform(
        tweets
    )
    print(polarities.shape)
    print(features.shape)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        polarities,
        train_size=0.95)


    classifier = tree.DecisionTreeClassifier()
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))


decision_tree_classifier()
