from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
import pandas as pd
from math import sqrt
from useful_components import TwitterDataSet, save_persistant_model


def decision_tree_classifier():
    data_set = TwitterDataSet(TfIdfTransform_Bool=True)


    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    n_features = data_set.get_num_features()

    max_feat = int(sqrt(n_features))
    classifier = tree.DecisionTreeClassifier(max_features=n_features, min_samples_leaf=1,
                                             max_depth=1000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    vec_path = 'DecisionTree_persistent_model/vectorizer.joblib'
    model_path = 'DecisionTree_persistent_model/model.joblib'
    save_persistant_model(vec_path, model_path, classifier, data_set)


decision_tree_classifier()
