from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from useful_components import TwitterDataSet
from sklearn.svm import LinearSVC
from sklearn import svm


def support_vec_classifier():
    data_set = TwitterDataSet(TfIdfTransform_Bool=True)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    classifier = LinearSVC(max_iter=10000, C=.8, loss='hinge', intercept_scaling=1)
    # classifier = svm.SVC(C=.5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))


support_vec_classifier()
