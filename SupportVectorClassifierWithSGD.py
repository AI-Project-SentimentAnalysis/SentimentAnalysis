from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from useful_components import TwitterDataSet
from sklearn import linear_model
from joblib import dump, load


def support_vec_classifier_with_stochastic_gradient_descent():
    data_set = TwitterDataSet(TfIdfTransform_Bool=True)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    classifier = linear_model.SGDClassifier(loss='hinge', # hinge is what makes it default to Support Vector Machine
                                            max_iter=1000, tol=1e-3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    dump(classifier, 'support_vec_cfr_with_sgd.joblib')
    return classifier




support_vec_classifier_with_stochastic_gradient_descent()
