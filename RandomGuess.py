from sklearn.metrics import accuracy_score
from math import sqrt
from useful_components import TwitterDataSet, save_persistant_model
from sklearn.dummy import DummyClassifier


def random_guess():
    data_set = TwitterDataSet(TfIdfTransform_Bool=True)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    classifier = DummyClassifier(strategy="uniform", random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    vec_path = 'RandomGuess_persistent_model/vectorizer.joblib'
    model_path = 'RandomGuess_persistent_model/model.joblib'
    save_persistant_model(vec_path, model_path, classifier, data_set)

random_guess()
