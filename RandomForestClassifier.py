from sklearn.metrics import accuracy_score
from math import sqrt
from useful_components import TwitterDataSet, save_persistant_model
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier():
    data_set = TwitterDataSet(TfIdfTransform_Bool=True)

    # split data set for training and testing
    X_train, X_test, y_train, y_test = data_set.get_test_train_split()

    n_features = data_set.get_num_features()

    max_feat = int(sqrt(n_features))
    classifier = RandomForestClassifier(max_depth=1002, random_state=0, n_estimators=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    vec_path = 'RandomForest_persistent_model/vectorizer.joblib'
    model_path = 'RandomForest_persistent_model/model.joblib'
    save_persistant_model(vec_path, model_path, classifier, data_set)

random_forest_classifier()
