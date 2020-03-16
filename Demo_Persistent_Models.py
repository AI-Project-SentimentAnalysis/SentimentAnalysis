from os import path
from joblib import dump, load
import pandas as pd


'''
list of tuples in the following format:
    (name_of_model, 
    path_to_vectorizor, 
    path_to_model)
where each entry is a string
'''
list_of_models = [
    ('Naive Bayesian Classifier',
     'NB_persistent_model/vectorizer.joblib',
     'NB_persistent_model/trained_NB_model.joblib'),
    ('Logistic Regression Classifier',
     'LogRegression_persistent_model/vectorizer.joblib',
     'LogRegression_persistent_model/model.joblib'),
    ('Decision Tree Classifier',
     'DecisionTree_persistent_model/vectorizer.joblib',
     'DecisionTree_persistent_model/model.joblib'),
    ('Support Vector Machine - Stochastic Gradient Descent Version',
     'SVM_SGD_persistent_model/vectorizer.joblib',
     'SVM_SGD_persistent_model/model.joblib'),
    ('Support Vector Machine - LinearSVC version',
     'LinearSVC_persistent_model/vectorizer.joblib',
     'LinearSVC_persistent_model/model.joblib'),
    ('Random Forest Classifier',
     'RandomForest_persistent_model/vectorizer.joblib',
     'RandomForest_persistent_model/model.joblib')
]


def run_all_saved_models():
    for (model_name, path_to_vectorizer, path_to_model) in list_of_models:
        print("Loading the Model: " + model_name)
        run_pers_model(model_name, path_to_vectorizer, path_to_model)
        print("*****************************************************\n")





def run_pers_model(classifier_name, path_to_vec, path_to_model_obj):
    if not path.exists(path_to_vec) or not path.exists(path_to_model_obj):
        print("Missing required persistent data to run this model")
        print("Model: " + classifier_name)
        return
    #load model and vectorizor
    model = load(path_to_model_obj)
    vec = load(path_to_vec)
    print("Running the Model")


    #load data into well-labeled variables
    classifier = model.cfr
    test_tweets = model.test_tweets
    X_test = model.X_test
    y_test = model.y_test
    #get the map for mapping tweet to polarity
    get_polarity = model.get_polarity

    #print out accuracy for demo
    print("Accuracy Measured for \'" + classifier_name +"\': ")
    print(classifier.score(X_test, y_test))



    print("Saving a sample set, with the model predictions and actual polarity")
    #get a random sample of tweets FROM THE TEST SET, do NOT use the training set
    #If you choose not to use the given methods and objects
    tweets_to_test = test_tweets.sample(10).array

    #Turn sample into a labeled Series, prepped for csv file
    tweets_series = pd.Series(tweets_to_test, name="Tweet Text")

    # vectorize the tweets (in order to make them a valid input to the classifier
    tweet_vecs = vec.transform(tweets_to_test).toarray()

    # get predictions
    preds = classifier.predict(tweet_vecs)
    # build series for predictions
    pred_series = pd.Series(preds, name="model-predicted polarity")


    # build series for actual polarity ratings
    pols = list()
    for item in tweets_to_test:
        pols.append(get_polarity[item])
    pol_series = pd.Series(pols, name="Actual Polarity")

    # build dataFrame
    columns = [tweets_series.name, pred_series.name, pol_series.name]
    results = pd.concat([tweets_series, pred_series, pol_series], ignore_index=True, axis=1)
    results.columns = columns

    #write out the sample data to a csv file, so we can use it in the report as illustration
    #if desired
    file_path_for_results = classifier_name.replace(" ", "") + "_results.csv"
    print("FilePath for sample data: " + file_path_for_results)
    results.to_csv(file_path_for_results)
    print("Sample Data Saved")




run_all_saved_models()