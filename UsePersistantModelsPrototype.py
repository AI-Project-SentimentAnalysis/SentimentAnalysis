from joblib import dump, load
from useful_components import TrainedModel
import pandas as pd

'''
class TrainedModel:
    def __init__(self, classifier, test_tweets, X_test, y_test, tweet_to_polarity_map):
        self.cfr = classifier
        self.test_tweets = test_tweets
        self.X_test = X_test
        self.y_test = y_test
        self.get_polarity = tweet_to_polarity_map
'''
trained_model_path = "NB_persistent_model/trained_NB_model.joblib"
vec_path = 'NB_persistent_model/vectorizer.joblib'

model = load(trained_model_path)
vec = load(vec_path)
classifier = model.cfr
test_tweets = model.test_tweets
X_test = model.X_test
y_test = model.y_test
get_polarity = model.get_polarity

tweets_to_test = test_tweets.sample(20).array

tweets_series = pd.Series(tweets_to_test, name="Tweet Text")
# tweets_to_test = twitter_data_obj.test_tweets.head(10).array

# vectorize the tweets
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
print(columns)
results = pd.concat([tweets_series, pred_series, pol_series], ignore_index=True, axis=1)
results.columns = columns

print(results)
results.to_csv('bayesian_results.csv')

print("Accuracy: ")
print(classifier.score(X_test, y_test))
