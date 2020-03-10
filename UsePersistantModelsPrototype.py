from joblib import dump, load
from useful_components import TwitterDataSet

dataset = TwitterDataSet()

SVM_with_SGD_file = 'support_vec_cfr_with_sgd.joblib'

classifier = load(SVM_with_SGD_file)

classifier.predict(dataset.data_frame['tweet_text'][90])
