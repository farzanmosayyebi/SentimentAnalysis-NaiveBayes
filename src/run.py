from template import NaiveBayesClassifier
import pandas as pd
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os


def preprocess(tweet_string):
    # clean the data and tokenize it
    stop_words = set(stopwords.words("english"))

    words = word_tokenize(tweet_string.lower().strip()) 
    features = [word for word in words if word.isalpha() and word not in stop_words]
    
    return features

def load_data(data_path):
    # load the csv file and return the data
    
    data = []
    read = pd.read_csv(data_path)
    
    for i in range(len(read)):
        data.append([str(read["text"][i]), str(read["label_text"][i]).strip()])
    
    return data

def load_test_data(data_path):
    data = []
    read = pd.read_csv(data_path)
    
    for i in range(len(read)):
        data.append(str(read["text"][i]))
    
    return data

def preprocess_train_data(train_data):
    for i in range(len(train_data)):
        train_data[i][0] = preprocess(train_data[i][0])
    return list(map(tuple, train_data))

def run_evaluation(eval_data, nb_classifier: NaiveBayesClassifier):
    total_data = len(eval_data)
    correct_classifications = 0
    wrong_classifications = 0

    for tweet in eval_data:
        label = nb_classifier.classify(preprocess(tweet[0]))
        if label == tweet[1]:
            correct_classifications += 1
        else:
            wrong_classifications += 1

    accuracy = (correct_classifications / total_data) * 100

    return accuracy

def run_test(test_data, nb_classifier: NaiveBayesClassifier):
    labels = []
    
    for tweet in test_data:
        label = nb_classifier.classify(preprocess(tweet))
        labels.append(label)
    
    return labels
    
def save_results(labels, result_path):

    with open(result_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

# train your model and report the duration time
train_data_path = os.path.join(os.getcwd(), "train_data.csv") 
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)

print("tokenizing train data...")
train_data = preprocess_train_data(load_data(train_data_path))
print("done.")

print("Training in progress...")
tic = time.perf_counter()

nb_classifier.train(train_data)

toc = time.perf_counter()
print(f"Training completed in {toc - tic:0.04f} seconds.")

print("Evaluation in progress...")
eval_data_path = os.path.join(os.getcwd(), "eval_data.csv")
eval_accuracy = run_evaluation(load_data(eval_data_path), nb_classifier)
print(f"Evaluation completed with an accuracy of %{eval_accuracy:0.04f}")

print("Test in progress...")
test_data_path_noLabel = os.path.join(os.getcwd(), "test_data_nolabel.csv")
result = run_test(load_test_data(test_data_path_noLabel), nb_classifier)
print(f"Test completed.")

result_path = os.path.join(os.getcwd(), "result.txt")
print("saving results...")
save_results(result, result_path)
print(f"results were saved at {result_path}")