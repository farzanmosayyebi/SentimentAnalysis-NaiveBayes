# Naive Bayes 3-class Classifier 
# Authors: Baktash Ansari - Sina Zamani 

# complete each of the class methods  

import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  

        self.classes = classes
        self.class_word_counts = {}
        self.class_counts = {}
        self.vocab = set()

        for c in classes:
            self.class_counts[c] = 0
            self.class_word_counts[c] = {}

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            self.class_counts[label] += 1

            for word in features:
                if word not in self.vocab:
                    self.vocab.add(word)

                if word in list(self.class_word_counts[label].keys()):
                    self.class_word_counts[label][word] += 1
                else:
                    self.class_word_counts[label][word] = 1

    def calculate_prior(self):
        # calculate log prior
        # you can add some attributes to this method

        num_of_positives = self.class_counts["positive"]
        num_of_negatives = self.class_counts["negative"]
        num_of_neutrals = self.class_counts["neutral"]

        alpha = 2 # smoothing factor in order to avoid divide-by-zero situations
        
        log_prior = np.log((num_of_positives + num_of_neutrals + alpha) / (num_of_negatives + num_of_neutrals + alpha))

        return log_prior

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        # return the corresponding value

        v_class = 6

        if (word not in list(self.class_word_counts[label].keys())):
            self.class_word_counts[label][word] = 1
        
        freq = self.class_word_counts[label][word]
        likelihood = (freq + 1) / (self.class_counts[label] + v_class)

        return likelihood

    def classify(self, features):
        # predict the class
        # inputs: features(list) --> words of a tweet 
        best_class = None

        score = 0
        
        log_prior = self.calculate_prior()

        for feature in features:
            p_positive = self.calculate_likelihood(feature, "positive")
            p_negative = self.calculate_likelihood(feature, "negative")
            p_neutral = self.calculate_likelihood(feature, "neutral")
           
            lambda_param = np.log((p_positive + p_neutral) / (p_negative + p_neutral))
            score += lambda_param
            
        score += log_prior

        if (score < -0.5):
            best_class = "negative"
        elif (-0.5 <= score <= 0.5):
            best_class = "neutral"
        else:
            best_class = "positive"

        return best_class
    

# Good luck :)
