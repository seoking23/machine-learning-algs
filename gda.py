"""
Author: Nathan Miller
Institution: UC Berkeley
Date: Spring 2022
Course: CS189/289A

A Template file for CS 189 Homework 3 question 8.

Feel free to use this if you like, but you are not required to!
"""
from locale import normalize
import numpy as np
import scipy
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

# TODO: Import any dependencies

class GDA:
    """Perform Gaussian discriminant analysis (both LDA and QDA)."""
    def __init__(self, *args, **kwargs):
        self._fit = False
        self.training_data_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        self.mean_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        self.cov_mat_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        self.linspace = np.linspace(0,1000)
        self.pi_c_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        #TODO: Possibly add new instance variables

    def evaluate(self, X, y, mode="lda"):
        """Predict and evaluate the accuracy using zero-one loss.

        Args:
            X (np.ndarray): The feature matrix shape (n, d)
            y (np.ndarray): The true labels shape (n,)

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            float: The accuracy loss of the learner.

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
        """
        #TODO: Compute predictions of trained model and calculate accuracy
        #Hint: call `predict` to simplify logic
        predictions = predict(X)
        accuracy = 0
        total_predictions = len(y)
        for x in range(total_predictions):
            if predictions[x] == y[x]:
                accuracy += 1
        
        return accuracy

    def fit(self, X, y):
        """Train the GDA model (both LDA and QDA).

        Args:
            X (np.ndarray): The feature matrix (n, d)
            y (np.ndarray): The true labels (n, )
        """
        #TODO: Train both the QDA and LDA model params based on the training data passed in
        # This will most likely involve setting instance variables that can be accessed at test time
        newX = normalize(X)
        for index, value in enumerate(newX):
            self.training_data_dict[y[index].tolist()[0]].append(value)
        for m_class in range(10):
            m_class_data = np.vstack(self.training_data_dict[m_class])
            mean = np.mean(m_class_data, axis=0)
            self.pi_c_dict[m_class] = np.count_nonzero(y == m_class)/len(y)
            cov_mat = np.cov(m_class_data.T)
            self.cov_mat_dict[m_class] = cov_mat
            self.mean_dict[m_class] = mean

        self._fit = True


    def predict(self, X, mode="lda"):
        """Use the fitted model to make predictions.

        Args:
            X (np.ndarray): The feature matrix of shape (n, d)

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            np.ndarray: The array of predictions of shape (n,)

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
            RuntimeError: If called before model is trained
        """
        if not self._fit:
            raise RuntimeError("Cannot predict for a model before `fit` is called")

        preds = None
        if mode == "lda":
            #TODO: Compute test-time preditions for LDA model trained in 'fit'
            combined_weighted_matrix = self.cov_mat_dict[0]
            mean_hat = self.mean_dict[0]
            for x in range(9):
                combined_weighted_matrix += self.cov_mat_dict[x+1]
            combined_weighted_matrix = combined_weighted_matrix/10
            multivariate_normal_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
            for x in range(10):
                mean_C = self.mean_dict[x]
                multivariate_normal_dict[x] = scipy.stats.multivariate_normal(self.linspace, mean_C, combined_weighted_matrix)
            for sample in X:
                prob_C = 0
                most_likely_c = 0
                for x in range(10):
                    prob_x = multivariate_normal_dict[x].logpdf(sample)
                    if prob_C < prob_x:
                        most_likely_c = x
                preds.append(most_likely_c)
                prob_C = 0
            
        elif mode == "qda":
            preds = None
        else:
            raise RuntimeError("Unknown mode!")
        return preds

    def normalize(data):
        normalized_data = []
        i = 0
        for sample in data:
            print(i)
            l2_norm = np.linalg.norm(sample)
            #print(sample)
            normalized_sample = [sample/l2_norm]
            #print(normalized_sample)
            normalized_data += normalized_sample
            i += 1
        return np.vstack(normalized_data)

