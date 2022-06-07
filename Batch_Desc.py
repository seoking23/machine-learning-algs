from collections import defaultdict
import glob
from math import comb
from operator import matmul
import random
from re import L
from termios import VDISCARD
from textwrap import fill
from urllib.parse import _NetlocResultMixinBase
import scipy
from scipy import stats
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

np.random.seed(3)



def data_partition():
    

    mat_contents = scipy.io.loadmat("data_wine.mat")
    
    training_data = mat_contents["X"]
    validation_set_size = int(6000 * 0.2)
    mean_tr = np.mean(training_data, axis=0)
    
    std_tr = np.std(training_data, axis=0)
    test_data = mat_contents["X_test"]
    test_data = normalize(test_data, mean_tr, std_tr)
    
    training_data_normalized = normalize(mat_contents["X"], mean_tr, std_tr)
    training_images = np.array(tuple(zip(training_data_normalized, np.array(mat_contents["y"]))), dtype=object)
    shuffled_training = np.random.permutation(training_images)
    validation_set = shuffled_training[:validation_set_size]
    shuffled_training = shuffled_training[validation_set_size:]

    return validation_set, shuffled_training, test_data

def normalize(data, mean, std):
    normalized_data = []
    i = 0
    return (data-mean)/std
    
    
validation_set, shuffled_training, test_data = data_partition()
validation_data, validation_labels = validation_set.T
validation_data = np.stack(validation_data, axis=0)
validation_labels = np.concatenate(validation_labels, axis = 0)
training_data, training_labels = shuffled_training.T
training_labels = np.concatenate( training_labels, axis=0)
training_data = np.stack(training_data, axis=0)





def find_s(X,w):
    
    return scipy.special.expit(X @ w)

def new_Omega(s):
    return np.diag(s*(1-s))

def find_e_batch(X, y, s, l_variable, w, step_size):
    w_prime = w
    w_prime[len(w_prime)-1] = 0
    
    e = (step_size * X.T) @ (y-s) + 2*l_variable*w
    return e



def cost(z, y, w, alpha):
    return -y.T @ np.log(z) - (np.ones(len(y))-y).T @ np.log(np.ones(len(y)-z) + alpha * np.linalg.norm(w,2))



def grad_descent_batch(iteration, lambda_value, step_size):
    

    X = training_data
    y = training_labels
    w_0 = np.full(shape = len(training_data[0]), fill_value=0)
    lambda_array = np.full(shape = len(training_data[0]), fill_value=lambda_value)
    
    w = w_0
    
    for i in range(iteration):

        s = find_s(X, w)
        
        
        e = find_e_batch(X, y, s, lambda_array, w, step_size)
        w = w + e
        if np.allclose(e, w_0, atol=1e-1, equal_nan=True):
            print("converged")
            break
    return w


def predict(test_data, w_set):
    s = find_s(test_data, w_set)
    predictions = []
    for sample in s:
        
        if sample > 0.5:
            predictions += [1]
            
        else:
            predictions += [0]
            
    return predictions


def evaluate(test_label, predictions):
    total_accurate = 0
    total_samples = len(test_label)
    
    for i in range(len(test_label)):
        if test_label[i] == predictions[i]:
            total_accurate += 1
    accuracy = total_accurate/total_samples
    print(accuracy)
    return accuracy


    

def visualize_iteration():
    number_of_iterations = np.arange(50)
    w_set = []
    accuracy_results = []
    for iteration in number_of_iterations:
        print(iteration)
        print("waiting...")
        w = grad_descent_batch(iteration, 0, 1)
        w_set += [w]
        accuracy_results += [evaluate(validation_labels, predict(validation_data, w))]


    plt.plot(number_of_iterations, accuracy_results, marker = 'o')
    plt.xlabel("Iteration Size #:")
    plt.ylabel("Accuracy Rate %:")
    plt.title("Logistic Gradient Descent + l2 Training Model: ")
    
    plt.show()

def visualize_lambda():
    iteration = 40
    w_set = []
    accuracy_results = []
    lambda_values = [0, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    for lambda_value in lambda_values:
        print("lambda:" + str(lambda_value))
        print("waiting...")
        w = grad_descent_batch(iteration, lambda_value, 1)
        w_set += [w]
        accuracy_results += [evaluate(validation_labels, predict(validation_data, w))]


    plt.plot(lambda_values, accuracy_results, marker = 'o')
    plt.xlabel("lambda value #:")
    plt.ylabel("Error Rate %:")
    plt.title("Logistic Gradient Descent + l2 Training Model: ")
    
    plt.show()

def visualize_step_size():
    iteration = 40
    lambda_value = 0.005
    w_set = []
    accuracy_results = []
    step_size_values = np.arange(10, step=0.1)
    for step_size in step_size_values:
        print("step size:" + str(step_size))
        print("waiting...")
        w = grad_descent_batch(iteration, lambda_value, step_size)
        w_set += [w]
        accuracy_results += [evaluate(validation_labels, predict(validation_data, w))]


    plt.plot(step_size_values, accuracy_results, marker = 'o')
    plt.xlabel("step size #:")
    plt.ylabel("Error Rate %:")
    plt.title("Logistic Gradient Descent + l2 Training Model: ")
    
    plt.show()

def kaggle():
    iteration = 40
    w = grad_descent_batch(iteration, 0.005, 0.1)
    csv_file = open('wine_predict.csv', 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'category'])

    id = 1
    predictions = predict(test_data, w)
    all_rows = []
    for prediction in predictions:
        row = [id, prediction]
        all_rows.append(row)
        id+=1
    writer.writerows(all_rows)

kaggle()