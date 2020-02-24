# =============================================================================
# Shreya Date
# CMSC-678: Introduction to Machine Learning
# Homework 2
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTS:

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# GLOBAL VARIABLES

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size
X = y = x_train = x_test = y_train = y_test = w = wx = np.empty(0)
y_predicted = accuracy = 0

# -----------------------------------------------------------------------------
# APIs

def img(row, data):
	image = np.zeros((image_size,image_size))
	for i in range(0,image_size):
		for j in range(0,image_size):
			pix = image_size*i+j
			image[i,j] = data[row, pix]
	plt.imshow(image, cmap = 'gray')
	plt.show()
            
def calculate_accuracy():
    global accuracy, y_test
    return (accuracy/y_test.shape[0])
    
def calculate_wx(x):
    global w, wx
    for i in range(len(w)):
        wx[i] = (np.dot(x, w[i]))
    return np.argmax(wx)
    
def test_perceptron():
    global w, x_test, y_test, wx, y_predicted, accuracy
    for i in range(len(x_test)):
        y_predicted = calculate_wx(x_test[i])
        y_actual    = int(y_test[i])
        if (y_predicted == y_actual):
            accuracy = accuracy + 1

def perceptron():
    global w, x_train, y_train, wx, y_predicted
    for i in range(len(x_train)):
        y_predicted = calculate_wx(x_train[i])
        y_actual    = int(y_train[i])
        if (y_predicted == y_actual):
            print("no change")
        else:
            w[y_predicted] = w[y_predicted] - x_train[i] 
            w[y_actual]    = w[y_actual]    + x_train[i]

def pre_perceptron():
    global w, x_train, wx, no_of_different_labels, x_test
    x_train = np.append(x_train, np.ones([len(x_train),1]),1)
    x_test  = np.append(x_test, np.ones([len(x_test),1]),1)
    a, b = x_train.shape
    w = np.zeros((no_of_different_labels, x_train.shape[1]))
    wx =  np.zeros((no_of_different_labels, 1))
    img(5, x_train)
    
def print_arrays():
    print(X.shape)
    print(y.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

def load_mnist(data_path = "./"):
    global X, y, x_train, x_test, y_train, y_test
    X    = np.loadtxt(data_path + "mnist_data.txt", delimiter = " ")
    y    = np.loadtxt(data_path + "mnist_labels.txt", delimiter = " ")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
   
# -----------------------------------------------------------------------------
# DRIVER
    
def main(): 
    load_mnist()
    #print_arrays()
    pre_perceptron()
    perceptron()
    test_perceptron()
    print("Accuracy is: ", calculate_accuracy())

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------