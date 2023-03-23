#-------------------------------------------------------------------------
# AUTHOR: Kyle Just
# FILENAME: svm.py
# SPECIFICATION: demonstrate the use of SVM with test data
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c_set = [1, 5, 10, 100]
degree_set = [1, 2, 3]
kernel_set = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

x_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

x_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
highest_accuracy = 0
for c in c_set:
    for degree in degree_set:
        for kernel in kernel_set:
           for shape in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #--> add your Python code here
                model = svm.SVC(C=c, degree=degree, kernel=kernel, decision_function_shape = shape)
                #Fit SVM to the training data
                #--> add your Python code here
                model.fit(x_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                correct = 0
                for (x_testSample, y_testSample) in zip(x_test, y_test):
                    prediction = model.predict([x_testSample])
                    if prediction == y_testSample:
                        correct += 1
                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                accuracy = correct / len(x_test)
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    print("Highest SVM accuracy so far: " + str(highest_accuracy) + ", Parameters: c=" + str(c) + ", degree=" + str(degree) + ", kernel=" + kernel + ", decision_function_shape = '" + shape + "'")
               




