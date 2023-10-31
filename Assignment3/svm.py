#-------------------------------------------------------------------------
# AUTHOR: Lynn Takahashi
# FILENAME: svm.py
# SPECIFICATION: see README.txt
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
best_accuracy = 0
best_parameters = None

for c_val in c:# c
    for d_val in degree:# degree
        for k_val in kernel:# kernel
           for dfs_val in decision_function_shape:# decision function shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=c_val, degree=d_val, kernel=k_val, decision_function_shape=dfs_val)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                correct_predictions = 0
                total_samples = len(X_test)

                #hint: to iterate over two collections simultaneously, use zip()
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    prediction = clf.predict([x_testSample])
                    if prediction[0] == y_testSample:
                        correct_predictions += 1

                accuracy = correct_predictions / total_samples

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_parameters = (c_val,d_val,k_val,dfs_val)

                print(f"Highest SVM accuracy so far: {best_accuracy:.2f}, Parameters: a={best_parameters[0]}, degree={best_parameters[1]}, kernel={best_parameters[2]}, decision_function_shape={best_parameters[3]}")

print("done")



