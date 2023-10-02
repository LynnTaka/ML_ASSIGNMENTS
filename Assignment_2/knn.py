# -------------------------------------------------------------------------
# AUTHOR: Lynn Takahashi
# FILENAME: knn.py
# SPECIFICATION: a simple implementatin of a knn classifier from scratch.
# uses csv file named binary_points.csv with data points and labels
# calculates the error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []  # og
error = 0  # count

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# loop your data to allow each instance to be your test set
for row in db:
    X = []  # training
    Y = []  # labels

    # add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    for train in db:
        if row != train:
            x, y = float(train[0]), float(train[1])
            X.append([x,y])

    # transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    # --> add your Python code here
    for label in db:
        if row != label:
            temp = label[2]
            if temp == '-':
                Y.append(-1)
            else:
                Y.append(1)

    # store the test sample of this iteration in the vector testSample
    # --> add your Python code here
    test_sample = [float(row[0]), float(row[1])]

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([test_sample])[0]

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    if row[2] == '-':
        true_label = -1
    else:
        true_label = 1

    if class_predicted != true_label:
        error += 1

# print the error rate
error_rate = error / len(db)
print('ERROR RATE: ', error_rate)
