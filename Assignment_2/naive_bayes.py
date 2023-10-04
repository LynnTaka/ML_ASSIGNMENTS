#-------------------------------------------------------------------------
# AUTHOR: Lynn Takahashi
# FILENAME: naive_bayes.py
# SPECIFICATION: a program that builds a naive bayes classifier by scratch to
# to predict whether or not to play tennis.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []#og
testdb = [] #test set
X = [] # training
Y = [] #label

#reading the training data in a csv file
with open('weather_training.csv', 'r') as train_file:
    reader = csv.reader(train_file)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row)

# dictionaries
dict_outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
dict_temp = {'Hot': 1, 'Cool': 2, 'Mild': 3}
dict_humid = {'Normal': 1, 'High': 2}
dict_wind = {'Weak': 1, 'Strong':2}
dict_label = {'Yes': 1, 'No': 2}

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
for row in db:
    outlook = dict_outlook[row[1]]
    temp = dict_temp[row[2]]
    humid = dict_humid[row[3]]
    wind = dict_wind[row[4]]

    X.append([outlook, temp, humid, wind])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for row in db:
    label = dict_label[row[5]]
    Y.append(label)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
with open('weather_test.csv', 'r') as test_file:
    reader = csv.reader(test_file)
    for i, row in enumerate(reader):
        if i > 0:
            testdb.append(row)

#printing the header os the solution
#formatting
print("{:<3} {:<10} {:<12} {:<10} {:<10} {:<10} {:<10}".format("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in testdb:
    day = row[0]
    outlook = dict_outlook[row[1]]
    temp = dict_temp[row[2]]
    humid = dict_humid[row[3]]
    wind = dict_wind[row[4]]

    pred_probability = clf.predict_proba([[outlook, temp, humid, wind]])[0]

    for label, value in dict_label.items():
        # print(pred_probability[value])
        pred_label = label #get label
        confidence = pred_probability[value-1]
        if confidence > 0.75:
            print("{:<3} {:<10} {:<12} {:<10} {:<10} {:<10} {:.4f}"
                  .format(day, row[1], row[2], row[3], row[4], label, confidence))
