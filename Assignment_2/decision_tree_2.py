# -------------------------------------------------------------------------
# AUTHOR: Lynn Takahashi
# FILENAME: decision_tree_2.py
# SPECIFICATION: program reads csv files: contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv.
# This program trains, tests, and output the performance of the 3 models created by using each training set on the test set provided
# (contact_lens_test.csv). The average accuracy is printed as the final classification performance of each model.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2.5 hrs
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

# datasets with diff number of instances
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []#og
    X = [] # training
    Y = [] # labels

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # dictionaries and other variables
    dict_age = {'Young': 1, 'Presbyopic': 3, 'Prepresbyopic': 2}  # row[0] age
    dict_spec = {'Myope': 1, 'Hypermetrope': 2}  # row[1] spec
    dict_astig = {'Yes': 1, 'No': 2}  # row[2] astigma
    dict_tear = {'Normal': 1, 'Reduced': 2}  # row[3] tear
    dict_lenses = {'Yes': 1, 'No': 2}  # labels
    accuracies = []  # list of accuracies

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        age = row[0]
        spec = row[1]
        astig = row[2]
        tear = row[3]

        trans_row = [dict_age[age], dict_spec[spec], dict_astig[astig], dict_tear[tear]]

        X.append(trans_row)

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        trans_row = dict_lenses[row[4]]
        Y.append(trans_row)

    # loop your training and test tasks 10 times here
    for i in range(10):
        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as testfile:
            reader2 = csv.reader(testfile)
            for j, row in enumerate(reader2):
                if j > 0:  # skipping the header
                    dbTest.append(row)
                    # print(row)

        #  prediction count
        correct_pred = 0
        total_pred = 0

        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during training,
            age = data[0]
            spec = data[1]
            astig = data[2]
            tear = data[3]

            trans_data = [dict_age[age], dict_spec[spec], dict_astig[astig], dict_tear[tear]]

            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            prediction_made = clf.predict([trans_data])[0]

            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if prediction_made == dict_lenses[data[4]]:
                correct_pred += 1

            total_pred += 1

        # accuracy for round of training and testing
        accuracies.append(correct_pred/total_pred)

    # find the average of this model during the 10 runs (training and test set)
    # print(len(accuracies)) #check
    avg_accuracy = sum(accuracies) / len(accuracies)

    # print the average accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f'Final accuracy when training on {ds}: {avg_accuracy:.2f}')
