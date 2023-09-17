#-------------------------------------------------------------------------
# AUTHOR: Lynn Takahashi
# FILENAME: decision_tree.py
# SPECIFICATION: The program reads data from a csv file 'contact_lens.csv' and plots the decision tree.  In order for this program to work correctly
# the cols need to be 'Age', 'Spectacle', 'Astigmatism', 'Tear', 'Outcome'
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2.5 hrs (I had to go over dictionaries again)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = [] #og
X = [] #training features into numbers
Y = [] #output

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

# check dim
print(len(db))
print(len(db[0]))

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
for row in db:

    # make dictionaries for each features
    trans_age = {'Young': 1, 'Presbyopic': 2, 'Prepresbyopic': 3} #row[0] Age
    trans_spec = {'Myope': 1, 'Hypermetrope': 2} # row[1] Spec
    trans_astig = {'Yes': 1, 'No': 2} # row[2] Astigma
    trans_tear = {'Normal': 1, 'Reduced': 2} #row[3] Tear

    # every new row creates a new list
    trans_row = [trans_age[row[0]], trans_spec[row[1]], trans_astig[row[2]], trans_tear[row[3]]]
    # append to list
    X.append(trans_row)

# check 2
print('\nX = ', X)
# print(len(X))
# print(len(X[0]))


#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for row in db:
    # dictionary
    trans_lenses = {'Yes': 1, 'No': 2}
    trans_row = trans_lenses[row[4]]
    Y.append(trans_row)

print('\nY = ', Y)
# print(len(Y))

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()