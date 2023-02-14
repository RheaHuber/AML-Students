# SVM modelling of students data

# Read in the data
import pandas as pd
data = pd.read_csv("StudentsLessAbridged.csv", sep=',')

# Initialize variable factors
# Portion of data used for testing
TESTPART = 0.25
# Kernel types: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
KERNELTYPE = 'rbf'
# True/false for whether to standardize data
STANDARDIZE = True

# Constant list of attributes
ATTRIBUTES = [
    'age',
    'Pstatus',
    'Medu',
    'Fedu',
    'traveltime',
    'studytime',
    'failures',
    'schoolsup',
    'famsup',
    'activities',
    'romantic',
    'Walc',
    'health',
    'absences',
    'G1',
    'G2'
]

# Set independendent/dependent variables
X = data[ATTRIBUTES]
y = data['G3']

# Set train/test split
from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TESTPART, random_state=42)

#scale the dataset using mean and std; do training/testing separately to avoid data leakage problem
if STANDARDIZE:
    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test - X.mean())/X.std()

from sklearn import svm
clfsvm = svm.SVC(kernel=KERNELTYPE)
clfsvm.fit(X_train, y_train)

#### See which data points are critical #####
# get the support vectors
print("clfsvm support vectors: {}".format(clfsvm.support_vectors_))
# get indices of support vectors
print("clfsvm support vector indices: {}".format(clfsvm.support_))
# get number of support vectors for each class
print("clfsvm # of support vectors in each class: {}".format(clfsvm.n_support_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(clfsvm.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clfsvm.score(X_test, y_test)))

student_prediction = clfsvm.predict(X_test)
compare = pd.DataFrame({'true': y_test, 'predicted': student_prediction})
print("true vs predicted\n", compare)

#printing confusion matrix
import matplotlib.pyplot as plot
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

ConfusionMatrixDisplay.from_estimator(clfsvm, X_test, y_test)
plot.show()