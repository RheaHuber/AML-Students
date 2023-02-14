# Neural Network modelling of students data

# Read in the data
import pandas as pd
data = pd.read_csv("StudentsLessAbridged.csv", sep=',')

# Initialize variable factors
# Portion of data used for testing
TESTPART = 0.25
# Number of nodes in each hidden layer
HLSIZE1 = 20
HLSIZE2 = 5
# Regularization parameter
REGULARIZATION = 0.0001
# Initial learning rate
LEARNRATE = 0.001
# Maximum number of iterations
MAXITER = 200

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

# Set up neural network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(HLSIZE1, HLSIZE2), solver='sgd', alpha=REGULARIZATION, learning_rate='adaptive', learning_rate_init=LEARNRATE, max_iter=MAXITER)
clf.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))

student_prediction = clf.predict(X_test)
compare = pd.DataFrame({'true': y_test, 'predicted': student_prediction})
print("true vs predicted\n", compare)

#printing confusion matrix
import matplotlib.pyplot as plot
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plot.show()
