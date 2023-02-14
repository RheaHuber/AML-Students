# Linear regression modelling of students data

# Read in the data
import pandas as pd
data = pd.read_csv("StudentsLessAbridged.csv", sep=',')

# Initialize variable factors
# Portion of data used for testing
TESTPART = 0.25

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

# Create regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=40, max_iter=10000)
lr.fit(X_train, y_train)

#########################################
# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#printing confusion matrix
import matplotlib.pyplot as plot
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test)
plot.show()