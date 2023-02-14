import pandas as pd
###################################################################
# Separating fruits
###################################################################
fruits = pd.read_csv('fruit_data_with_colors.txt', sep='\t')
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

fruits.head()

X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

#scale the dataset using mean and std; do training/testing separately to avoid data leakage problem
standardize = True
if standardize:
    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test - X.mean())/X.std()

from sklearn import svm
# kernals could be: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’. Try them.
clfsvm = svm.SVC(kernel='rbf')
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

fruit_prediction = clfsvm.predict(X_test)
compare = pd.DataFrame({'true': y_test, 'predicted': fruit_prediction})
print("true vs predicted\n", compare)

# Once you are happy with the model, it can be deployed
# Train the model use all data
clfsvm.fit((X-X.mean())/X.std(), y)
## Use the trained SVM classifier model to classify new, previously unseen objects
# first example: a small fruit
testFruit = pd.DataFrame([[8.7, 5.8, 130, 0.72]], columns=['height', 'width', 'mass', 'color_score'])
fruit_prediction = clfsvm.predict((testFruit-X.mean())/X.std())
print("small one:", lookup_fruit_name[fruit_prediction[0]])

# second example: a larger one
testFruit = pd.DataFrame([[8.5, 6.3, 190, 0.53]], columns=['height', 'width', 'mass', 'color_score'])
fruit_prediction = clfsvm.predict((testFruit-X.mean())/X.std())
print("large one:", lookup_fruit_name[fruit_prediction[0]])
