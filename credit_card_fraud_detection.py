import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import pickle

# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# load the dataset from the csv file
data = pd.read_csv('C:/Users/super/OneDrive/Documents/School/Spring 2020/CS 657/Final Project/creditcard.csv')

# getting the columns from the dataset
print(data.columns)

# used to get a fraction of the data for testing
#data = data.sample(frac=0.1, random_state = 1)
#print(data.shape)
#print(data.describe())

#  get a plot histogram of each parameter 
data.hist(figsize = (20, 20))
plt.show()

# get number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

# get the columns from the dataset
columns = data.columns.tolist()

# filter the columns
columns = [c for c in columns if c not in ["Class"]]

# store variable to be predicting on
target = "Class"

X = data[columns]
Y = data[target]

# print shapes
print(X.shape)
rint(Y.shape)

# define random states
state = 1

# define 
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=10,
        contamination=outlier_fraction)}

# fit model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)

# save moedels
def save_model(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

fold_auc = []

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
        save_model(clf, 'Outlier.pk2') #save the Local Outlier Factor
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X) 
        save_model(clf, 'Isolation.pk2') #save the Isolation Forest
    
    # reshape pred values 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
###############################################################    
    #** Having trouble here saving the AUC**#
#    s = clf.evaluate(Y, y_pred)
#    fold_auc.append(s[1])
#    print("AUC =", s[1])
###############################################################
    # run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

    
# load models to be rerun 
def load_model(filename):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    return clf