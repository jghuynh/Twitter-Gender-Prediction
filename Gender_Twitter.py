# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:20:49 2020

@author: jghuynh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection

#%%

def removeUnnecessaryCols(myDataFrame):
    # more than 99% of values is False
    myDataFrame.drop("_unit_state", axis = 1, inplace = True) 
    
    # more than 99% of these values are 3
    myDataFrame.drop("_trusted_judgments", axis = 1, inplace = True)
    
    # we don't need links. All these links are like an ID--unique for every person
    myDataFrame.drop("profileimage", axis = 1, inplace = True)
    
    # 100% of values are "yes"
    myDataFrame.drop("profile_yn", axis = 1, inplace = True)
    
    # 97% of values are 0
    myDataFrame.drop("retweet_count", axis = 1, inplace = True)
    
    # all names of tweets are IDs. They are unique, so they don't affect data
    myDataFrame.drop("name", axis = 1, inplace = True)
    
    # link bar color, side bar color--they have the same values, so eliminate
    myDataFrame.drop("link_color", axis = 1, inplace = True )
    
    # name and text are all unique values. None of them are the same, so eliminate
    myDataFrame.drop("text", axis = 1, inplace = True)
    
    # so many of tweet_location are invalid or written in weird symbols
    myDataFrame.drop("tweet_location", axis = 1, inplace = True)
    
    return myDataFrame

def oneHotEncode(myDataFrame):
    #stringCols = []
    #X = myDataFrame
    
    for column in myDataFrame.columns:
        #print("column =", column, " type = ", myDataFrame[column].values.dtype)
        if "O" == myDataFrame[column].values.dtype:
            
            '''
            if column == "tweet_location":
                if len(myDataFrame[column]) > 5:
                    myDataFrame.drop("tweet_location", axis = 0, inplace = True)
            '''
            #stringCols.append(column)
            oneHot = pd.get_dummies(myDataFrame[column])
            myDataFrame = myDataFrame.drop(column, axis = 1)
            myDataFrame = myDataFrame.join(oneHot)
            
            
    #myDataFrame = myDataFrame.drop(myDataFrame.std()[myDataFrame.std() < 0.6].index.values, axis = 1)
    #print(stringCols)
    print(myDataFrame)
    return myDataFrame



# Calculates the error between the predicted value and 
# the truth value
# AKA l(yi, qi)
#
# @param yi the actual ground truth
# @param qi the predicted probablity 
# @return the calculated error/binary cross entropy
def binaryCrossEntropy(yi, qi):
    return -(yi * np.log(qi) + (1 - yi)* np.log(1 - qi))

# Converts a scaler into a probability
# @param u the scaler (also, the predicted value)
# @return the probability
def sigmoid(u):
    #e = np.exp(1)
    expu = np.exp(u)
    return expu/(1 + expu)
    #return e**u/(1 + e**u)
    

# Calculates the binary cross entropy
# @param yi the ground truth value
# @param u the predicted probability 
def hi(u, yi):
    
    exp = np.exp(u)
    return -yi*u + np.log(1 + exp)

# Gets the sum of all the hi's/errors
def L(beta, X, Y):
    N = X.shape[0] # numer of rows\
    #cols = X.shape[1]
    mySumHi = 0
    
    # for every row in X
    for i in range(N):
        xihat = X[i] # the ith row of X
        yi = Y[i] # 1 row in X => 1 scaler (0 or 1) in Y
        #print("xihat", xihat)
        #print("beta", beta)
        dotProduct = np.vdot(xihat, beta)
        mySumHi += hi(dotProduct, yi)
    return mySumHi


# Calculats the clip of beta with alpha step size
# Basically, beta will go close to origin with alpha step size
def clip(beta, alpha):
  # so find the min: beta or alpha
  clipped = np.minimum(beta, alpha)

  # find max: clipped or -alpha
  clipped = np.maximum(clipped, -alpha)

  return clipped

# Calculates the proximal norm of betaHat with stepsize alpha
# @betaHat betaHat
# @alpha the stepsize
def proxL1Norm(betaHat, alpha, penalizeAll = True):
  
  # definition of prox operator
  out = betaHat - clip(betaHat, alpha)

  if not penalizeAll:
    # set the first value of prox as beta0
    out[0] = betaHat[0]
  
  return out

# Does Logistic regression with L1 regularized term
  # and proximal gradient descent

# @param X the X-values, augmented
def LogRegL1Regularized_proxGrad(X, y, myLambda):
    
  N, d = X.shape
    # N = num rows, d = num columns
  
  maxIter = 50
  # increasing helps
  
  # learn rate
  alpha = 0.00005

    # note: d is already the augmented size of columns
  beta = np.zeros(d)


  costFunVals = np.zeros(maxIter)
  # DOES PROXIMAL gradient method iteration
  for t in range(maxIter):
    # compute gradient of L
    #grad = X.T @ (X @ beta - y) # the smooth part of our objective function,
    # that was for linear regress
    
    grad = np.zeros(d)
        
    # computing grad
    
    # so slow because N is so darn huuge
    for i in range(N):
        Xi = X[i, :]
        Yi = y[i] # the ith y label
        qi= sigmoid(np.vdot(Xi, beta)) # predicted probability
        
        # computes the dot product of 2 vectors
        
        grad += (qi - Yi)*Xi # look at formula for gradient of L(B)
    
    # now my beta is huge
    beta = proxL1Norm(beta - alpha*grad, alpha*myLambda) 

    # calcualte and save the current objective value
    # compute norm of that vector (norm of error), then square that norm
    # that's the smooth part of our objective function
    # also, add the regularization term
    # which is lamda * L1 norm of beta
    # L1norm of beta = np.sum(np.abs(beta)) = sum of all the abs value of compoenents of beta
    
    # for linear reg
    #costFunVals[t] = 0.5 * np.linalg.norm(X @ beta - y)**2 + myLambda * np.sum(np.abs(beta)) # objective function
    # since beta is not zero, then costFunVal (error) will not approach 0
    
    # for log reg
    costFunVals[t] = L(beta, X, y)
    
    if t % 10 == 0:
        print("iteration: ", t, " Objective Function value: ", costFunVals[t])
        
  return beta, costFunVals

# Calculates the accuracy percentage of my predictions
# against Y_test
# @param numTest the number of test subjects
# @param predictions my predictions
# @param Y_test the testing data solutions/ground truth
def findAccuracy(numTest, predictions, Y_test):
  numCorrect = 0
  for i in range(numTest):
    if predictions[i] == Y_test[i]:
      numCorrect += 1
  accuracy = numCorrect/numTest
  return accuracy

def getPredictionsLog(numTest, beta):
    myPred = np.zeros(numTest)
    for index in range(numTest):
        Xi = X_test[index]
        myProb = sigmoid(Xi @ beta)
        if myProb < 0.5:
            myPred[index] = 0
        else:
            myPred[index] = 1
    return myPred

#myGenderFile = pd.read_csv("...\\gender-classifier-DFE-791531\\gender-classifier-DFE-791531.csv", encoding = "ISO-8859-1")

# Absolute path
myGenderFile = pd.read_csv("C:\\Users\\jghuynh\\Documents\\Machine_Learning\\Project_#5_Final_Gender\\gender-classifier-DFE-791531\\gender-classifier-DFE-791531.csv", encoding = "ISO-8859-1")
# type: dataframe

# All the columns that have more than half the rows/people answered "NA"
fewNAs = myGenderFile.columns[myGenderFile.isna().sum() <= 0.5*myGenderFile.shape[0]]
myGenderFile = myGenderFile[fewNAs]
# eradicated 3 columns that had waaay too many NAss

# columns with too big std
bigSTD = ["_golden", "gender:confidence", "profile_yn:confidence"]

# dropping those columns that have too big STD
myGenderFile.drop(bigSTD, axis = 1, inplace = True)

# drop the columns in which the person's gender is "unknown" or "brand" (??)
# in other words, keep the columns where gender is "female" or "male"

myGenderFile = myGenderFile.loc[(myGenderFile.gender == "female") | (myGenderFile["gender"] == "male")]

myGenderFile = removeUnnecessaryCols(myGenderFile)

# Using numpy to split
# len(myGenderFile) = rows = 12894, for now
# train: 80%
# validate: 10%
# test: 10%
# first param: 0.8 - 80% dataframe for train
# 2nd param: 0.9: 1 - 0.9 = 10% for test
# validate is 10% because 2nd param - 1st param = 0.9 - 0.8 = 0.1 = 10%
train, validate, test = np.split(myGenderFile.sample(frac=1), [int(.8*len(myGenderFile)), int(.9*len(myGenderFile))])
#X_train = train
train = oneHotEncode(train)
validate = oneHotEncode(validate)
test = oneHotEncode(test)

# since our gender column is one-hot encoded, we do not have column name "gender"
# so now, look at female column
# if value = 1, person is female; male otherwise

Y_train = train["female"].values
# Y_oneHot = pd.get_dummies(Y_train).values
#K = Y_oneHot.shape[1]

X_train = train.drop("female", axis = 1)
X_train.drop("male", axis = 1, inplace = True)
X_train = X_train.values

Y_valid = validate["female"].values
#X_valid = validate
X_valid = validate.drop("female", axis = 1)
X_valid.drop("male", axis = 1, inplace = True)
X_valid = X_valid.values

Y_test = test["female"].values
X_test = test.drop("female", axis = 1)
X_test.drop("male", axis = 1, inplace = True)

# getting ride of NAs in the numeric columns
colsWithNAs = X_train.columns[X_train.isna().sum() > 0]
for col in colsWithNAs:
    X_train[col] = X_train[col].fillna(X_train[col].mean())
    X_test[col] = X_test[col].fillna(X_test[col].mean())
#X["Age"] = X["Age"].fillna(X["Age"].mean())
colsWithNAs = X_test.columns[X_test.isna().sum() > 0]
for col in colsWithNAs:
    X_train[col] = X_train[col].fillna(X_train[col].mean())
    X_test[col] = X_test[col].fillna(X_test[col].mean())



# Wow that took a long time!


# how penalized are we to make beta sparse
# mylabda = 100, 200, ... 10000
myLambda = 900
#beta, costFunVals = solveLasso_proxGrad(X, y, myLambda)
# TODO: Make X_train all numbers
beta, costFunVals = LogRegL1Regularized_proxGrad(X_train, Y_oneHot, myLambda)



'''


X = myGenderFile.loc[:, :]
X.drop("gender", axis = 1, inplace = True)
X = X.values # transform into numpy array

y = myGenderFile["gender"].values
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, y, train_size = 0.8) #random_state = 1)

# using Skleanr to split
X_train = (X_train - X_train.mean(axis = 0))/X_train.std(axis = 0) # (xvalue  - mean)/standard deviation

# Using numpy to split
# len(myGenderFile) = rows = 12894, for now
train, validate, test = np.split(df.sample(frac=1), [int(.8*len(myGenderFile)), int(.9*len(myGenderFile))])
X_train = train
X_train.drop("gender", axis = 1, inplace = True)
X_train = X_train.values

Y_train = train["gender"].values


# pretend as if we had never seen the X_test data before.
# our predictions are more slightly accurate
# if use X_test.mean, we are cheating!
X_test = (X_test - X_train.mean(axis = 0))/X_train.std(axis = 0)

# Augment the X data
X_test = np.insert(X_test, 0, 1, axis = 1)
X_train = np.insert(X_train, 0, 1, axis = 1)

N = X.shape[0] # number of training examples/rows
d = X.shape[1] # number of characteristics/columns
'''
