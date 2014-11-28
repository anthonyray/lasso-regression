# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

winedata = pd.read_csv("data/winequality-red.csv",delimiter=";")
winedata.dropna()

quality = np.reshape(winedata["quality"].values,[winedata["quality"].values.shape[0],1])

data = winedata.drop("quality",1)
data = np.reshape(data.values,[data.values.shape[0],data.values.shape[1]])

# Question 1
# 1.a) 

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(data,quality)

print "Modèle linaire : "
print("réalisé avec " + str(data.shape[0]) + " observations.")
print("Coefficients : \n",regr.coef_)

# 1.b
y = quality - np.mean(quality)

data_means = np.mean(data,axis=0)
for i in range(data.shape[1]):
    data[:,i] = data[:,i] - np.mean(data[:,i])

X = data

print "X:"
print X[158],X[159],X[213]

print "Y:"
print  y[158],y[159],y[213]

# Question 2
# Switching to the second dataset

winedata = pd.read_csv("data/winequality-white.csv",delimiter=";")

# Question 3
def Lasso(X,y,penalisation): # Where X are the features, Y the observations, and the penalisation factor
    lass = linear_model.Lasso(fit_intercept=False,alpha=penalisation)
    lass.fit(X,y)
    return (lass.coef_,lass.predict(X))

# Question 4

# Plotting coefficients amplitude according to penalisation

penalisations = alphas = np.logspace(-10, 2, 50)


coeffs = list()
for penalisation in penalisations:
    output = Lasso(X,y,penalisation)
    coeffs.append(output[0])

fig1=plt.figure(figsize=(12,8))
ax1 = fig1.add_subplot(111)
ax1.set_xscale('log')
ax1.plot(penalisations,coeffs)
plt.show()

