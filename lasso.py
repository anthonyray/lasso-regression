# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation

# Loading data
winedata = pd.read_csv("data/winequality-red.csv",delimiter=";")
winedata.dropna()  # Dropping useless data.

quality = np.reshape(winedata["quality"].values,[winedata["quality"].values.shape[0],]) # Preparing data for numpy

data = winedata.drop("quality",1) # We want to predict quality, so we leave from the features. 
data = np.reshape(data.values,[data.values.shape[0],data.values.shape[1]]) # Preparing data for numpy

# Question 1
# 1.a) 

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(data,quality)

print "Modèle linaire : "
print("réalisé avec " + str(data.shape[0]) + " observations.")
print("Coefficients : \n",regr.coef_)

# 1.b

y = (quality - np.mean(quality)) / np.std(quality) # Normalizing features

for i in range(data.shape[1]): # Normalizing observations
    data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])

X = data

print "X:"
print X[158],X[159],X[213]

print "Y:"
print  y[158],y[159],y[213]

# Question 2
# Switching to the second dataset

winedata = pd.read_csv("data/winequality-white.csv",delimiter=";")
winedata.dropna()

y = np.reshape(winedata["quality"].values,[winedata["quality"].values.shape[0],])
y = (y - np.mean(y)) / np.std(y) # Normalizing observations

data = winedata.drop("quality",1)
X = np.reshape(data.values,[data.values.shape[0],data.values.shape[1]])

for i in range(X.shape[1]): # Normalizing features
    X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])


# Question 3

def Lasso(X,y,penalisation): # Where X are the features, Y the observations, and the penalisation factor
    lass = linear_model.Lasso(fit_intercept=False,alpha=penalisation)
    lass.fit(X,y)
    return (lass.coef_,lass.predict(X))

# Question 4

# Plotting coefficients amplitude according to penalisation
# We decide on a range for penalisations : for 10e-5 to 10e2. 

penalisations = np.logspace(-6, 3, 200) # We take 200 penalisations in the range defined before

coeffs = list() 

for penalisation in penalisations:
    output = Lasso(X,y,penalisation)
    coeffs.append(output[0])

# Plotting the result

fig1=plt.figure(figsize=(12,8))
ax1 = fig1.add_subplot(111)
ax1.set_xscale('log')
ax1.plot(penalisations,coeffs)
plt.xlabel("Penalisations")
plt.ylabel("thetas")

# Question 5

# Determinitation of the penalisation factor with Cross Validation

#lasscv = linear_model.LassoCV(cv=20)

lassCV = linear_model.LassoCV(alphas=penalisations,fit_intercept=False,normalize=False)
lassCV.fit(X,y)

# Question 5b
print "Lasso with CV : "
print "Penalisation trouvée par CV : " + str(lassCV.alpha_)

# Determining the smallest penalisation factor so that all coefficients equal to 0. 

# Question 5c
x_test = np.array([6,0.3,0.2,6,0.053,25,149,0.9934,3.24,0.35,10])

print "Score : "
print lassCV.predict(x_test)

# Question 5d
# Using OLS, without penalisation

ols = linear_model.LinearRegression(fit_intercept=False)
ols.fit(X,y)

print("Somme des carrés des erreurs OLS:")
print np.mean( (ols.predict(X) - y) ** 2)

print("Somme des carrés des erreurs LassoCV:")
print np.mean( (lassCV.predict(X) - y) ** 2)

print("Bias for OLS :")
print np.mean(ols.coef_)

print("Bias for LassoCV : ")
print np.mean(lassCV.coef_)


# Question 5e
# Applying OLS sur les variables actives en sortie du Lasso (on ne garde que les 

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(lassCV.predict(X),y)



print (reg.coef_)