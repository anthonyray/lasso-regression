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


