# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:27:04 2023

@author: MRUTYUNJAY
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
dataset=pd.read_csv(r'D:\class\14th march\MLR\House_data.csv')
dataset.head()
print(dataset.isnull().any())
print(dataset.dtypes)
dataset=dataset.drop(['id','date'],axis=1)
with sns.plotting_context("notebook",font_scale=2.5):
    g=sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],hue='bedrooms',palette='tab20',size=6)
g.set(xticklabels=[]);    
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((21613,1)).astype(int),values=X,axis=1)
import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
