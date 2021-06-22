# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 01:20:44 2021

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('StudentsPerformance.csv')
x=dataset.iloc[:,0:5].values
y=dataset.iloc[:,5:8].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_0=LabelEncoder()
x[:,0]=labelencoder_x_0.fit_transform(x[:,0])
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
labelencoder_x_3=LabelEncoder()
x[:,3]=labelencoder_x_3.fit_transform(x[:,3])
labelencoder_x_4=LabelEncoder()
x[:,4]=labelencoder_x_4.fit_transform(x[:,4])
from sklearn.compose import ColumnTransformer

#ct = ColumnTransformer([("race/ethnicity", OneHotEncoder(), [1])], remainder = 'drop')
#x = ct.fit_transform(x).toarray()
# ct = ColumnTransformer([("parental level of education", OneHotEncoder(), [2])], remainder = 'passthrough')
# x = ct.fit_transform(x)
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.float)
#columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
#x = np.array(columnTransformer.fit_transform(x), dtype = np.float)
#from sklearn.compose import make_column_transformer
#onehotencoder=OneHotEncoder(categories='auto', sparse=False)
#x=onehotencoder.fit_transform(x)
#onehotencoder=OneHotEncoder(categorical_features=[2])
#x=onehotencoder.fit_transform(x).toarray()
x=x[:,4:]
#x=x[:,2:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

clasificador=Sequential()
clasificador.add(Dense(units=10,kernel_initializer='uniform',activation='relu',input_dim=5))
clasificador.add(Dense(units=10,kernel_initializer='uniform',activation='relu'))
clasificador.add(Dense(units=3,kernel_initializer='uniform',activation='sigmoid'))

clasificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

clasificador.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred=clasificador.predict(X_test())

from sklearn.metrics import confusion_matrix
mc=confusion_matrix(y_test, y_pred)
