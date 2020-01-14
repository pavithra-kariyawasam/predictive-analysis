# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:40:41 2018

@author: pavit
"""

import numpy as np
import pandas as pd


df = pd.read_csv('C:/Users/pavit/Desktop/test.csv', header= 0)
original_headers = list(df.columns.values)
#print(df)

#pre processing
df=df.dropna()

df.dtypes


#correlations=df[['TenantId','CompanyId',' SiteId','CapturedDate','CapturedMonth','CapturedDay',' ShiftId',' ShiftHourId',' ItemClassificationId',' GarmentStatusId',' BandId']].corr()    
#print(correlations)

#segmentation of data set

temp_df=df[['TenantId','CompanyId',' SiteId','CapturedDate','CapturedMonth','CapturedDay',' ShiftId',' ShiftHourId',' ItemClassificationId',' GarmentStatusId',' BandId']]
temp_df.dtypes

#y_train = df[" BandId"] 

#columns_to_drop = ["TenantId", "CompanyId", "CapturedDate"," SiteId"]
#df.drop(labels=columns_to_drop, axis=1, inplace=True)

# Data Preprocessing
include = ['CapturedMonth','CapturedDay',' ShiftId',' ShiftHourId',' ItemClassificationId',' GarmentStatusId',' BandId']
df_ = temp_df[include]

# Data Preprocessing


df_ohe = pd.get_dummies(df_, columns=["CapturedMonth","CapturedDay"," ItemClassificationId"])

print(df_ohe)
dependent_variable = ' BandId'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
##train_test_dummies = pd.get_dummies(df, columns=["CapturedMonth","CapturedDay"," ItemClassificationId"," BandId"])




#X_train = train_test_dummies.values[:]
#X_test = train_test_dummies.values[:]

##print(type(X_train))
"""
from sklearn.model_selection import StratifiedShuffleSplit    

sss = StratifiedShuffleSplit(train_size=0.8, n_splits=1, 
                             test_size=0.2, random_state=0) 
for train_index, test_index in sss.split(train_test_dummies, y_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_test_dummies.iloc[train_index], train_test_dummies.iloc[test_index]
    Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]

"""

#print(Y_train)

df.isnull()
print(df.isnull().sum())



# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
print("%%%%%%%%%%%")
#print(x[1])
gb = GradientBoostingClassifier()
gb.fit(x,y)
predictions = gb.predict(x)


print(predictions)


# Save your model
from sklearn.externals import joblib
joblib.dump(gb, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
gb = joblib.load('model.pkl') 


# Saving the data columns from training
dff = pd.DataFrame(x)
print(dff.columns)
model_columns = dff.columns.tolist()
#model_columns = df.columns.tolist()
#model_columns.remove(' BandId')
print(model_columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")