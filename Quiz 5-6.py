#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error as mse, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns


data = pd.read_csv("C:/Users/Vangust/Downloads/titanic.csv")

data.head()


# In[ ]:


# N1,N2,N3:


# In[87]:


# Simple Linear Regression
# X = data[['Embarked']].values
# Y = data[['Survived']].values


# In[88]:


# Multiple linear Regression/Decision Tree Regressor
# X = data[['Embarked', 'Pclass', 'Age']]  
# Y = data['Survived']


# In[89]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[90]:


# Linear Regression
# model = LinearRegression()
# model.fit(X_train, Y_train)


# In[91]:


# Decision tree
# model = DecisionTreeRegressor()
# model.fit(X_train, Y_train)


# In[92]:


# Y_pred = model.predict(X_test)


# In[93]:


# mse = mean_squared_error(Y_test, Y_pred)
# r2 = r2_score(Y_test, Y_pred)


# In[94]:


# print(f'Mean Squared Error (MSE): {mse}')
# print(f'R-squared (R2): {r2}')


# In[95]:


# plt.scatter(X_test, Y_test, color='black')
# plt.plot(X_test, Y_pred, color='blue', linewidth=3)
# plt.xlabel('Embarked (X)')
# plt.ylabel('Survived (Y)')
# plt.title('Simple Linear Regression')
# plt.show()


# In[96]:


# Multiple linear Regression/Decision Tree Regressor
# new_data = pd.DataFrame({'Embarked': [1], 'Pclass': [2], 'Age': [3]})
# new_prediction = model.predict(new_data)
# print('Prediction on new data:', new_prediction[0])


# In[ ]:


# N4,N5


# In[97]:


X = data[['Embarked', 'Pclass', 'Age']] 
Y = data['Sex']


# In[98]:



X_train, X_test, Y_train, Y_test = train_test_split(data[['Age']],data.Survived,train_size=0.8, random_state=20)


# In[99]:


# Logistic Regression
# model = LogisticRegression()
# model.fit(X_train, Y_train)


# In[100]:


# Decision Tree Classifier
# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)


# In[101]:


Y_pred = model.predict(X_test)


# In[102]:


accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)


# In[103]:


print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', classification_rep)


# In[104]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[105]:


new_data = pd.DataFrame({'Sex': [2]})
print('Prediction on new data:', new_prediction[0])


# In[ ]:





# In[ ]:




