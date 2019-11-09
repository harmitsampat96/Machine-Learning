
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('titanic.csv')
data = data.dropna()
data['age'] = data['age'].astype(int)
data.head()


# In[3]:


data.head()
data.sex = data.sex.map(dict(female=1,male=0))
data.embarked = data.embarked.map(dict(C=0,Q=1,S=2))
data['embarked'] = data['embarked'].astype(int)
data=data.drop('name',axis=1)


# In[4]:


data.head()


# In[5]:


newdata=data


# In[6]:


newdata.head()


# In[7]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[8]:


def splitdata1(data):
    X=data.values[:,1:8]
    Y=data.values[:,0]
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
    return X_train,X_test,y_train,y_test


# In[9]:


X_train1,X_test1,y_train1,y_test1=splitdata1(newdata)


# In[10]:


X_train1


# In[11]:


# Using all features:
# Here, we use all the features of the given dataset to create the decision tree and check the accuracy


# In[12]:


def generateGiniClassifierObject(X_train,X_test,y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5) 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 


# In[13]:


clf1 = generateGiniClassifierObject(X_train1, X_test1, y_train1)


# In[14]:


def findPredictionForY(X_test, clf_object): 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred


# In[15]:


# Function to calculate accuracy 
def findAccuracyOfModel(y_test, y_pred): 
    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred)) 
    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100) 
    print("Report : ",classification_report(y_test, y_pred))


# In[16]:


# Using a subset of features:
# Here, we use a subset of the features of the given dataset to create the decision tree and check the accuracy
# Selected subset attributes: survived, pclass, age, sex, sibsp, parch
# Ignored attributes: fare, embarked


# In[17]:


newdata1=data.drop(['fare','embarked'],axis=1)


# In[18]:


newdata1.head()


# In[19]:


def splitdata2(data):
    X=data.values[:,1:6]
    Y=data.values[:,0]
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
    return X_train,X_test,y_train,y_test


# In[20]:


X_train2,X_test2,y_train2,y_test2=splitdata2(newdata1)


# In[21]:


X_train2


# In[22]:


clf2=generateGiniClassifierObject(X_train2,X_test2,y_train2)


# In[23]:


# Using a modified subset of features (Changing the values of features and adding new attributes):
# Here, we use a modified subset of the features of the given dataset to create the decision tree and check the accuracy
# Modification criteria is:
# new feature family = sibsp + parch
# new feature weighted_class = pclass*2 if pclass =1 ; pclass*3 if pclass =2 ; pclass*4 if pclass =3, etc


# In[24]:


newdata2=data


# In[25]:


newdata2.head()


# In[26]:


newdata2['family']=newdata2['sibsp']+newdata2['parch']


# In[27]:


newdata2['pclass'].unique()


# In[28]:


newdata2.weighted_class=int(0)


# In[29]:


newdata2.loc[newdata2.pclass==1,'weighted_class']=2
newdata2.loc[newdata2.pclass==2,'weighted_class']=6
newdata2.loc[newdata2.pclass==3,'weighted_class']=12


# In[30]:


newdata2['weighted_class'].unique()


# In[31]:


newdata2.head()


# In[32]:


def splitdata3(data):
    X=data.values[:,1:10]
    Y=data.values[:,0]
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
    return X_train,X_test,y_train,y_test


# In[33]:


X_train3,X_test3,y_train3,y_test3=splitdata3(newdata2)


# In[34]:


X_train3


# In[35]:


clf3=generateGiniClassifierObject(X_train3,X_test3,y_train3)


# In[36]:


def main():
    print('Decision Tree Classifier:\n')
    print('Case 1: Using all attributes:\n')
    y_pred1=findPredictionForY(X_test1,clf1)
    findAccuracyOfModel(y_test1,y_pred1)
    print('--------------------------------\n')
    print('Case 2: Using a subset of attributes:\n')
    y_pred2=findPredictionForY(X_test2,clf2)
    findAccuracyOfModel(y_test2,y_pred2)
    print('--------------------------------\n')
    print('Case 3: Using modified atttributes:\n')
    y_pred3=findPredictionForY(X_test3,clf3)
    findAccuracyOfModel(y_test3,y_pred3)
    print('--------------------------------\n')


# In[37]:


if __name__=='__main__':
    main()

