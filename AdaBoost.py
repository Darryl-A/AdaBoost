#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading in the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[3]:


#Read in the training file
df = pd.read_csv("http://miner2.vsnet.gmu.edu/files/uploaded_files/1646102559_507402_train35.txt")


# In[4]:


print(df)


# In[5]:


#Set the X and y for the table
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


# In[6]:


X = np.asarray(X)


# In[7]:


print(X)


# In[8]:


print(y)


# In[9]:


#Convert classes to -1 and 1
for i in range(len(y)):
    if y[i] == 3:
        y[i] = -1
    elif y[i] == 5:
        y[i] = 1


# In[10]:


print(y)


# In[11]:


print(y[:20])


# In[12]:


print(X)


# In[13]:


#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)


# In[14]:


print(X_train)


# In[63]:


#Decision Tree with depth 1
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DStump:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        #self.polarity = 1
        
        
    #Fit the tree using buildStump method
    def fit(self, X, y):
        self.classesN = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self.buildStump(X, y)

        
    #Given the dataframe X, predict the target variables 1 or -1
    def predict(self, X):   
        node = self.tree_
        predictedClass = []
        for inputs in X:
            while node.left:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictedClass.append(node.predicted_class)
        return predictedClass
            
    

    #Find the best feature and threshold to split on using gini index
    def bestSplit(self, X, y):
        num_parent = [np.sum(y == c) for c in range(self.classesN)]
        best_gini = 1.0 - sum((n / len(y)) ** 2 for n in num_parent)
        splitIndex, splitThres = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            leftNums = [0] * self.classesN
            rightNums = num_parent.copy()
            for i in range(1, len(y)):
                label = classes[i-1]
                leftNums[label] += 1
                rightNums[label] -= -1
                leftGini = 1.0 - sum((leftNums[x] / i) ** 2 for x in range(self.classesN))
                rightGini = 1.0 - sum((rightNums[x] / (len(y)-i)) ** 2 for x in range(self.classesN))
                weightedGini = (i * leftGini + (len(y) - i) * rightGini) / len(y)
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if weightedGini < best_gini:
                    best_gini = weightedGini
                    splitIndex = idx
                    splitThres = (float)((thresholds[i] + thresholds[i - 1]) / 2)
                    self.feature_index = splitIndex
                    self.threshold = splitThres
        return splitIndex, splitThres

    #Build the stump give a depth of 1
    def buildStump(self, X, y, depth=0):
        totalSamples = [np.sum(y == i) for i in range(self.classesN)]
        predicted_class = np.argmax(totalSamples)
        node = Node(predicted_class = predicted_class)
        if depth < self.max_depth:
            featureIndex, threshold = self.bestSplit(X, y)
            if featureIndex is not None:
                leftNode = X[:, featureIndex] < threshold
                rightNode = X[:, featureIndex] > threshold
                X_left = X[leftNode]
                y_left = y[leftNode]
                X_right = X[rightNode]
                y_right = y[rightNode]
                node.feature_index = featureIndex
                node.threshold = threshold
        return node


# In[64]:


#Using my decision tree with max depth 1
cf = DStump(max_depth = 1)


# In[65]:


#Fitting the stump on the given training set
cf.fit(X_train,y_train)


# In[58]:


ans = cf.predict(X_test)


# In[59]:


print(ans)


# In[32]:


#Checking the accuracy of the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm2 = confusion_matrix(y_test, ans)
print(cm2)
accuracy_score(y_test,ans)


# In[33]:


#Comparing my implementation with sklearn's
clf = DecisionTreeClassifier(max_depth = 1)
clf.fit(X_train, y_train)
check = clf.predict(X_test)


# In[34]:


com2 = confusion_matrix(y_test, check)
print(com2)
accuracy_score(y_test,check)


# In[44]:


#Implementing AdaBoost
class AdaBoost():
    def __init__(self, iters = 200):
        self.n_clf = iters
        self.clfs= None
        self.alpha = None
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        
        # Create list to store weak learners and their respective alpha values
        self.clfs = []
        self.alpha = []
        
        #Go over all the weak learners
        for i in range(0,self.n_clf):
            
            # creating stump classifier
            stump = DecisionTreeClassifier(max_depth=1)
            
            # Fit using new weights
            stump.fit(X,y, sample_weight = w)
            
            # Get predictions of the model
            predictions = stump.predict(X)
            
            # Calculating error
            errors = np.array([predictions[i]!= y[i] for i in range(n_samples)])
            e = np.sum(errors * w) / np.sum(w)
            
            #If error is greater than 0.5, reset weight for all instances
            if e > 0.5:
                w = np.full(n_samples, (1 / n_samples))
            
            #Calculate alpha value
            alpha = 0.5 * np.log((1.0 - e) / (e))
            
            # Add it to the alpha list
            self.alpha.append(alpha)
            
            #Calculate new weights - more weight for the misclassified
            w *= np.exp(-alpha * y * predictions)
            
            # Normalize to one
            w /= np.sum(w)

            #Add the weak learner to the stump list.
            self.clfs.append(stump)

            
    #Predict target variable given the matrix X        
    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        
        # Iterating through each classifier
        for i in range(self.n_clf):
            predictions = np.expand_dims(self.clfs[i].predict(X),1)
           
            #Combine all the weak classifiers with their importance -- alpha value
            y_pred += self.alpha[i] * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred


# In[45]:


#Assign the class to the model variable
model = AdaBoost()


# In[46]:


#Fit the training set on the adaboost model
model.fit(X_train,y_train)


# In[47]:


preds = model.predict(X_test)


# In[48]:


print(preds)


# In[49]:


#Check accuracy with confusion matrix
co = confusion_matrix(y_test, preds)
print(co)
accuracy_score(y_test,preds)


# In[41]:


#Getting the accuracy of the model:
def accuracy(y_pred,y_test):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count = count + 1
    return count / len(y_test)


# In[42]:


#Getting the classification error
classError = 1 - accuracy(preds, y_test)
print(classError)


# In[656]:


#Importing the test file
testDf = pd.read_csv("http://miner2.vsnet.gmu.edu/files/uploaded_files/1646102559_513849_test35-nolabels.txt", header = None)


# In[657]:


X2 = np.asarray(testDf)


# In[658]:


print(X2)


# In[663]:


y_pred = clf.predict(X2)


# In[664]:


print(y_pred)


# In[665]:


#Convert classes back to 3 and 5
for i in range(len(y_pred)):
    if y_pred[i] == -1:
        y_pred[i] = 3
    elif y_pred[i] == 1:
        y_pred[i] = 5


# In[666]:


y_pred = y_pred.astype(int)


# In[667]:


print(y_pred)


# In[668]:


len(y_pred)


# In[669]:


#Write the prediction file to another file
y_pred.tofile('AdaboostPred7.csv', sep = '\n')


# In[ ]:




