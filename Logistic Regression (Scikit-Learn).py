#!/usr/bin/env python
# coding: utf-8

# # NLP Mafia Project (Logistic Regression)
# 
# Here, we implement a Logistic Regression model using Scikit-Learn that serves as a Baseline model for the NLP Mafia project.

# In[68]:


# Cloning into the repository to obtain files
get_ipython().system('git clone https://bitbucket.org/bopjesvla/thesis.git')


# In[69]:


get_ipython().system('cp thesis/src/* .')


# ## Import the required modules

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# In[72]:


# Read in the data from the pkl file and reset the indices
# in order to get rid of any indices that might be empty
docs = pd.read_pickle('docs.pkl', compression='gzip')
docs = docs.reset_index()
docs


# ## Instantiate Model

# In[ ]:


# Instantiate a model and specify the number of splits of the
# dataset that we will be using

model = LogisticRegression(class_weight = {0: 0.25, 1: 0.75})

kf = KFold(n_splits = 20)


# ## FastText Wiki

# In[ ]:


# We have the data in the form of a Pandas Series. Convert this
# to NumPy arrays in order to pass to our model
vectorFastTextwiki = pd.Series(docs['vector_FastText_wiki']).to_numpy()
scum = pd.Series(docs['scum']).to_numpy()


# In[76]:


score_final = 0.0
auroc_final = 0.0
ap_final = 0.0

for train_index, test_index in kf.split(vectorFastTextwiki):
    X_train, X_test = vectorFastTextwiki[train_index], vectorFastTextwiki[test_index]
    Y_train, Y_test = scum[train_index], scum[test_index]

    # Train the model on the Training Dataset
    model.fit(X_train.tolist(), Y_train.tolist())

    # Model makes predictions based on input from the Test Dataset
    predictions = model.predict(X_test.tolist())

    # Compute the percentage accuracy of the model's predictions
    score = model.score(X_test.tolist(), Y_test.tolist())

    # Compute the AUROC of the model
    auroc = roc_auc_score(Y_test.tolist(), predictions)

    # Compute the Average Precision of the model
    average_precision = average_precision_score(Y_test.tolist(), predictions)

    # Stores the above results so as to obtain the mean performance
    # of the model on the total dataset
    score_final += score
    auroc_final += auroc
    ap_final += average_precision

print("Score:", score_final / 20.0)
print("AUROC:", auroc_final / 20.0)
print("Average Precision:", ap_final / 20.0)


# ## GloVe Wiki

# In[ ]:


# We have the data in the form of a Pandas Series. Convert this
# to NumPy arrays in order to pass to our model
vectorGloVewiki = pd.Series(docs['vector_GloVe_wiki']).to_numpy()


# In[79]:


score_final_2 = 0.0
auroc_final_2 = 0.0
ap_final_2 = 0.0

for train_index, test_index in kf.split(vectorGloVewiki):
    X_train, X_test = vectorGloVewiki[train_index], vectorGloVewiki[test_index]
    Y_train, Y_test = scum[train_index], scum[test_index]

    # Train the model on the Training Dataset
    model.fit(X_train.tolist(), Y_train.tolist())

    # Model makes predictions based on input from the Test Dataset
    predictions = model.predict(X_test.tolist())

    # Compute the percentage accuracy of the model's predictions
    score = model.score(X_test.tolist(), Y_test.tolist())

    # Compute the AUROC of the model
    auroc = roc_auc_score(Y_test.tolist(), predictions)

    # Compute the Average Precision of the model
    average_precision = average_precision_score(Y_test.tolist(), predictions)

    # Stores the above results so as to obtain the mean performance
    # of the model on the total dataset
    score_final_2 += score
    auroc_final_2 += auroc
    ap_final_2 += average_precision

print("Score:", score_final_2 / 20.0)
print("AUROC:", auroc_final_2 / 20.0)
print("Average Precision:", ap_final_2 / 20.0)

