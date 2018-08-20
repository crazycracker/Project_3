
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
import xgboost as xgb
from scipy import stats
from scipy.stats import randint
from sklearn import cross_validation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score


# In[2]:


train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
'occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# In[3]:


train_set.head(10)


# In[5]:


train_set_converted = pd.get_dummies(train_set)


# In[13]:


train_set_converted.columns


# In[9]:


test_set_converted = pd.get_dummies(test_set)


# In[16]:


test_set_converted.columns


# # one target class is enough. dropping wage_class > 50k

# In[14]:


train_set_converted.drop(columns=['wage_class_ >50K'],inplace=True)


# In[17]:


test_set_converted.drop(columns=['wage_class_ >50K.'],inplace=True)


# In[18]:


X_train = train_set_converted.drop(columns=['wage_class_ <=50K'])
y_train = train_set_converted['wage_class_ <=50K']
X_test = test_set_converted.drop(columns=['wage_class_ <=50K.'])
y_test = test_set_converted['wage_class_ <=50K.']


# In[22]:


train_cols = X_train.columns
test_cols = X_test.columns


# In[25]:


train_cols


# In[26]:


test_cols


# In[27]:


list(set(train_cols)-set(test_cols))


# In[20]:


X_test.


# In[31]:


params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1.0,
    'n_estimators': 5
}


# In[39]:


X_train.drop(columns=['native_country_ Holand-Netherlands'],inplace=True)


# In[37]:


X_test.drop(columns=['native_country_ Holand-Netherlands'],inplace=True)


# In[40]:


bst = xgb.XGBClassifier(**params).fit(X_train, y_train)


# In[41]:


preds = bst.predict(X_test)
preds


# In[44]:


correct = 0

for i in range(len(preds)):
    if (y_test[i] == preds[i]):
        correct += 1
        
acc = accuracy_score(y_test, preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-acc))
print(accuracy_score(y_test,preds))
print(classification_report(y_test,preds))


# In[67]:


train_set_converted.drop(columns=['native_country_ Holand-Netherlands'],inplace=True)


# In[68]:


clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }

numFolds = 5
kfold_5 = cross_validation.KFold(n = len(train_set_converted), shuffle = True, n_folds = numFolds)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 5,
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)


# In[69]:


clf.fit(X_train,y_train)


# In[70]:


clf.best_estimator_


# In[71]:


clf.best_params_


# In[72]:


best_params = {
    'objective':'binary:logistic',
    'colsample_bytree': 0.53708973447052,
 'learning_rate': 0.20891896852998107,
 'max_depth': 4,
 'min_child_weight': 2,
 'n_estimators': 608,
 'subsample': 0.7497395581379729
}


# In[73]:


clf_xgb = xgb.XGBClassifier(**best_params).fit(X_train,y_train)


# In[74]:


new_preds = clf_xgb.predict(X_test)
new_preds


# In[75]:


correct = 0

for i in range(len(preds)):
    if (y_test[i] == new_preds[i]):
        correct += 1
        
acc = accuracy_score(y_test, new_preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(new_preds)))
print('Error: {0:.4f}'.format(1-acc))
print(accuracy_score(y_test,new_preds))
print(classification_report(y_test,new_preds))


# In[82]:


clf_xgb.feature_importances_ > 0.1


# In[86]:


clf_xgb.booster


# In[87]:


clf_xgb.score


# In[88]:


X_test.columns


# ## from feature importances we can conclude that age, fnlwgt and hours_per_week have score greater than 0.1

# In[90]:


clf_xgb.feature_importances_ > 0.1

Decision Tree Classifier, XGBClassifier, Random Forest Classifier all the algorithms would perform best.
Decision Tree Classifier was giving an accuracy of 71%
Random Forest Classifier was giving an accuracy of 73%
XGBClassifier was giving an accuracy of 85%

I have increased the accuracy to 87% of XGBClassifier by tuning the hyperparameters using RandomizedSearchCV
# # I can conclude that XGBClassifier along with RandomizedSearchCV is the best . XGBClassifier - 87% Accuracy.
