

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
import xgboost as xgb
from scipy import stats
from scipy.stats import randint
from sklearn import cross_validation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score
```


```python
train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
'occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels
```


```python
train_set.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>wage_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>5</th>
      <td>37</td>
      <td>Private</td>
      <td>284582</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>6</th>
      <td>49</td>
      <td>Private</td>
      <td>160187</td>
      <td>9th</td>
      <td>5</td>
      <td>Married-spouse-absent</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>Jamaica</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>7</th>
      <td>52</td>
      <td>Self-emp-not-inc</td>
      <td>209642</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>8</th>
      <td>31</td>
      <td>Private</td>
      <td>45781</td>
      <td>Masters</td>
      <td>14</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>14084</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>9</th>
      <td>42</td>
      <td>Private</td>
      <td>159449</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>5178</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_set_converted = pd.get_dummies(train_set)
```


```python
train_set_converted.columns
```




    Index(['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
           'hours_per_week', 'workclass_ ?', 'workclass_ Federal-gov',
           'workclass_ Local-gov', 'workclass_ Never-worked',
           ...
           'native_country_ Scotland', 'native_country_ South',
           'native_country_ Taiwan', 'native_country_ Thailand',
           'native_country_ Trinadad&Tobago', 'native_country_ United-States',
           'native_country_ Vietnam', 'native_country_ Yugoslavia',
           'wage_class_ <=50K', 'wage_class_ >50K'],
          dtype='object', length=110)




```python
test_set_converted = pd.get_dummies(test_set)
```


```python
test_set_converted.columns
```




    Index(['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
           'hours_per_week', 'workclass_ ?', 'workclass_ Federal-gov',
           'workclass_ Local-gov', 'workclass_ Never-worked',
           ...
           'native_country_ Scotland', 'native_country_ South',
           'native_country_ Taiwan', 'native_country_ Thailand',
           'native_country_ Trinadad&Tobago', 'native_country_ United-States',
           'native_country_ Vietnam', 'native_country_ Yugoslavia',
           'wage_class_ <=50K.', 'wage_class_ >50K.'],
          dtype='object', length=109)



# one target class is enough. dropping wage_class > 50k


```python
train_set_converted.drop(columns=['wage_class_ >50K'],inplace=True)
```


```python
test_set_converted.drop(columns=['wage_class_ >50K.'],inplace=True)
```


```python
X_train = train_set_converted.drop(columns=['wage_class_ <=50K'])
y_train = train_set_converted['wage_class_ <=50K']
X_test = test_set_converted.drop(columns=['wage_class_ <=50K.'])
y_test = test_set_converted['wage_class_ <=50K.']
```


```python
train_cols = X_train.columns
test_cols = X_test.columns
```


```python
train_cols
```




    Index(['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
           'hours_per_week', 'workclass_ ?', 'workclass_ Federal-gov',
           'workclass_ Local-gov', 'workclass_ Never-worked',
           ...
           'native_country_ Portugal', 'native_country_ Puerto-Rico',
           'native_country_ Scotland', 'native_country_ South',
           'native_country_ Taiwan', 'native_country_ Thailand',
           'native_country_ Trinadad&Tobago', 'native_country_ United-States',
           'native_country_ Vietnam', 'native_country_ Yugoslavia'],
          dtype='object', length=108)




```python
test_cols
```




    Index(['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
           'hours_per_week', 'workclass_ ?', 'workclass_ Federal-gov',
           'workclass_ Local-gov', 'workclass_ Never-worked',
           ...
           'native_country_ Portugal', 'native_country_ Puerto-Rico',
           'native_country_ Scotland', 'native_country_ South',
           'native_country_ Taiwan', 'native_country_ Thailand',
           'native_country_ Trinadad&Tobago', 'native_country_ United-States',
           'native_country_ Vietnam', 'native_country_ Yugoslavia'],
          dtype='object', length=107)




```python
list(set(train_cols)-set(test_cols))
```




    ['native_country_ Holand-Netherlands']




```python
X_test.
```




    (16281, 107)




```python
params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1.0,
    'n_estimators': 5
}
```


```python
X_train.drop(columns=['native_country_ Holand-Netherlands'],inplace=True)
```


```python
X_test.drop(columns=['native_country_ Holand-Netherlands'],inplace=True)
```


```python
bst = xgb.XGBClassifier(**params).fit(X_train, y_train)
```


```python
preds = bst.predict(X_test)
preds
```




    array([1, 1, 1, ..., 0, 1, 0], dtype=uint8)




```python
correct = 0

for i in range(len(preds)):
    if (y_test[i] == preds[i]):
        correct += 1
        
acc = accuracy_score(y_test, preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-acc))
print(accuracy_score(y_test,preds))
print(classification_report(y_test,preds))
```

    Predicted correctly: 13839/16281
    Error: 0.1500
    0.8500092131932928
                 precision    recall  f1-score   support
    
              0       0.76      0.53      0.63      3846
              1       0.87      0.95      0.91     12435
    
    avg / total       0.84      0.85      0.84     16281
    
    


```python
train_set_converted.drop(columns=['native_country_ Holand-Netherlands'],inplace=True)
```


```python
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
```


```python
clf.fit(X_train,y_train)
```

    Fitting 5 folds for each of 5 candidates, totalling 25 fits
    

    [Parallel(n_jobs=-1)]: Done  25 out of  25 | elapsed:  4.1min finished
    




    RandomizedSearchCV(cv=sklearn.cross_validation.KFold(n=32561, n_folds=5, shuffle=True, random_state=None),
              error_score=0,
              estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1),
              fit_params=None, iid=True, n_iter=5, n_jobs=-1,
              param_distributions={'subsample': <scipy.stats._distn_infrastructure.rv_frozen object at 0x00000248E681BC50>, 'min_child_weight': [1, 2, 3, 4], 'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x00000248E681B2B0>, 'max_depth': [3, 4, 5, 6, 7, 8, 9], 'learning_rate': <scipy.stats._distn_infrastructure.rv_frozen object at 0x00000248E681B940>, 'colsample_bytree': <scipy.stats._distn_infrastructure.rv_frozen object at 0x00000248E681BEF0>},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='roc_auc', verbose=3)




```python
clf.best_estimator_
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=0.53708973447052, gamma=0,
           learning_rate=0.20891896852998107, max_delta_step=0, max_depth=4,
           min_child_weight=2, missing=None, n_estimators=608, n_jobs=1,
           nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=0.7497395581379729)




```python
clf.best_params_
```




    {'colsample_bytree': 0.53708973447052,
     'learning_rate': 0.20891896852998107,
     'max_depth': 4,
     'min_child_weight': 2,
     'n_estimators': 608,
     'subsample': 0.7497395581379729}




```python
best_params = {
    'objective':'binary:logistic',
    'colsample_bytree': 0.53708973447052,
 'learning_rate': 0.20891896852998107,
 'max_depth': 4,
 'min_child_weight': 2,
 'n_estimators': 608,
 'subsample': 0.7497395581379729
}
```


```python
clf_xgb = xgb.XGBClassifier(**best_params).fit(X_train,y_train)
```


```python
new_preds = clf_xgb.predict(X_test)
new_preds
```




    array([1, 1, 1, ..., 0, 1, 0], dtype=uint8)




```python
correct = 0

for i in range(len(preds)):
    if (y_test[i] == new_preds[i]):
        correct += 1
        
acc = accuracy_score(y_test, new_preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(new_preds)))
print('Error: {0:.4f}'.format(1-acc))
print(accuracy_score(y_test,new_preds))
print(classification_report(y_test,new_preds))
```

    Predicted correctly: 14169/16281
    Error: 0.1297
    0.8702782384374425
                 precision    recall  f1-score   support
    
              0       0.76      0.66      0.71      3846
              1       0.90      0.94      0.92     12435
    
    avg / total       0.87      0.87      0.87     16281
    
    


```python
clf_xgb.feature_importances_ > 0.1
```




    array([ True,  True, False, False, False,  True, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False])




```python
clf_xgb.booster
```




    'gbtree'




```python
clf_xgb.score
```




    <bound method ClassifierMixin.score of XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=0.53708973447052, gamma=0,
           learning_rate=0.20891896852998107, max_delta_step=0, max_depth=4,
           min_child_weight=2, missing=None, n_estimators=608, n_jobs=1,
           nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=0.7497395581379729)>




```python
X_test.columns
```




    Index(['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
           'hours_per_week', 'workclass_ ?', 'workclass_ Federal-gov',
           'workclass_ Local-gov', 'workclass_ Never-worked',
           ...
           'native_country_ Portugal', 'native_country_ Puerto-Rico',
           'native_country_ Scotland', 'native_country_ South',
           'native_country_ Taiwan', 'native_country_ Thailand',
           'native_country_ Trinadad&Tobago', 'native_country_ United-States',
           'native_country_ Vietnam', 'native_country_ Yugoslavia'],
          dtype='object', length=107)



## from feature importances we can conclude that age, fnlwgt and hours_per_week have score greater than 0.1


```python
clf_xgb.feature_importances_ > 0.1
```




    array([ True,  True, False, False, False,  True, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False])


Decision Tree Classifier, XGBClassifier, Random Forest Classifier all the algorithms would perform best.
Decision Tree Classifier was giving an accuracy of 71%
Random Forest Classifier was giving an accuracy of 73%
XGBClassifier was giving an accuracy of 85%

I have increased the accuracy to 87% of XGBClassifier by tuning the hyperparameters using RandomizedSearchCV
# I can conclude that XGBClassifier along with RandomizedSearchCV is the best . XGBClassifier - 87% Accuracy.
