import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import imblearn
import xgboost as xgb # conda install py-xgboost
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt


data = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
y_data = data[['ctype']]
X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'change_1m', 'change_3m', 'MRS_TX_3', 'ctype'], axis=1)
model = LogisticRegression(max_iter=1000)
model.fit(X_data, y_data)
print(model.intercept_)
print(model.coef_)
#
# for i in list(X_data.columns):
#     print(X_data[[i]].dtypes)

print('done')