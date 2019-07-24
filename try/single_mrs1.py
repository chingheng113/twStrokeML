import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from my_utils import data_util, performance_util
from sklearn.metrics import roc_auc_score


id_train_all, x_train_all, y_train_all = data_util.get_poor_god_downsample(
            'training_is_9.csv', sub_class='ischemic')
id_hold, x_hold, y_hold = data_util.get_poor_god(
            'hold_is_9.csv', sub_class='ischemic')
lm = LogisticRegression()
x = x_train_all[['MRS_TX_1']]
y = y_train_all
lm.fit(x, y)
test_x = x_hold[['MRS_TX_1']]
predictions = lm.predict(test_x)
logit_roc_auc = roc_auc_score(y_hold, lm.predict(test_x))
print(logit_roc_auc)