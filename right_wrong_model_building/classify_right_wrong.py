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
from lime import submodular_pick


data = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
y_data = data[['ctype']]
X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'change_1m', 'change_3m', 'MRS_TX_1', 'MRS_TX_3', 'ctype'], axis=1)

for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=i, stratify=y_data)
    # id_test = id_data.loc[X_test.index]

    # scaling
    # scaler = preprocessing.MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # if no scaling, do dataframe to ndarray for lime function
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # over-sampling
    print('before', y_train.groupby(['ctype']).size())
    # sm = imblearn.over_sampling.SMOTE(random_state=42)
    sm = imblearn.under_sampling.RandomUnderSampler(random_state=888)
    # sm = imblearn.combine.SMOTETomek(random_state=369)

    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('after', y_train.groupby(['ctype']).size())

    # model = ExtraTreesClassifier(n_estimators=250, random_state=0)
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.5)
    # model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred_proba[:, 1]))
    # print(model.intercept_)
    # print(model.coef_)


    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                       mode='classification', training_labels=y_train,
                                                       feature_names=X_data.columns.values, random_state=369)

    # idx = 12 #9
    # exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
    # print('id: %d' % idx)
    # # xgboost(like sklearn) expects X as 2D data(n_samples, n_features).If you want to predict only one sample
    # # you can reshape your feature vector to a 2D array
    # print('Probability (right case) =', model.predict_proba(np.array(X_test[idx]).reshape((1, -1)))[0, 1])
    # print('True class: %s' % y_test.values[idx])
    # print('Explanation for class right')
    # print('\n'.join(map(str, exp.as_list())))
    # fig = exp.as_pyplot_figure()
    # plt.show()

    # Submodular Pick
    sp_obj = submodular_pick.SubmodularPick(explainer, X_train, model.predict_proba,
                                            sample_size=50,
                                            num_features=5,
                                            num_exps_desired=1)
    for exp in sp_obj.sp_explanations:
        # https://stackoverflow.com/questions/60914598/keyerror-1-in-using-sp-lime-with-lightgbm
        exp.as_pyplot_figure(label=exp.available_labels()[0])
        plt.show()
        print(exp.as_list(label=exp.available_labels()[0]))
    print('done')