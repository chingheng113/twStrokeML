# https://scikit-learn.org/stable/modules/feature_selection.html
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from my_utils import data_util

if __name__ == '__main__':
    subtype = 'is'
    if subtype == 'is':
        sub_class = 'ischemic'
    else:
        sub_class = 'hemorrhagic'

    for i in range(0,10,1):
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('training_'+subtype+'_'+str(i)+'.csv', sub_class=sub_class)
        print(y_data_all[y_data_all == 0].shape)
        print(y_data_all[y_data_all == 1].shape)
        print(y_data_all.shape)
        # clf = ExtraTreesClassifier(n_estimators=50)
        # clf = clf.fit(X, y)
        # clf.feature_importances_
        #
        # model = SelectFromModel(clf, prefit=True)
        # X_new = model.transform(X)
        # X_new.shape
