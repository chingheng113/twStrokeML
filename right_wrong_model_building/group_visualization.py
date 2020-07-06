import pandas as pd
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import prince
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
y_data = data[['ctype']]
x_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'change_1m', 'change_3m', 'MRS_TX_3', 'ctype'], axis=1)
# x_data = data[['MRS_TX_1', 'Transfers', 'Feeding', 'Mobility', 'Bathing']]
x_data = preprocessing.StandardScaler().fit_transform(x_data)

# # pca
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x_data)
# principalDf = pd.DataFrame({'principal component 1':principalComponents[:,0],
#                             'principal component 2':principalComponents[:,1]
#                             })
# finalDf = pd.concat([principalDf, y_data], axis=1)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = [0, 1]
# colors = ['r', 'g']
# for target, color in zip(targets, colors):
#     indicesToKeep = finalDf['ctype'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
#                finalDf.loc[indicesToKeep, 'principal component 2'],
#                c=color,
#                s=1,
#                alpha=0.5)
# ax.legend(targets)
# ax.grid()
# plt.show()
# print(pca.explained_variance_ratio_)

# tsne
tsne = TSNE(n_components=2, perplexity=3)
tsne_results = tsne.fit_transform(x_data)
principalDf = pd.DataFrame({'principal component 1':tsne_results[:,0],
                            'principal component 2':tsne_results[:,1]})
finalDf = pd.concat([principalDf, y_data], axis=1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component t-sne', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['ctype'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=1,
               alpha=0.5)
ax.legend(targets)
ax.grid()
plt.show()
