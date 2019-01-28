from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from my_utils import data_util, plot_fig, performance_util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


def get_ischemic(df):
    return df[(df['ICD_ID_1.0'] == 1) | (df['ICD_ID_2.0'] == 1)]


def get_hemorrhagic(df):
    return df[(df['ICD_ID_3.0'] == 1) | (df['ICD_ID_4.0'] == 1)]


if __name__ == '__main__':
    np.random.seed(999)
    # df_all = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')
    #
    # df_is = get_ischemic(df_all)
    # id_is, x_is, y_is = data_util.get_x_y_data(df_is)
    # y_is = data_util.get_binarized_label(y_is)
    #
    # df_he = get_hemorrhagic(df_all)
    # id_he, x_he, y_he = data_util.get_x_y_data(df_he)
    # y_he = data_util.get_binarized_label(y_he)
    #
    # # is
    # x_data_train = data_util.scale(x_is)
    # t_sne = TSNE(n_components=2, perplexity=30).fit_transform(x_data_train)
    # df = pd.DataFrame(t_sne, columns=['x', 'y'])
    # df['p'] = y_is
    # df.to_csv('tsne_is.csv', sep=',', encoding='utf-8', index=False)
    # # he
    # x_data_train = data_util.scale(x_he)
    # t_sne = TSNE(n_components=2, perplexity=30).fit_transform(x_data_train)
    # df = pd.DataFrame(t_sne, columns=['x', 'y'])
    # df['p'] = y_he
    # df.to_csv('tsne_he.csv', sep=',', encoding='utf-8', index=False)

    df = pd.read_csv('tsne_is.csv', encoding='utf8')
    plt.figure()
    plt.scatter(df.ix[:,0], df.ix[:,1], c=df.ix[:, 2], s=0.1, cmap=plt.cm.get_cmap("jet", 2))
    plt.colorbar(ticks=range(2))
    plt.title('t-SNE 2D visualization of Taiwan stoke registry data')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.savefig("t-sne.png", dpi=300)
    plt.show()
