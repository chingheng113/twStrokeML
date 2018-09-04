from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from my_utils import data_util, plot_fig, performance_util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


def tsne_features_add(x_data, seed):
    t_sne = TSNE(n_components=2, perplexity=30, learning_rate=200.0, random_state=seed, verbose=1)
    new_features = t_sne.fit_transform(x_data)
    new_features = data_util.scale(new_features)
    new_x_data = np.concatenate([x_data, new_features], axis=1)
    return new_x_data
