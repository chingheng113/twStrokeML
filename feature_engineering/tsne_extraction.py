from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from my_utils import data_util, plot_fig, performance_util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


def tsne_features_add(x_data, seed):
    t_sne = TSNE(n_components=3, perplexity=30, learning_rate=200.0, random_state=seed)
    new_features = t_sne.fit_transform(x_data)
    # df = pd.DataFrame(new_features, columns=['tsne_x', 'tsne_y', 'tsne_z'])
    new_x_data = np.concatenate([x_data, new_features], axis=1)
    return new_x_data
