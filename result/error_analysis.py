import pandas as pd
from my_utils import performance_util as pu
import os

df = pd.read_csv(os.path.join('mlp', 'all', 'mlp_hemorrhagic_h_0_hold.csv'))
df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
diff = df.loc[df['predict_y'] != df['label']]

df2 = pd.read_csv(os.path.join('mlp_cnn', 'all', 'mlp_cnn_hemorrhagic_h_0_hold.csv'))
df2['predict_y'] = pu.labelize(df2[['0', '1']].values).astype(int)
diff2 = df2.loc[df2['predict_y'] != df2['label']]

a = list(set(diff.index.values) & set(diff2.index.values))

print('done')