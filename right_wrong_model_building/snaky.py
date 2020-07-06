import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os


look_at = 'w'

data = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
data = data.assign(MRS_TX_3=pd.cut(data['MRS_TX_3'], [-1, 2, 7], labels=[0, 1]))
data = data[['discharged_mrs', 'MRS_TX_1', 'MRS_TX_3', 'ctype']].astype(int)
data = pd.get_dummies(data, columns=['discharged_mrs', 'MRS_TX_1', 'MRS_TX_3'])

label = list(data.columns.values)
label.remove('ctype')

right_data = data[data.ctype == 1]
wrong_data = data[data.ctype == 0]

source = []
target = []
value = []

for i in ['0', '1', '2', '3', '4', '5']:
    for j in ['0', '1', '2', '3', '4', '5', '6']:
        source.append(label.index('discharged_mrs_'+i))
        target.append(label.index('MRS_TX_1_'+j))
        if look_at == 'r':
            value.append(right_data[(right_data['discharged_mrs_' + i] == 1) & (right_data['MRS_TX_1_' + j] == 1)].shape[0])
        else:
            value.append(wrong_data[(wrong_data['discharged_mrs_'+i] == 1) & (wrong_data['MRS_TX_1_'+j] == 1)].shape[0])

for i in ['0', '1', '2', '3', '4', '5', '6']:
    for j in ['0', '1']:
        source.append(label.index('MRS_TX_1_'+i))
        target.append(label.index('MRS_TX_3_'+j))
        if look_at == 'r':
            value.append(right_data[(right_data['MRS_TX_1_' + i] == 1) & (right_data['MRS_TX_3_' + j] == 1)].shape[0])
        else:
            value.append(wrong_data[(wrong_data['MRS_TX_1_'+i] == 1) & (wrong_data['MRS_TX_3_'+j] == 1)].shape[0])


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = label
    ),
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A2, B1, ...
      target = target,
      value = value
  ))])

if look_at == 'r':
    fig.update_layout(title_text="Right group", font_size=10)
else:
    fig.update_layout(title_text="Wrong group", font_size=10)
fig.show()