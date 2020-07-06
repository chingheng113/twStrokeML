import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

data = pd.read_csv(os.path.join('..', 'data_source', 'TSR_2018_3m_noMissing_validated.csv'))

data = data[(data['ICD_ID'] == 1)]

data = data.assign(MRS_TX_3=pd.cut(data['MRS_TX_3'], [-1, 2, 7], labels=[0, 1]))
# data = data.assign(MRS_TX_1=pd.cut(data['MRS_TX_1'], [-1, 2, 7], labels=[0, 1]))
data = data[['discharged_mrs', 'MRS_TX_1', 'MRS_TX_3']].astype(int)
data = pd.get_dummies(data, columns=['discharged_mrs', 'MRS_TX_1', 'MRS_TX_3'])

label = list(data.columns.values)
names = []
for s in list(data.columns.values):
    if 'MRS_TX_1' in s:
        names.append(s.replace('MRS_TX_1', 'mRS_30days'))
    elif 'MRS_TX_3' in s:
        names.append(s.replace('MRS_TX_3', 'mRS_90days'))
    else:
        names.append(s)

source = []
target = []
value = []


for i in ['0', '1', '2', '3', '4', '5', '6']:
    for j in ['0', '1']:
        source.append(label.index('MRS_TX_1_'+i))
        target.append(label.index('MRS_TX_3_'+j))
        value.append(data[(data['MRS_TX_1_'+i] == 1) & (data['MRS_TX_3_'+j] == 1)].shape[0])


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = names
    ),
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A2, B1, ...
      target = target,
      value = value
  ))])

fig.update_layout(title_text="", font_size=10)
fig.show()