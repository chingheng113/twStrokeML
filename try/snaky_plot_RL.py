import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os


data = pd.read_csv('TSR_group_sankey.csv')
# data = data.head(100)
data = data.dropna(axis=0)
cols = ['ICD', 'NIHSS_in_class', 'NIHSS_out_class', 'MRS_0', 'MRS_1', 'MRS_3']
data = data[cols]
data = pd.get_dummies(data, columns=cols)
label = data.columns.values.tolist()

source = []
target = []
value = []
# ICD -> NIHSS_in
for i in ['1', '2', '3', '4', '5']:
    for j in ['0', '1', '2', '3']:
        source.append(label.index('ICD_'+i))
        target.append(label.index('NIHSS_in_class_'+j))
        value.append(data[(data['ICD_'+i] == 1) & (data['NIHSS_in_class_'+j] == 1)].size)
# NIHSS_in -> NIHSS_out
for i in ['0', '1', '2', '3']:
    for j in ['0', '1', '2', '3']:
        source.append(label.index('NIHSS_in_class_'+i))
        target.append(label.index('NIHSS_out_class_'+j))
        value.append(data[(data['NIHSS_in_class_'+i] == 1) & (data['NIHSS_out_class_'+j] == 1)].size)
# NIHSS_out -> mrs_0
for i in ['0', '1', '2', '3']:
    for j in ['0', '1', '2', '3', '4', '5']:
        source.append(label.index('NIHSS_out_class_'+i))
        target.append(label.index('MRS_0_'+j))
        value.append(data[(data['NIHSS_out_class_'+i] == 1) & (data['MRS_0_'+j] == 1)].size)
# mrs_0 -> mrs_1
for i in ['0', '1', '2', '3', '4', '5']:
    for j in ['0', '1', '2', '3', '4', '5']:
        source.append(label.index('MRS_0_'+i))
        target.append(label.index('MRS_1_'+j))
        value.append(data[(data['MRS_0_'+i] == 1) & (data['MRS_1_'+j] == 1)].size)
# mrs_1 -> mrs_3
for i in ['0', '1', '2', '3', '4', '5']:
    for j in ['0', '1', '2', '3', '4', '5']:
        source.append(label.index('MRS_1_'+i))
        target.append(label.index('MRS_3_'+j))
        value.append(data[(data['MRS_1_'+i] == 1) & (data['MRS_3_'+j] == 1)].size)

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

fig.update_layout(title_text="", font_size=10)
fig.show()