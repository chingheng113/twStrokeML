import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os


data = pd.read_csv('TSR_group_sankey.csv')
# data = data.head(100)
data = data.dropna(axis=0)
cols = ['ICD', 'NIHSS_in', 'NIHSS_out', 'MRS_0', 'MRS_1', 'MRS_3']
data = data[cols]

data['NIHSS_change'] = data['NIHSS_out'] - data['NIHSS_in']
data['NIHSS_change'] = pd.cut(data['NIHSS_change'], [-99, -1, 0, 99], labels=[-1, 0, 1])
cols = ['ICD', 'NIHSS_change', 'MRS_0', 'MRS_1', 'MRS_3']


data = pd.get_dummies(data, columns=cols)
label = data.columns.values.tolist()

source = []
target = []
value = []
# ICD -> NIHSS_change
for i in ['1', '2', '3', '4', '5']:
    for j in ['-1', '0', '1']:
        source.append(label.index('ICD_'+i))
        target.append(label.index('NIHSS_change_'+j))
        value.append(data[(data['ICD_'+i] == 1) & (data['NIHSS_change_'+j] == 1)].size)
# NIHSS_change -> mrs_0
for i in ['-1', '0', '1']:
    for j in ['0', '1', '2', '3', '4', '5']:
        source.append(label.index('NIHSS_change_'+i))
        target.append(label.index('MRS_0_'+j))
        value.append(data[(data['NIHSS_change_'+i] == 1) & (data['MRS_0_'+j] == 1)].size)
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


label = ['Infarct' if x == 'ICD_1' else x for x in label]
label = ['TIA' if x == 'ICD_2' else x for x in label]
label = ['ICH' if x == 'ICD_3' else x for x in label]
label = ['SAH' if x == 'ICD_4' else x for x in label]
label = ['Other stroke' if x == 'ICD_5' else x for x in label]
label = ['NIHSS_difference_decline' if x == 'NIHSS_change_-1' else x for x in label]
label = ['NIHSS_difference_increase' if x == 'NIHSS_change_1' else x for x in label]
label = ['NIHSS_difference_unchanged' if x == 'NIHSS_change_0' else x for x in label]
label = ['mRS_discharge_0' if x == 'MRS_0_0' else x for x in label]
label = ['mRS_discharge_1' if x == 'MRS_0_1' else x for x in label]
label = ['mRS_discharge_2' if x == 'MRS_0_2' else x for x in label]
label = ['mRS_discharge_3' if x == 'MRS_0_3' else x for x in label]
label = ['mRS_discharge_4' if x == 'MRS_0_4' else x for x in label]
label = ['mRS_discharge_5' if x == 'MRS_0_5' else x for x in label]
label = ['mRS_30days_0' if x == 'MRS_1_0' else x for x in label]
label = ['mRS_30days_1' if x == 'MRS_1_1' else x for x in label]
label = ['mRS_30days_2' if x == 'MRS_1_2' else x for x in label]
label = ['mRS_30days_3' if x == 'MRS_1_3' else x for x in label]
label = ['mRS_30days_4' if x == 'MRS_1_4' else x for x in label]
label = ['mRS_30days_5' if x == 'MRS_1_5' else x for x in label]
label = ['mRS_90days_0' if x == 'MRS_3_0' else x for x in label]
label = ['mRS_90days_1' if x == 'MRS_3_1' else x for x in label]
label = ['mRS_90days_2' if x == 'MRS_3_2' else x for x in label]
label = ['mRS_90days_3' if x == 'MRS_3_3' else x for x in label]
label = ['mRS_90days_4' if x == 'MRS_3_4' else x for x in label]
label = ['mRS_90days_5' if x == 'MRS_3_5' else x for x in label]


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