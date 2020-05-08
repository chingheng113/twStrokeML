import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os


data = pd.read_csv(os.path.join('..', 'data_source', 'TSR_2018_3m(ck).csv'))
# data = data.head(100)

data = data[['Gender', 'onset_age', 'NIHS_1a_in', 'NIHS_1b_in', 'NIHS_1c_in', 'NIHS_2_in', 'NIHS_3_in', 'NIHS_4_in',
             'NIHS_5aL_in', 'NIHS_5bR_in', 'NIHS_6aL_in', 'NIHS_6bR_in', 'NIHS_7_in', 'NIHS_8_in', 'NIHS_9_in',
             'NIHS_10_in', 'NIHS_11_in', 'NIHS_1a_out', 'NIHS_1b_out', 'NIHS_1c_out', 'NIHS_2_out', 'NIHS_3_out',
             'NIHS_4_out', 'NIHS_5aL_out', 'NIHS_5bR_out', 'NIHS_6aL_out', 'NIHS_6bR_out', 'NIHS_7_out', 'NIHS_8_out',
             'NIHS_9_out', 'NIHS_10_out', 'NIHS_11_out']]
data = data.dropna(axis=0)

# data = data[data.Gender == 0]

cols = ['NIHS_1a', 'NIHS_1b', 'NIHS_1c', 'NIHS_2', 'NIHS_3', 'NIHS_4', 'NIHS_5aL', 'NIHS_5bR', 'NIHS_6aL', 'NIHS_6bR',
        'NIHS_7', 'NIHS_8', 'NIHS_9', 'NIHS_10', 'NIHS_11']

status = ['decrease', 'remain', 'increase']
source = []
target = []
value = []
for inx, col in enumerate(cols):
    data[col+'_diff'] = data[col+'_out'] - data[col+'_in']
    data[col+'_diff'] = pd.cut(data[col+'_diff'], [-99, -1, 0, 99], labels=[-1, 0, 1])
    agroup_count = data[[col+'_in', col+'_diff']].groupby([col+'_diff']).count()
    for index, row in agroup_count.iterrows():
        source.append(inx)
        target.append(len(cols)+index+1)
        value.append(row[col+'_in'])

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = cols+status
    ),
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A2, B1, ...
      target = target,
      value = value
  ))])

fig.update_layout(title_text="Entire cohort", font_size=10)
fig.show()