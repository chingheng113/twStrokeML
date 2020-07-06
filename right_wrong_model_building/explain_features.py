import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('reinvers_explain.csv')
col = list(data.columns.values)
features = {}
fe = {}
for s in col:
    c = re.sub(r"[<>=0-9.\s+]", "", s)
    if c in features:
        features[c] = data[s].dropna().shape[0]+features[c]
    else:
        features[c] = data[s].dropna().shape[0]
        fe[s] = data[s].dropna().shape[0]
f_df = pd.DataFrame.from_dict(features.items())
f_df.to_csv('a.csv', index=False)
# sort by value count
fe = {k: v for k, v in sorted(fe.items(), key=lambda item: item[1], reverse=True)}
data = data[list(fe.keys())]
plt.figure(figsize=(10, 12))
ax = sns.heatmap(data.T, yticklabels=True)
plt.show()

print('done')