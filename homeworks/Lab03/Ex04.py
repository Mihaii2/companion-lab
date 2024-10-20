import math
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

from tools.pd_helpers import apply_counts
exoplanets = pd.DataFrame([
  ('Big', 'Near', 'Yes', 20),
  ('Big', 'Far', 'Yes', 170),
  ('Small', 'Near', 'Yes', 139),
  ('Small', 'Far', 'Yes', 45),
  ('Big', 'Near', 'No', 130),
  ('Big', 'Far', 'No', 30),
  ('Small', 'Near', 'No', 11),
  ('Small', 'Far', 'No', 255)
],
columns=['Big', 'Orbit', 'Habitable', 'Count'])
exoplanets = apply_counts(exoplanets, 'Count')

exoplanets['Orbit'] = exoplanets['Orbit'].map({'Near': 0, 'Far': 1})
exoplanets['Big'] = exoplanets['Big'].map({'Small':0, 'Big':1})
exoplanets['Habitable'] = exoplanets['Habitable'].map({'No':0, 'Yes':1})
exoplanets_X = exoplanets.take([0,1], axis=1)
exoplanets_Y = exoplanets.take([2], axis=1)

dt = tree.DecisionTreeClassifier(criterion='entropy').fit(exoplanets_X, exoplanets_Y)
print(f'Training accuracy: {dt.score(exoplanets_X, exoplanets_Y)}')

fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=exoplanets_X.columns)
plt.show()