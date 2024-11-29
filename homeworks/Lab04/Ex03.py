import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from tools.plots import plot_decision_surface

d = pd.DataFrame({'X1': [1, 2, 3, 3, 3, 4, 5, 5, 5],
                  'X2': [2, 3, 1, 2, 4, 4, 1, 2, 4],
                  'Y':  [1, 1, 0, 0, 0, 0, 1, 1, 0]})

# 1
plt.scatter(d['X1'], d['X2'], c=d['Y'])
plt.show()

# 2
X = d.loc[:, ['X1', 'X2']]
Y = d.loc[:,'Y']
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X, Y)

plot_decision_surface(dt, X, Y)

# 3
X1 = np.random.random_sample(1000) * 10
X2 = np.random.random_sample(1000) * 10

Y = X1 > X2

new_data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

plt.scatter(new_data['X1'], new_data['X2'], c=new_data['Y'])
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# 4
X = new_data[['X1', 'X2']]
Y = new_data['Y']
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X, Y)

plot_decision_surface(dt, X, Y)