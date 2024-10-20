import math
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt


def avg_conditional_entropy(atr_known, df_X, df_Y):
    total_samples = len(df_Y)
    weighted_entropies = []

    for x2 in df_X[atr_known].unique():
        Y_conditioned = df_Y[df_X[atr_known] == x2]
        total_labels = len(Y_conditioned)
        weight = total_labels / total_samples

        entropy = sum(
            [-math.log2(Y_conditioned.value_counts()[y] / total_labels) * Y_conditioned.value_counts()[y] / total_labels
             for y in Y_conditioned.unique()])
        weighted_entropies.append(weight * entropy)

    return sum(weighted_entropies)


def entropy(df):
    total_labels = len(df)
    return sum(
        [-math.log2(df.value_counts()[y] / total_labels) * df.value_counts()[y] / total_labels for y in df.unique()])


X = pd.DataFrame({'X1': [1, 1, 1, 1, 0, 0],
                  'X2': [1, 1, 1, 0, 0, 0]})
Y = pd.Series([1, 1, 2, 3, 2, 3])

print(f'H(Y|X1): {avg_conditional_entropy("X1", X, Y)}')
print(f'H(Y|X2): {avg_conditional_entropy("X2", X, Y)}')
print(f'H(Y|X2) < H(Y|X1) => Attribute X2 wil be used for the first node of the ID3 tree')

dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X, Y)
new_instance = pd.DataFrame({'X1': [0], 'X2': [1]})
print(f'Prediction for X1:0, X2:1 is {dt.predict(new_instance)}')

fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=X.columns)
plt.show()