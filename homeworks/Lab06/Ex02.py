import pandas as pd
from scipy.stats import bernoulli
from sklearn.naive_bayes import BernoulliNB
import numpy as np

n_samples = 1000

y = bernoulli.rvs(0.5, size=n_samples)
x1 = [bernoulli.rvs(0.7) if label == 0 else bernoulli.rvs(0.2) for label in y]
x2 = [bernoulli.rvs(0.9) if label == 0 else bernoulli.rvs(0.5) for label in y]

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

X = df[['x1', 'x2']]
y = df['y']

cl = BernoulliNB(alpha=1e-10).fit(X, y)

predictions = cl.predict(X)

print(f'Accuracy: {np.array(predictions == y).mean()}')

df['x3'] = df['x2']

X = df[['x1', 'x2', 'x3']]
y = df['y']

cl = BernoulliNB(alpha=1e-10).fit(X, y)

predictions = cl.predict(X)

print(f'New Accuracy: {np.array(predictions == y).mean()}')
