import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import pandas as pd

def correlated_df(corr):
    size = 10000
    w1 = bernoulli.rvs(0.5, size=size, random_state=1)
    d = pd.DataFrame({'w1': w1})
    mask = bernoulli.rvs(corr, size=size, random_state=2)
    random = bernoulli.rvs(0.5, size=size, random_state=3)
    d['w2'] = d['w1'] & mask | random & ~mask
    d['mask'] = mask
    d['random'] = random
    d['y'] = d['w1'] & ~d['w2']
    return d

# Exercise 1: Naive Bayes performance vs correlation
corr_values = np.linspace(0, 1, 30)
nb_train_errors = []

for corr in corr_values:
    df = correlated_df(corr)
    model = BernoulliNB()
    model.fit(df[['w1', 'w2']], df['y'])
    y_pred = model.predict(df[['w1', 'w2']])
    nb_train_errors.append(1 - accuracy_score(df['y'], y_pred))

plt.figure(figsize=(8, 6))
plt.plot(corr_values, nb_train_errors)
plt.xlabel('Correlation')
plt.ylabel('Training Error')
plt.title('Naive Bayes Performance vs Correlation')
plt.show()

# Exercise 2: Decision Tree performance vs correlation
dt_train_errors = []

for corr in corr_values:
    df = correlated_df(corr)
    model = DecisionTreeClassifier()
    model.fit(df[['w1', 'w2']], df['y'])
    y_pred = model.predict(df[['w1', 'w2']])
    dt_train_errors.append(1 - accuracy_score(df['y'], y_pred))

plt.figure(figsize=(8, 6))
plt.plot(corr_values, dt_train_errors)
plt.xlabel('Correlation')
plt.ylabel('Training Error')
plt.title('Decision Tree Performance vs Correlation')
plt.show()