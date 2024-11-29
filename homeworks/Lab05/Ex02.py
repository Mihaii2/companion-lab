import pandas as pd
from tools.pd_helpers import apply_counts
from sklearn.naive_bayes import BernoulliNB

d_grouped = pd.DataFrame({
    'X1': [0, 0, 1, 1, 0, 0, 1, 1],
    'X2': [0, 0, 0, 0, 1, 1, 1, 1],
    'C' : [2, 18, 4, 1, 4, 1, 2, 18],
    'Y' : [0, 1, 0, 1, 0, 1, 0, 1]})
d = apply_counts(d_grouped, 'C')

X = d[['X1', 'X2']]
Y = d['Y']
X_predict = pd.DataFrame({'X1': [0], 'X2': [0]})

cl = BernoulliNB().fit(X, Y)
# 2
print("\nPoint 1 & 2:")
print(cl.predict(X_predict))
print(cl.predict_proba(X_predict))
print(cl.classes_)
# 3
print("\nPoint 3:")
print(f'class_log_prior_: {cl.class_log_prior_}')
print(f'feature_log_prob_: {cl.feature_log_prob_}')
# 4
print("\nPoint 4:")
print("We need P(Xᵢ=1|Y=0), P(Xᵢ=1|Y=1) for each feature Xi.\nSo the number of probabilities is 2n for n features(in cl.feature_log_prob_). Total: 2n + 2 probabilities(with the class prior probabilities)")