from sklearn.naive_bayes import BernoulliNB
import pandas as pd
d = pd.DataFrame({'A': [0, 0, 1, 0, 1, 1, 1],
                  'B': [0, 1, 1, 0, 1, 0, 1],
                  'C': [1, 0, 0, 1, 1, 0, 0],
                  'Y': [0, 0, 0, 1, 1, 1, 1]})
X = d[['A', 'B', 'C']]
Y = d['Y']
X_predict = pd.DataFrame({'A': [0], 'B': [0], 'C': [1]})

cl = BernoulliNB(alpha=1e-10).fit(X, Y)
print(cl.predict(X_predict))
print(cl.predict_proba(X_predict))
print(cl.classes_)

print("""
For Y=0:
P(Y=0) * P(A=0|Y=0) * P(B=0|Y=0) * P(C=1|Y=0)
= (3/7) * (2/3) * (1/3) * (1/3) = 0.03174
For Y=1:
P(Y=1) * P(A=0|Y=1) * P(B=0|Y=1) * P(C=1|Y=1)
= (4/7) * (1/4) * (2/4) * (2/4) = 0.03571

Normalize these probabilities:
Total = 0.0952 + 0.0714 = 0.06745

P(Y=0|A=0,B=0,C=1) = 0.0952/0.1666 = 0.4705
P(Y=1|A=0,B=0,C=1) = 0.0714/0.1666 = 0.5293
""")
