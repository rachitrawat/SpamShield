# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:57:17 2017

@author: RudradeepGuha
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

data = pd.read_csv(dataset_dir+dataset_dir_questions, encoding='latin1').iloc[::100, :]

X = np.zeros((12643, 2), dtype=int)
Y = np.zeros((12643, 1), dtype=int)
t = data.Title
counter = 0

# For all titles, we count the number of characters and add that to X and depending on the length
# classify them as 0(less likely to be upvoted) or 1(more likely to be upvoted) 
for i in t:
    f1 = len(i) - i.count(" ")
    f2 = data.loc[data['Title'] == i, 'OwnerUserId'].iloc[0]
    X[counter] = np.array([f1, f2])
    score = data.loc[data['Title'] == i, 'Score'].iloc[0]
    if score < 20:
        Y[counter] = 0
    else:
        Y[counter] = 1

print(X)
print(Y)

model = GaussianNB()

model.fit(X, Y)

print(model.predict_proba(np.array([180, 345768])))