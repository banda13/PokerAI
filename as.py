import os
import random
import numpy as np
import pydotplus
import pandas as pd
from IPython.core.display import Image
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import tree, export_graphviz

poker_hands = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
feature_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
hand_name = ['Nothing in hand',
    'One pair',
     'Two pairs',
     'Three of a kind',
     'Straight',
     'Flush',
     'Full house',
     'Four of a kind',
     'Straight flush',
     'Royal flush']

data = np.loadtxt("data/poker_hand_extended_train.data", delimiter=",")

data_distribution = 0.8
train_length = round(data_distribution * len(data))
validation_length = round(len(data) - train_length)

train = data[:train_length,:]
cls = {}
for i in range(10):
    cls[i] = len([i for x in train[:,10] if x == i])
print(cls)
train_data = train[:,:10]
train_label = train[:,10]

validation_data = data[train_length:, :10]
validation_label = data[train_length:, 10]


print('Training decision tree')
dtree = RandomForestClassifier(n_estimators=3)
dtree = dtree.fit(train_data, train_label)

print("Validating decision tree")
scores = cross_val_score(dtree, validation_data, validation_label)
print(scores.mean())

print("Testing decision tree")
predictions = dtree.predict(train_data)
print(classification_report(train_label, predictions))
print("Accuracy:" , accuracy_score(train_label, predictions))

dot_data = StringIO()
export_graphviz(dtree.estimators_[0], out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_names, class_names=poker_hands)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('v4.png')
Image(graph.create_png())