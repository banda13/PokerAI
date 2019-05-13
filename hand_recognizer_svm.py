import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

feature_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
hand_name = ['Nothing in hand','One pair','Two pairs','Three of a kind','Straight','Flush','Full house','Four of a kind','Straight flush','Royal flush']

data = np.loadtxt("data/poker_hand_extended_train.data", delimiter=",")

data_distribution = 0.8
train_length = round(data_distribution * len(data))
validation_length = round(len(data) - train_length)

print("%d train and %d validation data" % (train_length, validation_length))

train_data = data[:train_length,:10]
train_label = data[:train_length,10]

validation_data = data[train_length:, :10]
validation_label = data[train_length:, 10]

print("Training SVM")
linsvm = svm.SVC()
linsvm.fit(train_data, train_label)

print("Validating SVM")
scores = cross_val_score(linsvm, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing SVM")
predictions = linsvm.predict(validation_data)
print("Accuracy:" , accuracy_score(validation_label, predictions))