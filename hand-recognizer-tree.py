import numpy as np
import pydotplus
from IPython.core.display import Image
from sklearn import tree, svm
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.utils import class_weight

poker_hands = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
feature_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

data = np.loadtxt("data/poker-hand-training-true.data", delimiter=",")
np.random.shuffle(data)

data_distribution = 0.8
train_length = round(data_distribution * len(data))
validation_length = round(len(data) - train_length)
print("%d train and %d validation data" % (train_length, validation_length))

train_data = data[:train_length,:10]
train_label = data[:train_length,10]

validation_data = data[train_length:, :10]
validation_label = data[train_length:, 10]


print('Training decision tree')
dtree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7, class_weight="balanced")
dtree.fit(train_data, train_label)

print("Validating decision tree")
scores = cross_val_score(dtree, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing decision tree")
predictions = dtree.predict(validation_data)
# print(classification_report(validation_label, predictions))
print("Accuracy:" , accuracy_score(validation_label, predictions))

print("-------------------")

print("Training random forrest")
rforrest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
rforrest.fit(train_data, train_label)

print("Validating random forrest")
scores = cross_val_score(rforrest, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing random forrest")
predictions = rforrest.predict(validation_data)
print("Accuracy:" , accuracy_score(validation_label, predictions))

print("-------------------")

print("Training extra trees")
etree = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
etree.fit(train_data, train_label)

print("Validating extra trees")
scores = cross_val_score(etree, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing extra trees")
predictions = etree.predict(validation_data)
print("Accuracy:" , accuracy_score(validation_label, predictions))

print("-------------------")

print("Training Ada boost")
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(train_data, train_label)

print("Validating  Ada boost")
scores = cross_val_score(ada, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing Ada boost")
predictions = ada.predict(validation_data)
print("Accuracy:" , accuracy_score(validation_label, predictions))

print("-------------------")

print("Training tree with boost")
aboost = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)
aboost.fit(train_data, train_label)

print("Validating  boosted tree")
scores = cross_val_score(aboost, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing boosted tree")
predictions = aboost.predict(validation_data)
print("Accuracy:" , accuracy_score(validation_label, predictions))

# print("Training gradient boost")
# gboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)#.fit(tra, y_train)
# scores = cross_val_score(gboost, train_data, train_label, cv=5)
# print(scores.mean())

print("-------------------")

print("Training SVM")
linsvm = svm.SVC()
linsvm.fit(train_data, train_label)

print("Validating SVM")
scores = cross_val_score(linsvm, validation_data, validation_label, cv=5)
print(scores.mean())

print("Testing SVM")
predictions = linsvm.predict(validation_data)
print("Accuracy:" , accuracy_score(validation_label, predictions))



# match = 0
# for data, label in zip(validation_data, validation_label):
#     if dtree.predict([data]) == label:
#         match += 1
# print("%d match from %d. %d missed. %f accuracy" % (match, train_length, train_length-match, match / train_length))



#dot_data = StringIO()
#export_graphviz(dtree, out_file=dot_data,
#                filled=True, rounded=True,
#                special_characters=True,feature_names = feature_names, class_names=poker_hands)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('v2.png')
#Image(graph.create_png())