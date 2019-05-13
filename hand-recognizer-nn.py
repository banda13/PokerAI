import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout

print("Reading data sources")
train_data_source = np.loadtxt("data/poker-hand-training-true.data", delimiter=",")
test_data_source = np.loadtxt("data/poker-hand-testing.data", delimiter=",")
nb_classes = 10

random.shuffle(train_data_source)
random.shuffle(test_data_source)

data_distribution = 0.8
train_length = round(data_distribution * len(train_data_source))
validation_length = round(len(train_data_source) - train_length)
print("%d train and %d validation data" % (train_length, validation_length))

train_data = train_data_source[:train_length, :10]
train_label = train_data_source[:train_length, 10]

validation_data = train_data_source[train_length:, :10]
validation_label = train_data_source[train_length:, 10]

test_data = test_data_source[:, :10]
test_labels = test_data_source[:, 10]

model = Sequential()
model.add(Dense(1000, input_shape=(10,), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1000, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model created, training started")

callbacks = [TensorBoard(log_dir="logs/{}".format("temp1"))]

history = model.fit(train_data, train_label,
                    callbacks=callbacks,
                    batch_size=32,
                    epochs=200,
                    validation_data=(validation_data, validation_label),
                    shuffle=True)

loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print("Test: accuracy=%f loss=%f" % (accuracy, loss))

version_name = str(time.time())
os.makedirs("results/%s" % version_name)
model.save("model/%s.h5" % version_name)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('results/%s/train.jpg' % version_name)

plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('results/%s/test.jpg' % version_name)
