import os
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.regularizers import l2

data = np.loadtxt("data/poker-hand-training-true.data", delimiter=",")
nb_classes = 10

data_distribution = 0.8
train_length = round(data_distribution * len(data))
validation_length = round(len(data) - train_length)
print("%d train and %d validation data" % (train_length, validation_length))

train_data = data[:train_length,:10]
train_label = data[:train_length,10]

validation_data = data[train_length:, :10]
validation_label = data[train_length:, 10]

# Our first Keras Model
model = Sequential()
model.add(Dense(400, input_shape=(10,), activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.01)))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_label, validation_split=0.33, shuffle=True, epochs=100, batch_size=32)

loss, accuracy = model.evaluate(validation_data, validation_label, verbose=0)
print("Test: accuracy=%f loss=%f" % (accuracy, loss))

version_name = str(time.time())
os.makedirs("results/%s" % version_name)
model.save("model/%s.h5" % version_name)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('results/%s/train.jpg' % version_name)

plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('results/%s/test.jpg' % version_name)





