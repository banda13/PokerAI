from keras import optimizers, Model, Input
from keras.layers import Dense, LSTM
from keras.models import  load_model
import numpy as np
import json


class NameGenerator(object):

    soruce_file =  "data/animals.txt"
    model_directory = "model/name_generator.h5"

    def __init__(self):
        self.m = 6

        self.seq = open(NameGenerator.soruce_file, 'r').read()

        chars = sorted(set(self.seq))
        self.d = len(chars)

        self.enc = dict(zip(chars, range(self.d)))
        self.dec = dict(zip(range(self.d), chars))

        self.step = 3
        self.epoch = 50

        self.n = (len(self.seq) - (self.m - 1)) // self.step - 1

    def train(self):

        chars_in = np.zeros((self.n, self.m, self.d))
        chars_out = np.zeros((self.n, self.d))

        for i in range(self.n):
            for j in range(self.m):
                chars_in[i, j, self.enc[self.seq[i * self.step + j]]] = 1
            chars_out[i, self.enc[self.seq[i * self.step + self.m]]] = 1

        inp = Input(shape=(self.m, self.d))
        net = LSTM(64, return_sequences=True)(inp)
        net = LSTM(64)(inp)
        net = Dense(self.d, activation='softmax')(net)

        model = Model(inp, net)
        model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
        model.fit(chars_in, chars_out, epochs=self.epoch)
        model.save(NameGenerator.model_directory)
        print("Saved model to disk")


    def get_name(self):
        print("Getting random name..")
        model = load_model(NameGenerator.model_directory)
        model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
        print("Loaded model from disk")

        pred = np.zeros((1, self.m, self.d))
        pred[0, self.m - 1, self.enc['\n']] = 1
        out = ""

        for i in range(200):
            pr = np.random.choice(range(self.d), p=model.predict(pred)[0])
            pred = np.roll(pred, -1, 1)
            pred[0, -1:] = np.zeros(self.d)
            pred[0, -1, pr] = 1
            if self.dec[pr] == '\n':
                break
            out += self.dec[pr]
        print("Name generated: %s" % out)
        return out

# n = NameGenerator()
# n.train()
# n.get_name()