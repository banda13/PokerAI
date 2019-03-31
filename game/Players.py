import random
import numpy as np

from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.regularizers import l2

HAND_RECOGNIZER_MODEL_DIR = "model/1553883621.801545.h5"

class AbstractPlayer(object):

    def __init__(self, name, start_balance):
        self.name = name
        self.start_balance = start_balance
        self.hands = []
        self.bets = []

        self.round_wins = []
        self.game_lost = []
        self.recognizer_model = load_model(HAND_RECOGNIZER_MODEL_DIR)


class DummyPlayer(object):

    def __init__(self, name, start_balance):
        self.money = start_balance
        self.hands = []
        self.bets = []
        self.name = name
        self.lose = 0

        self.recognizer_model = load_model(HAND_RECOGNIZER_MODEL_DIR)
        print("%s: Hi, im %s, a dummy player and im ready to play" % (name, name))

    def make_bet(self, cards, min_pot, monies):
        t = np.array([cards])
        hand = self.recognizer_model.predict(t)
        self.hands.append(np.argmax(hand))
        bet = random.randint(1, self.money)
        bet = bet if bet > min_pot else self.money
        self.bets.append(bet)
        self.money -= bet
        print("%s: My bet is %s" % (self.name, bet))
        return bet

    def update(self, reward):
        self.money += reward
        print("%s: My balance is %s" % (self.name, self.money))
        if self.money == 0:
            self.lose += 1
            return False
        else:
            return True


class NotToCleverPlayer(object):

    def __init__(self, name, start_balance):
        self.money = start_balance
        self.hands = []
        self.bets = []
        self.name = name
        self.lose = 0

        self.recognizer_model = load_model(HAND_RECOGNIZER_MODEL_DIR)
        print("%s: Hi, im %s, a dummy player and im ready to play" % (name, name))

    def make_bet(self, cards, min_pot, monies):
        t = np.array([cards])
        hand = self.recognizer_model.predict(t)
        self.hands.append(np.argmax(hand))
        bet = random.randint(1, self.money)
        bet = bet if bet > min_pot else self.money
        self.bets.append(bet)
        self.money -= bet
        print("%s: My bet is %s" % (self.name, bet))
        return bet

    def update(self, reward):
        self.money += reward
        print("%s: My balance is %s" % (self.name, self.money))
        if self.money == 0:
            self.lose += 1
            return False
        else:
            return True


class NeuralPlayer(object):

    def __init__(self, name, start_balance):
        self.money = start_balance
        self.hands = []
        self.bets = []
        self.name = name
        self.lose = 0

        self.features = []
        self.rewards = []

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(4,), activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.load_weights('model/%s.h5' % self.name)

        self.recognizer_model = load_model(HAND_RECOGNIZER_MODEL_DIR)
        print("%s: Hi, im %s, a neural network based player and im ready to play" % (name, name))

    def make_bet(self, cards, min_pot, monies, enemy_hands):
        t = np.array([cards])
        hand = self.recognizer_model.predict(t)
        self.hands.append(np.argmax(hand))
        m = np.append(monies, enemy_hands)
        features = np.append(np.argmax(hand), m)
        self.features.append(features)
        ff = np.array([features])
        prediction = self.model.predict(ff)[0][0] * 100
        bet = min(int(prediction), self.money)
        if min_pot <= bet < self.money:
            bet = bet
        elif bet <= min_pot < self.money:
            bet = min_pot
        elif bet <= min_pot and self.money < min_pot:
            bet = self.money
        elif bet >= min_pot and bet > self.money:
            bet = self.money
        else:
            bet = bet
        self.bets.append(bet)
        self.money -= bet
        print("%s: My bet is %s" % (self.name, bet))
        return bet

    def update(self, reward):
        self.money += reward
        print("%s: My balance is %s" % (self.name, self.money))

        self.rewards.append(np.array([0 if reward == 0 else reward / 200.0]))
        # res = self.model.fit(np.array(self.features), np.array(self.rewards), batch_size=32, epochs=10)
        x = np.array([self.features[-1]])
        y = np.array([self.rewards[-1]])
        self.model.train_on_batch(x,y)
        self.model.save_weights('model/%s.h5' % self.name)

        if self.money == 0:
            self.lose += 1
            return False
        else:
            return True

def test_player(name):
    pass