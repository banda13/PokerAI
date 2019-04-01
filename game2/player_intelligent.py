import random
import time
import numpy as np
from keras import Sequential
from keras.layers import Dense

from game2.player_abstract import AbstractPlayer


class IntelligentPlayer(AbstractPlayer):

    def __init__(self, name, start_balance):
        super().__init__(name, start_balance)

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(8,), activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        try:
            self.model.load_weights('model/%s.h5' % self.name)
        except OSError:
            pass  # its ok, then we start from scratch

        print("%s: Hi, im %s, a clever player and im ready to play" % (name, name))

    def make_bet(self, cards, hands, min_pot, enemy_balances, enemy_rises):
        start = time.time()
        self.hands.append(hands)

        avg_balance, max_balance, min_balance = np.average(enemy_balances), np.max(enemy_balances), np.min(enemy_balances)
        if len(enemy_rises) > 0:
            avg_rise, max_rise, min_rise = np.average(enemy_rises), np.max(enemy_rises), np.min(enemy_rises)
        else:
            avg_rise, max_rise, min_rise = 0, 0, 0
        features = np.array([hands, min_pot, avg_balance, max_balance, min_balance, avg_rise, max_rise, min_rise])
        self.features.append(features)

        bet = round(self.model.predict(np.array([features]))[0][0] * 100)
        bet = self.validate_bet(bet, min_pot)
        self.bets.append(bet)
        self.balance -= bet
        print("%s: My bet is %s" % (self.name, bet))
        self.bet_times.append(time.time() - start)
        return bet

    def update(self, reward):
        start = time.time()
        self.balance += reward
        print("%s: My balance is %s" % (self.name, self.balance))
        self.rewards.append(np.array([0 if reward == 0 else reward / 300]))

        self.model.train_on_batch(np.array([self.features[-1]]), np.array([self.rewards[-1]]))
        self.model.save_weights('model/%s.h5' % self.name)

        self.round_played += 1
        if reward > 0:
            self.round_wins += 1
        self.balance_history.append(self.balance)
        self.update_times.append(time.time() - start)

    def check_status(self, min_balance):
        return self.balance > min_balance + 1

    def end_game(self, winner):
        if winner:
            self.game_wins += 1
        self.game_played += 1

    def __str__(self):
        return self.name