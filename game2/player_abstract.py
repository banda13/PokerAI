import numpy as np
from abc import ABC # require python 3.4+
from keras.engine.saving import load_model

HAND_RECOGNIZER_MODEL_DIR = "model/1556041857.1677544.h5"

'''
Abstract Base class for all player
'''


class AbstractPlayer(ABC):

    def __init__(self, name, start_balance):
        self.name = name
        self.balance = start_balance
        self.hands = []
        self.bets = []
        self.rewards = []
        self.features = []

        self.round_wins = 0
        self.round_played = 0

        self.game_wins = 0
        self.game_played = 0

        self.bet_times = []
        self.update_times = []

        self.balance_history = []
        self.loss_history = []
        self.accuracy_history = []

        self.recognizer_model = load_model(HAND_RECOGNIZER_MODEL_DIR)

    def validate_bet(self, bet, min_bet):
        if min_bet <= bet < self.balance:
             return bet
        elif bet <= min_bet < self.balance:
            return min_bet
        elif bet <= min_bet and self.balance < min_bet:
            return self.balance
        elif bet >= min_bet and bet > self.balance:
            return self.balance
        else:
            return bet

    def evaluate_player(self):
        avg_hand = np.average(self.hands)
        biggest_hand = np.max(self.hands)

        avg_bets = np.average(self.bets)
        avg_rewards = np.average(self.rewards)

        win_rate = self.game_wins / float(self.game_played)
        round_win_rate = self.round_wins / float(self.round_played)

        avg_bet_time = np.average(self.bet_times)
        avg_update_time = np.average(self.update_times)

        return self.name, avg_hand, biggest_hand, avg_bets, avg_rewards, win_rate, round_win_rate, avg_bet_time, avg_update_time

    def load_balance(self, balance):
        self.balance = balance

    def get_training_history(self):
        return self.loss_history, self.accuracy_history

    def __str__(self):
        return self.name