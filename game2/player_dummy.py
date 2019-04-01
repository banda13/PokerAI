import random
import time
import numpy as np

from game2.player_abstract import AbstractPlayer


class DummyPlayer(AbstractPlayer):

    def __init__(self, name, start_balance):
        super().__init__(name, start_balance)
        print("%s: Hi, im %s, a dummy player and im ready to play" % (name, name))

    def make_bet(self, cards, hands, min_pot, enemy_money, enemy_rise):
        start = time.time()
        self.hands.append(hands)

        try:
            bet = random.randint(min_pot, self.balance)
        except ValueError:
            bet = min_pot # TODO fixme it should not happened
        self.bets.append(bet)
        self.balance -= bet
        print("%s: My bet is %s" % (self.name, bet))
        self.bet_times.append(time.time() - start)
        return bet

    def update(self, reward):
        start = time.time()
        self.balance += reward
        print("%s: My balance is %s" % (self.name, self.balance))
        self.rewards.append(reward)
        self.round_played += 1
        if reward > 0:
            self.round_wins += 1
        self.balance_history.append(self.balance)
        self.update_times.append(time.time() - start)

    def check_status(self, min_balance):
        return self.balance > (min_balance + 1)

    def end_game(self, winner):
        if winner:
            self.game_wins += 1
        self.game_played += 1

    def __str__(self):
        return self.name
