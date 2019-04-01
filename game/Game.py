import random
import time

import numpy as np
from random import choices

from keras.engine.saving import load_model

from game.Dealers import Dealer
from game.Players import DummyPlayer, NeuralPlayer
from name_generator import NameGenerator

from keras import backend as K

HAND_RECOGNIZER_MODEL_DIR = "model/1553883621.801545.h5"

hand_name = {
    0: 'Nothing in hand',
    1: 'One pair',
    2: 'Two pairs',
    3: 'Three of a kind',
    4: 'Straight',
    5: 'Flush',
    6: 'Full house',
    7: 'Four of a kind',
    8: 'Straight flush',
    9: 'Royal flush',
}


class Game(object):

    num_players = 2
    players = []
    dealer = None
    start_balance = 100
    round_counter = 0
    player_lost = None
    card_count = 300
    games = 10
    start_time = None

    model = None

    def __init__(self):
        print("Simulation started, setting up players and the dealer")

        self.dealer = Dealer(self.card_count)

        name_gen = NameGenerator()
        # name_gen.get_name()
        self.players.append(NeuralPlayer('Lili', self.start_balance))
        # self.players.append(DummyPlayer('Pics', self.start_balance))
        self.players.append(DummyPlayer('Sapi', self.start_balance))
        # for k in range(self.num_players):
        #     if random.randint(0, 100) > 50:
        #         self.players.append(DummyPlayer(name_gen.get_name(), self.start_balance))
        #     else:
        #         self.players.append(NeuralPlayer(name_gen.get_name(), self.start_balance))

        self.model = load_model(HAND_RECOGNIZER_MODEL_DIR)
        self.start_time = time.time()

        print("Player's are ready, lets start the match!")

    def get_winner_and_pot(self, bets):
        pot = 0
        winner = ''
        winner_hand = 0
        for name, info in bets.items():
            pot += info['bet']
            t = np.array([info['cards']])
            hand = np.argmax(self.model.predict(t))
            print("Player %s has %s" % (name, hand_name[hand]))
            if hand > winner_hand:
                winner_hand = hand
                winner = name
            elif hand == winner_hand:
                winner = '' # TODO only works with 2 player

        print("%d is the pot in the end of the round %d" % (pot, self.round_counter))
        print("%s is the winner of the round with %s" % (winner, (hand_name[winner_hand])))
        return winner, pot

    def play(self):
        print("Game started")
        for i in range(self.games):
            while len(self.dealer.cards) > self.num_players and self.player_lost is None:
                self.round_counter += 1
                print("%d round started" % self.round_counter)
                bets = {}
                monies = []
                for p in self.players:
                    monies.append(p.money / (len(self.players) * 100.0))

                c1 = self.dealer.get_cards()
                c2 = self.dealer.get_cards()
                e_hand = np.argmax(self.model.predict(np.array([c1])))
                bets[self.players[1].name] = {'bet': self.players[1].make_bet(c1, 10, monies), 'cards': c1}
                bets[self.players[0].name] = {'bet': self.players[0].make_bet(c2, 10, monies, e_hand), 'cards': c2}
                winner, pot = self.get_winner_and_pot(bets)

                for p in self.players:
                    if p.name == winner:
                        p.update(pot)
                    else:
                        if not p.update(0):
                            self.player_lost = p

            print("Game ended")
            if self.player_lost is not None:
                print("%s is a looser.." % self.player_lost.name)
            self.dealer = Dealer(self.card_count)
            for p in self.players:
                p.money = 100
            self.player_lost = None

    def evaluate(self):
        for p in self.players:
            wins = self.games - p.lose
            avg_bet = np.average(np.array(p.bets))
            avg_hands = np.average(p.hands)
            print("%s: Winrate: %f with %f avg. bets %f hand avrage" % (p.name, wins / float(self.games), avg_bet, avg_hands))
            print("%s runtime " % str(time.time() - self.start_time))



