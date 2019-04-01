import random
import time
import numpy as np

from keras.engine.saving import load_model

from game.Dealers import Dealer
from game2.game_utils import get_player_balances

from keras import backend as K

HAND_RECOGNIZER_MODEL_DIR = "model/1553883621.801545.h5"


class Game(object):

    def __init__(self, players, model = None, cards_in_deck=52):
        print("Game simulation started, setting up players and the dealer")

        self.dealer = Dealer(cards_in_deck)
        self.active_players = players
        self.inactive_players = []
        self.players = players
        self.players_count = len(players)
        self.num_players = len(players)

        self.model = model if model is not None else load_model(HAND_RECOGNIZER_MODEL_DIR)
        self.start_time = time.time()

        self.round_counter = 0

        self.bet_increase_rate = 3  # each 3 turn min bot is growing

        print("Player's are ready, lets start the match!")

    def play(self):
        print("Game started")
        game_in_progress = True

        while len(self.dealer.cards) > self.num_players and game_in_progress:
            self.round_counter += 1
            print("%d round started, Please take your bets!" % self.round_counter)

            # reorder players
            self.active_players.insert(0, self.active_players.pop())
            balances = get_player_balances(self)
            min_bet = round(self.round_counter / 5)
            cards = []
            bets = []
            hands = []

            # Players making bets
            for p in self.active_players:
                card = self.dealer.get_cards()
                # for optimization reasons game makes hands recognition instead of players
                # in this case game speed will not strongly depend on hand recognizer model complexity
                hand = np.argmax(self.model.predict(np.array([card])))
                bet = p.make_bet(card, hand, min_bet, balances, bets)
                bets.append(bet)
                cards.append(cards)
                hands.append(hand)

            # Calculating winners
            players_idx_order = np.argsort(hands)
            winner_hand = hands[players_idx_order[-1]]
            winners = []
            for player_idx in players_idx_order:
                if hands[player_idx] >= winner_hand:
                    winners.append(self.active_players[player_idx])

            # Players updating they balances and may learn something
            pot = round(np.sum(bets) / len(winners))
            print("%s are the winner(s) %d is the pot" % (str(winners), pot))
            for p in self.active_players:
                if p in winners:
                    p.update(pot)
                else:
                    p.update(0)

                if not p.check_status(round((self.round_counter+1) / 5)):
                    self.active_players.remove(p)
                    self.inactive_players.append(p)

            # Checking if game ended
            if len(self.active_players) == 1:
                game_in_progress = False

        for p in self.active_players:
            p.end_game(True)
        for p in self.inactive_players:
            p.end_game(False)

    def evaluate(self):
        for p in (self.inactive_players + self.active_players):
            print("\n\n%s player: \n"
                  "%.2f average hands\n"
                  "%.2f biggest hand\n"
                  "%.2f average bets\n"
                  "%.2f average reward\n"
                  "%.2f%% win rate\n"
                  "%.2f%% round_win rate\n"
                  "%.2f sec average bet time\n"
                  "%.2f sec average update time" % p.evaluate_player())

    def __del__(self):
        pass
        # del self.model
        # K.clear_session()

    def get_players(self):
        return self.inactive_players + self.active_players
