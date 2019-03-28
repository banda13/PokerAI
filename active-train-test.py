import random
from random import choices

from keras.engine.saving import load_model

from name_generator import NameGenerator

MODEL_DIR = "model/1553703294.3642073.h5"


class Game(object):

    num_players = 2
    players = []
    dealer = None
    start_balance = 100
    round_counter = 0
    player_lost = None

    def __init__(self):
        print("Simmulation started, setting up players and the dealer")

        self.dealer = Dealer(300)

        name_gen = NameGenerator()
        for k in range(self.num_players):
            self.players.append(Player(name_gen.get_name(), self.start_balance))

        print("Player's are ready, lets start the match!")

    def get_winner(self, bets):
        pot = 0
        for name, money in bets.items():
            pot += money
        print("%d is the pot in the end of the round %d" % (pot, self.round_counter))
        winner = random.choice(list(bets.keys())) # TODO find out real winner based on the cards!
        print("%s is the winner" % winner)
        return winner, pot

    def play(self):
        while len(self.dealer.cards) > (self.num_players * 5) and self.player_lost is None:
            self.round_counter += 1
            print("%d round started" % self.round_counter)
            bets = {}
            for p in self.players:
                b = p.make_bet(self.dealer.get_cards())
                bets[p.name] = b

            winner, pot = self.get_winner(bets)

            for p in self.players:
                if p.name == winner:
                    p.update(pot)
                else:
                    if not p.update(0):
                        self.player_lost = p

        print("Game ended")
        if self.player_lost is not None:
            print("%s is a looser.." % self.player_lost.name)


class Dealer(object):

    def __init__(self, card_count):
        self.cards = []
        # TODO initializer deck based on real desk distribution
        for c in range(card_count):
            k = []
            for i in range(5):
                k.extend([random.randint(1, 4), random.randint(1, 13)])
            self.cards.append(k)
        random.shuffle(self.cards)
        print("Hi im the dealer, and im ready to play")

    def get_cards(self):
        c = []
        for i in range(5):
            c.append(self.cards.pop())


class Player(object):

    def __init__(self, name, start_balance):
        self.money = start_balance
        self.cards = []
        self.bets = []
        self.model = load_model(MODEL_DIR)
        self.name = name
        print("%s: Hi, im %s, and im ready to play" % (name, name))

    def make_bet(self, cards):
        self.cards.append(cards)
        bet = random.randint(1, self.money) # TODO make clever decisions!
        self.bets.append(bet)
        self.money -= bet
        print("%s: My bet is %s" % (self.name, bet))
        return bet

    def update(self, reward):
        self.money += reward
        print("%s: My balance is %s" % (self.name, self.money))
        if self.money == 0:
            return False
        else:
            return True


g = Game()
g.play()







