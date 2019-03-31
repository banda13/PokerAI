import random


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
        return self.cards.pop()
