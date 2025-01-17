import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/poker_hand_extended_train.data", delimiter=",") # poker-hand-training-true.data
# Some statistics to get acquainted with the data

nb_classes = 10  # we have 10 classes of poker hands
cls = {}
cls2 = []
for i in range(nb_classes):
    cls[i] = len([i for x in data[:,10] if x == i])
    cls2.append(cls[i])

poker_hands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

for i in poker_hands:
    print("%s: %d" % (hand_name[i], cls[i]))


plt.bar(poker_hands, cls2)
plt.title("Poker hands distribution")
plt.savefig("distribution_new.png")