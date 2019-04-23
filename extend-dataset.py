import math
import operator
import random
import numpy as np


# get all data in a specified category
def get_all_data_from_class(hand, m_data):
    data_in_class = []
    for x in m_data:
        if x[10] == hand:
            data_in_class.append(x)
    return data_in_class


# shuffle the order of the cards, but keep the pairs together
def shuffle_pairs(generated):
    shuffled_generated = []
    for g in generated:
        g = g.tolist()
        hand = g.pop(-1)
        shifted = g.copy()
        shifted.append(shifted.pop(0))  # O(?)
        pairs = []
        for x, y in list(zip(g, shifted))[::2]:  # speed? seems a bit slow..
            pairs.append([x, y])
        random.shuffle(pairs)
        res = [item for sublist in pairs for item in sublist]
        res.append(hand)
        shuffled_generated.append(res)
    return shuffled_generated


# generates more valid poker hands (if needed)
def generate_data(count, values):
    if count < len(values):
        return random.sample(values, count)
    else:
        generated = []
        diff_scale = int(math.ceil(count / len(values)))
        for j in range(diff_scale):
            generated.extend(random.sample(values, int(count / diff_scale)))
        return shuffle_pairs(generated)


def extend_data_in_category_if_needed(data, most_data_count):
    diff = most_data_count - value
    if diff > 0:
        origin_data = get_all_data_from_class(key, data)
        print("Generating %d data from hand %d" % (diff, key))
        generated_data = generate_data(diff, origin_data)
        return origin_data + generated_data
    else:
        return get_all_data_from_class(key, data)


print("Loading data")
data = np.loadtxt("data/poker-hand-training-true.data", delimiter=",", dtype=int)

nb_classes = 10
cls = {}
for i in range(nb_classes):
    cls[i] = len([i for x in data[:,10] if x == i])
print(cls)

most_data_count = max(cls.items(), key=operator.itemgetter(1))[1]
print("Biggest sample count %d " % most_data_count)
extended_data = []

print("Extending data by generating new data if needed")
for key, value in cls.items():
    extended_data.extend(extend_data_in_category_if_needed(data, most_data_count))

random.shuffle(extended_data)

print("Writing new data-set")
with open("data/poker_hand_extended_train.data", "w") as new_data_file:
    for d in extended_data:
        new_data_file.write(','.join(map(str, d)) + "\n")

print("Checking results")
data = np.loadtxt("data/poker_hand_extended_train.data", delimiter=",")


nb_classes = 10
cls = {}
for i in range(nb_classes):
    cls[i] = len([i for x in data[:,10] if x == i])
print(cls)





