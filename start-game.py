from keras.engine.saving import load_model

from game2.game import Game
from game2.player_dummy import DummyPlayer
from game2.player_intelligent import IntelligentPlayer
from game2.player_manual import ManualPlayer
from game2.player_not_dummy import NotDummyPlayer

p = []
# p.append(DummyPlayer('Gyozo', 100))
p.append(ManualPlayer(100))
p.append(IntelligentPlayer('Lili', 100))
# p.append(NotDummyPlayer('Kapcsa', 100))
# p.append(NotDummyPlayer('Herold', 100))
# p.append(DummyPlayer('Sara', 100))
# p.append(IntelligentPlayer('Andy', 100))

model = load_model("model/1553883621.801545.h5")

for i in range(10):
    print("GAME %d starting " % i)
    for x in p:
        x.load_balance(100)
    game = Game(p, model=model, cards_in_deck=104)
    game.play()
    game.evaluate()
    p = game.get_players()
