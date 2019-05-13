from keras.engine.saving import load_model

from game2.game import Game
from game2.player_dummy import DummyPlayer
from game2.player_intelligent import IntelligentPlayer
from game2.player_manual import ManualPlayer
from game2.player_not_dummy import NotDummyPlayer

p = []
# p.append(DummyPlayer('Gyozo', 500))
# p.append(ManualPlayer(100))
p.append(IntelligentPlayer('Lili3', 500))
p.append(NotDummyPlayer('Kapcsa2', 500))
# p.append(NotDummyPlayer('Herold', 100))
# p.append(DummyPlayer('Sara', 100))
p.append(IntelligentPlayer('Andy3', 500))

model = load_model("model/1556041857.1677544.h5")

game = None
for i in range(100):
    print("GAME %d starting " % i)
    for x in p:
        x.load_balance(100)
    game = Game(p, model=model, cards_in_deck=104)
    game.play()
    game.evaluate()
    p = game.get_players()
game.final_evaluate()

