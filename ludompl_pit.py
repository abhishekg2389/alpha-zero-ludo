import Arena
from MCTS import MCTS
from ludo_mpl.LudoMPLGame import LudoMPLGame
from ludo_mpl.LudoMPLPlayers import *
from ludo_mpl.keras_v2.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True
g = LudoMPLGame()

# all players
rp = RandomPlayer(g).play
hp1 = HumanPlayer(g).play
hp2 = HumanPlayer(g).play

# nnet players
#n1 = NNet(g)
#if mini_othello:
#    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
#else:
#    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
#args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

#if human_vs_cpu:
#    player2 = hp
#else:
#    n2 = NNet(g)
#    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
#    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#    mcts2 = MCTS(g, n2, args2)
#    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

#    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(hp1, hp2, g, display=LudoMPLGame.display)

print(arena.playGames(2, verbose=True))
