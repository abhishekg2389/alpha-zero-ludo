""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras, Tensorflow.

     [ Games ]      Pytorch     Tensorflow  Keras
      -----------   -------     ----------  -----
    - Othello       [Yes]       [Yes]       [Yes]
    - TicTacToe                             [Yes]
    - Connect4                  [Yes]
    - Gobang                    [Yes]       [Yes]
    - Santorini                 [Yes]

"""

import unittest

import Arena
from MCTS import MCTS

from ludo_mpl.LudoMPLGame import LudoMPLGame
from ludo_mpl.LudoMPLPlayers import *
from ludo_mpl.keras_v2.NNet import NNetWrapper as LudoMPLKerasNNet

import numpy as np
from utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game, game.display)
        print(arena.playGames(2, verbose=True))

    def test_ludompl_keras(self):
        self.execute_game_test(LudoMPLGame(), LudoMPLKerasNNet)

    '''
    def test_othello_pytorch(self):
        self.execute_game_test(OthelloGame(6), OthelloPytorchNNet)

    def test_othello_tensorflow(self):
        self.execute_game_test(OthelloGame(6), OthelloTensorflowNNet)

    def test_othello_keras(self):
        self.execute_game_test(OthelloGame(6), OthelloKerasNNet)

    def test_tictactoe_keras(self):
        self.execute_game_test(TicTacToeGame(), TicTacToeKerasNNet)

    def test_connect4_tensorflow(self):
        self.execute_game_test(Connect4Game(), Connect4TensorflowNNet)

    def test_gobang_keras(self):
        self.execute_game_test(GobangGame(), GobangKerasNNet)

    def test_gobang_tensorflow(self):
        self.execute_game_test(GobangGame(), GobangTensorflowNNet)

    def test_santorini_tensorflow(self):
        self.execute_game_test(SantoriniGame(5), SantoriniTensorflowNNet)
    '''
if __name__ == '__main__':
    unittest.main()