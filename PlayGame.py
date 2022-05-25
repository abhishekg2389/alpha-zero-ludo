import numpy as np
import sys
import os

from ludo_mpl.LudoMPLGame import LudoMPLGame
from ludo_mpl.LudoMPLPlayers import AggressivePlayer as LudoMPLAP
from ludo_mpl.LudoMPLPlayers import RandomPlayer as LudoMPLRP
from ludo_mpl.keras_v2.NNet import NNetWrapper as LudoMPLNNP
from utils import *
from MCTS import MCTS

def _getPlayer(p, g, tempFolder):
    if p == 'rp':
        return LudoMPLRP(g).play
    if p == 'ap':
        return LudoMPLAP(g).play
    if p == 'nnp':
        n1 = LudoMPLNNP(g)
        n1.load_checkpoint(tempFolder,'best.pth.tar')
        args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    if p == 'p_nnp':
        n1 = LudoMPLNNP(g)
        n1.load_checkpoint(tempFolder, 'temp.prev.pth.tar')
        args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    if p == 'n_nnp':
        n1 = LudoMPLNNP(g)
        n1.load_checkpoint(tempFolder, 'temp.new.pth.tar')
        args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

def playGame(p1, p2, tempFolder):
    g = LudoMPLGame()

    p1 = _getPlayer(p1, g, tempFolder)
    p2 = _getPlayer(p2, g, tempFolder)

    players = [p2, None, p1]
    curPlayer = 1
    board = g.getInitBoard()
    while g.getGameEnded(board, curPlayer) == 0:
        action = players[curPlayer + 1](g.getCanonicalForm(board, curPlayer))
        board, curPlayer = g.getNextState(board, curPlayer, action)
    return curPlayer * g.getGameEnded(board, curPlayer)

def main():
    tempFolder = sys.argv[1]
    p1 = sys.argv[2]
    p2 = sys.argv[3]
    game_idx = int(sys.argv[4])

    game_result = playGame(p1, p2, tempFolder)
    filename = os.path.join(tempFolder, "game_" + str(game_idx) + "_")
    if game_result == 1:
        filename += str(1)
    elif game_result == -1:
        filename += str(-1)
    else:
        filename += str(0)
    with open(filename, "wb+") as f:
        f.write('x')

if __name__ == '__main__':
    main()