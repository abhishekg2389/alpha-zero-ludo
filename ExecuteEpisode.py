from ludo_mpl.keras_v2.NNet import NNetWrapper as LudoMPLNNP
from ludo_mpl.LudoMPLGame import LudoMPLGame
from MCTS import MCTS

import os
from pickle import Pickler
import numpy as np
import sys

from utils import *

def executeEpisode(args):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                       pi is the MCTS informed policy vector, v is +1 if
                       the player eventually won the game, else -1.
    """
    game = LudoMPLGame()

    nnet = LudoMPLNNP(game)
    nnet.load_checkpoint(folder=args.checkpointFolder, filename=args.checkpointFile)
    mcts = MCTS(game, nnet, args)

    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < args.tempThreshold)

        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        sym = game.getSymmetries(canonicalBoard, pi)
        for b, p in sym:
            trainExamples.append([b, curPlayer, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)

        r = game.getGameEnded(board, curPlayer)

        if r != 0:
            trainExamples = [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]

            folder = args.checkpointFolder
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, "episode-" + str(args.iteration) + ".pkl")
            with open(filename, "wb+") as f:
                Pickler(f).dump(trainExamples)
            f.closed

            return

def main():
    args = dotdict({
        'checkpointFolder'  : sys.argv[1],
        'checkpointFile'    : sys.argv[2],
        'tempThreshold'     : int(sys.argv[3]),
        'iteration'         : int(sys.argv[4]),
        'numMCTSSims'       : int(sys.argv[5]),
        'cpuct'             : float(sys.argv[6])
    })
    executeEpisode(args)

if __name__ == '__main__':
    main()