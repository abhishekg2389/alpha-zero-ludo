import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

import ParallelArena
from MCTS import MCTS
from ludo_mpl.keras_v2.NNet import NNetWrapper as LudoMPLNNP

log = logging.getLogger(__name__)

class ParallelCoach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(self.args.iter + 1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > self.args.iter + 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                with Pool() as pool:
                    future_results = pool.starmap_async(executeEpisode, [(self.game, self.args) for _ in range(self.args.numEps)])
                    results = future_results.get()

                for result in results:
                    # save the iteration examples to the history
                    iterationTrainExamples += result

                self.trainExamplesHistory.append(iterationTrainExamples)

            while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.prev.pth.tar')
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.new.pth.tar')

            log.info('PITTING AGAINST PREVIOUS VERSION')
            pwins, nwins, draws = ParallelArena.playGames('p_nnp', 'n_nnp', self.args.arenaCompare, self.game)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.prev.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.getCheckpointFile(self.args.iter))
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

def executeEpisode(game, args):
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
    nnet = LudoMPLNNP(game)
    nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
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
            return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]