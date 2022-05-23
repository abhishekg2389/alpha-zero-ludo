from multiprocessing import Pool
import numpy as np

from ludo_mpl.LudoMPLPlayers import AggressivePlayer as LudoMPLAP
from ludo_mpl.LudoMPLPlayers import RandomPlayer as LudoMPLRP
from ludo_mpl.keras_v2.NNet import NNetWrapper as LudoMPLNNP
from utils import *
from MCTS import MCTS

def _getPlayer(p, g):
    if p == 'rp':
        return LudoMPLRP(g).play
    if p == 'ap':
        return LudoMPLAP(g).play
    if p == 'nnp':
        n1 = LudoMPLNNP(g)
        n1.load_checkpoint('./temp/','best.pth.tar')
        args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    if p == 'p_nnp':
        n1 = LudoMPLNNP(g)
        n1.load_checkpoint('./temp/', 'temp.prev.pth.tar')
        args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    if p == 'n_nnp':
        n1 = LudoMPLNNP(g)
        n1.load_checkpoint('./temp/', 'temp.new.pth.tar')
        args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

def playGame(p1, p2, g):
    p1 = _getPlayer(p1, g)
    p2 = _getPlayer(p2, g)

    players = [p2, None, p1]
    curPlayer = 1
    board = g.getInitBoard()
    while g.getGameEnded(board, curPlayer) == 0:
        action = players[curPlayer + 1](g.getCanonicalForm(board, curPlayer))
        board, curPlayer = g.getNextState(board, curPlayer, action)
    return curPlayer * g.getGameEnded(board, curPlayer)

def playGames(p1, p2, num_games, g):
    p1Won = 0
    p2Won = 0
    draw = 0

    num = int(num_games / 2)
    args = []
    for i in range(num):
        args.append((p1, p2, g))

    with Pool() as pool:
        future_results = pool.starmap_async(playGame, args)
        results = future_results.get()

    for r in results:
        if r == 1:
            p1Won += 1
        if r == -1:
            p2Won += 1
        else:
            draw += 1

    args = []
    for i in range(num):
        args.append((p2, p1, g))

    with Pool() as pool:
        future_results = pool.starmap_async(playGame, args)
        results = future_results.get()

    for r in results:
        if r == 1:
            p2Won += 1
        if r == -1:
            p1Won += 1
        else:
            draw += 1

    return p1Won, p2Won, draw