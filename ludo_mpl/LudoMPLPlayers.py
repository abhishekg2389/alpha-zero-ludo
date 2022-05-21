import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1, debug=True)
        print(valids)
        while True:
            input_move = int(input())
            if valids[input_move]:
                return input_move
        assert False

class AggressivePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        self.game.setGameGivenBoard(board)
        playing_seq = np.argsort(self.game.getBoard().pieces_away_from_home[:4])

        valids = self.game.getValidMoves(board, 1, debug=True)
        print(valids)

        for i in range(len(valids)):
            if valids[playing_seq[i]]:
                return playing_seq[i]

        assert False