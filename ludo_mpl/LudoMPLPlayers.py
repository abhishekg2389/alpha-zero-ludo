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
        # display(board)
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1, debug=True)
        print(valids)
        while True:
            input_move = int(input())
            if valids[input_move]:
                a = input_move
                break
        return a
