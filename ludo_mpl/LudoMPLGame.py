from __future__ import print_function
import sys

sys.path.append('..')
from Game import Game
from .LudoMPLLogic import Board
import numpy as np
import random


class LudoMPLGame(Game):
    def __init__(self):
        Game.__init__(self)
        self._base_board = Board(2)

        dice_nums = [1, 2, 3, 4, 5, 6]
        self.player1_dices = random.choices(dice_nums, k=24)
        self.player2_dices = np.copy(self.player1_dices)
        random.shuffle(self.player2_dices)

        self.player1_score = 0
        self.player2_score = 0
        self.curr_throw = 0

    def getInitBoard(self):
        b = Board(2)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (2, 65)

    def getActionSize(self):
        return 4

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = self._base_board.copy_board()

        score = b.execute_action(action, player)
        if player == 1:
            self.player1_score += score
        elif player == -1:
            self.player2_score += score

        if player == -1:
            self.curr_throw += 1

        _board = self._add_future_dice_values_to_board(b.convert_board_to_vector, player)
        return _board, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        if player == 1:
            num_pos_to_move = self.player1_dices[self.curr_throw]
            for i in [0, 1, 2, 3]:
                valids[i] = self._base_board.is_legal_move(i, num_pos_to_move)
        elif player == -1:
            num_pos_to_move = self.player2_dices[self.curr_throw]
            for i in [0, 1, 2, 3]:
                valids[i] = self._base_board.is_legal_move(4 + i, num_pos_to_move)

        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if (player == -1 and self.curr_throw == 25):
            if self.player1_score == self.player2_score:
                return 0.1
            elif self.player1_score > self.player2_score:
                return 1
            else:
                return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        btv = board.convert_board_to_vector;
        if player == -1:
            _btv            = np.copy(btv)
            btv[0, :52]     = _btv[1, :52]
            btv[0, 52:58]   = _btv[1, 58:64]
            btv[0, 64]      = _btv[1, 64]
            btv[1, :52]     = _btv[0, :52]
            btv[1, 58:64]   = _btv[0, 52:58]
            btv[1, 64]      = _btv[0, 64]
        return btv

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self):
        board_s = "".join(self._base_board.pieces_away_from_home)
        return board_s

    def _add_future_dice_values_to_board(self, board, player):
        if player == 1:
            player1_dice_sum = self.player1_dices[self.curr_throw + 1].sum()
            player2_dice_sum = self.player1_dices[self.curr_throw].sum()
        else:
            player1_dice_sum = self.player1_dices[self.curr_throw].sum()
            player2_dice_sum = self.player1_dices[self.curr_throw].sum()
        return np.append(board, [[player1_dice_sum], [player2_dice_sum]], 1)

    @staticmethod
    def display(board):
        print(board)
        print("-----------------------")