from __future__ import print_function

import sys

sys.path.append('..')
from Game import Game
from .LudoMPLLogic import *
import numpy as np
import random


class LudoMPLGame(Game):
    def __init__(self):
        Game.__init__(self)
        self._base_board = Board(2)

    def getInitBoard(self):
        dice_nums = [1, 2, 3, 4, 5, 6]
        self.player1_dices = np.array(random.choices(dice_nums, k=24))
        self.player2_dices = np.copy(self.player1_dices)
        random.shuffle(self.player2_dices)

        self.player1_score = 0
        self.player2_score = 0
        self.curr_throw = 0

        b = Board(2)
        return _add_additional_params_to_board(
            convert_board_to_vector(b.pieces_away_from_home),
            self.curr_throw, self.player1_dices, self.player2_dices, 0
        )

    def getBoardSize(self):
        return (2, 71)

    def getActionSize(self):
        return 4

    def setGameGivenBoard(self, board):
        self._set_board_from_bvf(board)

        player1_dices = np.zeros(24, dtype=np.int8)
        player2_dices = np.zeros(24, dtype=np.int8)
        _set_player_dices_from_board(player1_dices, player2_dices, board)
        self.player1_dices = player1_dices
        self.player2_dices = player2_dices

        self._set_curr_throw_from_board(board)

    def getMoveFromAction(self, board, player, action, boardSetAlready=False):
        if not boardSetAlready:
            self.setGameGivenBoard(board)

        b = self._base_board.copy_board()
        if player == 1:
            move = (1, action, self.player1_dices[self.curr_throw])
        else:
            move = (-1, action, self.player2_dices[self.curr_throw])

        return move

    def getNextState(self, board, player, action, boardSetAlready=False):
        if not boardSetAlready:
            self.setGameGivenBoard(board)
        b = self._base_board.copy_board()
        move = self.getMoveFromAction(board, player, action, boardSetAlready=True)
        score = execute_action(b.pieces, b.pieces_away_from_home, move, player)

        '''
        self._base_board = b

        if player == 1:
            self.player1_score += score
        elif player == -1:
            self.player2_score += score
        '''
        if player == -1:
            self.curr_throw += 1

        _board = _add_additional_params_to_board(
            convert_board_to_vector(b.pieces_away_from_home),
            self.curr_throw, self.player1_dices, self.player2_dices, player
        )
        return _board, -player

    def getValidMoves(self, board, player, boardSetAlready=False, debug=False):
        if not boardSetAlready:
            self.setGameGivenBoard(board)
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        if player == 1 and list(self.player1_dices).count(0) < list(self.player2_dices).count(0):
            _set_player_dices_from_board(self.player1_dices, self.player2_dices, board, isCanonicalBoard=True)
            player = -1

        if player == 1:
            if debug:
                print(self.player1_dices)
            num_pos_to_move = self.player1_dices[self.curr_throw]
            for i in [0, 1, 2, 3]:
                valids[i] = self._base_board.pieces_away_from_home[i] >= num_pos_to_move
        elif player == -1:
            if debug:
                print(self.player2_dices)
            num_pos_to_move = self.player2_dices[self.curr_throw]
            for i in [0, 1, 2, 3]:
                valids[i] = self._base_board.pieces_away_from_home[i] >= num_pos_to_move

        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if board[0, 70] == 0 and board[1, 70] == 0:
            self._set_board_from_bvf(board)
            board = self._base_board

            player1_score = 0
            player2_score = 0

            for i in [0, 1, 2, 3]:
                player1_score += (56 - board.pieces_away_from_home[i])
                if board.pieces_away_from_home[i] == 0:
                    player1_score += 56

            for i in [4, 5, 6, 7]:
                player2_score += (56 - board.pieces_away_from_home[i])
                if board.pieces_away_from_home[i] == 0:
                    player2_score += 56

            if player1_score == player2_score:
                return 0.00001
            elif player1_score > player2_score:
                return 1
            else:
                return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        _board = np.copy(board)
        if player == -1:
            _board[0, :26] = board[1, 26:52]
            _board[0, 26:52] = board[1, :26]
            _board[0, 52:58] = board[1, 58:64]
            _board[0, 64:] = board[1, 64:]
            _board[1, :26] = board[0, 26:52]
            _board[1, 26:52] = board[0, :26]
            _board[1, 58:64] = board[0, 52:58]
            _board[1, 64:] = board[0, 64:]
        return _board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    def stringRepresentationReadable(self):
        board_s = "".join(self._base_board.pieces_away_from_home)
        return board_s

    def _set_board_from_bvf(self, bvf):
        p, pafh = convert_vector_to_board(bvf)
        self._base_board = Board(num_players=2, pieces=p, pieces_away_from_home=pafh)

    def _set_curr_throw_from_board(self, bvf):
        curr_throw = int(min(24 - bvf[0, 70], 24 - bvf[1, 70]))
        self.curr_throw = curr_throw

    @staticmethod
    def display(board):
        print(board)
        print(convert_vector_to_board(board))
        print("-----------------------")

@jit(nopython=True)
def _set_player_dices_from_board(player1_dices, player2_dices, bvf, isCanonicalBoard=False):
    player1_dices_rem = np.zeros(int(bvf[0, 64:70].sum()), dtype=np.int8)
    counter = 0
    for i in [64, 65, 66, 67, 68, 69]:
        for j in range(int(bvf[0, i])):
            player1_dices_rem[counter] = i - 63
            counter += 1

    player2_dices_rem = np.zeros(int(bvf[1, 64:70].sum()), dtype=np.int8)
    counter = 0
    for i in [64, 65, 66, 67, 68, 69]:
        for j in range(int(bvf[1, i])):
            player2_dices_rem[counter] = i - 63
            counter += 1

    if isCanonicalBoard:
        _x = player1_dices_rem
        player1_dices_rem = player2_dices_rem
        player2_dices_rem = _x

    np.random.seed(0)
    np.random.shuffle(player1_dices_rem)
    np.random.shuffle(player2_dices_rem)

    if len(player1_dices_rem) != 0:
        player1_dices[-len(player1_dices_rem):] = player1_dices_rem
    if len(player2_dices_rem) != 0:
        player2_dices[-len(player2_dices_rem):] = player2_dices_rem

#@jit(nopython=True)
def _add_additional_params_to_board(board, curr_throw, player1_dices, player2_dices, player):
    player1_dice_counts = [0, 0, 0, 0, 0, 0]
    player2_dice_counts = [0, 0, 0, 0, 0, 0]

    player1_moves_left = 0
    player2_moves_left = 0

    if player == 0:
        for i in range(len(player1_dices)):
            player1_dice_counts[player1_dices[i] - 1] += 1
            player2_dice_counts[player2_dices[i] - 1] += 1

        player1_moves_left = 24
        player2_moves_left = 24
    elif player == 1:
        for i in range(curr_throw + 1, len(player1_dices)):
            if player1_dices[i] == 0:
                continue
            player1_dice_counts[player1_dices[i] - 1] += 1
            player1_moves_left += 1
        for i in range(curr_throw, len(player2_dices)):
            if player2_dices[i] == 0:
                continue
            player2_dice_counts[player2_dices[i] - 1] += 1
            player2_moves_left += 1
    else:
        for i in range(curr_throw, len(player1_dices)):
            if player1_dices[i] == 0:
                continue
            player1_dice_counts[player1_dices[i] - 1] += 1
            player1_moves_left += 1

        for i in range(curr_throw, len(player2_dices)):
            if player2_dices[i] == 0:
                continue
            player2_dice_counts[player2_dices[i] - 1] += 1
            player2_moves_left += 1

    addtnl_params = np.zeros((2, 7))

    for i in [0, 1]:
        for j in [1, 2, 3, 4, 5, 6]:
            if i == 0:
                addtnl_params[i, j - 1] = player1_dice_counts[j - 1]
            else:
                addtnl_params[i, j - 1] = player2_dice_counts[j - 1]

    addtnl_params[0, 6] = player1_moves_left
    addtnl_params[1, 6] = player2_moves_left

    _board = np.append(board, addtnl_params, 1)

    if _board[0, -7:-1].sum() != _board[0, -1] or _board[1, -7:-1].sum() != _board[1, -1]:
        assert (False)

    return _board