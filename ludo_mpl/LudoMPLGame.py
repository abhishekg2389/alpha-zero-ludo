from __future__ import print_function
import sys

sys.path.append('..')
from Game import Game
from .LudoMPLLogic import Board
import numpy as np
import random
from collections import Counter


class LudoMPLGame(Game):
    def __init__(self):
        Game.__init__(self)
        self._base_board = Board(2)

        dice_nums = [1, 2, 3, 4, 5, 6]
        self.player1_dices = np.array(random.choices(dice_nums, k=24))
        self.player2_dices = np.copy(self.player1_dices)
        random.shuffle(self.player2_dices)

        self.player1_score = 0
        self.player2_score = 0
        self.curr_throw = 0

    def getInitBoard(self):
        b = Board(2)
        return self._add_additional_params_to_board(b.convert_board_to_vector, 0)

    def getBoardSize(self):
        return (2, 65)

    def getActionSize(self):
        return 4

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        self._set_board_from_bvf(board)
        self._set_player_dices_from_board(board)
        self._set_curr_throw_from_board(board)
        b = self._base_board.copy_board()

        move = [-1, -1]
        if player == 1:
            move[0] = b.pieces[action]
            move[1] = move[0] + self.player1_dices[self.curr_throw]
        else:
            move[0] = b.pieces[4 + action]
            move[1] = move[0] + self.player2_dices[self.curr_throw]

        score = b.execute_action(move, player)

        '''
        self._base_board = b

        if player == 1:
            self.player1_score += score
        elif player == -1:
            self.player2_score += score

        if player == -1:
            self.curr_throw += 1
        '''

        _board = self._add_additional_params_to_board(b.convert_board_to_vector, player)
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

            if self.player1_score == self.player2_score:
                return 0.00001
            elif self.player1_score > self.player2_score:
                return 1
            else:
                return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        if player == -1:
            _board            = np.copy(board)
            board[0, :52]     = _board[1, :52]
            board[0, 52:58]   = _board[1, 58:64]
            board[0, 64]      = _board[1, 64]
            board[1, :52]     = _board[0, :52]
            board[1, 58:64]   = _board[0, 52:58]
            board[1, 64]      = _board[0, 64]
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self):
        board_s = "".join(self._base_board.pieces_away_from_home)
        return board_s

    def _set_board_from_bvf(self, bvf):
        p, pafh = self._base_board.convert_vector_to_board(bvf)
        self._base_board = Board(num_players=2, pieces=p, pieces_away_from_home=pafh)

    def _set_curr_throw_from_board(self, bvf):
        self.curr_throw = min(24 - bvf[0, 70], 24 - bvf[1, 70])

    def _set_player_dices_from_board(self, bvf):
        player1_dices = [0] * 24
        player2_dices = [0] * 24

        player1_dices_rem = []
        for i in [64, 65, 66, 67, 68, 69]:
            for j in range(bvf[0, i]):
                player1_dices_rem.append(i - 63)

        player2_dices_rem = []
        for i in [64, 65, 66, 67, 68, 69]:
            for j in range(bvf[1, i]):
                player2_dices_rem.append(i - 63)

        random.shuffle(player1_dices_rem)
        random.shuffle(player2_dices_rem)

        player1_dices[-len(player1_dices_rem):] = player1_dices_rem
        player2_dices[-len(player2_dices_rem):] = player2_dices_rem

        self.player1_dices = player1_dices
        self.player2_dices = player2_dices

    def _add_additional_params_to_board(self, board, player):
        if player == 0:
            player1_dice_counts = Counter(self.player1_dices)
            player2_dice_counts = Counter(self.player2_dices)
            player1_moves_left = 24
            player2_moves_left = 24
        elif player == 1:
            player1_dice_counts = Counter(self.player1_dices[self.curr_throw + 1 : ])
            player2_dice_counts = Counter(self.player1_dices[self.curr_throw : ])
            player1_moves_left = 23 - self.curr_throw
            player2_moves_left = 24 - self.curr_throw
        else:
            player1_dice_counts = Counter(self.player1_dices[self.curr_throw : ])
            player2_dice_counts = Counter(self.player1_dices[self.curr_throw : ])
            player1_moves_left = 24 - self.curr_throw
            player2_moves_left = 24 - self.curr_throw

        addtnl_params = np.zeros((2, 7))

        for i in [0, 1]:
            for j in [1, 2, 3, 4, 5, 6]:
                if i == 0:
                    if j in player1_dice_counts:
                        addtnl_params[i, j] = player1_dice_counts[j]
                else:
                    if j in player2_dice_counts:
                        addtnl_params[i, j] = player2_dice_counts[j]

        addtnl_params[0, 6] = player1_moves_left
        addtnl_params[1, 6] = player2_moves_left

        return np.append(board, [addtnl_params], 1)

    @staticmethod
    def display(board):
        print(board)
        print("-----------------------")