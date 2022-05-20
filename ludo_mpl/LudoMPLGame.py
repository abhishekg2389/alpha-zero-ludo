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

    def getInitBoard(self):
        dice_nums = [1, 2, 3, 4, 5, 6]
        self.player1_dices = np.array(random.choices(dice_nums, k=24))
        self.player2_dices = np.copy(self.player1_dices)
        random.shuffle(self.player2_dices)

        self.player1_score = 0
        self.player2_score = 0
        self.curr_throw = 0

        b = Board(2)
        return self._add_additional_params_to_board(b.convert_board_to_vector, 0)

    def getBoardSize(self):
        return (2, 71)

    def getActionSize(self):
        return 4

    def setGameGivenBoard(self, board):
        self._set_board_from_bvf(board)
        self._set_player_dices_from_board(board)
        self._set_curr_throw_from_board(board)

    def getMoveFromAction(self, board, player, action):
        isCanonicalBoard = player == 1 and list(self.player1_dices).count(0) < list(self.player2_dices).count(0)
        player1_dices, player2_dices = self._set_player_dices_from_board(board, isCanonicalBoard=isCanonicalBoard, onlyGet=True)

        if not isCanonicalBoard:
            if player == 1:
                num_pos_to_move = player1_dices[self.curr_throw]
            elif player == -1:
                num_pos_to_move = player2_dices[self.curr_throw]
        else:
            num_pos_to_move = player1_dices[self.curr_throw]

        return (player, action, num_pos_to_move)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        self.setGameGivenBoard(board)

        b = self._base_board.copy_board()

        move = self.getMoveFromAction(board, player, action)
        score = b.execute_action(move, player)

        '''
        self._base_board = b

        if player == 1:
            self.player1_score += score
        elif player == -1:
            self.player2_score += score
        '''
        if player == -1:
            self.curr_throw += 1

        _board = self._add_additional_params_to_board(b.convert_board_to_vector, player)
        return _board, -player

    def getValidMoves(self, board, player, debug=False):
        self.setGameGivenBoard(board)

        assert player == 1

        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        isCanonicalBoard = player == 1 and list(self.player1_dices).count(0) < list(self.player2_dices).count(0)
        player1_dices, player2_dices = self._set_player_dices_from_board(board, isCanonicalBoard=isCanonicalBoard, onlyGet=True)

        if isCanonicalBoard:
            if debug:
                print(player2_dices)
            num_pos_to_move = player2_dices[self.curr_throw]
        else:
            if debug:
                print(player1_dices)
            num_pos_to_move = player1_dices[self.curr_throw]

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
            _board[0, :26]       = board[1, 26:52]
            _board[0, 26:52]     = board[1, :26]
            _board[0, 52:58]     = board[1, 58:64]
            _board[0, 64:]       = board[1, 64:]
            _board[1, :26]       = board[0, 26:52]
            _board[1, 26:52]     = board[0, :26]
            _board[1, 58:64]     = board[0, 52:58]
            _board[1, 64:]       = board[0, 64:]
        return _board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    def stringRepresentationReadable(self):
        board_s = "".join(self._base_board.pieces_away_from_home)
        return board_s

    def _set_board_from_bvf(self, bvf):
        p, pafh = self._base_board.convert_vector_to_board(bvf)
        self._base_board = Board(num_players=2, pieces=p, pieces_away_from_home=pafh)

    def _set_curr_throw_from_board(self, bvf):
        curr_throw = int(min(24 - bvf[0, 70], 24 - bvf[1, 70]))
        self.curr_throw = curr_throw

    def _set_player_dices_from_board(self, bvf, isCanonicalBoard=False, onlyGet=False):
        player1_dices = [0] * 24
        player2_dices = [0] * 24

        player1_dices_rem = []
        for i in [64, 65, 66, 67, 68, 69]:
            for j in range(int(bvf[0, i])):
                player1_dices_rem.append(i - 63)

        player2_dices_rem = []
        for i in [64, 65, 66, 67, 68, 69]:
            for j in range(int(bvf[1, i])):
                player2_dices_rem.append(i - 63)

        if isCanonicalBoard:
            _x = np.copy(player1_dices_rem)
            player1_dices_rem = np.copy(player2_dices_rem)
            player2_dices_rem = _x

        random.seed(0)
        random.shuffle(player1_dices_rem)
        random.shuffle(player2_dices_rem)

        player1_dices[-len(player1_dices_rem):] = player1_dices_rem
        player2_dices[-len(player2_dices_rem):] = player2_dices_rem

        if onlyGet:
            return player1_dices, player2_dices
        else:
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
            player2_dice_counts = Counter(self.player2_dices[self.curr_throw : ])
            player1_moves_left = sum([v for k, v in player1_dice_counts.items() if k != 0])
            player2_moves_left = sum([v for k, v in player2_dice_counts.items() if k != 0])
        else:
            player1_dice_counts = Counter(self.player1_dices[self.curr_throw : ])
            player2_dice_counts = Counter(self.player2_dices[self.curr_throw : ])
            player1_moves_left = sum([v for k, v in player1_dice_counts.items() if k != 0])
            player2_moves_left = sum([v for k, v in player2_dice_counts.items() if k != 0])

        addtnl_params = np.zeros((2, 7))

        for i in [0, 1]:
            for j in [1, 2, 3, 4, 5, 6]:
                if i == 0:
                    if j in player1_dice_counts:
                        addtnl_params[i, j-1] = player1_dice_counts[j]
                else:
                    if j in player2_dice_counts:
                        addtnl_params[i, j-1] = player2_dice_counts[j]

        addtnl_params[0, 6] = player1_moves_left
        addtnl_params[1, 6] = player2_moves_left

        _board = np.append(board, addtnl_params, 1)

        if _board[0, -7:-1].sum() != _board[0, -1] or _board[1, -7:-1].sum() != _board[1, -1]:
            return

        return _board

    @staticmethod
    def display(bv):
        print(bv)
        board = Board.convert_vector_to_board(bv)
        print(board)
        print("---------------")

        db = [
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 0], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 1], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 1], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [], [0, 0, 0], [], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [], [], [], [], [], [], [], [], [], [], [], [], [], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [], [0, 0, 0], [], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 1], [0, 0, 1], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 1], [0, 0, 1], [0, 0, 0], [], [], [], [], [], []],
            [[], [], [], [], [], [], [0, 0, 0], [0, 0, 0], [0, 0, 0], [], [], [], [], [], []],
        ]
        for i in range(8):
            if board[0][i] == -1:
                pafh = board[1][i]
                if i < 4:
                    r = 8+pafh
                    c = 7
                else:
                    r = 6-pafh
                    c = 7
            else:
                if board[0][i] <= 4:
                    r = 13 - board[0][i]
                    c = 6
                elif board[0][i] <= 10:
                    r = 8
                    c = 10 - board[0][i]
                elif board[0][i] == 11:
                    r = 7
                    c = 0
                elif board[0][i] <= 17:
                    r = 6
                    c = board[0][i] - 12
                elif board[0][i] <= 23:
                    r = 23 - board[0][i]
                    c = 6
                elif board[0][i] == 24:
                    r = 0
                    c = 7
                elif board[0][i] <= 30:
                    r = board[0][i] - 25
                    c = 8
                elif board[0][i] <= 36:
                    r = 6
                    c = 8 + board[0][i] - 30
                elif board[0][i] == 37:
                    r = 7
                    c = 14
                elif board[0][i] <= 43:
                    r = 8
                    c = 8 + 44 - board[0][i]
                elif board[0][i] <= 49:
                    r = 8 + board[0][i] - 43
                    c = 8
                elif board[0][i] == 50:
                    r = 14
                    c = 7
                elif board[0][i] == 51:
                    r = 14
                    c = 6
                else:
                    assert False

            db[r][c][i // 4] += 1

        dbs = ""
        for i in range(15):
            for j in range(15):
                if len(db[i][j]) == 0:
                    dbs += "    "
                else:
                    _dbs = str(db[i][j][0]) + str(db[i][j][1])
                    if db[i][j][2] == 1:
                        dbs += "(" + _dbs + ")"
                    else:
                        dbs += " " + _dbs + " "
            dbs += "\n"

        print(dbs)