'''
Author: Abhishek Gupta
Date: May 5, 2022
'''

import numpy as np
from numba import njit

class Board():
    def __init__(self, num_players=2, pieces_away_from_home=None):
        self.num_players = num_players
        if pieces_away_from_home is None:
            self.pieces_away_from_home = [56] * 8
        else:
            self.pieces_away_from_home = pieces_away_from_home

    def copy_board(self):
        return Board(
            self.num_players,
            pieces_away_from_home=np.copy(self.pieces_away_from_home)
        )

    def execute_action(self, move, player):
        assert(move[0] == player)

        if player == 1:
            player_piece_idx = move[1]
        else:
            player_piece_idx = move[1] + 4
        steps_to_move = move[2]

        self.pieces_away_from_home[player_piece_idx] -= steps_to_move
        assert (self.pieces_away_from_home[player_piece_idx] >= 0)

        piece_cut = check_piece_cut(player_piece_idx, player)
        if piece_cut != -1:
            self.pieces_away_from_home[piece_cut] = 56

        return 56 * (self.pieces_away_from_home[player_piece_idx] == 0) + 2 * (piece_cut != -1) + steps_to_move

@njit
def find_piece_pos_board(piece_pos_away_from_home, player):
    if player == 1:
        if piece_pos_away_from_home <= 5:
            piece_pos_board = -1
        else:
            piece_pos_board = 56 - piece_pos_away_from_home
        assert(piece_pos_board != 51)
    elif player == -1:
        if piece_pos_away_from_home <= 5:
            piece_pos_board = -1
        else:
            piece_pos_board = (26 + 56 - piece_pos_away_from_home) % 52
        assert (piece_pos_board != 25)
    assert(piece_pos_board is not None)
    return piece_pos_board

@njit
def check_piece_cut(pieces_away_from_home, player_piece_idx, player):
    player_piece_pos_board = find_piece_pos_board(pieces_away_from_home[player_piece_idx])

    if player_piece_pos_board == -1:
        return -1

    if player_piece_pos_board in [8, 13, 21, 26, 34, 39, 47]:
        return -1

    pieces_pos_board = [None] * 8
    for i in range(8):
        pieces_pos_board[i] = find_piece_pos_board(pieces_away_from_home[i])

    player_piece_idx_pos_count = 0
    for i in range(8):
        if player_piece_pos_board == pieces_pos_board[i]:
            player_piece_idx_pos_count += 1

    if player_piece_idx_pos_count > 2:
        return -1
    elif player_piece_idx_pos_count == 1:
        return -1
    elif player_piece_idx_pos_count == 0:
        assert False
    else:
        if player == 1:
            for i in [4, 5, 6, 7]:
                if pieces_pos_board[i] == player_piece_pos_board:
                    return i
        else:
            for i in [0, 1, 2, 3]:
                if pieces_pos_board[i] == player_piece_pos_board:
                    return i
    return -1

@njit
def convert_board_to_board_state(pieces_away_from_home):
    pieces_pos_board = [None] * 8
    for i in range(8):
        pieces_pos_board[i] = find_piece_pos_board(pieces_away_from_home[i])

    bs = np.zeros((2, 64))

    for i in range(8):
        if i < 4:
            if pieces_pos_board[i] == -1:
                bs[0, 57 - pieces_away_from_home[i]] += 1
            else:
                bs[0, pieces_pos_board[i]] += 1
        else:
            if pieces_pos_board[i] == -1:
                bs[1, 63 - pieces_away_from_home[i]] += 1
            else:
                bs[1, pieces_pos_board[i]] += 1

    return bs

@njit
def convert_board_state_to_board(bs):
    _bs = np.copy(bs)

    pieces_away_from_home = [-1] * 8

    marker_idx = 0

    for i in [0, 1]:
        for j in range(52):
            while _bs[i, j] > 0:
                if i == 0:
                    pieces_away_from_home[marker_idx] = 56 - j
                else:
                    assert marker_idx >= 4
                    if j < 26:
                        pieces_away_from_home[marker_idx] = 30 - j
                    else:
                        pieces_away_from_home[marker_idx] = 52 - j + 30
                marker_idx += 1
                _bs[i, j] -= 1
        for j in range(52, 58):
            while _bs[i, j] > 0:
                pieces_away_from_home[marker_idx] = 57 - j
                marker_idx += 1
                _bs[i, j] -= 1
        for j in range(58, 64):
            while _bs[i, j] > 0:
                pieces_away_from_home[marker_idx] = 63 - j
                marker_idx += 1
                _bs[i, j] -= 1

    return pieces_away_from_home
