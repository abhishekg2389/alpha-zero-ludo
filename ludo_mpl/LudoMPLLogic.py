'''
Author: Abhishek Gupta
Date: May 5, 2022.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

import numpy as np
from numba import jit

class Board():

    def __init__(self, num_players=2, pieces=None, pieces_away_from_home=None):
        self.num_players = num_players
        assert(self.num_players == 2)

        if pieces is None:
            self.pieces = [0, 0, 0, 0]
            if num_players == 2:
                self.pieces += [26, 26, 26, 26]
        else:
            self.pieces = pieces

        if pieces_away_from_home is None:
            self.pieces_away_from_home = [56] * len(self.pieces)
        else:
            self.pieces_away_from_home = pieces_away_from_home

    def copy_board(self):
        """Create copy of board with specified pieces."""
        return Board(
            self.num_players,
            pieces=np.copy(self.pieces),
            pieces_away_from_home=np.copy(self.pieces_away_from_home)
        )

@jit(nopython=True)
def execute_action(pieces, pieces_away_from_home, move, player):
    '''
    move = [player, player_piece_idx, steps_to_move]
    '''
    assert (move[0] == player)

    if player == 1:
        player_piece_idx = move[1]
    else:
        player_piece_idx = move[1] + 4
    steps_to_move = move[2]

    pieces_away_from_home[player_piece_idx] -= steps_to_move
    if player == 1:
        if pieces_away_from_home[player_piece_idx] <= 5:
            pieces[player_piece_idx] = -1
        else:
            pieces[player_piece_idx] += steps_to_move
    else:
        if pieces_away_from_home[player_piece_idx] <= 5:
            pieces[player_piece_idx] = -1
        else:
            pieces[player_piece_idx] += steps_to_move
            pieces[player_piece_idx] = pieces[player_piece_idx] % 52

    assert (pieces_away_from_home[player_piece_idx] >= 0)
    piece_cut = check_piece_cut(pieces, player_piece_idx, player)
    if piece_cut != -1:
        if piece_cut >= 4:
            pieces[piece_cut] = 26
        elif piece_cut < 4:
            pieces[piece_cut] = 0
        pieces_away_from_home[piece_cut] = 56

    return 56 * (pieces_away_from_home[player_piece_idx] == 0) + 2 * (piece_cut != -1) + steps_to_move

@jit(nopython=True)
def check_piece_cut(pieces, player_piece_idx, player):
    if pieces[player_piece_idx] == -1:
        return -1

    player_piece_idx_pos = pieces[player_piece_idx]

    if player_piece_idx_pos in [8, 13, 21, 26, 34, 39, 47]:
        return -1

    player_piece_idx_pos_count = 0
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        if pieces[i] == player_piece_idx_pos:
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
                if pieces[i] == player_piece_idx_pos:
                    return i
        else:
            for i in [0, 1, 2, 3]:
                if pieces[i] == player_piece_idx_pos:
                    return i
    return -1

@jit(nopython=True)
def convert_board_to_vector(pieces_away_from_home):
    bv = np.zeros((2, 64))

    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        if i // 4 == 0:
            if pieces_away_from_home[i] <= 5:
                bv[0, 57 - pieces_away_from_home[i]] += 1
            else:
                bv[0, 56 - pieces_away_from_home[i]] += 1
        else:
            if pieces_away_from_home[i] <= 5:
                bv[1, 63 - pieces_away_from_home[i]] += 1
            else:
                bv[1, (26 + 56 - pieces_away_from_home[i]) % 52] += 1

    return bv

@jit(nopython=True)
def convert_vector_to_board(bv):
    _bv = np.copy(bv)

    pieces = [-2] * 8
    pieces_away_from_home = [-1] * 8

    marker_idx = 0

    for i in [0, 1]:
        for j in range(52):
            while _bv[i, j] > 0:
                pieces[marker_idx] = j
                if i == 0:
                    pieces_away_from_home[marker_idx] = 56 - j
                else:
                    if j < 26:
                        pieces_away_from_home[marker_idx] = 30 - j
                    else:
                        pieces_away_from_home[marker_idx] = 52 - j + 30
                marker_idx += 1
                _bv[i, j] -= 1
        for j in range(52, 58):
            while _bv[i, j] > 0:
                pieces[marker_idx] = -1
                pieces_away_from_home[marker_idx] = 57 - j
                marker_idx += 1
                _bv[i, j] -= 1
        for j in range(58, 64):
            while _bv[i, j] > 0:
                pieces[marker_idx] = -1
                pieces_away_from_home[marker_idx] = 63 - j
                marker_idx += 1
                _bv[i, j] -= 1

    return pieces, pieces_away_from_home
