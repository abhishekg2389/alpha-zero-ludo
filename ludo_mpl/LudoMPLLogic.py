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

class Board():

    __safe_pos = [8, 13, 21, 26, 34, 39, 47]
    __board_size = 56
    __round_off_pos = 52

    def __init__(self, num_players=2, pieces=None, pieces_away_from_home=None):
        "Set up initial board configuration."

        self.num_players = num_players
        if pieces is None:
            self.pieces = [0, 0, 0, 0]
            if num_players == 2:
                self.pieces += [26, 26, 26, 26]
        else:
            self.pieces = pieces

        if pieces_away_from_home is None:
            self.pieces_away_from_home = [self.__board_size] * len(self.pieces)
        else:
            self.pieces_away_from_home = pieces_away_from_home

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def copy_board(self):
        """Create copy of board with specified pieces."""
        return Board(
            self.num_players,
            pieces=np.copy(self.pieces),
            pieces_away_from_home=np.copy(self.pieces_away_from_home)
        )

    def execute_action(self, action, player):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        action_from_pos = action[0]
        action_to_pos = action[1]
        plyr_idx = None

        pos_diff = action_to_pos - action_from_pos

        if player == 1:
            for i in [0, 1, 2, 3]:
                if self.pieces[i] == action_from_pos:
                    plyr_idx = i
        else :
            for i in [4, 5, 6, 7]:
                if self.pieces[i] == action_from_pos:
                    plyr_idx = i
            action_to_pos = action_to_pos % 52

        assert(plyr_idx is not None)
        self.pieces[plyr_idx] = action_to_pos
        self.pieces_away_from_home[plyr_idx] -= pos_diff
        assert(self.pieces_away_from_home[plyr_idx] >= 0)

        if self.pieces_away_from_home[plyr_idx] <= 5:
            self.pieces[plyr_idx] = -1

        piece_cut = self.check_piece_cut(action_to_pos, player)
        if piece_cut != -1:
            if piece_cut >= 4:
                self.pieces[piece_cut] = 26
            elif piece_cut < 4:
                self.pieces[piece_cut] = 0
            self.pieces_away_from_home[piece_cut] = self.__board_size

        return 56 * (self.pieces_away_from_home[plyr_idx] == 0) + 2 * (piece_cut != -1) + pos_diff

    def check_piece_cut(self, action_to_pos, color):
        if action_to_pos in self.__safe_pos:
            return -1

        if color == 1:
            if self._count_pieces_on_pos(action_to_pos) > 2:
                return -1
            elif self._count_pieces_on_pos(action_to_pos) == 2:
                for i in [4, 5, 6, 7]:
                    if self.pieces[i] == action_to_pos:
                        return i
                return -1
            else:
                return -1
        elif color == -1:
            if self._count_pieces_on_pos(action_to_pos) > 2:
                return -1
            elif self._count_pieces_on_pos(action_to_pos) == 2:
                for i in [0, 1, 2, 3]:
                    if self.pieces[i] == action_to_pos:
                        return i
                return -1
            else:
                return -1

    def is_legal_move(self, plyr_idx, num_pos_to_move):
        return self.pieces_away_from_home[plyr_idx] >= num_pos_to_move

    def _count_pieces_on_pos(self, pos):
        pieces_count = 0
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            if self.pieces[i] == pos:
                pieces_count += 1
        return pieces_count

    @property
    def convert_board_to_vector(self):
        bv = np.zeros((2, 64))

        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            if i // 4 == 0:
                if self.pieces_away_from_home[i] <= 5:
                    bv[0, 57 - self.pieces_away_from_home[i]] += 1
                else:
                    bv[0, 56 - self.pieces_away_from_home[i]] += 1
            else:
                if self.pieces_away_from_home[i] <= 5:
                    bv[1, 63 - self.pieces_away_from_home[i]] += 1
                elif self.pieces_away_from_home[i] - 30 > 0:
                    bv[1, self.pieces_away_from_home[i] - 30] += 1
                else:
                    bv[1, 30 - self.pieces_away_from_home[i]] += 1

        return bv

    @staticmethod
    def convert_vector_to_board(bv):
        _bv = np.copy(bv)

        pieces = [-2]*8
        pieces_away_from_home = [-1]*8

        marker_idx = 0
        for i in [0, 1]:
            for j in range(52):
                while _bv[i, j] > 0:
                    pieces[marker_idx] = j
                    pieces_away_from_home[marker_idx] = 56 - j
                    marker_idx += 1
                    _bv[i, j] -= 1
            for j in range(52, 58):
                while _bv[i, j] > 0:
                    pieces[marker_idx] = -1
                    pieces_away_from_home[marker_idx] = 57 - j
                    marker_idx += 1
                    _bv[i, j] -= 1
            for j in range(59, 64):
                while _bv[i, j] > 0:
                    pieces[marker_idx] = -1
                    pieces_away_from_home[marker_idx] = 63 - j
                    marker_idx += 1
                    _bv[i, j] -= 1

        assert(-2 not in pieces)
        assert (-1 not in pieces_away_from_home)

        return pieces, pieces_away_from_home