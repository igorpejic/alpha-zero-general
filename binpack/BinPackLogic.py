from collections import namedtuple
import numpy as np

from data_generator import DataGenerator
from solution_checker import SolutionChecker

DEFAULT_HEIGHT = 8
DEFAULT_WIDTH = 8
ORIENTATIONS = 2

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    BinPack Board.
    """

    def __init__(self, height, width, tiles, board=None):
        "Set up initial board configuration."
        """
        tiles - array of tiles with (width, height)
        state - current bin configuration with 1 indicating locus is taken
                    0 indicating that it is free
        """

        self.height = height
        self.width = width
        self.tiles = tiles
        self.orientations = ORIENTATIONS


        if board is None:
            board = np.zeros([self.height, self.width])
            self.board = board
        else:
            self.board = board

        self.tiles = tiles
        assert self.board.shape == (self.height, self.width)
        assert len(self.tiles) == len(tiles)

    def add_tile(self, tile_index, player):
        """
        Create copy of board containing new tile.
        Position is index (?) on which to place tile.
        We always place the tile which is located at position 1 or 2. 
        """

        tile = self.tiles[tile_index]
        next_lfb = SolutionChecker.get_next_lfb_on_grid(self.board)
        success, grid = SolutionChecker.place_element_on_grid_given_grid(
            tile, next_lfb, val=1,
            grid=self.board, cols=self.board.shape[1],
            rows=self.board.shape[0]
        )
        self.board = grid
        self.tiles = SolutionChecker.eliminate_pair_tiles(
            [tuple(x) for x in self.tiles], tuple(tile))

        self.tiles = SolutionChecker.pad_tiles_with_zero_scalars(
            self.tiles, 2)
        return self.board, self.tiles


    def get_valid_moves(self):
        """
        Any drop on locus with zero value is a valid move
        If lower than self.height * self.width it is first orientation, 
        if not it is second
        """

        valid_tile_indexes = SolutionChecker.get_valid_tile_actions_indexes_given_grid(self.board, self.tiles)
        return valid_tile_indexes

    def all_tiles_placed(self):
        ret = (DataGenerator.get_n_tiles_placed(self.state) ==
               (len(self.state) - 1) // self.orientations)
        return ret

    def get_win_state(self):
        if self.all_tiles_placed() or not np.any(self.get_valid_moves()): 
            # game  has ended calculate reward
            locus_filled = np.sum(self.state[0])
            total_locus = self.state[0].shape[0] * self.state[0].shape[1]
            if locus_filled == total_locus:
                return 1
            else:
                return locus_filled / total_locus

        # game has not ended yet
        return False


    def with_state(self, tiles, board):
        """Create copy of board with specified pieces."""
        return Board(self.height, self.width, tiles, np.copy(board))

    def __str__(self):
        result_str = ''
        for _slice in self.state:
            for row in _slice:
                result_str += str(row)
        return result_str
