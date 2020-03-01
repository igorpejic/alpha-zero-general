import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .BinPackLogic import Board

from data_generator import DataGenerator
from solution_checker import SolutionChecker
from collections import namedtuple

ORIENTATIONS = 2
State = namedtuple('State', 'board tiles')

class BinPackGame(Game):
    """
    BinPack Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, n_tiles=None):
        self.initialize(height, width, n_tiles)

    def initialize(self, height, width, n_tiles):
        Game.__init__(self)

        self.height = height
        self.width = width
        self.n_tiles = n_tiles

        dg = DataGenerator()
        self.tiles, _ = dg.gen_instance(n_tiles, width, height)
        self.tiles = SolutionChecker.get_tiles_with_orientation(self.tiles)
        self._base_board = Board(height, width, self.tiles)

    def getInitBoard(self):
        # self.initialize(self.height, self.width, self.n_tiles)
        return State(self._base_board.board, self._base_board.tiles)

    def getBoardSize(self):
        ORIENTATIONS = 2
        return (
            self._base_board.height, self._base_board.width,
            self.n_tiles * ORIENTATIONS + 1
                )

    def getActionSize(self):
        # allowing moves without gravity physical support
        return len(self.tiles)
        # orientations are included in tiles
        # self._base_board.orientations)

    def getNextState(self, state, action, player):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_state(tiles=state.tiles, board=state.board)
        b.add_tile(action, player)
        return State(b.board, b.tiles), player

    def getValidMoves(self, board, tiles, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_state(tiles=tiles, board=board).get_valid_moves()

    def getGameEnded(self, state, player):
        non_placed_tiles = SolutionChecker.get_n_nonplaced_tiles(state.tiles)
        if non_placed_tiles == 0: # all tiles placed
            return 1
        possible_tile_actions = SolutionChecker.get_possible_tile_actions_given_grid(state.board, state.tiles)
        n_possible_actions = SolutionChecker.get_n_nonplaced_tiles(possible_tile_actions)
        if n_possible_actions != 0:
            # 0 used to represent unfinished game.
            return 0
        else:
            return self.getAreaReward(state.board)

    def getAreaReward(self, board):
        '''
        returns reward; higher area covered, bigger reward
        '''
        unique, counts = np.unique(board, return_counts=True)
        counts = dict(zip(unique, counts))
        return counts[1] / (board.shape[0] * board.shape[1])

    def getTilesReward(self, tiles):
        n_unplaced_tiles = SolutionChecker.get_n_nonplaced_tiles(tiles)
        return (len(tiles) - n_unplaced_tiles) / len(tiles)

    def getCanonicalForm(self, state, player):
        # Flip player from 1 to -1
        return state

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        """
        TODO:
        We need to rotate just the board and keep the pieces unrotated
        although I don't think rotating te pieces would hurt, it's just not  needed
        """
        # return [(board, pi), (board[:, ::-1], pi[::-1])]
        board_right = np.copy(board)
        board_right[0] = board_right[0][::-1]
        # return [(board, pi), (board_right, pi[::-1])]
        return [(board, pi)]

    def stringRepresentation(self, state):
        return (str(state.board), str(state.tiles))


def display(board):
    print(" -----------------------")
    print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")
