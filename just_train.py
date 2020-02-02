from Coach import Coach
import numpy as np
# from othello.pytorch.NNet import NNetWrapper as nn

# from othello.OthelloGame import OthelloGame as Game
# from othello.tensorflow.NNet import NNetWrapper as nn
from binpack.tensorflow.NNet import NNetWrapper as nn
from binpack.BinPackGame import BinPackGame as Game
from utils import *
from data_generator import DataGenerator
from solution_checker import SolutionChecker

args = dotdict({
    'numIters': 8,
    'numEps': 3,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 2,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 40,

})

def gen_state(width, height, n_tiles, dg):
    '''
    get tiles and solution
    '''

    tiles, solution = dg.gen_matrix_instance(N_TILES, WIDTH, HEIGHT, with_solution=True)
    board = np.zeros([1, height, width])
    state = np.concatenate((board, tiles), axis=0)
    return state, solution

def solution_to_solution_matrix(solution, rows, cols):
    '''
    transform solution to 2D matrix with 1 only there where the correct tile should be placed, 0 elsewhere
    This is the expected output of the residual network
    '''
    grid = np.zeros((rows, cols))
    position = solution[2]
    grid[position[0]: position[0] + solution[0], position[1]: position[1] + solution[1]] = 1
    return grid

def pad_tiles_with_zero_matrices(tiles, n_zero_matrices_to_add, rows, cols):
    '''
    add tiles with zero matrices to compensate for tiles which were already placed
    '''

    zero_matrices = np.zeros([n_zero_matrices_to_add, rows, cols])
    return np.concatenate((tiles, zero_matrices), axis=0)


if __name__=="__main__":
    N_TILES = 8 
    HEIGHT = 8
    WIDTH = 8
    g = Game(HEIGHT, WIDTH, N_TILES)
    ORIENTATIONS = 2

    dg = DataGenerator(WIDTH, HEIGHT)
    nnet = nn(g)

    if args.load_model and False:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    n_tiles, height, width = N_TILES, HEIGHT, WIDTH
    state, solution = gen_state(width, height, n_tiles, dg)
    # place tiles one by one
    # generate pair x and y where x is stack of state + tiles
    grid = np.zeros([height, width])
    examples = []
    for solution_index, solution_tile in enumerate(solution):
        success, grid = SolutionChecker.place_element_on_grid_given_grid(
            solution_tile[:ORIENTATIONS], solution_tile[2], val=1, grid=grid, cols=width, rows=height
        )
        tiles = dg._transform_instance_to_matrix([x[:ORIENTATIONS] for x in solution[solution_index:]])
        tiles = pad_tiles_with_zero_matrices(tiles,  ORIENTATIONS * N_TILES - len(tiles), width, height)
        state = np.concatenate((np.expand_dims(grid, axis=0), tiles), axis=0)
        examples.append([tiles, state])

    nnet.train(examples)
