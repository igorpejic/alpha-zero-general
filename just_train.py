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
    transform solution to 2D matrix with 1 only there where the correct tile should be placed, -1 elsewhere
    This is the expected output of the residual network
    '''
    grid = np.ones((rows, cols))
    grid *= -1
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
    print('Preparing examples')
    N_EXAMPLES = 20
    i = 0
    while i < N_EXAMPLES:
        print(f'{i}/{N_EXAMPLES}')
        state, solution = gen_state(width, height, n_tiles, dg)
        grid = np.zeros([height, width])
        for solution_index, solution_tile in enumerate(solution):
            tiles = dg._transform_instance_to_matrix([x[:ORIENTATIONS] for x in solution[solution_index:]])
            tiles = pad_tiles_with_zero_matrices(tiles,  ORIENTATIONS * N_TILES - len(tiles), width, height)
            pi = solution_to_solution_matrix(solution_tile, cols=width, rows=height).flatten()
            # v = N_TILES - solution_index
            v = 1
            if solution_index == len(solution) - 1 :
                continue
            state = np.concatenate((np.expand_dims(grid, axis=0), tiles), axis=0)
            examples.append([state, pi, v])

            success, grid = SolutionChecker.place_element_on_grid_given_grid(
                solution_tile[:ORIENTATIONS], solution_tile[2], val=1, grid=grid, cols=width, rows=height
            )

        i += 1
    _examples = examples

    nnet.train(_examples)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    for example in _examples[:1]:
        prediction = nnet.predict(example[0])
        _prediction = np.reshape(prediction[0], (width, height))
        expected = np.reshape(example[1], (width, height))
        print('prediction')
        print(_prediction)
        print('expected')
        print(expected)
        print('grid state')
        print(example[0][0])
    la = []
