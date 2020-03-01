from Coach import Coach
# from othello.pytorch.NNet import NNetWrapper as nn

# from othello.OthelloGame import OthelloGame as Game
# from othello.tensorflow.NNet import NNetWrapper as nn
# from binpack.tensorflow.NNet import NNetWrapper as nn
# from binpack.tensorflow.NNet import NNetWrapper as nn
from alpha.binpack.keras.NNet import NNetWrapper as nn
from binpack.BinPackGame import BinPackGame as Game
from utils import *

args = dotdict({
    'numIters': 20,
    'numEps': 9,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 40,
    'arenaCompare': 4,
    'cpuct': 1.5,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 40,

})

if __name__=="__main__":
    N_TILES = 10
    HEIGHT = 8
    WIDTH = 8
    g = Game(HEIGHT, WIDTH, N_TILES)

    nnet = nn(g, predict_move_index=True, scalar_tiles=True, predict_v=True)

    if args.load_model and False:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
