import unittest
from MCTS import CustomMCTS
from data_generator import DataGenerator

from asciitree import LeftAligned
from collections import OrderedDict as OD

def run_mcts():
    #w = 20
    #h = 20
    #n = 7
    n = 6
    w = 10
    h = 10

    N_simulations = 7

    dg = DataGenerator(w, h)
    tiles, board = dg.gen_tiles_and_board(n, w, h, order_tiles=True, from_file=False)
    print(f'Starting problem with width {w}, height {h} and {n} tiles')
    print(f'TILES: {tiles}')
    print(f'Performing: {N_simulations} simulations per possible tile-action')

    custom_mcts = CustomMCTS(tiles, board)

    ret = custom_mcts.predict(N=N_simulations)
    child = ret.children[0]
    tree = ret.render_children()

    tr = LeftAligned()
    print(tr(tree))

    return ret

if __name__=='__main__':
    state = run_mcts()
