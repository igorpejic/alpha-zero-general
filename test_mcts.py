import unittest
from MCTS import CustomMCTS
from data_generator import DataGenerator

class TestCustomMCTS(unittest.TestCase):

    def test_initialize_state(self):
        w = 20
        h = 20
        n = 20
        dg = DataGenerator(w, h)
        tiles, board = dg.gen_instance_and_board(n, w, h)
        custom_mcts = CustomMCTS(tiles, board)

        state = custom_mcts.predict()
