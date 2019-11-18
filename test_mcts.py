import unittest
from MCTS import CustomMCTS, State, render_to_dict
from data_generator import DataGenerator

class TestCustomMCTS(unittest.TestCase):

    def test_initialize_state(self):
        w = 20
        h = 20
        n = 20
        N = 1
        dg = DataGenerator(w, h)
        tiles, board = dg.gen_tiles_and_board(n, w, h)
        custom_mcts = CustomMCTS(tiles, board)

        state = custom_mcts.predict(N=N)

    def test_render_to_dict(self):
        w = 1
        h = 1
        n = 1
        dg = DataGenerator(w, h)
        tiles, board = dg.gen_tiles_and_board(n, w, h)

        # do not rely on tiles here, they are of wrong sizes
        state = State(board, [[1, 1]])
        child_state_1 = State(board, [[2, 3]], parent=state)
        state.children.append(child_state_1)

        child_state_1_1 = State(board, [[4, 5]], parent=child_state_1)
        child_state_1.children.append(child_state_1_1)

        child_state_1_2 = State(board, [[4, 6]], parent=child_state_1)
        child_state_1.children.append(child_state_1_2)

        child_state_1_1_1 = State(board, [[5, 6]], parent=child_state_1_1)
        child_state_1_1.children.append(child_state_1_1_1)


        ret = state.render_children()

        self.assertEqual(len(ret.keys()), 1)
        self.assertEqual(len(ret[str(state)].keys()), 1)
        self.assertEqual(str(child_state_1) in ret[str(state)], True)
        self.assertEqual(str(child_state_1_1) in ret[str(state)][str(child_state_1)], True)

        ret_child_1 = ret[str(state)][str(child_state_1)]
        self.assertEqual(str(child_state_1_1) in ret_child_1.keys(), True)

        self.assertEqual(str(child_state_1_2) in ret_child_1.keys(), True)

        ret_child_1_1 = ret[str(state)][str(child_state_1)][str(child_state_1_1)]
        self.assertEqual(str(child_state_1_1_1) in ret_child_1_1.keys(), True)
        self.assertEqual(ret_child_1_1[str(child_state_1_1_1)], child_state_1_1_1)
