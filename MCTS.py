import math
import numpy as np
# EPS = 1e-8
EPS = 0.1

from collections import OrderedDict

from data_generator import DataGenerator
from solution_checker import SolutionChecker

ORIENTATIONS = 2

def sort_key(x):
    return x.score

def render_to_dict(node, tree=None):
    if not node.children:
        return {}
    if tree is None:
        tree = OrderedDict([])

    if str(node) in tree:
        tree = tree[str(node)]
    else:
        tree[str(node)] = OrderedDict([])
        tree = tree[str(node)]
    for child in node.children:
        tree[str(child)] = render_to_dict(child, tree)

    return tree


class State(object):

    def __init__(self, board, tiles, parent=None):
        self.board = np.copy(board)
        self.tiles = tiles[:]
        self.parent = parent
        self.children = []
        self.score = None
        self.tile_placed = None


    def copy(self):
        return State(np.copy(self.board), self.tiles[:], parent=self.parent)

    def child_with_biggest_score(self):
        return sorted(self.children, key=sort_key, reverse=True)[0]

    def render_children(self):
        return {str(self): render_to_dict(self)}


    def __str__(self):
        return self.__repr__()

    def __repr__(self, short=False):
        if short:
            return f'{self.tiles}'

        output_list = [list(x) for x in self.tiles]
        if len(output_list) > 10:
            output_list = str(output_list[:10]) +  '(...)'

        output_board = ''
        if len(self.tiles) < 2:
            output_board = self.board
        return f'Remaining tiles:{len(self.tiles) / ORIENTATIONS}, tile placed: {self.tile_placed}, {output_list} sim. depth:({self.score}) {output_board}'

def get_cols(board):
    return board.shape[1]

def get_rows(board):
    return board.shape[0]

def eliminate_pair_tiles(tiles, index):
    '''
    search through the list to find rotated instance, 
    then remove both
    '''
    tile_to_remove = tiles[index]
    rotated_tile = (tile_to_remove[1], tile_to_remove[0])


    old_tiles = tiles[:]
    new_tiles = old_tiles[:index] + old_tiles[index + 1:]

    rotated_tile_index = new_tiles.index(rotated_tile)

    new_tiles = new_tiles[:rotated_tile_index] + new_tiles[rotated_tile_index + 1:]
    return new_tiles

def get_max_index(_list):
    max_index = 0
    max_value = -math.inf
    for i, el in enumerate(_list):
        if not el:
            continue
        if el.score > max_value:
            max_value = el.score
            max_index = i
    return max_index

class CustomMCTS():
    def __init__(self, tiles, board):
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

        self.initial_tiles, self.initial_board = tiles, board
        self.state = State(self.initial_board, self.initial_tiles)
        self.solution_checker = SolutionChecker(len(tiles), get_rows(board), get_cols(board))


    def predict(self, temp=1, N=3000):
        initial_state = self.state
        state = self.state
        available_tiles = state.tiles
        prev_state = state

        depth = 0
        while len(state.tiles):
            tile_placed = False
            states = []
            for i, tile in enumerate(state.tiles):
                val = 1
                new_board = np.copy(state.board)
                next_position = SolutionChecker.get_next_lfb_on_grid(new_board)
                if not next_position:
                    states.append(None)
                    continue

                success = SolutionChecker.place_element_on_grid_given_grid(
                    tile, SolutionChecker.get_next_lfb_on_grid(new_board),
                    val, new_board, get_cols(new_board), get_rows(new_board))

                if not success:
                    # cannot place the tile.  this branch will not be considered
                    states.append(None)
                    continue
                else:
                    tile_placed = True
                    new_tiles = eliminate_pair_tiles(state.tiles, i)
                    new_state = State(
                        board=new_board, tiles=new_tiles, parent=state)
                    state.children.append(new_state)
                    simulation_result = self.perform_simulations(new_state, N=N)
                    new_state.score = simulation_result
                    new_state.tile_placed = tile
                    states.append(new_state)
            if not tile_placed:
                # no tile was placed, it's a dead end; end game
                return initial_state

            depth += 1
            best_action = get_max_index(states) 
            prev_state = state
            new_state = states[best_action]

            state = new_state

        print('Solution found!')
        return initial_state
            

    def perform_simulations(self, state, N=3000):
        depths = []
        for n in range(N):
            depths.append(self.perform_simulation(state.copy()))
        _max = np.max(np.array(depths))
        return _max

    def perform_simulation(self, state):
        '''
        Given a state perform N simulations.
        One simulation consists of either filling container or having no more tiles to place.

        Returns average depth
        '''
        depth = 0
        if len(state.tiles) ==0:
            return 0
        while True:
            next_random_tile = np.random.randint(len(state.tiles))
            new_board = np.copy(state.board)
            next_position = SolutionChecker.get_next_lfb_on_grid(new_board)
            if not next_position:
                # cannot place next tile; return depth
                return depth
            success = SolutionChecker.place_element_on_grid_given_grid(
                state.tiles[next_random_tile], next_position, depth, new_board,
                get_cols(state.board), get_rows(state.board))
            if not success:
                # cannot place the tile. return depth reached
                return depth
            else:
                new_tiles = eliminate_pair_tiles(state.tiles, next_random_tile)
                state = State(board=new_board, tiles=new_tiles, parent=state)
                if len(new_tiles) == 0:
                    # no more tiles. return depth + 1
                    return depth + 1
            depth += 1

        return depth


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s, state = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        # IGOR changed this behaviour

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        _s, state = self.game.stringRepresentation(canonicalBoard)

        if _s not in self.Es:
            self.Es[_s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[_s]!=0:
            # terminal node
            return self.Es[_s]

        if _s not in self.Ps:
            # leaf node
            self.Ps[_s], v = self.nnet.predict(canonicalBoard[0])
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if valids.all():
                print(self.Ps[_s])
            self.Ps[_s] = self.Ps[_s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[_s])
            if sum_Ps_s > 0:
                self.Ps[_s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[_s] = self.Ps[_s] + valids
                self.Ps[_s] /= np.sum(self.Ps[_s])

            self.Vs[_s] = valids
            self.Ns[_s] = 0
            return v

        valids = self.Vs[_s]

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (_s,a) in self.Qsa:
                    u = self.Qsa[(_s,a)] + self.args.cpuct*self.Ps[_s][a]*math.sqrt(self.Ns[_s])/(1+self.Nsa[(_s,a)])
                    #if not state.state[0].any():
                    #    print(self.Qsa[(_s, a)], u, 'first', cur_best, u > cur_best, self.Nsa[(_s, a)])
                else:
                    # i Added + 100 - optimistic initialization??
                    u = self.args.cpuct*self.Ps[_s][a]*math.sqrt(self.Ns[_s] + EPS) + 100     # Q = 0 ?
                    #if not state.state[0].any():
                    #    print(u, 'second')

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player, next_vis_state = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (_s,a) in self.Qsa:
            self.Qsa[(_s,a)] = (self.Nsa[(_s,a)]*self.Qsa[(_s,a)] + v)/(self.Nsa[(_s,a)]+1)
            self.Nsa[(_s,a)] += 1

        else:
            self.Qsa[(_s,a)] = v
            self.Nsa[(_s,a)] = 1

        self.Ns[_s] += 1
        return v
