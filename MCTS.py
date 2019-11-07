import math
import numpy as np
# EPS = 1e-8
EPS = 0.1

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
            self.Ps[_s], v = self.nnet.predict(canonicalBoard)
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
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
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
