import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def playGame(self, player, verbose=True):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        curPlayer = 1
        state = self.game.getInitBoard()
        it = 0
        game_ended = self.game.getGameEnded(state, curPlayer)
        while game_ended==0:
            it+=1
            action = player(self.game.getCanonicalForm(state, curPlayer))

            np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)}, linewidth=115)

            state = self.game.getCanonicalForm(state, curPlayer)
            valids = self.game.getValidMoves(state.board, state.tiles, 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            state, curPlayer = self.game.getNextState(
                state, action, curPlayer)

            if verbose:
                print(action, state.tiles[action])
                print(state.board)
                print(state.tiles)
            game_ended = self.game.getGameEnded(state, curPlayer)

        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(state, 1)))

        return game_ended

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)
        oneWon = 0
        twoWon = 0
        draws = 0
        results_player_1 = []
        results_player_2 = []
        for _ in range(num):
            game_result_player_1 = self.playGame(self.player1, verbose=verbose)
            results_player_1.append(game_result_player_1)
            game_result_player_2 = self.playGame(self.player2, verbose=verbose)
            results_player_2.append(game_result_player_2)
            if game_result_player_1 > game_result_player_2:
                oneWon += 1
            elif game_result_player_1 < game_result_player_2:
                twoWon += 1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            # bar.next()

        print(f'Results player Prev: {results_player_1}')
        print(f'Results player New: {results_player_2}')
        bar.finish()

        return oneWon, twoWon, draws
