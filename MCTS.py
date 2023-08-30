import logging
import math
import torch
import numpy as np

from Game import Game
from utils import dotdict

torch.manual_seed(4)

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            #print(i, "mcts sim")
            self.search(canonicalBoard)

        s = canonicalBoard
        tupled_s = tuple([int(i) for i in s])
        counts = [self.Nsa[(tupled_s, a)] if (tupled_s, a) in self.Nsa else 0 for a in range(0, 180)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, num_of_sims_next_round=25):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = canonicalBoard
        game = Game.input_state_2p(s)
        next_steps, valids = game.next_states()
        if game.tiles.tiles_in_play() == 0:
            for player in game.players:
                player.score_round()
            if not any([i.board.game_ending() for i in game.players]):
                v = game.montecarlo(num_of_sims_next_round)
                v = sum([self.nnet.predict(i)[1] for i in v])/len(v)
                if s[-1] == 0:
                    return -v
                elif s[-1] == 1:
                    return v
                else:
                    raise Exception("Round has not ended.")
            else:
                for player in game.players:
                    player.score_game()
                if game.players[0].score > game.players[1].score:
                    return 1
                elif game.players[0].score == game.players[1].score:
                    return 0
                elif game.players[0].score < game.players[1].score:
                    return -1
                else:
                    raise Exception("Scoring Error!")

        def tuplify(ten):
            return tuple([int(i) for i in ten])

        tupled_s = tuplify(s)
        if tupled_s not in self.Ps:
            # leaf node
            self.Ps[tupled_s], v = self.nnet.predict(canonicalBoard)
            self.Ps[tupled_s] = torch.exp(self.Ps[tupled_s][0])
            self.Ps[tupled_s] = self.Ps[tupled_s] * valids  # masking invalid moves
            sum_Ps_s = self.Ps[tupled_s].sum()
            if sum_Ps_s > 0:
                self.Ps[tupled_s] = self.Ps[tupled_s].float() / sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[tupled_s] = self.Ps[tupled_s] + valids
                self.Ps[tupled_s] = self.Ps[tupled_s].float()/(self.Ps[tupled_s]).sum()
                self.Ps[tupled_s] = self.Ps[tupled_s].float()/(self.Ps[tupled_s]).sum()

            self.Vs[tupled_s] = valids
            self.Ns[tupled_s] = 0
            return -v[0,0]

        valids = self.Vs[tupled_s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(0, len(valids)):
            if valids[a]:
                if (tupled_s, a) in self.Qsa:
                    u = self.Qsa[(tupled_s, a)] + self.args.cpuct * self.Ps[tupled_s][a] * math.sqrt(self.Ns[tupled_s]) / (1 + self.Nsa[(tupled_s, a)])
                else:
                    u = self.args.cpuct * self.Ps[tupled_s][a] * math.sqrt(self.Ns[tupled_s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = next_steps[:, a]

        v = self.search(next_s)
        if (tupled_s, a) in self.Qsa:
            self.Qsa[(tupled_s, a)] = (self.Nsa[(tupled_s, a)] * self.Qsa[(tupled_s, a)] - v) / (self.Nsa[(tupled_s, a)] + 1)
            self.Nsa[(tupled_s, a)] += 1

        else:
            self.Qsa[(tupled_s, a)] = v
            self.Nsa[(tupled_s, a)] = 1

        self.Ns[tupled_s] += 1
        return -v


