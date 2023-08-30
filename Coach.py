import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from Game import Game, Arena
from NeuralNets import AzulNet, NNetWrapper
from Dists import Dist

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, nnet, args, dist):
        self.game = Game(2)
        self.nnet = nnet
        self.pnet = self.nnet.__class__(100)  # the competitor network
        self.dist = dist
        self.args = args
        self.mcts = MCTS(self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        self.curPlayer = 1
        episodeStep = 0
        initial_state = self.dist.sample()
        game = Game.input_state_2p(initial_state)

        while True:
            episodeStep += 1
            canonicalBoard = game.output_state_2p()
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            trainExamples.append([canonicalBoard, pi, None])

            action = np.random.choice(len(pi), p=pi)
            board = (game.next_states()[0])[:, action]
            game = Game.input_state_2p(board)

            round_ended = (game.tiles.tiles_in_play() == 0)

            if round_ended:
                for player in self.game.players:
                    player.board.score_round()

                game_ended =  any([player.board.game_ending() for player in game.players])

                if game_ended:
                    for player in game.players:
                        player.score_game()
                    players = [player for player in game.players]
                    players.sort(key=lambda x: -x.score)

                    return [(x[0], x[1], ((-1) ** (players[0].id == 0 + i))) for i, x in enumerate(trainExamples[::-1])]

                v = self.game.montecarlo(25)
                v = sum([self.nnet.predict(i)[1] for i in v]) / len(v)
                for player in self.game.players:
                    if player.is_first:
                        player = player
                        break

                return [(x[0], x[1], v * ((-1) ** (player.id + i)))
                        for i, x in enumerate(trainExamples[::-1])]


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            #self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            #self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            #self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')

            score = 0
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=1)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=1)),  initial_dist=self.dist, )
            score = arena.play_games(self.args.arenaCompare)

            print(score)

            #log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (score))
            if score < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                #self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print("Success!!!!!!")
                print("Success!!!!!!")
                log.info('ACCEPTING NEW MODEL')
                #self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                #self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


