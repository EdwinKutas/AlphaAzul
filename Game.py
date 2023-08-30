import torch
from torch import tensor
from numpy.random import permutation
from abc import ABCMeta, abstractmethod
import numpy as np


class TileException(ValueError):
    pass


class DefaultDictInt(dict):

    def __getitem__(self, item):
        if item in self.keys():
            return super().__getitem__(item)
        return 0


class Game:

    def __init__(self, num_of_players, tiles=None, players=None):
        self.num_of_players = num_of_players
        self.x = 0
        if tiles is None:
            self.tiles = TotalTiles(self.num_of_players)
        else:
            self.tiles = tiles

        if players is None:
            self.players = [Player() for _ in range(0, self.num_of_players)]
            self.players[0].is_first = True
            for i in range(0, self.num_of_players):
                self.players[i].id = i
        else:
            self.players = players

    def play_round(self):
        while True: # Checks if input actually works with board state
            row, color, location = self.players[0].choose()
            while True: # Checks if there is a sensible input
                if location > len(self.tiles.factories):
                    print("Incorrect location")
                    row, color, location = self.players[0].choose()
                elif (row < len(self.players[0].board.grid)
                      and self.players[0].board.rows[row][1] is not None
                      and self.players[0].board.rows[row][1] != color):
                    print("Row already has a different color")
                    row, color, location = self.players[0].choose()
                else:
                    break

            if location == len(self.tiles.factories):
                try:
                    self.take(self.tiles.middle, color, self.players[0], row)
                    break
                except TileException:
                    print("Trying to take -1 tile or tile not in tile holder")
            else:
                try:
                    self.take(self.tiles.factories[location], color, self.players[0], row)
                    break
                except TileException:
                    print("Tile not in tile holder")
        self.players = self.players[1:] + [self.players[0]]

    def play_whole_game(self):

        while not any(player.board.game_ending() for player in self.players):
            tiles = self.tiles
            tiles.populate()

            for i in range(0, len(self.players)):
                if self.players[i].is_first:
                    self.players = self.players[i:] + self.players[0:i]
                    break
            print("NEW ROUND")
            while tiles.tiles_in_play() != 0:

                for j, fac in enumerate(tiles.factories):
                    print(f"{j}'th factory is {fac}")
                print("Mid", tiles.middle)
                print("TIP", tiles.tiles_in_play())

                input_val = input("print board states? 2: All boards, 1: Your board")
                if input_val == "2":
                    for player in self.players:
                        print(player.board)
                elif input_val == "1":
                    print(self.players[i].board)

                self.play_round()

            for player in self.players:
                print("Scoring!")
                for row_length, row in enumerate(player.board.rows, start=1):
                    if row[0] == row_length:
                        self.tiles.discard += [[row[1]]*row_length]
                player.score_round()

        for player in self.players:
            player.score_game()
        self.players.sort(key=lambda x: -x.score)
        return self.players

    def take(self, tileholder, value: int, player, position: int):
        if value not in tileholder.keys() or tileholder[value] == 0:
            raise TileException("Need at least one tile")
        if value == -1:
            raise TileException("Cannot take the -1 tile")

        if (position < len(self.players[0].board.grid)
                and self.players[0].board.rows[position][1] is not None
                and self.players[0].board.rows[position][1] != value):
            raise TileException("This row already has a color")

        if (position < len(self.players[0].board.grid)
                and self.players[0].board.rows[position][0] == position + 1):
            raise TileException("This row is already full")

        if position < len(self.players[0].board.grid):
            grid_pos = self.players[0].board.grid[position]
            grid_pos = [i for i in grid_pos if i[0] == value][0]
            if grid_pos[1]:
                raise TileException("This row has already been filled.")

        if tileholder[-1] != 0:
            tileholder[-1] = 0
            player.is_first = True
            player.board.neg_row += 1

        if position < len(player.board.grid) and player.board.rows[position][1] is None:
            player.board.rows[position][1] = value

        if position < len(player.board.grid):
            current = player.board.rows[position][0]
            player.board.rows[position][0] = min(position + 1, current + tileholder[value])
            player.board.rows[position][1] = value
            player.board.neg_row += max(0, tileholder[value] - (position + 1 - current))
        else:
            player.board.neg_row += tileholder[value]

        if isinstance(tileholder, Factory):
            for i in tileholder.keys():
                if i != value:
                    self.tiles.middle[i] += tileholder[i]
                    tileholder[i] = 0
                else:
                    tileholder[i] = 0
        elif isinstance(tileholder, Middle):
            tileholder.pop(value)
        else:
            raise Exception("State actions for taking from non middle or factory.")

        return None

    def montecarlo(self, number_of_sims):
        out = []
        for i in range(0, number_of_sims):
            new_game = Game.input_state_2p(self.output_state_2p())
            new_game.tiles.populate()
            out.append(new_game.output_state_2p())
        return out

    def output_state_2p(self):
        """
        0-24 - factories
        25-30 middle, includes -1
        31-55 board player 1
        56-65 rows player 1, stored as color then amount
        66 -1 in current round player 1
        67 points player 1
        68-92 board player 2
        93-102 rows, player 2
        103 -1 in current round player 2
        104 points player 2
        105-109 tiles in play
        110 who is first, if -1 indicates it is undecided.
        111 is player id
        """
        out = torch.zeros((112, ), requires_grad=False)
        for i, factory in enumerate(self.tiles.factories):
            for j in range(0, 5):
                out[5*i+ j] = factory[j]

        for i in range(0, 5):
            out[25+i] = self.tiles.middle[i]
        out[30] = self.tiles.middle[-1]

        x = 31
        for player in self.players:
            for row in player.board.grid:
                for column in row:
                    out[x] = 1 if column[1] else 0
                    x += 1
            for amount, color in player.board.rows:
                if color is None:
                    out[x] = -1
                    out[x+1] = -1
                else:
                    out[x] = color
                    out[x+1] = amount
                x += 2
            out[x] = player.board.neg_row
            out[x+1] = player.score
            x += 2
        for i in range(0, 5):
            out[105 + i] = 20 - self.tiles.discard.count(i)

        out[110] = -1
        for i, player in enumerate(self.players):
            if player.is_first:
                out[110] = i

        out[111] = self.players[0].id
        return tensor(out, dtype=torch.float32, requires_grad=False)

    @classmethod
    def input_state_2p(cls, input_tensor):
        """
        Takes in an input state and outputs the class corresponding to it. This is the inverse function
        to output state.
        """
        tiles = TotalTiles.from_tensor(input_tensor[0:31])
        players = []
        for p_i in range(0, 2):
            grid_vals = []
            for x in range(0, 5):
                for y in range(0, 5):
                    if int(input_tensor[31 + 37*p_i + 5*x + y]) != 0:
                        grid_vals.append((x, y))
            rows = []
            for row in range(0, 5):
                if int(input_tensor[56+37*p_i + 2*row + 1]) not in [0, -1]:
                    rows.append((row,
                                 int(input_tensor[56+37*p_i + 2*row]),
                                 int(input_tensor[56+37*p_i + 2*row + 1])))

            player = Player()
            player.board = Board(val=5, row_values=rows, grid_values=grid_vals)
            player.board.neg_row = int(input_tensor[66+ 37*p_i])
            player.score = int(input_tensor[67+37*p_i])
            players.append(player)
        for i in range(0, 5):
            tiles.discard += [i]*max(20-int(input_tensor[104+i]), 0)

        if input_tensor[110] != -1:
            players[int(input_tensor[110])].is_first = True

        val = int(input_tensor[111])
        for i, player in enumerate(players):
            player.id = (val + i) % 2
        game = Game(num_of_players=2, tiles=tiles, players=players)
        return game


    def next_states(self, noise=0):
        """
        :return: A (112x180) tensor and a (180) dimensional tensor.
        The 180 dimensional tensor indicates which moves are valid. With noise=0 this will be
        zeroes and ones. With noise=1 this will be all 0.5.
        The 112x180 dimensional tensor indicates the positions on the board after each possible move.
        If a move is not valid, the column will be zero.
        """

        game_state = self.output_state_2p()
        out_states = torch.zeros((112, 180), requires_grad=False, dtype=torch.float32)
        out_valid = torch.zeros((180, ), requires_grad=False, dtype=torch.float32) # This may need to be changed to float
        for i in range(0, 180):
            new_game = Game.input_state_2p(game_state)
            color = i//36
            factory_num = (i - 36*color)//6
            row = (i - 36*color - factory_num*6)
            factory = (new_game.tiles.factories[factory_num] if factory_num < len(new_game.tiles.factories)
                       else new_game.tiles.middle)
            try:
                new_game.take(factory, color, new_game.players[0], row)
                new_game.players = new_game.players[::-1]
                vec = new_game.output_state_2p()
                vec[111] = (vec[111] + 1) % 2
                valid = 1
            except:
                vec = torch.zeros((112))
                valid = 0
            out_states[:, i] = vec
            out_valid[i] = valid
        return out_states, out_valid


class Player:

    def __init__(self):
        self.board = Board()
        self.turns = []
        self.is_first = False
        self.score = 0
        self.id = None

    def choose(self):

        def dummy_inputer(s):
            while True:
                try:
                    out = int(input(s))
                    if out in [i for i in range(0, 5)]:
                        return out
                except ValueError:
                    print("Incorrect Value!")
                    pass

        location = dummy_inputer("Location")
        color = dummy_inputer("Color")
        row = dummy_inputer("Row")
        return row, color, location

    def score_round(self):
        self.score += self.board.score_round()
        return None

    def score_game(self):

        self.score += self.board.score_rows()
        self.score += self.board.score_columns()
        self.score += self.board.score_colors()
        return None


class TotalTiles:

    def __init__(self, num_of_players, val=5, factory_size=4, amount_of_tiles=20):
        self.num_of_players = num_of_players
        self.val = val
        self.factory_size = factory_size
        self._factories = [Factory() for i in range(0, 2 * num_of_players + 1)]
        self.discard = []
        self.tiles_bag = sum([[i] * amount_of_tiles for i in range(0, self.val)], [])
        self.middle = Middle()
        self.middle[-1] = 1

    @property
    def factories(self):
        return [i for i in self._factories if i.size() != 0]

    def tiles_in_play(self):
        return sum(i.size() for i in self.factories) + self.middle.size()

    def populate(self):
        tiles_bag = list(permutation(self.tiles_bag))
        for factory in self._factories:
            if len(tiles_bag) < self.factory_size:
                tiles_bag += self.discard
                tiles_bag = list(permutation(tiles_bag))

            factory.populate(tiles_bag[0:self.factory_size])
            tiles_bag = tiles_bag[self.factory_size:]
        self.factories.sort(key=lambda x: x.sorting_key())
        self.tiles_bag = tiles_bag

    @classmethod
    def from_tensor(cls, tensor_input):
        tiles = TotalTiles(2)
        factories = [Factory() for _ in range(0, 5)]
        for i in range(0, 25):
            factories[i//5][i % 5] = int(tensor_input[i])
        factories.sort(key=lambda x: x.sorting_key())

        i += 1
        for j in range(0, 5):
            tiles.middle[j] = int(tensor_input[i])
            i += 1
        tiles._factories = factories
        tiles.middle[-1] = int(tensor_input[i])
        return tiles


class TileHolderAbstract(DefaultDictInt, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def size(self):
        return sum(self[i] for i in self.keys())


class Factory(TileHolderAbstract):

    def __init__(self):
        pass

    def populate(self, tiles):
        for i in set(tiles):
            self[i] = tiles.count(i)

    def size(self):
        return super().size()

    def sorting_key(self):
        # This only works for games with four tiles
        out = 1
        for i, p in zip(range(0, 5), [1, 2, 3, 5, 7]):
            out *= p**self[i]
        return out


class Middle(TileHolderAbstract):

    def __init__(self):
        pass

    def size(self):
        return super().size()


class Board:

    def __init__(self, val=5, row_values=[], grid_values=[]):

        def cycle(li, i):
            return li[-i:] + li[0:-i]

        self.grid = [cycle([(i, False) for i in range(0, val)], j) for j in range(0, val)]
        self.rows = [[0, None] for _ in range(0, val)]
        self.neg_row = 0
        self.moves = []

        for row, color, amount in row_values:
            if amount > row + 1:
                raise Exception(f"Cannot put {amount} tiles into a row with {row+1} tiles.")
            self.rows[row] = [amount, color]
        for row, column in grid_values:
            self.grid[row][column] = (self.grid[row][column][0], True)

    def score_round(self):
        """
        :param move: Tuple (i, j) coordinate
        :return: Updates the score and resets the board
        """

        def score_li(li, i):
            out = 1
            j = i+1
            while j < len(li):
                if li[j]:
                    out += 1
                else:
                    break
                j += 1

            j = i-1
            while j >= 0:
                if li[j]:
                    out += 1
                else:
                    break
                j += -1

            return out

        out = 0
        for row, val in enumerate(self.rows):
            # Check this function is not scoring partially completed rows.
            amount, color = val
            if amount == 0:
                continue
            column = (row+color) % len(self.rows)
            if amount == row+1:
                self.grid[row][column] = (color, True)
                column_indicators = [i[column][1] for i in self.grid]
                col_score = score_li(column_indicators, row)
                row_indicators = [i[1] for i in self.grid[row]]
                row_score = score_li(row_indicators, column)

                if col_score == 1 and row_score == 1:
                    out += 1
                elif col_score == 1 or row_score == 1:
                    out += col_score + row_score - 1
                else:
                    out += col_score + row_score

                self.rows[row] = [0, None]

        out -= (min(self.neg_row, 2)
                       + min(max(self.neg_row - 2, 0), 2)*2
                       + max(self.neg_row - 4, 0)*3)
        self.neg_row = 0
        return out

    def game_ending(self):
        for row in self.grid:
            row = [i[1] for i in row]
            if all(row):
                return True
        return False

    def score_rows(self):
        out = 0
        for row in self.grid:
            li = [i[1] for i in row]
            if all(li):
                out += 2
        return out

    def score_columns(self):
        out = 0
        for i in range(0, len(self.grid)):
            li = [row[i] for row in self.grid]
            li = [j[1] for j in li]
            if all(li):
                out += 7
        return out

    def score_colors(self):
        out = 0
        value = len(self.grid)
        for i in range(0, value):
            for j in range(0, value):
                if not self.grid[j][(i+j) % value][1]:
                    break
            else:
                out += 10
        return out

    def __repr__(self):
        out = ""
        for i, row in enumerate(self.rows):
            out += f"row {i}: {self.rows[i]} \n"
        out += "Board: \n"
        for i in self.grid:
            out += f"{i} \n"
        out += f"Negative Row: {self.neg_row} \n"
        return out


class Arena:

    def __init__(self, f, g, initial_dist, dists_evaluators=None, threshold=0):
        self.game = None
        self.function_1 = f
        self.function_2 = g
        self.initial_dist = initial_dist

        if dists_evaluators is not None:
            self.dists = [self.initial_dist] + [i[0] for i in dists_evaluators]
            self.evaluators = [g] + [i[1] for i in dists_evaluators]
        else:
            self.dists = [self.initial_dist]
            self.evaluators = [g]
        self.threshold = threshold

    def play_games(self, n):
        out = 0
        i = 0
        while i < n:
            for boolean_indexer in range(0, 2):
                funcs = [self.function_1, self.function_2]
                self.game = Game.input_state_2p(self.initial_dist.sample())
                while self.game.tiles.tiles_in_play() != 0:
                    board = self.game.output_state_2p()
                    action = funcs[boolean_indexer](board)
                    action = int(action)
                    board = self.game.next_states()[0][:, action]
                    self.game = Game.input_state_2p(board)
                    boolean_indexer += 1
                    boolean_indexer %= 2

                for player in self.game.players:
                    player.score_round()

                if any([player.board.game_ending() for player in self.game.players]):
                    for player in self.game.players:
                        player.score_game()
                    players = [player for player in self.game.players]
                    players.sort(key=lambda x: -x.score)
                    i += 1
                    out += (-1) ** (players[0].id)
                    print(f"Game ended! Winner={(players[0].id)}")

                elif self.dists is not None and self.evaluators is not None:
                    #  Checks if we have a training set 'nearby'.
                    dists = [dist for dist in self.dists if dist.pmf(self.game.output_state_2p()) >= self.threshold]
                    if len(dists) == 0:
                        print("Failed round, no nearby distributions - Be very suspicious!")
                    else:
                        dists = [i for i in enumerate(dists)]
                        dists.sort(key=lambda x: -x[1].pmf(self.game.output_state_2p())) # Be careful about minus signs
                        dists = dists[0:min(3, len(dists))]
                        evaluators = [self.evaluators[j[0]] for j in dists]
                        dists = [np.exp(i[1].pmf(self.game.output_state_2p())) for i in dists]
                        sims = self.game.montecarlo(25)
                        value = 0
                        for state in sims:
                            value += sum(ev(state)*j**2 for ev, j in zip(evaluators, dists))
                        value /= sum(j**2 for j in dists)
                        value /= 25
                        if self.game.players[0].is_first:
                            value *= (-1) ** self.game.players[0].id
                        else:
                            value *= (-1) ** self.game.players[1].id
                        out += value
                        i += 1
        return out/n






