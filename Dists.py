import numpy as np
from typing import List, Tuple


class Dist:
    """
    This class generates the boards at different stages of the game based on two parameters - p and q.
    """
    def __init__(self, p, q, tiles_in_play: List[tuple]):
        self.p = p
        self.q = q
        self.tip = tiles_in_play

    def pmf(self, vector):
        out = 0
        board_1 = [int(i) for i in vector[31:56]]
        board_2 = [int(i) for i in vector[68:93]]
        for i in (board_1 + board_2):
            if i:
                out += np.log(self.p)
            else:
                out += np.log(1-self.p)
        return out

    def sample_tiles(self):

        tile_dist = [sum([j[1] for j in self.tip[0:i]]) for i in range(1, len(self.tip)+1)]
        val = np.random.random()
        for i, j in enumerate(tile_dist):
            if val>j:
                break
        total_tiles = self.tip[i][0]
        tiles = np.random.randint(0, total_tiles, (4,))
        tiles = [i for i in tiles]
        tiles.sort()
        tiles = [0] + tiles + [total_tiles]
        tiles = [i-j for i, j in zip(tiles[1:], tiles[0:-1])]
        return tiles

    def sample_boards(self):

        board_1 = np.random.binomial(1, self.p, (5, 5))
        board_2 = np.random.binomial(1, self.p, (5, 5))
        while 5 in board_1.sum(1):
            board_1 = np.random.binomial(1, self.p, (5, 5))
        while 5 in board_2.sum(1):
            board_2 = np.random.binomial(1, self.p, (5, 5))
        return board_1, board_2

    def sample_rows(self):
        board_1, board_2 = self.sample_boards()

        def gen_row(board):
            rows = []
            for i, single_row in enumerate(board):
                vals = np.where(single_row == 0)
                colors = [j for j in range(5-i, 5)] + [j for j in range(0, 5-i)]
                colors = np.array(colors)
                colors = colors[vals]
                index = np.random.randint(0, len(colors))
                color = colors[index]
                amount = np.random.binomial(i, self.q)
                if amount == 0:
                    color = -1
                rows.append((color, amount))

            return rows

        rows_1, rows_2 = gen_row(board_1), gen_row(board_2)
        return rows_1, rows_2, board_1, board_2

    def sample(self):
        tiles = self.sample_tiles()
        tiles_in_play = tiles
        rows_1, rows_2, board_1, board_2 = self.sample_rows()
        out = np.zeros((112, ))

        tiles = sum([j*[i] for i, j in enumerate(tiles)], [])
        tiles = np.random.permutation(tiles)
        holders = []
        for i in range(0, 5):
            holders.append(tiles[4*i:4*(i+1)])
            for j in tiles[4*i: 4*(i+1)]:
                tiles_in_play[j] -= 1

        holders = [[len(np.where(i==j)[0]) for j in range(0, 5)] for i in holders]
        holders = sum(holders, [])
        out[0:25] = holders
        out[30] = 1  # -1 in the middle

        board_1, board_2 = board_1.flatten(), board_2.flatten()
        out[31:56] = board_1
        out[68:93] = board_2

        rows_1, rows_2 = sum((list(i) for i in rows_1), []), sum((list(i) for i in rows_2), [])
        out[56:66] = rows_1
        out[93:103] = rows_2

        scores = np.random.normal(self.p*60, 2, (2, ))
        scores = [round(i) for i in scores]
        out[67], out[104] = scores[0], scores[1]

        out[110] = 0

        for j in range(0, 5):
            out[105+j] = tiles_in_play[j]

        return out



