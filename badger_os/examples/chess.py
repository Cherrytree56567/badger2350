'''
MicroPython Sunfish Chess Engine for Badger 2040 with Persistence
- Ported by: Quan Lin
- Modified by: Jerzy Głowacki
- Persistence modifications by: Your Name
- License: GNU GPL v3
'''

import re
import time
import gc
import badger2040
import ujson
from collections import namedtuple
from micropython import const
import micropython

# Device screen dimensions.
WIDTH = badger2040.WIDTH
HEIGHT = badger2040.HEIGHT

badger2040.system_speed(badger2040.SYSTEM_FAST)

# Pieces sprite data.
pieces = bytearray(
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x80\x01\x80\x01\x80\x00\x00'
    b'\x00\x00\x00\x00\x01\x80\x01\x80\x01\x80\x00\x00\x00\x00\x00\x00'
    b'\x03\xc0\x0d\xb0\x03\xc0\x07\x80\x1d\xb8\x01\x80\x03\xc0\x0d\xb0'
    b'\x03\xc0\x07\x80\x1d\xb8\x01\x80\x02\x40\x6d\xb6\x02\x40\x0c\xe0'
    b'\x17\xe8\x02\x40\x03\xc0\x6d\xb6\x03\xc0\x0f\xe0\x1f\xf8\x03\xc0'
    b'\x02\x40\x6d\xb6\x06\x60\x18\x30\x10\x08\x02\x40\x03\xc0\x6d\xb6'
    b'\x07\xe0\x1f\x90\x1f\xf8\x03\xc0\x3e\x7c\x6d\xb6\x0c\x30\x14\x18'
    b'\x18\x18\x06\x60\x3f\xfc\x6d\xb6\x0f\xf0\x1b\xe8\x1f\xf8\x07\xe0'
    b'\x22\x44\x32\x4c\x09\x90\x30\x08\x08\x10\x04\x20\x23\xc4\x3f\xfc'
    b'\x0e\x70\x3f\xe8\x0f\xf0\x07\xe0\x41\x82\x20\x04\x08\x10\x21\x0c'
    b'\x08\x10\x04\x20\x5d\xba\x3f\xfc\x0f\xf0\x3e\xf4\x0f\xf0\x07\xe0'
    b'\x40\x02\x20\x04\x0c\x30\x42\x04\x08\x10\x04\x20\x5e\x7a\x3f\xfc'
    b'\x0f\xf0\x7d\xf4\x0f\xf0\x07\xe0\x30\x0c\x30\x0c\x04\x20\x4c\x04'
    b'\x08\x10\x0c\x30\x2f\xf4\x3f\xfc\x07\xe0\x7b\xf4\x0f\xf0\x0f\xf0'
    b'\x1c\x38\x1f\xf8\x07\xe0\x34\x04\x08\x10\x08\x10\x13\xc8\x10\x08'
    b'\x04\x20\x37\xf4\x0f\xf0\x0f\xf0\x16\x68\x10\x08\x04\x20\x04\x04'
    b'\x08\x10\x18\x18\x19\x98\x1f\xf8\x07\xe0\x07\xfc\x0f\xf0\x1f\xf8'
    b'\x13\xc8\x1f\xf8\x3f\xfc\x07\xfc\x1f\xf8\x10\x08\x1c\x38\x10\x08'
    b'\x3c\x3c\x04\x04\x18\x18\x1f\xf8\x18\x18\x10\x08\x40\x02\x04\x04'
    b'\x20\x04\x10\x08\x1f\xf8\x1f\xf8\x7f\xfe\x07\xfc\x3f\xfc\x1f\xf8'
    b'\x0f\xf0\x0f\xf0\x3f\xfc\x07\xfc\x3f\xfc\x1f\xf8\x0f\xf0\x0f\xf0'
    b'\x3f\xfc\x07\xfc\x3f\xfc\x1f\xf8\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
)

# Hourglass sprite data.
hourglass = bytearray(
    b'\x0f\xff\xff\xf0\x0f\xff\xff\xf0\x03\x00\x00\xc0\x03\x00\x00\xc0'
    b'\x03\x00\x00\xc0\x03\x00\x00\xc0\x01\xff\xff\x80\x01\xff\xff\x80'
    b'\x01\xff\xff\x80\x00\xff\xff\x00\x00\x7f\xfe\x00\x00\x3f\xfc\x00'
    b'\x00\x1f\xf8\x00\x00\x0f\xf0\x00\x00\x07\xe0\x00\x00\x07\xe0\x00'
    b'\x00\x07\xe0\x00\x00\x07\xe0\x00\x00\x0d\xb0\x00\x00\x19\x98\x00'
    b'\x00\x31\x8c\x00\x00\x61\x86\x00\x00\xc1\x83\x00\x01\xc3\xc3\x80'
    b'\x01\x9f\xf9\x80\x01\xff\xff\x80\x03\xff\xff\xc0\x03\xff\xff\xc0'
    b'\x03\xff\xff\xc0\x03\xff\xff\xc0\x0f\xff\xff\xf0\x0f\xff\xff\xf0'
)

###############################################################################
# Chess engine core: definitions, piece-square tables, and move generation
###############################################################################

@micropython.native
def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step

@micropython.native
def reverse(s):
    return ''.join(reversed(s))

@micropython.native
def swapcase(s):
    return ''.join(c.lower() if c.isupper() else c.upper() for c in s)

piece = {'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000}

pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

def padrow(row, k):
    return (0,) + tuple(x+piece[k] for x in row) + (0,)

# Pad tables and join piece and pst dictionaries.
for k, table in pst.items():
    pst[k] = sum((padrow(table[i*8:i*8+8], k) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

# Global board constants and initial board.
A1, H1, A8, H8 = const(91), const(98), const(21), const(28)
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# Directions for moves.
N, E, S, W = const(-10), const(1), const(10), const(-1)
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']
TABLE_SIZE = const(500)
QS_LIMIT = const(219)
EVAL_ROUGHNESS = const(13)
DRAW_TEST = True
TIME_LIMIT = const(5000)

# Position class defines a state of a chess game.
class Position(namedtuple('Position', 'board score wc bc ep kp')):
    '''
    board -- a 120 character representation of the board
    score -- the board evaluation
    wc -- castling rights (tuple: queen- and king-side for white)
    bc -- castling rights for black (tuple)
    ep -- en passant square
    kp -- king passant square
    '''
    __slots__ = ()

    @micropython.native
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    if q.isspace() or q.isupper():
                        break
                    if p == 'P' and d in (N, N+N) and q != '.':
                        break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'):
                        break
                    if p == 'P' and d in (N+W, N+E) and q == '.' and j not in (self.ep, self.kp, self.kp-1, self.kp+1):
                        break
                    yield (i, j)
                    if p in 'PNK' or q.islower():
                        break
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]:
                        yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]:
                        yield (j+W, j+E)

    @micropython.native
    def rotate(self):
        return Position(swapcase(reverse(self.board)), -self.score, self.bc, self.wc, 119-self.ep if self.ep else 0, 119-self.kp if self.kp else 0)

    @micropython.native
    def nullmove(self):
        return Position(swapcase(reverse(self.board)), -self.score, self.bc, self.wc, 0, 0)

    @micropython.native
    def put(self, board, i, p):
        return board[:i] + p + board[i+1:]

    @micropython.native
    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        board = self.put(board, j, board[i])
        board = self.put(board, i, '.')
        if i == A1:
            wc = (False, wc[1])
        if i == H1:
            wc = (wc[0], False)
        if j == A8:
            bc = (bc[0], False)
        if j == H8:
            bc = (False, bc[1])
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = self.put(board, A1 if j < i else H1, '.')
                board = self.put(board, kp, 'R')
        if p == 'P':
            if A8 <= j <= H8:
                board = self.put(board, j, 'Q')
            if j - i == 2*N:
                ep = i + N
            if j == self.ep:
                board = self.put(board, j+S, '.')
        return Position(board, score, wc, bc, ep, kp).rotate()

    @micropython.native
    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        score = pst[p][j] - pst[p][i]
        if q.islower():
            score += pst[q.upper()][119-j]
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score

###############################################################################
# Persistence functions using ujson
###############################################################################

STATE_FILE = "game_state.json"

def position_to_dict(pos):
    """Convert a Position to a dictionary representation."""
    return {
        "board": pos.board,
        "score": pos.score,
        "wc": list(pos.wc),
        "bc": list(pos.bc),
        "ep": pos.ep,
        "kp": pos.kp
    }

def dict_to_position(d):
    """Convert a dictionary back to a Position."""
    return Position(d["board"], d["score"], tuple(d["wc"]), tuple(d["bc"]), d["ep"], d["kp"])

def save_game_state(hist):
    """
    Save the game state to file.
    'hist' is a list of Position instances.
    """
    try:
        with open(STATE_FILE, "w") as f:
            state = {
                "history": [position_to_dict(h) for h in hist]
            }
            ujson.dump(state, f)
        print("Game state saved.")
    except Exception as e:
        print("Error saving state:", e)

def load_game_state():
    """
    Load and return the game history.
    Returns a list of Position instances, or None if no saved state.
    """
    try:
        with open(STATE_FILE, "r") as f:
            state = ujson.load(f)
        history = [dict_to_position(d) for d in state["history"]]
        print("Game state loaded.")
        return history
    except Exception as e:
        print("No saved game state found or error reading file:", e)
        return None

###############################################################################
# Search logic
###############################################################################

Entry = namedtuple('Entry', 'lower upper')

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    @micropython.native
    def bound(self, pos, gamma, depth, root=True):
        self.nodes += 1
        depth = max(depth, 0)
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER
        if DRAW_TEST:
            if not root and pos in self.history:
                return 0
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        @micropython.native
        def is_dead(pos):
            return any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())

        @micropython.native
        def moves():
            gc.collect()
            print((gc.mem_free(),))
            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, root=False)
            if depth == 0:
                yield None, pos.score
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                if len(self.tp_move) > TABLE_SIZE:
                    self.tp_move.clear()
                self.tp_move[pos] = move
                break
        if best < gamma and best < 0 and depth > 0:
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0
        if len(self.tp_score) > TABLE_SIZE:
            self.tp_score.clear()
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)
        return best

    @micropython.native
    def search(self, pos, history=()):
        self.nodes = 0
        if DRAW_TEST:
            self.history = set(history)
            self.tp_score.clear()
        for depth in range(1, 100):
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower+upper+1)//2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
            self.bound(pos, lower, depth)
            yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True)).lower

###############################################################################
# User interface and display functions
###############################################################################

display = badger2040.Badger2040()
display.set_font('bitmap6')
display.set_update_speed(badger2040.UPDATE_FAST)

def display_image(data, w, h, x, y):
    for oy in range(h):
        for ox in range(w):
            o = oy * (w >> 3) + (ox >> 3)
            bm = 0b10000000 >> (ox & 0b111)
            if data[o] & bm:
                display.display.pixel(x + ox, y + oy)

def display_clear():
    display.set_pen(15)
    display.clear()
    display.set_pen(0)

def display_legend(my_move, your_move):
    display.text('a - select', 200, 0, scale=2)
    display.text('b - left', 200, 16, scale=2)
    display.text('c - right', 200, 32, scale=2)
    display.text('Your move:', 200, 48, scale=2)
    display.text(your_move, 200, 64, scale=2)
    display.text('My move:', 200, 80, scale=2)
    display.text(my_move, 200, 96, scale=2)

def display_hourglass():
    display_image(hourglass, 32, 32, 16, 48)

def display_score(score):
    display.text(score, 200, 112, scale=2)
    display.partial_update(200, 112, 96, 16)

def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank

def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)

def display_pos(pos):
    offset = 64
    sprites = ['K', 'Q', 'B', 'N', 'R', 'P', 'k', 'q', 'b', 'n', 'r', 'p']
    for i, row in enumerate(pos.board.split()):
        for j, p in enumerate(row):
            if p != '.':
                display.icon(pieces, sprites.index(p), 192, 16, j * 16 + offset, i * 16)
    for i in range(0, 8):
        display.line(offset + i * 16, 0, offset + i * 16, 127)
        display.line(offset, i * 16, offset + 128, i * 16)
    display.line(offset + 128, 0, offset + 128, 127)
    display.line(offset, 127, offset + 128, 127)

def select_pos(pos, move=[0, 0], thickness=1):
    offset = 64
    if (pos[0] > 6 and move[0] > 0) or (pos[0] < 1 and move[0] < 0) or (pos[1] > 6 and move[1] > 0) or (pos[1] < 1 and move[1] < 0):
        return pos
    if move != [0, 0]:
        display.set_pen(15)
        display.line(offset + pos[0] * 16 + 1, pos[1] * 16 + 1, offset + pos[0] * 16 + 15, pos[1] * 16 + 1, thickness)
        display.line(offset + pos[0] * 16 + 15, pos[1] * 16 + 1, offset + pos[0] * 16 + 15, pos[1] * 16 + 15, thickness)
        display.line(offset + pos[0] * 16 + 1, pos[1] * 16 + 1, offset + pos[0] * 16 + 1, pos[1] * 16 + 15, thickness)
        display.line(offset + pos[0] * 16 + 1, pos[1] * 16 + 15, offset + pos[0] * 16 + 15, pos[1] * 16 + 15, thickness)
        display.set_pen(0)
        pos[0] += move[0]
        pos[1] += move[1]
    display.line(offset + pos[0] * 16 + 1, pos[1] * 16 + 1, offset + pos[0] * 16 + 15, pos[1] * 16 + 1, thickness)
    display.line(offset + pos[0] * 16 + 15, pos[1] * 16 + 1, offset + pos[0] * 16 + 15, pos[1] * 16 + 15, thickness)
    display.line(offset + pos[0] * 16 + 1, pos[1] * 16 + 1, offset + pos[0] * 16 + 1, pos[1] * 16 + 15, thickness)
    display.line(offset + pos[0] * 16 + 1, pos[1] * 16 + 15, offset + pos[0] * 16 + 15, pos[1] * 16 + 15, thickness)
    display.partial_update(offset + min(pos[0] - move[0], pos[0]) * 16, min(pos[1] - move[1], pos[1]) * 16, 16 * (1 + abs(move[0])), 16 * (1 + abs(move[1])))
    return pos

def wait_for_move():
    start = [4, 6]
    stop = []
    current = start
    select_pos(current)
    while True:
        if display.pressed(badger2040.BUTTON_A):
            if current == stop and start != stop:
                break
            else:
                select_pos(current, [0, 0], 4)
                current = stop = start.copy()
        elif display.pressed(badger2040.BUTTON_B):
            current = select_pos(current, [-1, 0])
        elif display.pressed(badger2040.BUTTON_C):
            current = select_pos(current, [1, 0])
        elif display.pressed(badger2040.BUTTON_UP):
            current = select_pos(current, [0, -1])
        elif display.pressed(badger2040.BUTTON_DOWN):
            current = select_pos(current, [0, 1])
    return chr(97 + start[0]) + str(8 - start[1]) + chr(97 + stop[0]) + str(8 - stop[1])

###############################################################################
# Main game loop with persistence integration
###############################################################################

def draw_chess():
    # Load saved game state, if available.
    hist = load_game_state()
    if hist is None or len(hist) == 0:
        hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
    searcher = Searcher()
    my_move = ''
    your_move = '...'
    while True:
        display_clear()
        display_legend(my_move, your_move)
        display_pos(hist[-1])
        display.update()

        if hist[-1].score <= -MATE_LOWER:
            display_score('You lost!')
            break

        move = None
        while move not in hist[-1].gen_moves():
            your_move = wait_for_move()
            move = (parse(your_move[:2]), parse(your_move[2:]))
        hist.append(hist[-1].move(move))
        # Save state after player's move.
        save_game_state(hist)
        my_move = '...'

        # Trim history to last eight moves.
        hist_remained = hist[-8:]
        hist.clear()
        hist.extend(hist_remained)

        display_clear()
        display_legend(my_move, your_move)
        display_pos(hist[-1].rotate())
        display_hourglass()
        display.update()

        if hist[-1].score <= -MATE_LOWER:
            display_score('You won!')
            break

        @micropython.native
        def fire():
            start_time = time.ticks_ms()
            for _depth, move, score in searcher.search(hist[-1], hist):
                diff = time.ticks_diff(time.ticks_ms(), start_time)
                print('Depth:', _depth, 'Time:', diff, 'ms')
                if diff > TIME_LIMIT:
                    break
            return _depth, move, score

        _depth, move, score = fire()

        if score == MATE_UPPER:
            display_score('Checkmate!')

        my_move = render(119-move[0]) + render(119-move[1])
        your_move = '...'

        hist.append(hist[-1].move(move))
        # Save state after computer's move.
        save_game_state(hist)
        gc.collect()
        print((gc.mem_free(),))

changed = False

while True:
    display.keepalive()
    if display.pressed(badger2040.BUTTON_UP):
        changed = True
    if display.pressed(badger2040.BUTTON_DOWN):
        changed = True
    if changed:
        draw_chess()
        changed = False
    display.halt()
