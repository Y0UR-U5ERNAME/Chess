# This is a recreation of the board game chess. Use Python 3.10+, preferably the latest version.
# How to play chess: https://www.chess.com/learn-how-to-play-chess

# Notes:
# - on some computers, the screen flickers when the screen updates
# - you can't return to the main menu while the ai is thinking
# - the 50 move rule is not implemented
# - the promotion choices take a little while to show up the first time

# todo: make class for pieces, fix bot giving up pawn for free and not noticing rook capture during ai vs ai

from turtle import *
from time import sleep
from math import inf
from random import uniform

# piece-square tables ---------------------------------------------
# from http://www.talkchess.com/forum3/viewtopic.php?f=2&t=68311&start=19

MG_VALS = {'p': 82, 'n': 337, 'b': 365, 'r': 477, 'q': 1025, 'k': 0}
EG_VALS = {'p': 94, 'n': 281, 'b': 297, 'r': 512, 'q': 936, 'k': 0}

MG_PST = {
'p': (
        0,   0,   0,   0,   0,   0,  0,   0,
        98, 134,  61,  95,  68, 126, 34, -11,
        -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
        0,   0,   0,   0,   0,   0,  0,   0,
    ),
'n': (
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73, -41,  72,  36,  23,  62,   7,  -17,
        -47,  60,  37,  65,  84, 129,  73,   44,
        -9,  17,  19,  53,  37,  69,  18,   22,
        -13,   4,  16,  13,  28,  19,  21,   -8,
        -23,  -9,  12,  10,  19,  17,  25,  -16,
        -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ),
'b': (
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
        0,  15,  15,  15,  14,  27,  18,  10,
        4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ),
'r': (
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  26,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    ),
'q': (
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50,
    ),
'k': (
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    )
}

EG_PST = {
'p': (
        0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100,  85,  67,  56,  53,  82,  84,
        32,  24,  13,   5,  -2,   4,  17,  17,
        13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
        13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0,
    ),
'n': (
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ),
'b': (
        -14, -21, -11,  -8, -7,  -9, -17, -24,
        -8,  -4,   7, -12, -3, -13,  -4, -14,
        2,  -8,   0,  -1, -2,   6,   0,   4,
        -3,   9,  12,   9, 14,  10,   3,   2,
        -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    ),
'r': (
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
        7,  7,  7,  5,  4,  -3,  -5,  -3,
        4,  3, 13,  1,  2,   1,  -1,   2,
        3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    ),
'q': (
        -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
        3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ),
'k': (
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
        10,  17,  23,  15,  20,  45,  44,  13,
        -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
    )
}

MIN_EG = 518
MAX_MG = 6192
PHASE_RANGE = MAX_MG - MIN_EG


# setup -----------------------------------------------------------
WIDTH = 960
HEIGHT = 720
AI_DEPTH = 3 # if the ai is too slow, decrease this
MATE_VALUE = 11000
SEARCH_MIN = -MATE_VALUE - AI_DEPTH - 1
SEARCH_MAX = -SEARCH_MIN
MAX_SEARCH_EXTENSIONS = 8
EG_MATERIAL_START = MG_VALS['r'] * 2 + MG_VALS['b'] + MG_VALS['n']

screen = Screen()
#screen._root.iconbitmap('icon.ico')
screen.cv._rootwindow.resizable(False, False)
setup(WIDTH, HEIGHT)
title('Chess')
bgcolor('#111822')
tracer(0)
speed(0)
pu()
ht()
shape('square')
seth(90)
listen()
processing = False

# lookup tables
DIRECTION_OFFSETS = (-8, 8, -1, 1, -9, 9, -7, 7)
KING_MOVES = []
KNIGHT_MOVES = []
DIST_TO_EDGE = []

for j in range(8):
    for i in range(8):
        l = []
        for off_y in (-1, 1, 0):
            for off_x in (-1, 1, 0):
                if (off_x or off_y) and 0 <= i + off_x <= 7 and 0 <= j + off_y <= 7:
                    l.append(i + off_x + (j + off_y) * 8)
        KING_MOVES.append(l)

for j in range(8):
    for i in range(8):
        l = []
        for off_x, off_y in ((1, 2), (2, 1), (-1, 2), (-2, 1), (-1, -2), (-2, -1), (1, -2), (2, -1)):
            if (off_x or off_y) and 0 <= i + off_x <= 7 and 0 <= j + off_y <= 7:
                l.append(i + off_x + (j + off_y) * 8)
        KNIGHT_MOVES.append(l)

for j in range(8):
    for i in range(8):
        dist_north = j
        dist_south = 7 - j
        dist_west = i
        dist_east = 7 - i
        DIST_TO_EDGE.append([dist_north, dist_south, dist_west, dist_east, min(dist_north, dist_west), min(dist_south, dist_east), min(dist_north, dist_east), min(dist_south, dist_west)])

class Board():
    def __init__(self):
        self.load_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    
    def make_fen(self, half_moves, total_moves):
        board, col, castling, ep_tile = self.board, 'wb'[self.turn], self.castling, self.ep_tile
        fen = ''
        b = [board[i*8:i*8+8] for i in range(8)]
        for i in b:
            for j in i:
                if not j:
                    if (' ' + fen)[-1].isnumeric():
                        fen = fen[:-1] + str(int(fen[-1]) + 1)
                    else: fen += '1'
                else: fen += ' PRNBQKprnbqk'[j]
            fen += '/'
        fen = f"{fen[:-1]} {col} {''.join('KQkq'[i] for i in range(4) if castling & (0b1000 >> i))} {to_file_rank(ep_tile)} {half_moves} {total_moves}"
        return fen

    def load_fen(self, fen):
        data = fen.split()

        b = data[0].split('/')
        board = []
        for i in b:
            for j in i:
                if j.isnumeric(): board += [0 for k in range(int(j))]
                else: board.append(' PRNBQKprnbqk'.index(j))
        self.board = board
        
        pieces = {}
        counts = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [board[i*8:i*8+8] for i in range(8)]
        for c, i in enumerate(b):
            for d, j in enumerate(i):
                if not j: continue
                p = 'wb'[j > 6] + 'prnbqk'[(j - 1) % 6]
                if p[-1] == 'r': # for correct castling
                    if (d, c) == ((0, 7), (0, 0))[j > 6]: p += 'a'
                    elif (d, c) == ((7, 7), (7, 0))[j > 6]: p += 'h'
                    elif counts[j]: p += str(counts[j])
                elif counts[j]:
                    p += str(counts[j])
                counts[j] += 1
                pieces[p] = d + c * 8
        self.pieces = pieces

        col = data[1]
        turn = 'wb'.index(col)
        self.turn = turn

        self.castling = sum((i in data[2]) << (c ^ 0b11) for c, i in enumerate('KQkq'))

        if data[3] == '-': self.ep_tile = None
        else: self.ep_tile = from_file_rank(data[3])

        check_data = self.get_check_data()
        self.check_data = check_data

        in_check = check_data[0]
        poss_move = next(self.all_targets(), None) is not None
        if in_check:
            if poss_move: state = ('White', 'Black')[turn] + ' in check'
            else: state = ('Black', 'White')[turn] + ' wins!'
        else:
            bishops = [i for i in pieces if i[1] == 'b']
            insuff = (len(pieces) == 3 and 'n' in ''.join(pieces)) or (len(pieces) == len(bishops) + 2 and len({(pieces[i]&0b001001).bit_count()%2 for i in bishops}) <= 1)
            if poss_move and not insuff: state = ('White', 'Black')[turn] + "'s turn"
            else: state = 'Draw!'
        
        self.state = state
        self.history = [self.make_history()]

    def perft(self, depth):
        moves = tuple(self.all_moves())
        if depth == 1: return len(moves)
        count = 0
        for m in moves:
            unmake_data = self.make_move(*m)
            if self.state[-1] != '!': count += self.perft(depth - 1)
            self.unmake_move(*m, *unmake_data)
        return count

    def check_perft(self, max_depth):
        from time import time
        for i in range(1, max_depth):
            t = time()
            print(c:=self.perft(i))
            tt = time() - t
            print(tt, c / tt)

    def make_history(self):
        return self.board.copy(), self.ep_tile, self.castling, self.turn

    def make_move(self, selected, pos, promote=0, add_captured=False):
        pieces, board, ep_tile, castling, turn = self.pieces, self.board, self.ep_tile, self.castling, self.turn
        col = 'wb'[turn]
        did_capture = 0
        old_check_data, old_state = self.check_data, self.state

        def p_at(p):
            return list(pieces.keys())[list(pieces.values()).index(p)]
        def capture(cap_pos):
            nonlocal did_capture, captured_p
            did_capture = board[cap_pos]
            captured_p = p_at(cap_pos)
            pieces.pop(captured_p)
            if add_captured:
                global all_captured
                if board[cap_pos] in all_captured[turn]:
                    all_captured[turn][board[cap_pos]] += 1
                else:
                    all_captured[turn][board[cap_pos]] = 1
                    all_captured[turn] = dict(sorted(all_captured[turn].items(), key=lambda x: (1, 4, 3, 2, 5, 7, 10, 9, 8, 11).index(x[0])))

        moved_p = p_at(selected)
        captured_p = None

        if board[pos]: capture(pos)
        is_ep = False
        if board[selected] in {1, 7} and pos == ep_tile:
            # en passant capture
            is_ep = True
            captured = (pos & 0b000111) | (selected & 0b111000)
            capture(captured)
            board[captured] = 0
        if board[selected] in {1, 7} and abs((pos & 0b111000) - (selected & 0b111000)) == 2 << 3:
            # en passant capture available
            ep_tile = selected + (-1<<3, 1<<3)[turn]
        else: ep_tile = None
        self.ep_tile = ep_tile
        
        # update castling rights
        if castling & 0b1000 and 63 in {selected, pos}: castling &= 0b0111
        if castling & 0b0100 and 56 in {selected, pos}: castling &= 0b1011
        if castling & 0b0010 and 7 in {selected, pos}: castling &= 0b1101
        if castling & 0b0001 and 0 in {selected, pos}: castling &= 0b1110
        if castling & 0b1100 and selected == 60: castling &= 0b0011
        if castling & 0b0011 and selected == 4: castling &= 0b1100
        self.castling = castling

        is_castle = False
        if board[selected] in {6, 12} and abs(pos - selected) == 2:
            # castling
            y = pos & 0b111000
            if pos < selected:
                is_castle = 1
                board[3 | y] = (2, 8)[turn]
                board[y] = 0
                pieces[col + 'ra'] = 3 | y
            else:
                is_castle = 2
                board[5 | y] = (2, 8)[turn]
                board[7 | y] = 0
                pieces[col + 'rh'] = 5 | y

        if promote:
            board[pos] = promote
            p = moved_p
            pieces.pop(p)
            pieces[col + ' prnbqkprnbqk'[promote] + p] = pos
        else:
            board[pos] = board[selected]
            pieces[moved_p] = pos
        
        board[selected] = 0

        # get new state
        turn = not turn
        self.turn = turn
        col = 'wb'[turn]
        check_data = self.get_check_data()
        self.check_data = check_data

        in_check = check_data[0]
        poss_move = next(self.all_targets(), None) is not None
        
        if in_check:
            if poss_move: state = ('White', 'Black')[turn] + ' in check'
            else: state = ('Black', 'White')[turn] + ' wins!'
        else:
            bishops = [i for i in pieces if i[1] == 'b']
            insuff = (len(pieces) == 3 and 'n' in ''.join(pieces)) or (len(pieces) == len(bishops) + 2 and len({(pieces[i]&0b001001).bit_count()%2 for i in bishops}) <= 1)
            if poss_move and not insuff: state = ('White', 'Black')[turn] + "'s turn"
            else: state = 'Draw!'
        
        hist = self.make_history()
        if self.history.count(hist) == 2: state = 'Draw!'
        self.history.append(hist)
        
        self.state = state
        return moved_p, captured_p, did_capture, is_ep, is_castle, old_check_data, old_state

    def unmake_move(self, start, end, promote, moved_p, captured_p, captured, is_ep, is_castle, old_check_data, old_state):
        board, pieces = self.board, self.pieces
        if promote:
            board[start] = (1, 7)[board[end] > 6]
            pieces.pop(moved_p[0] + ' prnbqkprnbqk'[promote] + moved_p)
        else:
            board[start] = board[end]
        pieces[moved_p] = start

        if is_ep:
            board[end] = 0
            board[end & 0b000111 | start & 0b111000] = captured
            pieces[captured_p] = end & 0b000111 | start & 0b111000
        else:
            board[end] = captured
            if captured_p: pieces[captured_p] = end
        
        if is_castle:
            col = moved_p[0]
            rook = (8, 2)[col == 'w']
            rook_p = col + ('ra', 'rh')[is_castle - 1]
            rook_pos = pieces[rook_p]
            board[rook_pos] = 0
            if is_castle == 1:
                board[rook_pos - 3] = rook
                pieces[rook_p] = rook_pos - 3
            elif is_castle == 2:
                board[rook_pos + 2] = rook
                pieces[rook_p] = rook_pos + 2
        
        self.turn = not self.turn
        self.history.pop()
        self.ep_tile, self.castling, self.check_data, self.state = self.history[-1][1], self.history[-1][2], old_check_data, old_state

    def get_check_data(self):
        col = 'wb'[self.turn]
        king_pos, pieces, board = self.pieces[col + 'k'], self.pieces, self.board

        w = col == 'w'

        # get checkers
        checkers = 0x0000000000000000
        king_x = king_pos & 0b000111
        if king_x:
            pos = king_pos + (7, -9)[w]
            if 0 <= pos <= 63 and board[pos] == (1, 7)[w]:
                checker = pos
                checkers |= 0b1 << 63 >> pos
        if king_x != 7:
            pos = king_pos + (9, -7)[w]
            if 0 <= pos <= 63 and board[pos] == (1, 7)[w]:
                checker = pos
                checkers |= 0b1 << 63 >> pos
        for i in legal_moves({col+'n': king_pos}, col+'n', board, check=False):
            if board[i] == (3, 9)[w]:
                checker = i
                checkers |= 0b1 << 63 >> i
        for i in legal_moves({col+'b': king_pos}, col+'b', board, check=False):
            if board[i] in ({4, 5}, {10, 11})[w]:
                checker = i
                checkers |= 0b1 << 63 >> i
        for i in legal_moves({col+'r': king_pos}, col+'r', board, check=False): 
            if board[i] in ({2, 5}, {8, 11})[w]:
                checker = i
                checkers |= 0b1 << 63 >> i

        # king danger tiles
        board[king_pos] = 0
        king_danger = 0x0000000000000000
        for p in pieces:
            if p[0] == 'wb'[w]:
                if p[1] == 'p':
                    pawn_pos = pieces[p]
                    pawn_x = pieces[p] & 0b000111
                    if pawn_x: king_danger |= 0b1 << 63 >> pawn_pos + (-9, 7)[w]
                    if pawn_x != 7: king_danger |= 0b1 << 63 >> pawn_pos + (-7, 9)[w]
                else:
                    piece = col + p[1] # pretend this opponent piece is of the same color as the king, so that pieces protected by this piece are also king danger tiles
                    for m in legal_moves({piece: pieces[p]}, piece, board, check=False):
                        king_danger |= 0b1 << 63 >> m
        board[king_pos] = (12, 6)[w]

        # capture and push tiles
        capture_tiles = 0xFFFFFFFFFFFFFFFF
        push_tiles = 0xFFFFFFFFFFFFFFFF
        if checkers.bit_count() == 1:
            capture_tiles = checkers
            push_tiles = 0x0000000000000000
            if board[checker] in {2, 4, 5, 8, 10, 11}:
                check_dir = get_dir(checker, king_pos)
                p = king_pos + check_dir
                while p != checker:
                    push_tiles |= 0b1 << 63 >> p
                    p += check_dir
        
        # pinned pieces
        pinned = 0x0000000000000000
        for i in pieces:
            if i[0] != col and i[1] in 'brq':
                pos = pieces[i]
                dir = None
                if i[1] != 'b' and (pos & 0b111000 == king_pos & 0b111000 or pos & 0b000111 == king_pos & 0b000111): # king and rook/queen on same horizontal/vertical
                    dir = get_dir(pos, king_pos)
                if i[1] != 'r':
                    diff = (pos & 0b000111) - (king_pos & 0b000111), (pos & 0b111000) - (king_pos & 0b111000) >> 3
                    if diff[0] == diff[1] or diff[0] == -diff[1]: # king and bishop/queen on same diagonal
                        dir = get_dir(pos, king_pos)
                if dir:
                    p = king_pos + dir
                    num = 0
                    pin_pos = None
                    while p != pos:
                        if (t:=board[p]):
                            if num or (t > w*6 and t < w*6+7): pin_pos = None; break # it can't be a pin if more than one piece is in between or if the piece is not the king's color
                            pin_pos = p
                            num = 1
                        p += dir
                    if pin_pos: pinned |= 0b1 << 63 >> pin_pos

        return checkers, king_danger, capture_tiles, push_tiles, pinned

    def all_moves(self, only_captures=False):
        pieces, board, ep_tile, castling, check_data = self.pieces, self.board, self.ep_tile, self.castling, self.check_data
        col = 'wb'[self.turn]
        for p in pieces:
            if p[0] == col:
                for m in legal_moves(pieces, p, board, ep_tile, castling, check_data, only_captures=only_captures):
                    if p[1] == 'p' and m & 0b111000 == (7<<3, 0<<3)[col == 'w']:
                        # promotion
                        for i in (5, 3, 2, 4):
                            yield pieces[p], m, i + (6, 0)[col == 'w']
                    else: yield pieces[p], m, 0

    def all_targets(self, check=True):
        pieces, board, ep_tile, castling, check_data = self.pieces, self.board, self.ep_tile, self.castling, self.check_data
        col = 'wb'[self.turn]
        for p in pieces:
            if p[0] == col:
                for m in legal_moves(pieces, p, board, ep_tile, castling, check_data, check=check):
                    yield m

    # fail-soft minimax/negamax with alpha-beta pruning
    def minimax(self, depth, alpha, beta, get_move=False, num_extensions=0):
        board, state = self.board, self.state
        if not depth: return self.quiescence_search(alpha, beta) # quiescence search
        if state[-1] == '!': return (-MATE_VALUE - depth, 25)[state[0] == 'D'] # encourage mating sooner than later, contempt factor for drawin

        best_score = -inf
        best_move = None
        for m in sorted(self.all_moves(), key=lambda x: (1, 4, 2, 3, 5, 0)[(board[x[1]]-1)%6], reverse=True):
            unmake_data = self.make_move(*m)
            if self.history.count(self.make_history()) > 1: score = -25
            else:
                extension = num_extensions < MAX_SEARCH_EXTENSIONS and self.state[-1] == 'k'
                score = -self.minimax(depth - 1 + extension, -beta, -alpha, num_extensions=num_extensions + extension)
            self.unmake_move(*m, *unmake_data)
            
            if score >= beta:
                if get_move: return score, m
                return score
            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score
                    if get_move: best_move = m
                    
        if get_move: return alpha, best_move
        return alpha

    def quiescence_search(self, alpha, beta):
        turn, pieces, board, state = self.turn, self.pieces, self.board, self.state
        if state[-1] == '!': best_score = (-MATE_VALUE, 25)[state[0] == 'D'] # contempt factor for drawing
        else: best_score = (1, -1)[turn] * evaluate(pieces) # must negate for black so that positive scores will always be good
        if best_score >= beta: return best_score
        if best_score > alpha: alpha = best_score
        
        for m in sorted(self.all_moves(only_captures=True), key=lambda x: (1, 4, 2, 3, 5, 0)[(board[x[1]]-1)%6], reverse=True):
            unmake_data = self.make_move(*m)
            score = -self.quiescence_search(-beta, -alpha)
            self.unmake_move(*m, *unmake_data)
            
            if score >= beta: return score
            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score
                    
        return alpha

    def get_ai_move(self, depth, lower, upper):
        # uses NegaC* algorithm (https://www.chessprogramming.org/NegaC*), which is supposed to be faster but isn't in this program
        '''
        score = lower
        move = None
        while lower < upper:
            alpha = (lower + upper) // 2
            score, _move = minimax(depth, alpha, alpha + 1, turn, pieces, board, ep_tile, castling, state, check_data, True)
            if _move: move = _move
            if score > alpha: lower = score
            else: upper = score
        '''
        move = self.minimax(depth, lower, upper, True)[1]
        return move

# todo: create class for pieces like https://joeyrobert.org/2016/01/06/optimizing-move-generation/


# functions -------------------------------------------------------
pt_within = lambda px, py, x, y, w, h: x <= px <= x + w and y <= py <= y + h
pt_within_center = lambda px, py, x, y, w, h: x - w/2 <= px <= x + w/2 and y - h/2 <= py <= y + h/2

to_file_rank = lambda pos: 'abcdefgh'[pos[0]] + str(8 - pos[1])
from_file_rank = lambda pos: ('abcdefgh'.index(pos[0]), 8 - int(pos[1]))

get_dir = lambda p1, p2: min(max((p1 & 0b000111) - (p2 & 0b000111), -1), 1) + min(max((p1 & 0b111000) - (p2 & 0b111000), -1<<3), 1<<3)

def write_at(x, y, arg, move=False, align='left', font=('Arial', 8, 'normal')):
    goto(x, y)
    write(arg, move, align, font)

def menu():
    onscreenclick(None)
    clear()
    pencolor('white')
    write_at(0, HEIGHT/4, '♙   CHESS   ♟', align='center', font=('Arial', 72, 'bold'))
    pencolor('#cb9f5c')
    write_at(0, 0, '( 2-Player Game )', align='center', font=('Arial', 24, 'normal'))
    write_at(0, -64, '( White vs. AI )', align='center', font=('Arial', 24, 'normal'))
    write_at(0, -128, '( Black vs. AI )', align='center', font=('Arial', 24, 'normal'))
    write_at(0, -192, '( Quit )', align='center', font=('Arial', 24, 'normal'))
    pencolor('#aabbcc')
    write_at(-WIDTH/2 + 5, -HEIGHT/2 + 10, 'Made by Y0UR-U5ERNAME', align='left', font=('Arial', 16, 'normal'))
    update()

    def menu_onclick(x, y):
        global vs_ai, ai_turn
        if pt_within_center(x, y - 18, 0, 0, 250, 36): vs_ai = False; ai_turn = 0; game()
        elif pt_within_center(x, y - 18, 0, -64, 250, 36): vs_ai = True; ai_turn = 1; game()
        elif pt_within_center(x, y - 18, 0, -128, 250, 36): vs_ai = True; ai_turn = 0; game()
        elif pt_within_center(x, y - 18, 0, -192, 250, 36): bye()
    onscreenclick(menu_onclick)

def piece_at(pos):
    pieces = board_state.pieces
    return list(pieces.keys())[list(pieces.values()).index(pos)]

def tile(x, y):
    shapesize(3.95)
    goto(x, y)
    stamp()

def small_tile(x, y):
    shapesize(160 / 5 / 20)
    goto(x, y)
    stamp()

def draw_game():
    turn, board, state, pieces = board_state.turn, board_state.board, board_state.state, board_state.pieces
    clear()

    # draw board
    col = 0
    side = (turn, not ai_turn)[vs_ai]
    for c, piece in enumerate(board):
        i, j = c & 0b000111, (c & 0b111000) >> 3
        col = (i + j) % 2
        pencolor('#111822')
        if selected == c: fillcolor(('#8889d4', '#8d67a4')[col])
        elif c == last_move[0]: fillcolor(('#af9d3c', '#7b6f34')[col])
        elif c == last_move[1]: fillcolor(('#bda941', '#8f813e')[col])
        else: fillcolor(('#cb9f5c', '#7c5444')[col])
        if side: tile((7 - i) * 80 - WIDTH/2 + 80, j * 80 - HEIGHT/2 + 80)
        else: tile(i * 80 - WIDTH/2 + 80, (7 - j) * 80 - HEIGHT/2 + 80)

        if c in poss_moves:
            shape('circle')
            t_col = fillcolor()
            color(tuple(t_col[i]*.7 + (20, 0, 0)[i]/255*.3 for i in range(3)))
            if piece:
                shapesize(3.85); stamp()
                color(t_col); shapesize(3.2); stamp()
            else: shapesize(1); stamp()
            shape('square')
        
        if piece:
            bk(36)
            pencolor(('white', 'black')[piece > 6])
            write(' ♙♖♘♗♕♔♟♜♞♝♛♚'[piece], align='center', font=('Arial', 48, 'normal'))
    
    # draw ranks and files
    pencolor('#aabbcc')
    goto(-WIDTH/2 + 20, -HEIGHT/2 + 80 - 9)
    for i in range(8):
        write('12345678'[7-i if side else i], align='center', font=('Arial', 12, 'normal'))
        fd(80)
    goto(-WIDTH/2 + 80, -HEIGHT/2 + 20 - 9)
    for i in range(8):
        write('abcdefgh'[7-i if side else i], align='center', font=('Arial', 12, 'normal'))
        setx(xcor() + 80)
    
    # draw UI
    pencolor('white')
    write_at((WIDTH - HEIGHT + 40)/2 + HEIGHT - 40 - WIDTH/2, -12, state, align='center', font=('Arial', 24, 'bold'))
    pencolor('#cb9f5c')
    write_at((WIDTH - HEIGHT + 40)/2 + HEIGHT - 40 - WIDTH/2, -48, '( Main Menu )', align='center', font=('Arial', 16, 'normal'))
    pencolor('#aabbcc')

    capt = lambda t: '\n'.join(f' {" ♙♖♘♗♕♔♟♜♞♝♛♚"[i]} {all_captured[t][i]}' for i in all_captured[t])

    val = lambda t: sum((1, 3, 3, 5, 9, 0)['pnbrqk'.index(i[1])] for i in pieces if i[0] == 'wb'[t])

    val_s = val(side)
    val_ns = val(not side)

    write_at(HEIGHT - 40 - WIDTH/2 + 40, HEIGHT/2 - 40 - 24*(1+len(all_captured[not side])), ('Black', 'White')[side] + (f' +{val_ns - val_s}' if val_ns > val_s else '') + ('\n' if all_captured[not side] else '') + capt(not side), font=('Arial', 16, 'normal'))
    write_at(HEIGHT - 40 - WIDTH/2 + 40, -HEIGHT/2 + 40, capt(side) + '\n' + ('White', 'Black')[side] + (f' +{val_s - val_ns}' if val_s > val_ns else ''), font=('Arial', 16, 'normal'))

    update()

def game_onclick(x, y):
    global selected, poss_moves, last_move, promote_tile, processing
    turn, state, board, pieces, ep_tile, castling, check_data = board_state.turn, board_state.state, board_state.board, board_state.pieces, board_state.ep_tile, board_state.castling, board_state.check_data

    if processing: return # prevents concurrent clicking
    if pt_within_center(x, y - 12, (WIDTH - HEIGHT + 40)/2 + HEIGHT - 40 - WIDTH/2, -48, 128, 24): menu(); return
    processing = True

    # click on board/promotion choice
    on_board = pt_within(x, y, -WIDTH/2 + 40, -HEIGHT/2 + 40, 640, 640)
    on_promotion = promote_tile and pt_within_center(x, y, promote_tile[0]*80+40 - WIDTH/2 + 40, promote_tile[1]*80+40+40+20 - HEIGHT/2 + 40, 160, 32)
    if state[-1] != '!' and (on_board or on_promotion):
        px = int(x - (-WIDTH/2 + 40))
        py = int(y - (-HEIGHT/2 + 40))
        tx = px//80
        ty = py//80
        pos = 7 - tx | ty << 3 if turn else tx | 7 - ty << 3
        if on_board and turn*6 < board[pos] < turn*6+7:
            if pos == selected: selected = None; poss_moves = []
            else: selected = pos; poss_moves = tuple(legal_moves(pieces, piece_at(selected), board, ep_tile, castling, check_data))
        else:
            if on_board and pos not in poss_moves: selected = None; poss_moves = []
            else:
                promote = 0
                if on_promotion:
                    # promote pawn
                    promote = ((x - (promote_tile[0]*80+40 - WIDTH/2 + 40 - 80))//32+1)%5
                    if not promote: promote_tile = None; draw_game(); processing = False; return
                    tx, ty = promote_tile
                    pos = 7 - tx | ty << 3 if turn else tx | 7 - ty << 3
                elif board[selected] in {1, 7} and pos & 0b111000 == (0<<3, 7<<3)[turn]:
                    # pawn promotion choice
                    draw_game()
                    for c, i in enumerate('♕♘♖♗♛♞♜♝'[turn*4:turn*4+4]+'❌'):
                        color('white')
                        small_tile(tx*80+40 - WIDTH/2 + 40 + (c*32 - 64), ty*80+40+40+20 - HEIGHT/2 + 40)
                        pencolor('black')
                        bk(15)
                        write(i, align='center', font=('Arial', 20, 'normal'))
                    promote_tile = (tx, ty)
                    update()
                    processing = False
                    return
                promote_tile = None
                old_state = state
                board_state.make_move(selected, pos, (5, 3, 2, 4)[int(promote) - 1] + turn*6 if promote else 0, True)
                new_state = board_state.state
                board_state.state = old_state
                last_move = (selected, pos)
                selected = None; poss_moves = []
                board_state.turn = not board_state.turn
                if new_state[-1] != '!':
                    draw_game()
                    if not vs_ai: sleep(0.3)
                board_state.turn = not board_state.turn
                board_state.state = new_state
                turn, state, board, pieces, ep_tile, castling, check_data = board_state.turn, board_state.state, board_state.board, board_state.pieces, board_state.ep_tile, board_state.castling, board_state.check_data

                if vs_ai and state[-1] != '!':
                    draw_game()
                    #from time import time
                    #t = time()
                    ai_move = board_state.get_ai_move(AI_DEPTH, SEARCH_MIN, SEARCH_MAX)
                    #print(time() - t)
                    board_state.make_move(*ai_move, True)
                    last_move = ai_move
                    turn, state, board, pieces, ep_tile, castling, check_data = board_state.turn, board_state.state, board_state.board, board_state.pieces, board_state.ep_tile, board_state.castling, board_state.check_data

        draw_game()
    processing = False

def game():
    global board_state, selected, poss_moves, last_move, promote_tile, all_captured
    onscreenclick(None)
    clear()

    board_state = Board()
    selected = None
    poss_moves = []
    last_move = (None, None)
    promote_tile = None
    all_captured = [{}, {}]

    draw_game()

    # ai vs. ai
    '''
    while 1:
        ai_move = board_state.get_ai_move(AI_DEPTH, SEARCH_MIN, SEARCH_MAX)
        board_state.make_move(*ai_move, True)
        last_move = ai_move
        draw_game()
        if board_state.state[-1] == '!': return
    '''
    
    onscreenclick(game_onclick)
    if vs_ai and not ai_turn:
        ai_move = board_state.get_ai_move(AI_DEPTH, SEARCH_MIN, SEARCH_MAX)
        board_state.make_move(*ai_move, True)
        last_move = ai_move
        draw_game()

def legal_moves(pieces, piece, board, ep_tile=None, castling=0b0000, check_data=None, check=True, only_captures=False):
    col = piece[0] # w, b
    w = col == 'w'
    typ = piece[1] # p, r, n, b, q, k
    pos = pieces[piece]

    validate = lambda move: (t:=board[move]) <= 6-w*6 or t > 12-w*6

    def is_legal(move):
        if only_captures and not (board[move] or (move == ep_tile and typ == 'p')): return False
        if not validate(move): return False
        if not check: return True

        if typ == 'k':
            if check_data[1] & 0b1 << 63 >> move: return False # king cannot move into king danger tiles
            return True

        king_pos = pieces[col + 'k']
        if check_data[4] & 0b1 << 63 >> pos:
            if check_data[0]: return False # pinned pieces cannot move when the king is in check
            d = get_dir(pos, move)
            if get_dir(pos, king_pos) not in {d, -d}: return False # pinned pieces can only move directly toward or away from king
            return True

        if move == ep_tile and typ == 'p': # move is an en passant move
            taken_pos = move & 0b000111 | pos & 0b111000
            if not check_data[2] & 0b1 << 63 >> taken_pos: return False # en passant can capture a checking pawn
            if not check_data[3] & 0b1 << 63 >> move: return False # en passant can block a sliding piece's check
            # edge case: en passant reveals a horizontal check from rook/queen
            if king_pos & 0b111000 == pos & 0b111000:
                board[pos] = 0 # remove taking pawn
                board[taken_pos] = 0 # remove taken pawn
                king_x = king_pos & 0b000111
                r = range(king_x-1, -1, -1) if pos & 0b000111 < king_x else range(king_x+1, 8) # start searching from the left or right of the king for a rook/queen
                ret = True
                for i in r:
                    if (t:=board[(pos & 0b111000) | i]):
                        if t in ({2, 5}, {8, 11})[w]: ret = False
                        break
                board[pos] = (7, 1)[w] # add back taking pawn
                board[taken_pos] = (1, 7)[w] # add back taken pawn
                return ret
        elif board[move]: # move is a capture, check capture_tiles
            if not check_data[2] & 0b1 << 63 >> move: return False
        elif not check_data[3] & 0b1 << 63 >> move: return False # move is not a capture, check push_tiles

        return True

    if check and check_data[0].bit_count() > 1 and typ != 'k' : return # only king can move if double check
    match typ:
        case 'p':
            f = (lambda off: pos - off) if w else (lambda off: pos + off)
            # capture
            pawn_x = pos & 0b000111
            if pawn_x != 7 and (w*6 < board[m:=f(8)+1] < w*6+7 or m == ep_tile) and is_legal(m): yield m
            if pawn_x and (w*6 < board[m:=f(8)-1] < w*6+7 or m == ep_tile) and is_legal(m): yield m
            if only_captures: return
            # forward
            if not board[m:=f(8)]:
                if is_legal(m): yield m
                if pos & 0b111000 == (1<<3, 6<<3)[w] and not board[m:=f(16)] and is_legal(m): yield m
        case 'n':
            # knight movement
            if check and check_data[4] & 0b1 << 63 >> pos: return # pinned knights cannot move
            for i in KNIGHT_MOVES[pos]:
                if is_legal(i): yield i
        case 'k':
            # king movement
            moves = KING_MOVES[pos]
            for i in moves:
                if is_legal(i): yield i
            # castling
            if not check or check_data[0]: return
            if castling & (0b0010 << (w << 1)) and not any(board[pos + i] for i in (1, 2)) and is_legal(pos + 1) and is_legal(m:=pos + 2): yield m
            if castling & (0b0001 << (w << 1)) and not any(board[pos - i] for i in (1, 2, 3)) and is_legal(pos - 1) and is_legal(m:=pos - 2): yield m
        case _:
            for i in range((typ == 'b') << 2, 4 + ((typ != 'r') << 2)):
                d = DIRECTION_OFFSETS[i]
                for j in range(DIST_TO_EDGE[pos][i]):
                    p = pos + d * (j + 1)
                    if not validate(p): break
                    if is_legal(p): yield p
                    if w*6 < board[p] < w*6+7: break

def evaluate(pieces):
    mg_val = sum((MG_VALS[i[1]] + (MG_PST[i[1]][pieces[i]] if i[0] == 'w' else MG_PST[i[1]][pieces[i] ^ 0b111000])) * (-1, 1)[i[0] == 'w'] for i in pieces)
    eg_val = sum((EG_VALS[i[1]] + (EG_PST[i[1]][pieces[i]] if i[0] == 'w' else EG_PST[i[1]][pieces[i] ^ 0b111000])) * (-1, 1)[i[0] == 'w'] for i in pieces)
    factor_mg = (min(max(sum(MG_VALS[i[1]] for i in pieces if i[1] != 'p'), MIN_EG), MAX_MG) - MIN_EG) / PHASE_RANGE
    val = mg_val * factor_mg + eg_val * (1 - factor_mg)
    
    w_mat = sum(MG_VALS[i[1]] for i in pieces if i[0] == 'w')
    b_mat = sum(MG_VALS[i[1]] for i in pieces if i[0] == 'b')
    w_eg_weight = 1 - min(1, (w_mat - sum(MG_VALS[i[1]] for i in pieces if i[:2] == 'wp')) / EG_MATERIAL_START)
    b_eg_weight = 1 - min(1, (b_mat - sum(MG_VALS[i[1]] for i in pieces if i[:2] == 'bp')) / EG_MATERIAL_START)

    val += uniform(-.1, .1)

    return val + mopupeval(pieces, 'w', 'b', w_mat, b_mat, b_eg_weight) - mopupeval(pieces, 'b', 'w', b_mat, w_mat, w_eg_weight)

def mopupeval(pieces, col1, col2, mat1, mat2, eg_weight):
    score = 0
    if mat1 > mat2 + 200 and eg_weight:
        k1 = pieces[col1 + 'k']
        k1x = k1 & 0b000111
        k1y = (k1 & 0b111000) >> 3
        k2 = pieces[col2 + 'k']
        k2x = k2 & 0b000111
        k2y = (k2 & 0b111000) >> 3
        score += (abs(k2x - 3.5) + abs(k2y - 3.5)) * 4.7 # encourage pushing opponent king farther from center
        score += (14 - (abs(k1x - k2x) + abs(k1y - k2y))) * 1.6 # encourage moving king closer to opponent king
        return score * eg_weight * 2 # maybe don't multiply by 2
    return 0


menu()
done()