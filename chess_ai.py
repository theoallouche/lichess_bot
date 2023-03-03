import copy
import time
import math
import threading
from typing import List, Tuple, Optional

import chess
import chess.pgn
from chess.polyglot import zobrist_hash


MATERIAL_VALUE = {chess.PAWN: 1, chess.ROOK: 5, chess.KNIGHT: 3, chess.BISHOP: 3, chess.QUEEN: 9}


class ThreadWithReturnValue(threading.Thread):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result = None

    def run(self):
        if self._target is not None:
            self.result = self._target(*self._args, **self._kwargs)


class ChessAI:

    def __init__(self, depth, color):
        self.max_depth = depth
        self.color = color
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.evaluated_positions = {}

    @staticmethod
    def _count_material_value(board: chess.Board, color: chess.Color) -> int:
        """ Count material value of `color` player based on the conventional evaluation"""
        return sum(value*len(board.pieces(piece, color)) for piece, value in MATERIAL_VALUE.items())

    @staticmethod
    def evaluation(board: chess.Board, result, depth) -> int:
        """ Evaluate `board`. The evaluation is based on material only.
        Positive sign for White, negative for Black."""
        if result == '1/2-1/2':
            return 0
        white_win_value = 1000 + depth
        if result == '1-0':
            return white_win_value
        if result == '0-1':
            return -white_win_value
        return ChessAI._count_material_value(board, chess.WHITE) - \
               ChessAI._count_material_value(board, chess.BLACK)

    @staticmethod
    def _get_move_priority(board: chess.Board, move: chess.Move):
        """ Give a move a priority score for the evaluation in order to sort the
        moves to make the Alpha-Beta pruning converge faster.
        This has no effect on the evaluation result.
        """
        score = 0
        # Taking a good opponent piece with a bad piece should prioritized
        if board.is_capture(move):
            taker = board.piece_at(move.from_square).piece_type
            taken = chess.PAWN if board.is_en_passant(move) else board.piece_at(move.to_square).piece_type
            score += 10*taken - taker

        # Moving to an attacked square should not be prioritized
        score -= 3*int(board.is_attacked_by(not board.turn, move.to_square))

        # score += 3*int(board.is_into_check(move))
        return score

    def negamax(self, board: chess.Board, depth: int, maximizer: chess.Color,
                alpha: float, beta: float, asked_depth: int, last_best_move: Optional[chess.Move]) -> Tuple[float, chess.Move, int, list[chess.Move]]:
        """ Negamax tree search. Return the best move of the position based on
        the evaluation function. It also returns its associated value and
        variant and the total visited positions.
        """
        alphaOrig = alpha
        key = zobrist_hash(board)
        if entry:= self.evaluated_positions.get(key):
            if entry['depth'] >= depth:
                if entry['flag'] == "EXACT":
                    return entry['value'], entry['bestmove']
                elif entry['flag'] == "LOWERBOUND":
                    alpha = max(alpha, entry['value'])
                elif entry['flag'] == "UPPERBOUND":
                    beta = min(beta, entry['value'])
                if alpha >= beta:
                    return entry['value'], entry['bestmove']

        result = board.result(claim_draw=True) # 3-fold repetition automatically triggered on Lichess
        if result != "*" or depth == 0:
            perspective = maximizer*2-1
            return perspective*ChessAI.evaluation(board, result, depth), None

        bestmove = None
        value = float("-inf")
        sorted_moves = list(board.legal_moves)
        sorted_moves.sort(key=lambda move: ChessAI._get_move_priority(board, move), reverse=True)
        if last_best_move is not None and last_best_move in sorted_moves:
            sorted_moves.remove(last_best_move)
            sorted_moves.insert(0, last_best_move)
        for move in sorted_moves:
            board.push(move)
            eval_, _, = self.negamax(board, depth-1, not maximizer, -beta, -alpha, asked_depth, last_best_move)
            board.pop()
            value = max(value, -eval_)
            if value > alpha:
                bestmove = move
                alpha = value
                if alpha >= beta:
                    break

        if value <= alphaOrig:
            flag = "UPPERBOUND"
        elif value >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"
        self.evaluated_positions[key] = {'value': value, 'depth': depth, 'flag': flag, 'bestmove': bestmove}
        return value, bestmove

    def find_move(self, timeout: float = 30) -> chess.Move:
        """ Find the best move according to the evaluation function."""
        if self.board.legal_moves.count() == 1:
            return list(self.board.legal_moves)[0]
        remaining_time = timeout
        last_best_move = None
        self.evaluated_positions = {}
        for depth in range(2, self.max_depth + 1):
            backup = copy.deepcopy(self.board) # Last thread will timeout. Attributes (like self.board) would be altered
            start = time.time()
            x = ThreadWithReturnValue(target=self.negamax, args=(backup, depth, self.color, float("-inf"), float("inf"), depth, last_best_move,))
            x.start()
            x.join(timeout=remaining_time)
            if x.result is None:
                break
            eval_, last_best_move = x.result
            last_run_duration = time.time() - start
            remaining_time -= last_run_duration
            # print(f"{eval_:>+3d} {self.board.san(move):<6} {self.board.variation_san(variant):<50} {nnodes:>6} nodes in {last_run_duration:.2f}s {len(self.evaluated_positions)}")
            print(f"{eval_:>+3d} {self.board.san(last_best_move):<6} in {last_run_duration:.2f}s {len(self.evaluated_positions)}")
            if remaining_time < timeout / 2:
                return last_best_move
        return last_best_move

    def time_manager(self, remaining_time):
        tau = 50
        n_moves_played = len(self.board.move_stack) // 2
        return remaining_time / (tau * math.exp(-2*n_moves_played/tau))

    def play(self):
        while not self.board.is_game_over():
            move = self.find_move(timeout=10)
            if move is None:
                break
            self.board.push(move)
            print(self.board.unicode(borders=False, empty_square='⭘', orientation=True))


if __name__ =='__main__':
    chess_ai = ChessAI(5, chess.WHITE)
    chess_ai.play()