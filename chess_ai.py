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
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        return (
            chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
            3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
            3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
            5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
            9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
        )

    @staticmethod
    def evaluation(board: chess.Board, result, depth, three_fold_repetition) -> int:
        """ Evaluate `board`. The evaluation is based on material only.
        Positive sign for White, negative for Black."""
        if result == '1/2-1/2':
            return 0
        white_win_value = 1000 + depth
        if result == '1-0':
            return white_win_value
        if result == '0-1':
            return -white_win_value
        if three_fold_repetition:
            return 0
        return ChessAI._count_material_value(board, chess.WHITE)

    def _get_move_priority(self, board: chess.Board, move: chess.Move):
        """ Give a move a priority score for the evaluation in order to sort the
        moves to make the Alpha-Beta pruning converge faster.
        This has no effect on the evaluation result.
        """
        score = 0
        # Taking a good opponent piece with a bad piece should prioritized
        if board.is_capture(move):
            taker = board.piece_type_at(move.from_square)
            taken = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
            mvv_lva_scores = [[0, 0, 0, 0, 0, 0, 0],        # victim K, attacker K, Q, R, B, N, P, None
                              [50, 51, 52, 53, 54, 55, 0],  # victim Q, attacker K, Q, R, B, N, P, None
                              [40, 41, 42, 43, 44, 45, 0],  # victim R, attacker K, Q, R, B, N, P, None
                              [30, 31, 32, 33, 34, 35, 0],  # victim B, attacker K, Q, R, B, N, P, None
                              [20, 21, 22, 23, 24, 25, 0],  # victim N, attacker K, Q, R, B, N, P, None
                              [10, 11, 12, 13, 14, 15, 0],  # victim P, attacker K, Q, R, B, N, P, None
                              [0, 0, 0, 0, 0, 0, 0]]        # victim None, attacker K, Q, R, B, N, P, None
            score += mvv_lva_scores[taken][taker]

        if move in self.evaluated_positions:
            score += 60

        return score

    def sort_moves(self, board: chess.Board, last_best_move=None):
        sorted_moves = list(board.legal_moves)
        sorted_moves.sort(key=lambda move: self._get_move_priority(board, move), reverse=True)
        if last_best_move is not None and last_best_move in sorted_moves:
            sorted_moves.remove(last_best_move)
            sorted_moves.insert(0, last_best_move)
        return sorted_moves

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
                    return entry['value'], entry['bestmove'], 0, []
                elif entry['flag'] == "LOWERBOUND":
                    alpha = max(alpha, entry['value'])
                elif entry['flag'] == "UPPERBOUND":
                    beta = min(beta, entry['value'])
                if alpha >= beta:
                    return entry['value'], entry['bestmove'], 0, []

        # 3-fold repetition automatically triggers the draw on Lichess so we have to take care of it
        # result(claim_draw=True) would call can_claim_threefold_repetition which is overkill as it would
        # check all the possible next moves of the position.
        # we only need to check if the current position is a 3fold repetition
        three_fold_repetition = board.is_repetition()
        result = board.result(claim_draw=False)
        if result != "*" or depth == 0 or three_fold_repetition:
            n_moves = asked_depth if depth == 0 else asked_depth - depth
            perspective = maximizer*2-1
            return perspective*ChessAI.evaluation(board, result, depth, three_fold_repetition), None, 1, board.move_stack[-n_moves:]

        bestmove = None
        best_variant = []
        n_evaluated_positions = 0
        value = float("-inf")
        for move in self.sort_moves(board):
            board.push(move)
            eval_, _, count, variant = self.negamax(board, depth-1, not maximizer, -beta, -alpha, asked_depth, last_best_move)
            n_evaluated_positions += count
            board.pop()
            value = max(value, -eval_)
            if value > alpha:
                bestmove = move
                alpha = value
                best_variant = variant
                if alpha >= beta:
                    break

        if value <= alphaOrig:
            flag = "UPPERBOUND"
        elif value >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"
        self.evaluated_positions[key] = {'value': value, 'depth': depth, 'flag': flag, 'bestmove': bestmove}
        return value, bestmove, n_evaluated_positions, best_variant

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
            eval_, last_best_move, nnodes, variant = x.result

            # eval_, last_best_move, nnodes, variant = self.negamax(backup, depth, self.color, float("-inf"), float("inf"), depth, last_best_move)

            last_run_duration = time.time() - start
            remaining_time -= last_run_duration
            print(f"{eval_:>+3d} {self.board.san(last_best_move):<6} {self.board.variation_san(variant):<50} {nnodes:>6} nodes in {last_run_duration:.2f}s (TT={len(self.evaluated_positions)})")
            if remaining_time < timeout / 2:
                return last_best_move
        return last_best_move

    def time_manager(self, remaining_time):
        tau = 50
        n_moves_played = len(self.board.move_stack) // 2
        return remaining_time / (tau * math.exp(-2*n_moves_played/tau))

    def play(self):
        while not self.board.is_game_over():
            move = self.find_move(timeout=1)
            if move is None:
                break
            self.board.push(move)
            print(self.board.unicode(borders=False, empty_square='⭘', orientation=True))


if __name__ =='__main__':
    chess_ai = ChessAI(8, chess.WHITE)
    chess_ai.play()
