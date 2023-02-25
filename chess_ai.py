import copy
import time
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
    def evaluation(board: chess.Board) -> int:
        """ Evaluate `board`. The evaluation is based on material only.
        Positive sign for White, negative for Black."""
        result = board.result()
        if result == '1-0':
            return 1000
        if result == '0-1':
            return -1000
        if result == '1/2-1/2': # or board.can_claim_draw():
            return 0
        eval_ = ChessAI._count_material_value(board, chess.WHITE) - \
            ChessAI._count_material_value(board, chess.BLACK)
        return eval_#0 if board.can_claim_draw() else eval_

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

    # def negamax(self, board: chess.Board, depth: int, maximizer: chess.Color,
    #             alpha: float, beta: float, asked_depth: int, last_best_move: Optional[chess.Move]) -> Tuple[float, chess.Move, int, list[chess.Move]]:
    #     """ Negamax tree search. Return the best move of the position based on
    #     the evaluation function. It also returns its associated value and
    #     variant and the total visited positions.
    #     """
    #     alphaOrig = alpha
    #     key = zobrist_hash(board)
    #     if entry:= self.evaluated_positions.get(key):
    #         if entry['depth'] >= depth:
    #             if entry['flag'] == "EXACT":
    #                 return entry['value'], entry['bestmove'], 0, []
    #             elif entry['flag'] == "LOWERBOUND":
    #                 alpha = max(alpha, entry['value'])
    #             elif entry['flag'] == "UPPERBOUND":
    #                 beta = min(beta, entry['value'])
    #             if alpha >= beta:
    #                 return entry['value'], entry['bestmove'], 0, []

    #     if depth == 0 or board.is_game_over():
    #         n_moves = asked_depth if depth == 0 else asked_depth - depth
    #         return (maximizer*2-1)*ChessAI.evaluation(board), None, 1, board.move_stack[-n_moves:]

    #     bestmove = None
    #     best_variant = []
    #     value = float("-inf")
    #     total_nodes = 0
    #     sorted_moves = list(board.legal_moves)
    #     sorted_moves.sort(key=lambda move: ChessAI._get_move_priority(board, move), reverse=True)
    #     if last_best_move is not None and last_best_move in sorted_moves:
    #         sorted_moves.remove(last_best_move)
    #         sorted_moves.insert(0, last_best_move)
    #     for move in sorted_moves:
    #         board.push(move)
    #         eval_, _, count, variant = self.negamax(board, depth-1, not maximizer, -beta, -alpha, asked_depth, last_best_move)
    #         total_nodes += count
    #         board.pop()
    #         value = max(value, -eval_)
    #         if value > alpha:
    #             bestmove = move
    #             alpha = value
    #             best_variant = variant
    #             if alpha >= beta:
    #                 break

    #     if value <= alphaOrig:
    #         flag = "UPPERBOUND"
    #     elif value >= beta:
    #         flag = "LOWERBOUND"
    #     else:
    #         flag = "EXACT"
    #     self.evaluated_positions[key] = {'value': value, 'depth': depth, 'flag': flag, 'bestmove': bestmove}
    #     assert bestmove is not None
    #     return value, bestmove, total_nodes, best_variant

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

        if depth == 0 or board.is_game_over():
            return (maximizer*2-1)*ChessAI.evaluation(board), None

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
        # assert bestmove is not None
        return value, bestmove


    # def find_move(self, timeout: float = 30) -> chess.Move:
    #     """ Find the best move according to the evaluation function."""
    #     remaining_time = timeout
    #     last_best_move = None
    #     backup = copy.deepcopy(self.board)
    #     for depth in range(1, self.max_depth + 1):
    #         start = time.time()
    #         x = ThreadWithReturnValue(target=self.negamax, args=(self.board, depth, self.color, float("-inf"), float("inf"), depth, last_best_move,))
    #         x.start()
    #         x.join(timeout=remaining_time)
    #         if x.result is None:
    #             break
    #         # eval_, move, nnodes, variant = x.result
    #         # print(eval_, move, nnodes, variant)
    #         eval_, move = x.result
    #         # print(eval_, move)
    #         # assert move is not None
    #         last_run_duration = time.time() - start
    #         last_best_move = move
    #         remaining_time -= last_run_duration
    #         # print(f"{eval_:>+3d} {self.board.san(move):<6} {self.board.variation_san(variant):<50} {nnodes:>6} nodes in {last_run_duration:.2f}s {len(self.evaluated_positions)}")
    #         print(f"{eval_:>+3d} {self.board.san(move):<6} in {last_run_duration:.2f}s {len(self.evaluated_positions)}")

    #     # Last thread has timeout. Attributes (like self.board) have been altered and must be restored
    #     self.board = backup
    #     return move

    def find_move(self, timeout: float = 30) -> chess.Move:
        """ Find the best move according to the evaluation function."""
        last_best_move = None
        eval_, move = self.negamax(self.board, self.max_depth, self.color, float("-inf"), float("inf"), self.max_depth, last_best_move)
        print(f"{eval_:>+3d} {self.board.san(move):<6} {len(self.evaluated_positions)}")
        return move



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