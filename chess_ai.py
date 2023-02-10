import time
from typing import List, Tuple, Optional

import chess
import chess.pgn


MATERIAL_VALUE = {chess.PAWN: 1, chess.ROOK: 5, chess.KNIGHT: 3, chess.BISHOP: 3, chess.QUEEN: 9}


class ChessAI:

    def __init__(self, depth, color):
        self.max_depth = depth
        self.color = color
        self.board = chess.Board()
        self.game = chess.pgn.Game()

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

    @staticmethod
    def negamax(board: chess.Board, depth: int, maximizer: chess.Color,
                alpha: float, beta: float, asked_depth: int, last_best_move: Optional[chess.Move]) -> Tuple[float, chess.Move, int, list[chess.Move]]:
        """ Negamax tree search. Return the best move of the position based on
        the evaluation function. It aloso return its associated value and
        variant and the total visited positions."""
        if depth == 0 or board.is_game_over():
            if depth == 0:
                path = board.move_stack[-asked_depth:]
            else:
                n_moves = asked_depth - depth
                path = board.move_stack[-n_moves:]
            return (maximizer*2-1)*ChessAI.evaluation(board), None, 1, path

        bestmove = None
        best_variant = []
        value = float("-inf")
        total_nodes = 0
        sorted_moves = list(board.legal_moves)
        sorted_moves.sort(key=lambda move: ChessAI._get_move_priority(board, move), reverse=True)
        if last_best_move is not None and last_best_move in sorted_moves:
            sorted_moves.remove(last_best_move)
            sorted_moves.insert(0, last_best_move)
        for move in sorted_moves:
            board.push(move)
            eval_, _, count, variant = ChessAI.negamax(board, depth-1, not maximizer, -beta, -alpha, asked_depth, last_best_move=None)
            total_nodes += count
            board.pop()
            value = max(value, -eval_)
            if value > alpha:
                bestmove = move
                alpha = value
                best_variant = variant
                if alpha >= beta:
                    break
        return value, bestmove, total_nodes, best_variant

    def find_move(self, timeout: float = 30) -> chess.Move:
        """ Find the best move according to the evaluation function."""
        depth = 1
        last_run_duration = 0.0
        consumed_time = 0.0
        last_best_move = None
        while consumed_time < timeout - last_run_duration and depth <= self.max_depth:
            start = time.time()
            eval_, move, nnodes, variant = ChessAI.negamax(self.board, depth, self.color, float("-inf"), float("inf"), depth, last_best_move=last_best_move)
            last_run_duration = time.time() - start
            consumed_time += last_run_duration
            print(f"{eval_:>+3d} {self.board.san(move):<6} {self.board.variation_san(variant):<50} {nnodes:>6} nodes in {last_run_duration:.2f}s")
            depth +=1
            last_best_move = move
        print("")
        return move

    def play(self):
        while not self.board.is_game_over():
            move = self.find_move(timeout=10)
            self.board.push(move)


if __name__ =='__main__':
    chess_ai = ChessAI(8, chess.WHITE)
    chess_ai.play()