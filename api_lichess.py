import threading

import chess
import berserk

from chess_ai import ChessAI


class LichessWorker(threading.Thread):

    def __init__(self, client, game_id, color, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.ai = ChessAI(7, chess.WHITE if color == 'white' else chess.BLACK)

    def have_to_move(self, moves, game_state_event):
        if game_state_event['status'] != 'started': # This fails to detect draw on opponent move as the event on opponent move is still status=started
            return False
        if len(moves) == self.ai.board.move_stack:
            return False
        last_player = chess.WHITE if len(moves) % 2 != 0 else chess.BLACK
        return last_player != self.ai.color

    def handle_state_change(self, game_state_event):
        # print(game_state_event)
        moves = game_state_event['moves'].split()
        played_move = chess.Move.from_uci(moves[-1])
        print(f"{self.ai.board.san(played_move)} played")
        self.ai.board.push(played_move)
        print("\n", self.ai.board.unicode(borders=False, empty_square='⭘', orientation=self.ai.color), "\n")
        if self.have_to_move(moves, game_state_event):
            move = self.ai.find_move(timeout=20)
            self.client.bots.make_move(self.game_id, move) # Can fail if opponent drew?

    def run(self):
         # If white, initiate the first move
        if self.ai.color == chess.WHITE:
            move = self.ai.find_move(timeout=20)
            self.client.bots.make_move(self.game_id, move)
        for event in self.stream:
            if event['type'] == 'gameState': # When a move is played, a draw offered or the game ends.
                self.handle_state_change(event)
            if event['type'] == 'gameFinish':
                return


class LichessBot:

    def __init__(self, token_session):
        self.client = berserk.Client(session=berserk.TokenSession(token_session))

    def handle_challenge(self, event):
        self.client.bots.accept_challenge(event['challenge']['id'])

    def launch(self):
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                self.handle_challenge(event)
            elif event['type'] == 'gameStart':
                game = LichessWorker(self.client, event['game']['id'], event['game']['color'])
                game.start()


if __name__ =='__main__':
    client = LichessBot(token_session="Your token")
    client.launch()
