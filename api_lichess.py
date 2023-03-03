import datetime
import threading
import time

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
        self.ai = ChessAI(10, chess.WHITE if color == 'white' else chess.BLACK)

    def have_to_move(self, moves, game_state_event):
        if game_state_event['status'] != 'started': # This fails to detect draw on opponent move as the event on opponent move is still status=started
            return False
        if len(moves) == self.ai.board.move_stack:
            return False
        last_player = chess.WHITE if len(moves) % 2 != 0 else chess.BLACK
        return last_player != self.ai.color

    def handle_state_change(self, game_state_event):
        moves = game_state_event['moves'].split()
        if len(moves) == len(self.ai.board.move_stack): # or check for wdraw or bdraw in event keys
            return # Could be a draw offer. Ignore it.
        played_move = chess.Move.from_uci(moves[-1])
        print(f"{self.ai.board.san(played_move)} played")
        self.ai.board.push(played_move)
        print("\n", self.ai.board.unicode(borders=False, empty_square='⭘', orientation=self.ai.color), "\n")
        if self.have_to_move(moves, game_state_event):
            time_key = 'wtime' if self.ai.color == chess.WHITE else 'btime'
            seconds_left = (game_state_event[time_key] - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)).total_seconds()
            allocate_time = self.ai.time_manager(seconds_left)
            print(f"{seconds_left}s left. Allocating {allocate_time:.2f}s")
            move = self.ai.find_move(timeout=allocate_time)
            self.client.bots.make_move(self.game_id, move)

    def run(self):
         # If white, initiate the first move
        if self.ai.color == chess.WHITE:
            move = self.ai.find_move(timeout=3)
            self.client.bots.make_move(self.game_id, move)

        for event in self.stream:
            if event['type'] == 'gameFinish' or ('status' in event and event['status'] != 'started'):
                print(f"End of game {self.game_id}") # Could also be the opponent disconnexion
                return
            if event['type'] == 'gameState': # When a move is played, a draw offered or the game ends.
                self.handle_state_change(event)


class LichessBot:

    def __init__(self, token_session):
        self.client = berserk.Client(session=berserk.TokenSession(token_session))

    def handle_challenge(self, event):
        # self.client.bots.accept_challenge(event['challenge']['id'])
        pass

    def launch(self):
        self.client.challenges.create('maia1', True, clock_limit=60, clock_increment=0)
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                self.handle_challenge(event)
            elif event['type'] == 'gameStart':
                game = LichessWorker(self.client, event['game']['id'], event['game']['color'])
                game.start()
            elif event['type'] == 'gameFinish':
                time.sleep(1)
                self.client.challenges.create('maia1', True, clock_limit=60, clock_increment=0)


if __name__ =='__main__':
    with open("token.txt", "r") as token_file:
        token = token_file.read()
        print(token)
    client = LichessBot(token_session=token)
    client.launch()
