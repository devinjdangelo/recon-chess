import random
from reconchess import *

PieceDict = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
        'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

class ReconBot(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        
    def _fen_to_obs(fen):
        obs = np.zeros(shape=(13,8,8),dtype=np.int32)
        row = 0
        col = 0
        for char in fen:
            if char.isalpha():
                charint = PieceDict[char]
                obs[charint,col,row] = 1
                row += 1
            elif char=='/':
                col += 1
                row = 0
            elif char.isdigit():
                row += int(char)
            else:
                return obs
            
    def _get_obs(self):
        self.obs = self._fen_to_obs(self.board.fen())
        

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        self._get_obs()

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        return None

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass
