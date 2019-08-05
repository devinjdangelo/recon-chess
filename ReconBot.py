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

    def _square_to_col_row(square: Square):
        return square//8,square%8

    def _piece_idx_at_col_row(col,row):
        return np.argmax(self.obs[:,col,row])
        

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        self._get_obs()

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        #when the opponent moves, we no longer have certainty that any previously known pieces
        #have not moved. Here we remove any piece that could have taken any legal move from the
        #known game state observation.

        legal_moves = [_square_to_col_row(move.from_square) for move in self.board.legal_moves]
        legal_col,legal_row = list(zip(*legal_moves))

        if self.color:
            self.obs[6:11,legal_col,legal_row] = 0
        else:
            self.obs[:6,legal_col,legal_row] = 0
            
        if captured_my_piece:
            col,row = self._square_to_col_row(capture_square)
            self.obs[:11,col,row] = 0
            self.obs[12,col,row] = 1
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)
            col,row = self._square_to_col_row(capture_square)
            piece_idx = PieceDict[piece.symbol]
            self.obs[piece_idx,col,row] = 1

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        action = random.choice(move_actions)
        return action

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        if taken_move is not None:
            self.board.push(taken_move)
            #update observation, zero old location and set new location to 1
            col,row = self._square_to_col_row(taken_move.from_square)
            piece_idx = self._piece_idx_at_col_row(col,row)
            self.obs[piece_idx,col,row] = 0
            col,row = self._square_to_col_row(taken_move.to_square)
            self.obs[piece_idx,col,row] = 1


    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass
