import numpy as np
from reconchess import *

from Network import ReconChessNet
from ReconBot import ReconBot


class ReconTrainer:
	# implements training procedures for ReconBot
	# by interfacing with reconchess api
	def __init__(self):
		self.train_net = ReconChessNet()
		self.train_agent = ReconBot(net=self.train_net)
		self.train_agent.init_net()

		self.opponent_net = ReconChessNet()
		self.opponent_agent = ReconBot(net=self.opponent_net)
		self.opponent_agent.init_net()

		self.opponent_net.set_weights(self.train_net.get_weights())

	def play_game(self,white,black):
		#adapted from reconchess.play.play_local_game()
		#white -> white player agent
		#black -> black player agent
		game = LocalGame()

		white_player.handle_game_start(chess.WHITE, game.board.copy(), white_name)
	    black_player.handle_game_start(chess.BLACK, game.board.copy(), black_name)
	    game.start()
	    

	def train(self):
		pass


if __name__=="__main__":
	trainer = ReconTrainer()
