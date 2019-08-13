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

	    players = [black, white]

	    maximum_moves = 50
	    obs_memory = np.zeros(shape=(maximum_moves*2,13,8,8),dtype=np.int32)
	    mask_memory = np.zeros(shape=(maximum_moves,13,8,8),dtype=np.int32)
	    #action,prob,value
	    action_memory = np.zeros(shape=(maximum_moves*2,3),dtype=np.int32)
	    rewards = np.zeros(shape=(maximum_moves*2,),dtype=np.float32)


	    turn = 0
	    while not game.is_over() and turn//2<maximum_moves:
	    	player = players[game.turn]
	    	sense_actions = game.sense_actions()
		    move_actions = game.move_actions()

		    notify_opponent_move_results(game, player)

		    if player is self.train_agent:
		    	obs_memory[turn,:,:,:] = np.copy(self.train_agent.obs)
		    	action_memory[turn,:] = np.copy(self.train_agent.action_memory)
		    	turn += 1

		    play_sense(game, player, sense_actions, move_actions)

		    if player is self.train_agent:
		    	obs_memory[turn,:,:,:] = np.copy(self.train_agent.obs)
		    	mask_memory[turn//2,:,:,:] = np.copy(self.train_agent.mask)
		    	action_memory[turn,:] = np.copy(self.train_agent.action_memory)
		    	turn += 1

		    play_move(game, player, move_actions)

		memory = [obs_memory,mask_memory,action_memory,rewards]
		memory = [np.resize(mem,(turn,)+mem.shape[1:]) for mem in memory]

		return memory

	def train(self):
		pass


if __name__=="__main__":
	trainer = ReconTrainer()
