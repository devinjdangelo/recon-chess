import numpy as np
from reconchess import *
import random
import time

from Network import ReconChessNet
from ReconBot import ReconBot


class ReconTrainer:
	# implements training procedures for ReconBot
	# by interfacing with reconchess api
	def __init__(self):
		self.train_net = ReconChessNet()
		self.train_agent = ReconBot(net=self.train_net,verbose=False)
		self.train_agent.init_net()

		self.opponent_net = ReconChessNet()
		self.opponent_agent = ReconBot(net=self.opponent_net,verbose=False)
		self.opponent_agent.init_net()

		self.opponent_net.set_weights(self.train_net.get_weights())

	def play_game(self,white,black):
		#adapted from reconchess.play.play_local_game() 
		#white -> white player agent
		#black -> black player agent
		game = LocalGame()

		white_name = white.__class__.__name__
		black_name = black.__class__.__name__

		white.handle_game_start(chess.WHITE, game.board.copy(), white_name)
		black.handle_game_start(chess.BLACK, game.board.copy(), black_name)
		game.start()

		players = [black, white]

		maximum_moves = 50
		obs_memory = np.zeros(shape=(maximum_moves*2,13,8,8),dtype=np.int32)
		mask_memory = np.zeros(shape=(maximum_moves,4096),dtype=np.int32)
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
				play_sense(game, player, sense_actions, move_actions)
				action_memory[turn,:] = np.copy(self.train_agent.action_memory)
				turn += 1
			else:
				play_sense(game, player, sense_actions, move_actions)

			if player is self.train_agent:
				obs_memory[turn,:,:,:] = np.copy(self.train_agent.obs)
				play_move(game, player, move_actions)
				mask_memory[turn//2,:] = np.copy(self.train_agent.mask)[0,:]
				action_memory[turn,:] = np.copy(self.train_agent.action_memory)
				turn += 1
			else:
				play_move(game, player, move_actions)

		memory = [obs_memory,mask_memory,action_memory,rewards]
		memory = [np.resize(mem,(turn,)+mem.shape[1:]) for mem in memory]

		return memory

	def train(self):
		games = 100
		game_memory = []
		for game in range(games):
			print('Now playing game ',game)
			color = random.choice([True,False])
			if color:
				outmem = self.play_game(self.train_agent,self.opponent_agent)
			else:
				outmem = self.play_game(self.opponent_agent,self.train_agent)

			game_memory.append(outmem)


if __name__=="__main__":
	trainer = ReconTrainer()
	t = time.time()
	trainer.train()
	elapsed_time = time.time()-t
	print('Played 100 games in ',elapsed_time,' seconds')
	print('That is ',elapsed_time/100,' games per second')

	time.sleep(20)
