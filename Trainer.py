import numpy as np
from reconchess import *
import random
import time
from copy import deepcopy

from Network import ReconChessNet
from ReconBot import ReconBot


class ReconTrainer:
    # implements training procedures for ReconBot
    # by interfacing with reconchess api
    def __init__(self):
        self.train_net = ReconChessNet('train')
        self.train_agent = ReconBot(net=self.train_net,verbose=False)
        self.train_agent.init_net()

        self.opponent_net = ReconChessNet('opponent')
        self.opponent_agent = ReconBot(net=self.opponent_net,verbose=False)
        self.opponent_agent.init_net()

        self.train_net.lstm_stateful.set_weights(self.train_net.lstm.get_weights())
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
        obs_memory = np.zeros(shape=(maximum_moves*2,13,8,8),dtype=np.float32)
        mask_memory = np.zeros(shape=(maximum_moves,4096),dtype=np.int32)
        #action,prob,value
        action_memory = np.zeros(shape=(maximum_moves*2,3),dtype=np.float32)
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
                rewards[turn] = 0
                turn += 1
            else:
                play_sense(game, player, sense_actions, move_actions)

            if player is self.train_agent:
                obs_memory[turn,:,:,:] = np.copy(self.train_agent.obs)
                play_move(game, player, move_actions)
                mask_memory[turn//2,:] = np.copy(self.train_agent.mask)[0,:]
                action_memory[turn,:] = np.copy(self.train_agent.action_memory)
                rewards[turn] = 0
                turn += 1
            else:
                play_move(game, player, move_actions)


        game.end()
        winner = game.get_winner_color()
        win_reason = game.get_win_reason()
        game_history = game.get_game_history()

        white.handle_game_end(winner, win_reason, game_history)
        black.handle_game_end(winner, win_reason, game_history)


        memory = [obs_memory,mask_memory,action_memory,rewards]
        newsize = [turn,turn//2,turn,turn]
        memory = [np.resize(mem,(newsize[i],)+mem.shape[1:]) for i,mem in enumerate(memory)]

        return memory,winner,win_reason

    def collect_exp(self,n_games,loop,score):
        game_memory = [[],[],[],[],[]]
        for game in range(n_games):
            color = random.choice([True,False])
            if color:
                outmem,winner,win_reason = self.play_game(self.train_agent,self.opponent_agent)
            else:
                outmem,winner,win_reason = self.play_game(self.opponent_agent,self.train_agent)

            if winner is None:
                score = 0.995*score
                print('loop ',loop,' Game ',game,' No winner after',outmem[0].shape[0]//2,' moves',' current score: ','{:.5f}'.format(score))
            elif winner==color:
                printcolor = 'white' if color else 'black'
                outmem[3][-1] += 1
                score = 0.995*score + 0.005*1
                print('loop ',loop,' Game ',game,' Trainbot wins as ',printcolor,' in',outmem[0].shape[0]//2,' moves by ',win_reason,' current score: ','{:.5f}'.format(score))
            elif winner!=color:
                printcolor = 'white' if color else 'black'
                outmem[3][-1] -= 1
                score = 0.995*score + 0.005*(-1)
                print('loop ',loop,' Game ',game,' Trainbot loses as ',printcolor,' in',outmem[0].shape[0]//2,' moves by ',win_reason,' current score: ','{:.5f}'.format(score))

            outmem.append(self.GAE(outmem[3],outmem[2][:,-1]))
            [mem.append(outmem[i]) for i,mem in enumerate(game_memory)]


        return game_memory

    def train(self,update_n,batch_size,epochs):
        assert update_n%batch_size==0
        n_batches = update_n//batch_size
        loop = 1
        score = 0
        while True:
            mem = self.collect_exp(update_n,loop,score)
            samples_available = list(range(update_n))
            for i in range(n_batches):
                sample_idx = np.random.choice(samples_available,replace=False,size=batch_size)
                if len(samples_available)<batch_size:
                    samples_available = list(range(update_n))
                else:
                    samples_available = [idx for idx in samples_available if idx not in sample_idx]
                batch = [[m[idx] for idx in sample_idx] for m in mem]
                loss,pg_loss,entropy,vf_loss,g_n = self.send_batch(batch)
                print('Loss: ',loss,' Policy Loss: ',pg_loss,' Entropy: ',entropy,' Value Loss: ',vf_loss,' Grad Norm: ',g_n)

            self.train_net.lstm_stateful.set_weights(self.train_net.lstm.get_weights())
            #if loop%5==0:
                #self.opponent_net.set_weights(self.train_net.get_weights)



    def send_batch(self,batch):
        inputs = batch[0]
        mask = batch[1]
        a_taken = [b[:,0] for b in batch[2]]
        lg_prob_old = [b[:,1] for b in batch[2]]
        old_v_pred = [b[:,2] for b in batch[2]]
        gae = batch[4]

        gae_flat = [i for episode in deepcopy(gae) for i in episode]
        gae_u,gae_o = np.mean(gae_flat),np.std(gae_flat)+1e-8

        gae = [(g-gae_u)/gae_o for g in gae]

        returns = [old_v_pred[i]+gae[i] for i in list(range(len(gae)))]
        clip = 0.2

        return self.train_net.update(inputs,mask,lg_prob_old,a_taken,gae,old_v_pred,returns,clip)


    @staticmethod
    def GAE(rewards,values,g=0.99,l=0.95,bootstrap=False):
    #rewards: length timesteps list of rewards recieved from environment
    #values: length timesteps list (timesteps +1 if bootstrap) of state values
    #g: gamma discount factor
    #l: lambda discount factor
    #bootstrap: if true, bootstrap estimated return after episode timeout (approximates continuing rather than episodic version of problem)
        tmax = len(rewards)
        lastadv = 0
        GAE = [0] * (tmax)
        for t in reversed(range(tmax)):
            if t == tmax - 1 and bootstrap==False:
                delta = rewards[t] - values[t]
                GAE[t] = lastadv = delta 
            else:
                delta = rewards[t] + g * values[t+1] - values[t]
                GAE[t] = lastadv = delta + g*l*lastadv
         
        return GAE


            




if __name__=="__main__":
    trainer = ReconTrainer()
    trainer.train(100,50,3)
