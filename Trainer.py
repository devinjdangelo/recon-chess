import numpy as np
from reconchess import *
import random
import time
import csv
from copy import deepcopy
from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

from Sharedmem import SharedArray
from Network import ReconChessNet
from ReconBot import ReconBot


class ReconTrainer:
    # implements training procedures for ReconBot
    # by interfacing with reconchess api
    def __init__(self,model_path,load_model,opponent_initial_model_path,score,score_smoothing,game_stat_path,net_stat_path):

        self.model_path = model_path
        self.game_stat_path = game_stat_path
        self.net_stat_path = net_stat_path


        self.bootstrap = SharedArray((workers,),dtype=np.int32)
        self.splits = SharedArray((workers,),dtype=np.int32)

        self.score = SharedArray((1,),dtype=np.float32)
        self.wins = SharedArray((1,),dtype=np.float32)
        self.losses = SharedArray((1,),dtype=np.float32)
        self.ties = SharedArray((1,),dtype=np.float32)
        self.train_color = SharedArray((1,),dtype=np.int32)

        self.score_smoothing = score_smoothing

        if rank==0:
            self.score = score
            self.wins = 0
            self.losses = 0
            self.ties =0

            self.train_net = ReconChessNet('train')
            self.train_agent = ReconBot(net=self.train_net,verbose=False,name='train')

            self.opponent_net = ReconChessNet('opponent')
            self.opponent_agent = ReconBot(net=self.opponent_net,verbose=False,name='opponent')

            if not load_model:
                self.train_agent.init_net()
                self.opponent_agent.init_net()
                self.train_net.lstm_stateful.set_weights(self.train_net.lstm.get_weights())
                self.opponent_net.set_weights(self.train_net.get_weights())            

            else:
                self.train_net.load_weights(self.model_path)
                self.train_net.lstm_stateful.set_weights(self.train_net.lstm.get_weights())
                if opponent_initial_model_path is not None:
                    self.opponent_net.load_weights(opponent_initial_model_path)
                    self.opponent_net.lstm_stateful.set_weights(self.opponent_net.lstm.get_weights())
                else:
                    self.opponent_net.set_weights(self.train_net.get_weights()) 


            with open(self.game_stat_path,'w',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerows([('Loop','Round','ngames','Wins','Losses','Ties','Score Avg')])

            with open(self.net_stat_path,'w',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerows([('Batch Size','Avg Game Len','Loss','Policy Loss','Entropy','Value Loss','Grad Norm')])

        else:
            #off rank 0, give full agents but with no network
            self.train_agent = ReconBot(verbose=False,name='train')
            self.opponent_agent = ReconBot(verbose=False,name='opponent')


    def play_n_moves(self,n_moves):
        #adapted from reconchess.play.play_local_game() 
        #gathers n_moves of experience, restarting the game as many times as needed
        #white -> white player agent
        #black -> black player agent

        
        if rank==0:
            self.splits = np.zeros(self.splits.shape,dtype=np.int32)
            self.bootstrap = np.ones(self.bootstrap.shape,dtype=np.int32)
            bootstrap = np.ones((workers,n_moves),dtype=np.int32)
            split_idx = []

            train_as_white = random.choice([True,False])
            if train_as_white:
                self.train_color[:] = 1
            else:
                self.train_color[:] = 0

        comm.Barrier()
        if self.train_color == 1:
            train_as_white = True
        else:
            train_as_white = False

        obs_memory = np.zeros(shape=(workers,n_moves*2,13,8,8),dtype=np.float32)
        mask_memory = np.zeros(shape=(workers,n_moves,4096),dtype=np.int32)
        #action,prob,value
        action_memory = np.zeros(shape=(workers,n_moves*2,3),dtype=np.float32)
        rewards = np.zeros(shape=(workers,n_moves*2,),dtype=np.float32)

        total_turns = 0

        
        if train_as_white:
            white = self.train_agent
            black = self.opponent_agent
        else:
            black = self.train_agent
            white = self.opponent_agent

        need_to_switch_colors = False

        while total_turns//2<n_moves:

            if need_to_switch_colors:
                white,black = black,white
                need_to_switch_colors = False

            game = LocalGame()


            white_name = white.__class__.__name__
            black_name = black.__class__.__name__

            white.handle_game_start(chess.WHITE, game.board.copy(), white_name)
            black.handle_game_start(chess.BLACK, game.board.copy(), black_name)
            game.start()

            players = [black, white]

            while not game.is_over() and total_turns//2<n_moves:
                    

                comm.Barrier()
                if rank==0:
                    split_idx += [i*total_turns for i in range(workers) if self.splits[i]==1]
                    self.splits[:] = [0]*workers

                player = players[game.turn]
                sense_actions = game.sense_actions()
                move_actions = game.move_actions()
                comm.Barrier()
                notify_opponent_move_results(game, player)
                comm.Barrier()


                if player is self.train_agent:
                    if rank==0:
                        obs_memory[:,total_turns,:,:,:] = np.copy(self.train_agent.obs)
                    comm.Barrier()
                    play_sense(game, player, sense_actions, move_actions)
                    comm.Barrier()
                    if rank==0:
                        action_memory[:,total_turns,:] = np.copy(self.train_agent.action_memory)
                        rewards[:,total_turns] = [0]*workers
                    comm.Barrier()
                    total_turns += 1
                else:
                    comm.Barrier()
                    play_sense(game, player, sense_actions, move_actions)
                    comm.Barrier()
                    comm.Barrier()

                if player is self.train_agent:
                    if rank==0:
                        obs_memory[:,total_turns,:,:,:] = np.copy(self.train_agent.obs)
                    comm.Barrier()
                    play_move(game, player, move_actions)
                    comm.Barrier()
                    if rank==0:
                        mask_memory[:,total_turns//2,:] = np.copy(self.train_agent.mask)
                        action_memory[:,total_turns,:] = np.copy(self.train_agent.action_memory)
                        rewards[:,total_turns] = [0]*workers
                    comm.Barrier()
                    
                    total_turns += 1
                else:
                    comm.Barrier()
                    play_move(game, player, move_actions)
                    comm.Barrier()
                    comm.Barrier()


            game.end()
            winner = game.get_winner_color()
            win_reason = game.get_win_reason()
            game_history = game.get_game_history()

            white.handle_game_end(winner, win_reason, game_history)
            black.handle_game_end(winner, win_reason, game_history)

            if winner is not None:
                #if white wins, need to switch 
                need_to_switch_colors = winner
                if (winner and train_as_white) or (not winner and not train_as_white):
                    rewards[rank,total_turns-1] += 1
                    self.splits[rank] = 1
                    self.score = 1*(1-self.score_smoothing)+ self.score * self.score_smoothing
                    self.wins += 1
                elif (not winner and train_as_white) or (winner and not train_as_white):
                    rewards[rank,total_turns-1] -= 1
                    self.splits[rank] = 1
                    self.score = -1*(1-self.score_smoothing)+ self.score * self.score_smoothing
                    self.losses += 1
            else:
                self.score = self.score * self.score_smoothing
                self.ties += 1

            if total_turns == n_moves*2:
                if rank==0:
                    terminal_state_value = self.train_agent.get_terminal_v()
                if winner is not None:
                    self.bootstrap[rank] = 0
                comm.Barrier()
                if rank == 0:
                    bootstrap[:,-1] = np.copy(self.bootstrap)
                    bootstrap[:,-2] = list(range(workers))

            
                if rank==0:
                    memory = [obs_memory,mask_memory,action_memory,rewards]
                    memory = [np.reshape(m,(-1,)+m.shape[2:]) for m in memory]
                    split_idx.sort()
                    memory = [np.split(m,split_idx,axis=0) for m in memory]

                    print(bootstrap)
                    bootstrap = np.split(np.reshape(bootstrap,(-1,)),split_idx)
                    print(bootstrap)
                    gae = []
                    for i in range(len(memory[0])):
                        should_bootstrap = bootstrap[i][-1]==1
                        print(should_bootstrap)
                        if should_bootstrap:
                            print(bootstrap[i])
                            tv = terminal_state_value[[bootstrap[i][-2]]]
                        gae += self.GAE(memory[3][i],memory[2][i][:,-1],bootstrap=bootstrap,terminal_state_value=tv)

                    memory.append(gae)
                else:
                    memory = [[],[],[],[],[],[]]




        return memory

    def collect_exp(self,n_rounds,n_moves,loop):
        game_memory = [[],[],[],[],[],[]]
        for game in range(n_rounds):
            outmem = self.play_n_moves(n_moves)
            if rank==0:
                ngames = len(outmem[0])
                print(ngames,' Games Played ',self.wins, ' Wins ',self.losses,' losses ',self.ties,' ties ',self.score, ' score')

                with open(self.game_stat_path,'a',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerow([loop,game,ngames,self.wins,self.losses,self.ties,self.score])

                self.wins = 0
                self.losses = 0
                self.ties = 0

                game_memory = [game_memory[i]+outmem[i] for i in range(len(game_memory))]

        return game_memory

    def train(self,n_rounds,n_moves,epochs,equalize_weights_every_n,save_every_n):
        loop = 1

        while True:
            mem = self.collect_exp(n_rounds,n_moves,loop)

            if rank==0:
                samples_available = list(range(len(mem[0])))
                batch_size = len(samples_available)//2
                n_batches = len(samples_available)//batch_size*epochs

                for i in range(n_batches):
                    sample_idx = np.random.choice(samples_available,replace=False,size=batch_size)
                    samples_available = [idx for idx in samples_available if idx not in sample_idx]
                    if len(samples_available)<batch_size:
                        samples_available = list(range(update_n))
                        
                    batch = [[m[idx] for idx in sample_idx] for m in mem]
                    loss,pg_loss,entropy,vf_loss,g_n = self.send_batch(batch)
                    print('batch_size',batch_size,'Loss: ',loss,' Policy Loss: ',pg_loss,' Entropy: ',entropy,' Value Loss: ',vf_loss,' Grad Norm: ',g_n)

                    with open(self.net_stat_path,'a',newline='', encoding='utf-8') as output:
                        wr = csv.writer(output)
                        wr.writerows([(loss,pg_loss,entropy,vf_loss,g_n)])

                self.train_net.lstm_stateful.set_weights(self.train_net.lstm.get_weights())

                #if loop%equalize_weights_every_n==0:
                    #self.opponent_net.set_weights(self.train_net.get_weights)

                if loop%save_every_n==0:
                    self.train_net.save_weights(self.model_path)

            loop += 1
            comm.Barrier()

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
    def GAE(rewards,values,g=0.99,l=0.95,bootstrap=False,terminal_state_value=None):
    #rewards: length timesteps list of rewards recieved from environment
    #values: length timesteps list of state values
    #g: gamma discount factor
    #l: lambda discount factor
    #terminal_state_value: network value of terminal state if episode cut off before true end
    #bootstrap: if true, bootstrap estimated return after episode timeout (approximates continuing rather than episodic version of problem)
        tmax = len(rewards)
        lastadv = 0
        GAE = [0] * (tmax)
        for t in reversed(range(tmax)):
            if t == tmax - 1 and not bootstrap:
                delta = rewards[t] - values[t]
                GAE[t] = lastadv = delta 
            elif t==tmax-1 and bootstrap:
                delta = rewards[t] + g * terminal_state_value - values[t]
                GAE[t] = lastadv = delta + g*l*lastadv
            else:
                delta = rewards[t] + g * values[t+1] - values[t]
                GAE[t] = lastadv = delta + g*l*lastadv
         
        return GAE


            
