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

from .Sharedmem import SharedArray
from .Network import ReconChessNet
from .ReconBot import ReconBot

class ReconPlayer:
    def __init__(self,name,max_batch_size,learning_rate,no_net=False):
        if not no_net:
            self.net = ReconChessNet(name,max_batch_size,learning_rate)
        else:
            self.net=None
        self.agent = ReconBot(net=self.net,verbose=False,name=name)

    def sync_lstm_weights(self):
        self.net.lstm_stateful.set_weights(self.net.lstm.get_weights())


class ReconTrainer:
    # implements training procedures for ReconBot
    # by interfacing with reconchess api
    def __init__(self,model_path,load_model,load_opponent_model,train_initial_model_path,opponent_initial_model_path,
        score,score_smoothing,game_stat_path,net_stat_path,max_batch_size,learning_rate,clip,n_opponents):

        self.model_path = model_path
        self.game_stat_path = game_stat_path
        self.net_stat_path = net_stat_path

        self.clip = clip
        self.n_opponents = n_opponents


        self.bootstrap = SharedArray((workers,),dtype=np.int32)
        self.splits = SharedArray((workers,),dtype=np.int32)

        self.score = SharedArray((workers,self.n_opponents),dtype=np.float32)
        self.wins = SharedArray((workers,),dtype=np.float32)
        self.losses = SharedArray((workers,),dtype=np.float32)
        self.ties = SharedArray((workers,),dtype=np.float32)
        self.train_color = SharedArray((1,),dtype=np.int32)
        self.next_opponet_to_play = SharedArray((1,),dtype=np.int32)

        self.score_smoothing = score_smoothing

        if rank==0:
            self.score[:,:] = score
            self.wins[:] = [0]*workers
            self.losses[:] = [0]*workers
            self.ties[:] = [0]*workers

            self.train_player = ReconPlayer('train',max_batch_size,learning_rate)
            self.opponents = [ReconPlayer('opponent '+str(i),max_batch_size,learning_rate) for i in range(self.n_opponents)]
            self.next_to_sync_with_train = 0

            self.train_player.agent.init_net()
            for opponent in self.opponents:
                opponent.agent.init_net()

            if not load_model:
                self.train_player.sync_lstm_weights()
                for opponent in self.opponents:
                    opponent.sync_lstm_weights()
            else:
                print('loading train: ',self.model_path+train_initial_model_path)
                self.train_player.net.load_weights(self.model_path+train_initial_model_path)
                self.train_player.sync_lstm_weights()
                if load_opponent_model and opponent_initial_model_path is not None:
                    print('loading opponents: ',opponent_initial_model_path+train_initial_model_path[-3:])
                    #load specific models as opponents
                    for i in range(self.n_opponents):
                        self.opponents[i].net.load_weights(self.model_path+opponent_initial_model_path+str(i)+train_initial_model_path[-3:])
                        self.opponents[i].sync_lstm_weights()  


            with open(self.game_stat_path,'w',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerows([('Loop','Round','ngames','Wins','Losses','Ties','Score Avg','Win Avg','Loss Avg','Tie Avg','scoremin','scoremax')])

            with open(self.net_stat_path,'w',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerows([('iter','Loss','Policy Loss','Entropy','Value Loss','Grad Norm')])

        else:
            #off rank 0, give full agents but with no network
            self.train_player = ReconPlayer('train',max_batch_size,learning_rate,no_net=True)
            self.opponents = [ReconPlayer('opponent'+str(i),max_batch_size,learning_rate,no_net=True) for i in range(self.n_opponents)]


    def play_n_moves(self,n_moves,max_turns_per_game,loop):
        #adapted from reconchess.play.play_local_game() 
        #gathers n_moves of experience, restarting the game as many times as needed
        #white -> white player agent
        #black -> black player agent

        max_replays_to_save = 5
        if rank==0:
            self.splits[:] = [0]*workers
            self.bootstrap[:] = [0]*workers
            bootstrap = np.ones((workers,n_moves*2),dtype=np.int32)
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
            white = self.train_player.agent
            black = self.opponents[self.next_opponet_to_play[0]].agent
        else:
            black = self.train_player.agent
            white = self.opponents[self.next_opponet_to_play[0]].agent

        need_to_switch_colors = False

        while total_turns//2<n_moves:
            if need_to_switch_colors:
                white,black = black,white
                need_to_switch_colors = False
                train_as_white = not train_as_white

            game = LocalGame()


            white_name = white.__class__.__name__
            black_name = black.__class__.__name__

            white.handle_game_start(chess.WHITE, game.board.copy(), white_name)
            black.handle_game_start(chess.BLACK, game.board.copy(), black_name)
            game.start()

            players = [black, white]
            game_turns = 0
            while not game.is_over() and total_turns//2<n_moves and game_turns//2<=max_turns_per_game:
                    

                comm.Barrier()
                if rank==0:
                    split_idx += [i*n_moves*2+total_turns for i in range(workers) if self.splits[i]==1]
                    self.splits[:] = [0]*workers

                #print('rank: ',rank, ' global turn: ',total_turns,' game_turn: ',game_turns)

                player = players[game.turn]
                sense_actions = game.sense_actions()
                move_actions = game.move_actions()
                comm.Barrier()
                notify_opponent_move_results(game, player)
                comm.Barrier()


                if player is self.train_player.agent:
                    if rank==0:
                        obs_memory[:,total_turns,:,:,:] = np.copy(self.train_player.agent.obs)
                    comm.Barrier()
                    play_sense(game, player, sense_actions, move_actions)
                    comm.Barrier()
                    if rank==0:
                        action_memory[:,total_turns,:] = np.copy(self.train_player.agent.action_memory)
                        rewards[:,total_turns] = [0]*workers
                    comm.Barrier()
                    total_turns += 1
                    game_turns += 1
                else:
                    comm.Barrier()
                    play_sense(game, player, sense_actions, move_actions)
                    comm.Barrier()
                    comm.Barrier()

                if player is self.train_player.agent:
                    if rank==0:
                        obs_memory[:,total_turns,:,:,:] = np.copy(self.train_player.agent.obs)
                    comm.Barrier()
                    play_move(game, player, move_actions)
                    comm.Barrier()
                    if rank==0:
                        mask_memory[:,total_turns//2,:] = np.copy(self.train_player.agent.mask)
                        action_memory[:,total_turns,:] = np.copy(self.train_player.agent.action_memory)
                        rewards[:,total_turns] = [0]*workers
                    comm.Barrier()
                    
                    total_turns += 1
                    game_turns += 1
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

            if rank==0 and loop%20==0 and max_replays_to_save>0:
                colorstr = 'white' if train_as_white else 'black'
                game_history.save('./replays/loop'+str(loop)+'step'+str(total_turns)+colorstr+'.json')
                max_replays_to_save -= 1


            if game_turns//2>max_turns_per_game and total_turns < n_moves*2:
                game_turns = 0
                self.splits[rank] = 1
                if player is white:
                    need_to_switch_colors = True

            if winner is not None:
                #if white wins, need to switch 
                need_to_switch_colors = winner
                if (winner and train_as_white) or (not winner and not train_as_white):
                    rewards[rank,total_turns-1] += 1
                    self.splits[rank] = 1
                    self.score[rank,self.next_opponet_to_play[0]] = 1*(1-self.score_smoothing)+ self.score[rank,self.next_opponet_to_play[0]] * self.score_smoothing
                    self.wins[rank] += 1
                elif (not winner and train_as_white) or (winner and not train_as_white):
                    rewards[rank,total_turns-1] -= 1
                    self.splits[rank] = 1
                    self.score[rank,self.next_opponet_to_play[0]] = -1*(1-self.score_smoothing)+ self.score[rank,self.next_opponet_to_play[0]] * self.score_smoothing
                    self.losses[rank] += 1
            else:
                self.score[rank,self.next_opponet_to_play[0]] = self.score[rank,self.next_opponet_to_play[0]] * self.score_smoothing
                self.ties[rank] += 1

            if total_turns == n_moves*2:
                if rank==0:
                    terminal_state_value = self.train_player.agent.get_terminal_v()
                if winner is not None:
                    self.bootstrap[rank] = 0
                comm.Barrier()
                if rank == 0:
                    bootstrap[:,-1] = np.copy(self.bootstrap)
                    bootstrap[:,-2] = list(range(workers))

            
                if rank==0:
                    memory = [obs_memory,mask_memory,action_memory,rewards]
                    memory = [np.reshape(m,(-1,)+m.shape[2:]) for m in memory]
                    split_idx += [i*n_moves*2+total_turns for i in range(workers-1)]
                    split_idx.sort()
                    splits_by_mem = [split_idx,[idx//2 for idx in split_idx],split_idx,split_idx]
                    memory = [np.split(m,splits_by_mem[i],axis=0) for i,m in enumerate(memory)]

                    bootstrap = np.split(np.reshape(bootstrap,(-1,)),split_idx)
                    gae = []
                    for i in range(len(memory[0])):
                        should_bootstrap = bootstrap[i][-1]==1
                        if should_bootstrap:
                            tv = terminal_state_value[[bootstrap[i][-2]]][0]
                        else:
                            tv = None
                        gae.append(self.GAE(memory[3][i],memory[2][i][:,-1],bootstrap=should_bootstrap,terminal_state_value=tv))

                    memory.append(gae)

                else:
                    memory = [[],[],[],[],[]]

        if rank==0:
            if self.next_opponet_to_play<self.n_opponents-1:
                self.next_opponet_to_play[:] += 1
            else:
                self.next_opponet_to_play[:] = 0




        return memory

    def collect_exp(self,n_rounds,n_moves,max_turns_per_game,loop):
        game_memory = [[],[],[],[],[]]
        for game in range(n_rounds):
            outmem = self.play_n_moves(n_moves,max_turns_per_game,loop)
            if rank==0:
                ngames = len(outmem[0])
                tot_wins,tot_losses,tot_ties = np.sum(self.wins),np.sum(self.losses),np.sum(self.ties)
                self.win_avg = self.win_avg*0.8 + (1-0.8)*tot_wins/ngames
                self.loss_avg = self.loss_avg*0.8 + (1-0.8)*tot_losses/ngames
                self.tie_avg = self.tie_avg*0.8 + (1-0.8)*tot_ties/ngames

                print('loop: ', loop,' Games Played: ',ngames,' Wins: ',tot_wins, ' losses: ',tot_losses,
                    ' ties: ',tot_ties,' score: ',np.mean(self.score),' Win pct: ','{0:.2f}'.format(self.win_avg),' loss pct: ','{0:.2f}'.format(self.loss_avg))

                with open(self.game_stat_path,'a',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerow([loop,game,ngames,tot_wins,tot_losses,tot_ties,np.mean(self.score),self.win_avg,self.loss_avg,self.tie_avg,np.amin(np.mean(self.score,0)),np.amax(np.mean(self.score,0))])

                self.wins[:] = [0]*workers
                self.losses[:] = [0]*workers
                self.ties[:] = [0]*workers

                game_memory = [game_memory[i]+outmem[i] for i in range(len(game_memory))]

        return game_memory

    def train(self,n_rounds,n_moves,epochs,equalize_weights_on_score,save_every_n,max_turns_per_game):
        loop = 1
        total_steps_gathered = 0
        start_time = time.time()
        if rank==0:
            self.win_avg = 0.45
            self.loss_avg = 0.45
            self.tie_avg = 0.1
        while True:
            if rank==0:
                samples_available = list(range(len(mem[0])))
                total_steps_gathered += sum([len(m) for m in mem[0]])
                batch_size = len(samples_available)//2
                n_batches = len(samples_available)//batch_size*epochs

                #print(len(samples_available),batch_size,n_batches)

                for i in range(n_batches):
                    sample_idx = np.random.choice(samples_available,replace=False,size=batch_size)
                    samples_available = [idx for idx in samples_available if idx not in sample_idx]
                    if len(samples_available)<batch_size:
                        samples_available = list(range(len(mem[0])))
                        
                    batch = [[m[idx] for idx in sample_idx] for m in mem]
                    loss,pg_loss,entropy,vf_loss,g_n = self.send_batch(batch)
                    print('iter: ',i, 'batch_size: ',batch_size,' Loss: ',loss,' Policy Loss: ',pg_loss,' Entropy: ',entropy,' Value Loss: ',vf_loss,' Grad Norm: ',g_n)

                    with open(self.net_stat_path,'a',newline='', encoding='utf-8') as output:
                        wr = csv.writer(output)
                        wr.writerows([(i,loss,pg_loss,entropy,vf_loss,g_n)])

                self.train_player.sync_lstm_weights

                if np.amin(np.mean(self.score,0)) >= equalize_weights_on_score:
                    #once desired performance is achieved, equalized opponent/train weights 
                    #and reset performance metrics
                    print('equalizing weights')
                    self.opponents[self.next_to_sync_with_train].net.set_weights(self.train_player.net.get_weights())
                    self.next_to_sync_with_train = 0 if self.next_to_sync_with_train==self.n_opponents-1 else self.next_to_sync_with_train+1
                    self.score[:,:] = 0
                    self.win_avg = 0.45
                    self.loss_avg = 0.45
                    self.tie_avg = 0.10

                if loop%save_every_n==0:
                    self.train_player.net.save_weights(self.model_path+'train_loop_'+str(loop))
                    for i in range(self.n_opponents):
                        self.opponents[i].net.save_weights(self.model_path+'opponent_loop_'+str(i)+str(loop))

                steps_per_second = total_steps_gathered/(time.time()-start_time)
                msteps_per_day = steps_per_second*60*60*24/1e6
                print('loop: ',loop,' steps per second: ','{0:.2f}'.format(steps_per_second),' million steps per day: ','{0:.2f}'.format(msteps_per_day))
            loop += 1
            comm.Barrier()

    def send_batch(self,batch):
        inputs = batch[0]
        mask = batch[1]
        a_taken = [b[:,0] for b in batch[2]]
        lg_prob_old = [b[:,1] for b in batch[2]]
        old_v_pred = [b[:,2] for b in batch[2]]
        gae = batch[4]

        returns = [old_v_pred[i]+gae[i] for i in list(range(len(gae)))]

        gae_flat = [i for episode in deepcopy(gae) for i in episode]
        gae_u,gae_o = np.mean(gae_flat),np.std(gae_flat)+1e-8

        gae = [(g-gae_u)/gae_o for g in gae]


        return self.train_player.net.update(inputs,mask,lg_prob_old,a_taken,gae,old_v_pred,returns,self.clip)


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


            
