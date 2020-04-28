
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

color = 0 if rank==0 else 1
cpu_comm = comm.Split(color,0)

from .Sharedmem import SharedArray
from .Network import ReconChessNet
from .ReconBot import ReconBot

class ReconPlayer:
    def __init__(self,name,max_batch_size,learning_rate,use_cpu):
        
        self.net = ReconChessNet(name,max_batch_size,learning_rate,use_cpu)
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

        self.score = SharedArray((workers-1,self.n_opponents),dtype=np.float32)
        self.wins = SharedArray((workers-1,),dtype=np.float32)
        self.losses = SharedArray((workers-1,),dtype=np.float32)
        self.ties = SharedArray((workers-1,),dtype=np.float32)
        #self.train_color = SharedArray((1,),dtype=np.int32)

        self.score_smoothing = score_smoothing

        if rank==0:
            self.score[:,:] = score
            self.wins[:] = [0]*(workers-1)
            self.losses[:] = [0]*(workers-1)
            self.ties[:] = [0]*(workers-1)

            with open(self.game_stat_path,'w',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerows([('Loop','Round','ngames','Wins','Losses','Ties','Score Avg','Win Avg','Loss Avg','Tie Avg','scoremin','scoremax')])

            with open(self.net_stat_path,'w',newline='', encoding='utf-8') as output:
                    wr = csv.writer(output)
                    wr.writerows([('iter','Loss','Policy Loss','Entropy','Value Loss','Grad Norm')])

            use_cpu = False #gpu used for training on rank 0, inference on rank 1+

        else:
            use_cpu = True
        
        self.train_player = ReconPlayer('train',max_batch_size,learning_rate,use_cpu)
        self.opponents = [ReconPlayer('opponent '+str(i),max_batch_size,learning_rate,use_cpu) for i in range(self.n_opponents)]
        self.next_to_sync_with_train = 0

        self.train_player.agent.init_net()
        for opponent in self.opponents:
            opponent.agent.init_net()

        if load_model:
            print('loading train: ',self.model_path+train_initial_model_path)
            self.train_player.net.load_weights(self.model_path+train_initial_model_path)
            self.train_player.sync_lstm_weights()
            if load_opponent_model and opponent_initial_model_path is not None:
                print('loading opponents: ',opponent_initial_model_path+train_initial_model_path[-3:])
                #load specific models as opponents
                for i in range(self.n_opponents):
                    self.opponents[i].net.load_weights(self.model_path+opponent_initial_model_path+str(i)+train_initial_model_path[-3:])
                    self.opponents[i].sync_lstm_weights()  
        else:
            self.train_player.sync_lstm_weights()
            self.synchronize_weights()


    def synchronize_weights(self):
        players = [self.train_player] + self.opponents
        for player in players:
            if rank==0:
                weights = player.net.get_weights()
            else:
                weights = None #need to sync from master

            weights = comm.bcast(weights,root=0)

            if rank!=0:
                player.net.set_weights(weights)
            



    def play_game(self,train_as_white,maximum_moves):
        #adapted from reconchess.play.play_local_game() 
        #white -> white player agent
        #black -> black player agent
        game = LocalGame()

        opponent_number = random.choice(list(range(len(self.opponents))))
        if train_as_white:
            white = self.train_player.agent
            black = self.opponents[opponent_number].agent
        else:
            black = self.train_player.agent
            white = self.opponents[opponent_number].agent

        white_name = white.__class__.__name__
        black_name = black.__class__.__name__

        white.handle_game_start(chess.WHITE, game.board.copy(), white_name)
        black.handle_game_start(chess.BLACK, game.board.copy(), black_name)
        game.start()

        players = [black, white]

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

            if player is self.train_player.agent:
                obs_memory[turn,:,:,:] = np.copy(self.train_player.agent.obs)
                play_sense(game, player, sense_actions, move_actions)
                action_memory[turn,:] = np.copy(self.train_player.agent.action_memory)
                rewards[turn] = 0
                turn += 1
            else:
                play_sense(game, player, sense_actions, move_actions)

            if player is self.train_player.agent:
                obs_memory[turn,:,:,:] = np.copy(self.train_player.agent.obs)
                play_move(game, player, move_actions)
                mask_memory[turn//2,:] = np.copy(self.train_player.agent.mask)[0,:]
                action_memory[turn,:] = np.copy(self.train_player.agent.action_memory)
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

        if winner is not None:
            #if white wins, need to switch 
            if (winner and train_as_white) or (not winner and not train_as_white):
                rewards[turn-1] += 1
                self.score[rank-1,opponent_number] = 1*(1-self.score_smoothing)+ self.score[rank-1,opponent_number] * self.score_smoothing
                self.wins[rank-1] += 1
            elif (not winner and train_as_white) or (winner and not train_as_white):
                rewards[turn-1] -= 1
                self.score[rank-1,opponent_number] = -1*(1-self.score_smoothing)+ self.score[rank-1,opponent_number] * self.score_smoothing
                self.losses[rank-1] += 1
        else:
            self.score[rank-1,opponent_number] = self.score[rank-1,opponent_number] * self.score_smoothing
            self.ties[rank-1] += 1

        if turn//2==maximum_moves and winner is None:
            terminal_state_value = self.train_player.agent.get_terminal_v()[0]
            should_bootstrap = True
        else:
            terminal_state_value = None
            should_bootstrap = False

        

        memory = [obs_memory,mask_memory,action_memory,rewards]
        newsize = [turn,turn//2,turn,turn]
        memory = [np.resize(mem,(newsize[i],)+mem.shape[1:]) for i,mem in enumerate(memory)]

        gae = self.GAE(memory[3],memory[2][:,-1],bootstrap=should_bootstrap,terminal_state_value=terminal_state_value)
        memory.append(gae)

        return memory,turn,game_history

    def collect_exp(self,n_moves,max_turns_per_game,loop):
        game_memory = [[],[],[],[],[]]
        moves_played = 0
        game_num = 0
        while moves_played<n_moves:
            game_num += 1
            if moves_played + max_turns_per_game*2 > n_moves:
                max_turns_per_game = (n_moves - moves_played)//2
            train_as_white = random.choice([True,False])
            outmem,moves,game_history = self.play_game(train_as_white,max_turns_per_game)
            game_memory = [game_memory[i]+[outmem[i]] for i in range(len(game_memory))]
            moves_played += moves
            #print(f'Rank {rank} playing game {game_num}, moves {moves_played} out of {n_moves}')
            if rank==1:
                if loop%10==0 and game_num<=10:
                    colorstr = 'white' if train_as_white else 'black'
                    game_history.save('./replays/loop'+str(loop)+'step'+str(game_num)+colorstr+'.json')

        cpu_comm.barrier()
        if rank==1:
            tot_wins,tot_losses,tot_ties = np.sum(self.wins),np.sum(self.losses),np.sum(self.ties)
            ngames = tot_wins + tot_losses + tot_ties
            self.win_avg = self.win_avg*0.8 + (1-0.8)*tot_wins/ngames
            self.loss_avg = self.loss_avg*0.8 + (1-0.8)*tot_losses/ngames
            self.tie_avg = self.tie_avg*0.8 + (1-0.8)*tot_ties/ngames

            print('loop: ', loop,' Games Played: ',ngames,' Wins: ',tot_wins, ' losses: ',tot_losses,
                ' ties: ',tot_ties,' score: ',np.mean(self.score),' Win pct: ','{0:.2f}'.format(self.win_avg),' loss pct: ','{0:.2f}'.format(self.loss_avg))

            with open(self.game_stat_path,'a',newline='', encoding='utf-8') as output:
                wr = csv.writer(output)
                wr.writerow([loop,game,ngames,tot_wins,tot_losses,tot_ties,np.mean(self.score),self.win_avg,self.loss_avg,self.tie_avg,np.amin(np.mean(self.score,0)),np.amax(np.mean(self.score,0))])

            self.wins[:] = [0]*(workers-1)
            self.losses[:] = [0]*(workers-1)
            self.ties[:] = [0]*(workers-1)



                

        return game_memory

    def train(self,n_moves,epochs,equalize_weights_on_score,save_every_n,max_turns_per_game):
        loop = 1
        total_steps_gathered = 0
        start_time = time.time()
        if rank==1:
            self.win_avg = 0.45
            self.loss_avg = 0.45
            self.tie_avg = 0.1
        while True:
            if rank != 0:
                mem = self.collect_exp(n_moves,max_turns_per_game,loop)
                mem_gathered = comm.gather(mem,root=0)
            elif rank==0:
                mem = None
                print('Rank 0 waiting for data...')
                mem_gathered = comm.gather(mem,root=0)
                mem = [[],[],[],[],[]]
                for g in mem_gathered:
                    if g is None:
                        continue
                    else:
                        for i,data in enumerate(g):
                            mem[i] += data
                print(f'Rank 0 got {len(mem[0])} episodes')
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
            self.synchronize_weights()

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


            

