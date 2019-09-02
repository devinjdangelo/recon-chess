from training.Trainer import ReconTrainer
from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

model_path = './models/'
load_model = False
load_opponent_model = False
train_initial_model_path = 'train_loop_20'
opponent_initial_model_path = 'opponent_loop_20'

score = 0
score_smoothing = 0.999

game_stat_path = 'Performance Stats 1.4.csv'
net_stat_path = 'Network Stats 1.4.csv'
max_batch_size = 64
learning_rate = 1e-2


trainer = ReconTrainer(model_path,load_model,load_opponent_model,train_initial_model_path,
	opponent_initial_model_path,score,score_smoothing,game_stat_path,net_stat_path,max_batch_size,learning_rate)

#n_rounds = 128//workers
n_rounds = 1
n_moves = 4112
max_turns_per_game = 96


epochs = 3
equalize_weights_on_score = 0.1 #approx 55% win rate
save_every_n = 20


trainer.train(n_rounds,n_moves,epochs,equalize_weights_on_score,save_every_n,max_turns_per_game)