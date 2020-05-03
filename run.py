from training.Trainer import ReconTrainer
from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

model_path = './models/'
load_model = False
load_opponent_model = load_model
train_initial_model_path = 'train_loop_62'
opponent_initial_model_path = 'opponent_loop_'

score = 0.00
score_smoothing = 0.995

game_stat_path = 'Performance Stats 2.1.csv'
net_stat_path = 'Network Stats 2.1.csv'
max_batch_size = 80
learning_rate = 1e-2
clip = 0.2
n_opponents = 5
strongest = 4


trainer = ReconTrainer(model_path,load_model,load_opponent_model,train_initial_model_path,
	opponent_initial_model_path,score,score_smoothing,game_stat_path,net_stat_path,max_batch_size,
	learning_rate,clip,n_opponents,strongest)

n_moves = 10240
max_turns_per_game = 70


epochs = 4
equalize_weights_on_score = 0.05 #0.12 #approx 55% win rate
save_every_n = 1


trainer.train(n_moves,epochs,equalize_weights_on_score,save_every_n,max_turns_per_game)