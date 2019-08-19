from Trainer import ReconTrainer
from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

model_path = './models/'
load_model = False
opponent_initial_model_path = None

score = 0
score_smoothing = 0.99

game_stat_path = 'Performance Stats 1.csv'
net_stat_path = 'Network Stats 1.csv'


trainer = ReconTrainer(model_path,load_model,opponent_initial_model_path,score,score_smoothing,game_stat_path,net_stat_path)

#n_rounds = 128//workers
n_rounds = 1
n_moves = 256*2
max_turns_per_game = 50

epochs = 3
equalize_weights_every_n = 100
save_every_n = 100


trainer.train(n_rounds,n_moves,epochs,equalize_weights_every_n,save_every_n,max_turns_per_game)