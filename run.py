# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


from training.Trainer import ReconTrainer
from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

model_path = './models/'
load_model = True
load_opponent_model = load_model
train_initial_model_path = 'train_loop_160'
opponent_initial_model_path = 'opponent_loop_'

score = 0.00
score_smoothing = 0.995

game_stat_path = 'Performance Stats 8.7.csv'
net_stat_path = 'Network Stats 8.7.csv'
max_batch_size = 80
learning_rate = 1e-2
clip = 0.2
n_opponents = 5


trainer = ReconTrainer(model_path,load_model,load_opponent_model,train_initial_model_path,
	opponent_initial_model_path,score,score_smoothing,game_stat_path,net_stat_path,max_batch_size,
	learning_rate,clip,n_opponents)

#n_rounds = 128//workers
n_rounds = 20
n_moves = 4112//n_opponents
max_turns_per_game = 70


epochs = 4
equalize_weights_on_score = 0.06 #0.12 #approx 55% win rate
save_every_n = 20


trainer.train(n_rounds,n_moves,epochs,equalize_weights_on_score,save_every_n,max_turns_per_game)