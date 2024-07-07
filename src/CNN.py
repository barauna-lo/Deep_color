import optuna
import functions as fn
import torch
import sys
import report as rp
import warnings
warnings.filterwarnings('ignore')

#LOAD DATA

beta = 2
win_in = 1024
win_ou = 2048
n_series = 2**8
data1 = torch.load(f'data_X{win_in}_Y{win_ou}_n{n_series}.pt')
trainX, trainY, trainX_test,trainY_test = data1[beta][0],data1[beta][1],data1[beta][2],data1[beta][3]


#EXPERIMENT

#n_cut = 1
low_freq_cut= 0
high_freq_cut= 0
train_size = len(trainX[0])
test_size =  len(trainY[0])

def objective(trial):
    data1 = torch.load(f'data_X{win_in}_Y{win_ou}_n{n_series}.pt')
    trainX, trainY, trainX_test,trainY_test = data1[beta][0],data1[beta][1],data1[beta][2],data1[beta][3]
    train_size = len(trainX[0])
    test_size =  len(trainY[0])
    title         = 'CNN' 
    epochs        = 10
    low_freq_cut  = 0
    high_freq_cut = 0
    
    neuron_min = 32
    neuron_max = 1024

    drop_max = 0.3
    drop_min = 0.0

    n_layers      = trial.suggest_int('n_layers', 1, 5)
    n_cut         = trial.suggest_int('n_cut', 1, 5)
    
    activations1    = trial.suggest_categorical('activations1',["sigmoid", "relu", "tanh","const"])
    activations2    = trial.suggest_categorical('activations2',["sigmoid", "relu", "tanh","const"])
    activations3    = trial.suggest_categorical('activations3',["sigmoid", "relu", "tanh","const"])
    activations4    = trial.suggest_categorical('activations4',["sigmoid", "relu", "tanh","const"])
    activations5    = trial.suggest_categorical('activations5',["sigmoid", "relu", "tanh","const"])
    
    dropout1       = trial.suggest_float('dropout1', drop_min, drop_max)
    dropout2       = trial.suggest_float('dropout2', drop_min, drop_max)
    dropout3       = trial.suggest_float('dropout3', drop_min, drop_max)
    dropout4       = trial.suggest_float('dropout4', drop_min, drop_max)
    dropout5       = trial.suggest_float('dropout5', drop_min, drop_max)

    kernel_size1    = 2*trial.suggest_int('kernel_size1', 1, 3) +1
    kernel_size2    = 2*trial.suggest_int('kernel_size2', 1, 3) +1
    kernel_size3    = 2*trial.suggest_int('kernel_size3', 1, 3) +1
    kernel_size4    = 2*trial.suggest_int('kernel_size4', 1, 3) +1
    kernel_size5    = 2*trial.suggest_int('kernel_size5', 1, 3) +1

    in_channels1    = 2*(kernel_size1 + trial.suggest_int('in_channels1', 1, 3))+1
    in_channels2    = 2*(kernel_size2 + trial.suggest_int('in_channels2', 1, 3))+1
    in_channels3    = 2*(kernel_size3 + trial.suggest_int('in_channels3', 1, 3))+1
    in_channels4    = 2*(kernel_size4 + trial.suggest_int('in_channels4', 1, 3))+1
    in_channels5    = 2*(kernel_size5 + trial.suggest_int('in_channels5', 1, 3))+1

    lr              = trial.suggest_float('lr', 1e-4, 5e-1)
    optimiser       = trial.suggest_categorical('optimiser',['LBFGS','SGD','Adam'])

    model = fn.CNN(n_layers=n_layers, 
                 activations     = [activations1,activations2,activations3,activations4,activations5],
                 dropouts        = [dropout1,dropout2,dropout3,dropout4,dropout5],
                 in_channels     = [in_channels1,in_channels2,in_channels3,in_channels4,in_channels5],
                 kernel_size     = [kernel_size1,kernel_size2,kernel_size3,kernel_size4,kernel_size5],
                 stride          = [1,1,1,1,1,1,1,1,1])

    l = fn.DBeta(n_cut,
              low_freq_cut=low_freq_cut,
              high_freq_cut=high_freq_cut)
    
    loss = fn.training_loop(epochs,
                         model, 
                         l, 
                         trainX.float(), trainY.float(), 
                         trainX_test.float(), trainY_test.float(),
                         optimiser = optimiser,
                         future=len(trainY[0]),
                         n_cut=n_cut,
                         lr = lr,
                         plot_results_during_traing=False,
                         save_results = True,
                         verbose = False,
                         file_name=f'out_{title}_t{trial.number}_train{train_size}_test{test_size}_beta{beta}.plk')
    return loss

if __name__ == "__main__":
	win_in = int(sys.argv[1])
	win_ou = int(sys.argv[2])
	beta   = int(sys.argv[3])
	study = optuna.create_study(direction='minimize')
	study.optimize(objective, n_trials=127,n_jobs=2)

	print('Number of finished trials:', len(study.trials))
	print(f'Best trial:',study.best_trial.params) 
	rp.report(f'CNN_{win_in}_{win_ou}_{beta}')
