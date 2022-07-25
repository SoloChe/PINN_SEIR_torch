# from socket import NI_DGRAM
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import random
import matplotlib.pylab as plt
# import matplotlib.gridspec as gridspec

from utility.data_simu_COVID import model
from utility.neural_net import DNN 
from utility.neural_net import Modified_DNN
from pinn import PINN_COVID

import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(torch.__version__)

np.random.seed(10)
random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

def state_init(n_x, n_y): 
    
    S_xy_0 = np.abs(np.random.normal(150, 30, size=(n_x, n_y)).astype(int))
    N = np.sum(S_xy_0)
    print(f'number of people = {N}')

    # Infected
    I_xy_0 = np.zeros((n_x, n_y))
    I_ind = []
    for _ in range(5):
        ia, ib = random.randint(0,n_x-1), random.randint(0,n_y-1)
        I_ind.append((ia, ib))
        I_xy_0[ia, ib] +=  int(S_xy_0[ia, ib]/5)
        S_xy_0[ia, ib] -= int(S_xy_0[ia, ib]/5)

    n_infected = np.sum(I_xy_0)    
    print(f'number of infected people = {n_infected}')


    # Recoverd
    R_xy_0 = np.zeros((n_x, n_y)) 
    E_xy_0 = np.zeros((n_x, n_y)) 

    u_0 = np.concatenate((np.expand_dims(S_xy_0,axis=(0)), 
                            np.expand_dims(E_xy_0,axis=(0)), 
                            np.expand_dims(I_xy_0,axis=(0)),
                            np.expand_dims(R_xy_0,axis=(0)),
                            ), axis=0)/N

    assert u_0.shape == (4, n_x, n_y)
    return u_0, N

def get_simu_data(u_0, h, dt, n_x, n_y, n_timestep, para_simu, N):
   
    m = model(n_x, n_y, n_timestep, para_simu, N)
    u_t = m.simu(u_0, h, dt)

    print(u_t.shape)

    X_simu = np.concatenate( [np.sum( u_t[i], axis=(1,2))[None,...] for i in range(n_timestep)], axis=0)
    return u_t, X_simu
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PINN on SEIR Model')
    parser.add_argument('--n_x', default=10, type=int, help='size x')
    parser.add_argument('--n_y', default=10, type=int, help='size y')
    parser.add_argument('--h', default=0.5, type=float, help='dx and dy')
    parser.add_argument('--dt', default=0.5, type=float, help='dt')
    parser.add_argument('--n_timestep', default=200, type=int, help='n_dt')
    parser.add_argument('--epochs', default=50000, type=int, help='epochs')
    parser.add_argument('--warm_up', default=50, type=int, help='warm-up epochs')
    parser.add_argument('--resume', default=False, type=bool, help='resume or not')
    parser.add_argument('--save', default=True, type=bool, help='save or not')
    # args = parser.parse_args('') # if .ipynb
    args = parser.parse_args()

    

    n_x = args.n_x
    n_y = args.n_y
    u_0, N = state_init(n_x, n_y) 

    h = args.h
    dt = args.dt
    n_timestep = args.n_timestep

    para_simu = {'nS':0.1, 'nE':0.1, 'nI':0.1, 'nR':0.1,  
                    'beta':50, 'a':0.2, 'gamma':0.1, 'd':0.}
    
    if not args.resume:

        u_t, X_simu = get_simu_data(u_0, h, dt, n_x, n_y, n_timestep, para_simu, N)
        np.save('./saved/simu_data', u_t)
        plt.figure()    
        plt.plot(X_simu[:,0], 'r-', linewidth = 2)
        plt.plot(X_simu[:,1], 'g-', linewidth = 2)
        plt.plot(X_simu[:,2], 'b-', linewidth = 2)
        plt.plot(X_simu[:,3], 'k-', linewidth = 2)
        plt.legend(['S', 'E', 'I', 'R'])
        plt.savefig('./pic/simu_data.tiff')
    else:
        u_t = np.load('./saved/simu_data.npy')
        X_simu = np.concatenate( [np.sum( u_t[i], axis=(1,2))[None,...] for i in range(n_timestep)], axis=0)

    ## data_fit
   
    x = np.array([h*i for i in range(n_x)], dtype=float)
    y = np.array([h*i for i in range(n_y)], dtype=float)
    t = np.array([dt*i for i in range(n_timestep)], dtype=float)

    Y, X = np.meshgrid(y,x) 
    X, Y= X.reshape((-1,1)), Y.reshape((-1,1))


    data_loc = np.hstack([X, Y])
    data_loc = np.tile(data_loc, (n_timestep,1))
    data_t = np.vstack( [np.reshape([t[i]]*n_x*n_y, (-1,1)) for i in range(n_timestep)])
    data = np.hstack((data_t, data_loc))

    # data_loc_test = np.tile(data_loc, (n_timestep-n_timestep_train,1))
    # data_t_test = np.vstack( [np.reshape([t[i]]*n_x*n_y, (-1,1)) for i in range(n_timestep_train, n_timestep)])
    # data_test = np.hstack((data_t_test, data_loc_test))

    u_train = np.vstack( [np.hstack([u_t[j,i,:,:].reshape((-1,1)) for i in range(4)]) for j in range(n_timestep)] )
    u_lb = u_train.min(0)
    u_ub = u_train.max(0)
    u_lb = torch.tensor(u_lb).float().to(device)
    u_ub = torch.tensor(u_ub).float().to(device)


    input_lb = data.min(0) 
    input_ub = data.max(0)  
    input_lb = torch.tensor(input_lb).float().to(device)
    input_ub = torch.tensor(input_ub).float().to(device)

    batch_size = 5000
    idx = np.random.choice(n_x*n_y*n_timestep, batch_size, replace=False)
    tensor_t = torch.tensor(data_t[idx,...], requires_grad=True).float()
    tensor_X = torch.tensor(data_loc[:,0:1][idx,...], requires_grad=True).float()
    tensor_Y = torch.tensor(data_loc[:,1:2][idx,...], requires_grad=True).float()
    tensor_U = torch.tensor(u_train[idx,...]).float()


    dataset = TensorDataset(tensor_t, tensor_X, tensor_Y, tensor_U) 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    # model 
    n_hidden = 256
    layers = [3, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, 4]
    net = Modified_DNN(layers)
    
    pinn = PINN_COVID(net, data_loader, \
                    input_lb=input_lb, input_ub=input_ub, u_lb=u_lb, u_ub=u_ub,\
                    n_x=n_x, n_y=n_y, n_timestep=n_timestep, data=data, \
                    X_simu=X_simu, device=device)
    pinn.warm_up = args.warm_up
    pinn.save = args.save

    if args.resume:
        checkpoint = torch.load('./saved/checkpoint.pth')
        pinn.net.load_state_dict(checkpoint['net_state_dic'])
        pinn.opt_Adam.load_state_dict(checkpoint['opt_state_dic'])
        pinn.epoch_start = checkpoint['epoch']
        history_resume = pinn.train(0)
        np.save('./saved/history_resume.npy', history_resume) 
    else:
        history = pinn.train(args.epochs)
        if args.save:
            np.save('./saved/history.npy', history)

   
    
    
