import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pylab as plt

class PINN_COVID:
    
    def __init__(self, net, data_loader, **kwargs):
        
        self.data_loader = data_loader
        self.input_lb = kwargs['input_lb']
        self.input_ub = kwargs['input_ub']
        self.u_lb = kwargs['u_lb']
        self.u_ub = kwargs['u_ub']

        self.device = kwargs['device'] # cuda
        # self.N = kwargs['N']
        self.kwargs = kwargs
        self.warm_up = 0
        self.save = False
        # net
        # net = nn.DataParallel(net)
        self.net = net.to(self.device)

        # pde parameters for learning
        self.para = nn.ParameterDict({
                                    'nS':nn.Parameter(torch.tensor([-2.])),
                                    'nE':nn.Parameter(torch.tensor([-2.])),
                                    'nI':nn.Parameter(torch.tensor([-2.])),
                                    'nR':nn.Parameter(torch.tensor([-2.])),
                                    'beta':nn.Parameter(torch.tensor([4.])), 
                                    'a':nn.Parameter(torch.tensor([-2.])), 
                                    'gamma':nn.Parameter(torch.tensor([-3.])),
                                    'd':nn.Parameter(torch.tensor([-3.]))
                                    }).to(self.device)

        for key, value in self.para.items():
            if key not in ['d']:
                 self.net.register_parameter(key, value)
        

        # opt: using both Adam 
        self.lr = 0.003
        self.opt_Adam = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        # self.opt_Adam = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.epoch_start = 0

        self.iter = 0
        self.opt_LBFGS = optim.LBFGS(
            self.net.parameters(), 
            lr=1., 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-8, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

    def net_U(self, x, y, t):
        xt = torch.cat((t,x,y), dim=1)
        xt = 2*(xt-self.input_lb)/(self.input_ub-self.input_lb) - 1
        u = self.net(xt)
        return u

    def net_F(self, x, y, t):
    
        u = self.net_U(x, y, t)
        u = u*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling
        U_t = []; U_xx = []; U_yy = []
        
        for i in range(4):
            u_t = torch.autograd.grad(u[:,i:i+1], t, 
                                        torch.ones_like(u[:,i:i+1]),
                                        retain_graph=True,
                                        create_graph=True)[0]
            
            u_x = torch.autograd.grad(u[:,i:i+1], x, 
                                        torch.ones_like(u[:,i:i+1]), 
                                        retain_graph=True,
                                        create_graph=True)[0]

            u_y = torch.autograd.grad(u[:,i:i+1], y, 
                                        torch.ones_like(u[:,i:i+1]), 
                                        retain_graph=True,
                                        create_graph=True)[0]

            u_xx = torch.autograd.grad(u_x, x, 
                                        torch.ones_like(u[:,i:i+1]),
                                        retain_graph=True,
                                        create_graph=True)[0]
           
            u_yy = torch.autograd.grad(u_y, y, 
                                        torch.ones_like(u[:,i:i+1]),
                                        retain_graph=True,
                                        create_graph=True)[0]
            U_t.append(u_t)
            U_xx.append(u_xx)
            U_yy.append(u_yy)

        S = u[:,0:1]; E = u[:,1:2]; I = u[:,2:3]; 
        

        nS = torch.exp(self.para['nS'])
        nE = torch.exp(self.para['nE']) 
        nI = torch.exp(self.para['nI']) 
        nR = torch.exp(self.para['nR'])

        beta = torch.exp(self.para['beta'])
        a = torch.exp(self.para['a'])
        d = 0 # torch.exp(self.para['d'])
        gamma = torch.exp(self.para['gamma'])
        
        
        N_S = nS*(U_xx[0] + U_yy[0]) - beta*S*I
        N_E = nE*(U_xx[1] + U_yy[1]) + beta*S*I - a*E
        N_I = nI*(U_xx[2] + U_yy[2]) + a*E - gamma*I
        N_R = nR*(U_xx[3] + U_yy[3]) + gamma*I 

       
        F_S = U_t[0] - N_S; 
        F_E = U_t[1] - N_E; 
        F_I = U_t[2] - N_I; 
        F_R = U_t[3] - N_R
        return F_S, F_E, F_I, F_R

    def _train_step(self, X, Y, t, U, X_col, Y_col, t_col):
        
        U_pred = self.net_U(X, Y, t)
        F_S, F_E, F_I, F_R = self.net_F(X_col, Y_col, t_col)
        
        assert U.shape == U_pred.shape
        loss1 = torch.mean( (U-U_pred)**2, dim=0)
        assert loss1.shape == (4,) 
        loss2 = (torch.mean(F_S**2) + \
                torch.mean(F_E**2) + \
                torch.mean(F_I**2) + \
                torch.mean(F_R**2)) 
        return loss1, loss2

    def _closure(self, X, Y, t, U):

        t = t.clone().detach().requires_grad_(True).to(self.device)
        X = X.clone().detach().requires_grad_(True).to(self.device)
        Y = Y.clone().detach().requires_grad_(True).to(self.device)
        U = U.clone().detach().to(self.device)

        loss1, loss_F = self._train_step(X, Y, t, U, X, Y, t)
        loss_U = torch.sum(loss1)
        self.opt_LBFGS.zero_grad()
        loss =  self.wu*loss_U + self.wf*loss_F
        loss.backward()

        self.iter += 1
        if self.iter % 1  == 0:
            print('epoch_LBFGS: {}  loss_U: {:.4e}  loss_F: {:.4e}'.format(
                                                    self.iter, 
                                                    loss_U.item(),
                                                    loss_F.item()
                                                    ), flush=True)
            
            print('epoch_LBFGS: {}  nS: {:.4f}  nE: {:.4f}  nI: {:.4f}  nR: {:.4f}'.format(
                                                    self.iter, 
                                                    torch.exp(self.para['nS']).item(),
                                                    torch.exp(self.para['nE']).item(),
                                                    torch.exp(self.para['nI']).item(),
                                                    torch.exp(self.para['nR']).item()
                                                    ), flush=True)

            print('epoch_LBFGS: {}  beta: {:.4f}  gamma: {:.4f}  a: {:.4f}  d: {:.4f}'.format(
                                                    self.iter,
                                                    torch.exp(self.para['beta']).item(),
                                                    torch.exp(self.para['gamma']).item(),
                                                    torch.exp(self.para['a']).item(),
                                                    torch.exp(self.para['d']).item()
                                                    ), flush=True)

        self.history['loss1'].append(loss_U.item())                                        
        self.history['loss2'].append(loss_F.item())                                        
        self.history['nX'].append([self.para['nS'].item(), self.para['nE'].item(), self.para['nI'].item(), self.para['nR'].item()])                                        
        self.history['beta'].append(self.para['beta'].item())
        self.history['gamma'].append(self.para['gamma'].item())                                        
        self.history['a'].append(self.para['a'].item())
        return loss

    def train(self, epochs):
        self.net.train() # training mode
      
        self.history = {'loss1':[], 'loss2':[], 'nX':[], 'beta':[], 'gamma':[], 'a':[]}

        self.wu, self.wf = 1, 1
        grad_U_mean, grad_F_mean, grad_nX_mean = 1, 1, 1

        self.grad_U = {'epoch':0}
        self.grad_F = {'epoch':0}
        self.grad_para = {'epoch':0}

        for epoch in range(self.epoch_start, self.epoch_start+epochs):

            total_num, total_loss_F, total_loss_U = 0, 0.0, 0.0
            total_loss_S, total_loss_E, total_loss_I, total_loss_R = 0.0, 0.0, 0.0, 0.0

            self.grad_U['epoch'] = epoch+1
            self.grad_F['epoch'] = epoch+1
            self.grad_para['epoch'] = epoch+1

            
            for t, X, Y, U in self.data_loader:

                t, X, Y = t.to(self.device), X.to(self.device), Y.to(self.device)
                U = U.to(self.device)
                U = (U-self.u_lb)/(self.u_ub-self.u_lb) # scaling

                if (epoch+1) < self.warm_up:
                    self.wf = 0
                elif (epoch+1) == self.warm_up:
                    self.wf = 1

                
                    
                loss1, loss2 = self._train_step(X, Y, t, U, X, Y, t) 
                loss_U = self.wu*torch.sum(loss1)
                loss_F = self.wf*loss2

                self.opt_Adam.zero_grad()

               
                loss_F.backward(retain_graph=True)
                for name, para in self.net.named_parameters():
                    if name in ['nS', 'nE', 'nI', 'nR', 'beta', 'gamma', 'a']:
                        self.grad_para[name] = torch.abs(para.grad.clone()/self.wf).item()
                    else:
                        self.grad_F[name] = para.grad.clone()/self.wf # copied

                loss_U.backward(retain_graph=True)
                for name, para in self.net.named_parameters():
                    if name not in ['nS', 'nE', 'nI', 'nR', 'beta', 'gamma', 'a']:
                        self.grad_U[name] = para.grad.clone() - self.grad_F[name]*self.wf

                self.opt_Adam.step()

                if (epoch+1) > self.warm_up:
                    grad_U_max = torch.max(
                        torch.abs(torch.cat([j.ravel() for i, j in self.grad_U.items() if i != 'epoch'])))
                    grad_U_mean = torch.mean(
                        torch.abs(torch.cat([j.ravel() for i, j in self.grad_U.items() if i != 'epoch'])))
                    grad_F_mean = torch.mean(
                        torch.abs(torch.cat([j.ravel() for i, j in self.grad_F.items() if i != 'epoch'])))

                    self.wf = 0.1 * self.wf + 0.9 * grad_U_mean / grad_F_mean

              
                batch_size = t.shape[0]
                total_num += batch_size
                total_loss_S += loss1[0].item() * batch_size
                total_loss_E += loss1[1].item() * batch_size
                total_loss_I += loss1[2].item() * batch_size
                total_loss_R += loss1[3].item() * batch_size
                total_loss_U += loss_U.item() * batch_size
                total_loss_F += loss_F.item() * batch_size

            self.adjust_learning_rate(epoch)
           
            if (epoch+1) % 5 == 0:
                print('----------------------------')
                print('epoch_Adam: {}  loss_S: {:.4e}  loss_E: {:.4e} loss_I: {:.4e}  loss_R: {:.4e}'.format(
                                                    epoch+1, 
                                                    total_loss_S/total_num,
                                                    total_loss_E/total_num,
                                                    total_loss_I/total_num,
                                                    total_loss_R/total_num
                                                    ), flush=True)

                print('epoch_Adam: {}  weight_F: {:.4e}  grad_U_mean: {:.4e}  grad_F_mean: {:.4e}'.format(
                                                    epoch+1, 
                                                    self.wf,
                                                    grad_U_mean,
                                                    grad_F_mean,
                                                    ), flush=True)

                print('epoch_Adam: {}  total_loss_U: {:.4e}  total_loss_F: {:.4e}'.format(
                                                    epoch+1, 
                                                    total_loss_U/total_num,
                                                    total_loss_F/total_num
                                                    ), flush=True)

                print('epoch_Adam: {}  nS: {:.4f}  nE: {:.4f}  nI: {:.4f}  nR: {:.4f}'.format(
                                                    epoch+1, 
                                                    torch.exp(self.para['nS']).item(),
                                                    torch.exp(self.para['nE']).item(),
                                                    torch.exp(self.para['nI']).item(),
                                                    torch.exp(self.para['nR']).item()
                                                    ), flush=True)
                print('epoch_Adam: {}  grad_nS: {:.4e}  grad_nS: {:.4e}  grad_nS: {:.4e}  grad_nS: {:.4e}'.format(
                                                    epoch+1,
                                                    self.grad_para['nS'],
                                                    self.grad_para['nE'],
                                                    self.grad_para['nI'],
                                                    self.grad_para['nR']
                                                ), flush=True)

                print('epoch_Adam: {}  beta: {:.4f}  gamma: {:.4f}  a: {:.4f}  d: {:.4f}'.format(
                                                    epoch+1, 
                                                    torch.exp(self.para['beta']).item(),
                                                    torch.exp(self.para['gamma']).item(),
                                                    torch.exp(self.para['a']).item(),
                                                    torch.exp(self.para['d']).item()
                                                    ), flush=True)

                print('epoch_Adam: {}  grad_beta: {:.4e}  grad_gamma: {:.4e}  grad_a: {:.4e}'.format(
                                                    epoch+1,
                                                    self.grad_para['beta'],
                                                    self.grad_para['gamma'],
                                                    self.grad_para['a']
                                                ), flush=True)


            self.history['loss1'].append(total_loss_U/total_num)                                        
            self.history['loss2'].append(total_loss_F/total_num)                                        
            self.history['nX'].append([self.para['nS'].item(), self.para['nE'].item(), self.para['nI'].item(), self.para['nR'].item()])                                        
            self.history['beta'].append(self.para['beta'].item())
            self.history['gamma'].append(self.para['gamma'].item())                                        
            self.history['a'].append(self.para['a'].item())

            if self.save:
                torch.save({'epoch':epoch+1, 'net_state_dic':self.net.state_dict(), 'opt_state_dic':self.opt_Adam.state_dict()}, \
                                './saved/checkpoint.pth' )

            if (epoch+1) % 2000 == 0:
               self.plot(self.kwargs['data'], epoch+1)

            

        # for t, X, Y, U in self.data_loader:
        #     U = U.to(self.device)
        #     U = (U-self.u_lb)/(self.u_ub-self.u_lb) # scaling
        #     self.opt_LBFGS.step(lambda: self._closure(X, Y, t, U)) 

        return self.history

    def adjust_learning_rate(self, epoch):
        lr = self.lr*0.90**(epoch/2000)
        for param_group in self.opt_Adam.param_groups:
            param_group['lr'] = lr
        
    def predict(self, x):
         x = torch.tensor(x, requires_grad=True).float().to(self.device)
         t = x[:,0:1]; X = x[:,1:2]; Y = x[:,2:3]

         self.net.eval()
         u = self.net_U(X, Y, t)
         u = u*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling
         return u.detach().cpu().numpy()

    def plot(self, x, epoch):
        pred = self.predict(x)
        temp = int(self.kwargs['n_x']*self.kwargs['n_y'])
        XT_pred = np.zeros((self.kwargs['n_timestep'], temp, 4)) 

        for i in range(self.kwargs['n_timestep']): 
            XT_pred[i] = pred[temp*i:temp*(i+1),:]
            

        X_simu_pred = np.sum(XT_pred, axis=1) 

        fig, axs = plt.subplots(2, 2, figsize=(20, 8))
        axs[0,0].plot(self.kwargs['X_simu'][:,0], 'b-', linewidth = 2)
        axs[0,0].plot(X_simu_pred[:,0], 'k--', linewidth = 2)
        axs[0,0].legend(['Exact', 'Predict'])
        axs[0,0].set_title('S')

        axs[0,1].plot(self.kwargs['X_simu'][:,1], 'b-', linewidth = 2)
        axs[0,1].plot(X_simu_pred[:,1], 'k--', linewidth = 2)
        axs[0,1].set_title('E')

        axs[1,0].plot(self.kwargs['X_simu'][:,2], 'b-', linewidth = 2)
        axs[1,0].plot(X_simu_pred[:,2], 'k--', linewidth = 2)
        axs[1,0].set_title('I')

        axs[1,1].plot(self.kwargs['X_simu'][:,3], 'b-', linewidth = 2)
        axs[1,1].plot(X_simu_pred[:,3], 'k--', linewidth = 2)
        axs[1,1].set_title('R')

        plt.savefig('./pic/Adam_{}.tiff'.format(epoch))


    
        





