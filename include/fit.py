from torch.autograd import Variable
import torch
import torch.optim
import copy
import skimage.measure
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import ortho_group
from .helpers import *
import torch.fft
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import os
import csv
import math
import pickle

if torch.cuda.device_count()==0:
    dtype = torch.FloatTensor
    device = 'cpu'
else:
    dtype = torch.cuda.FloatTensor
    device = 'cuda'

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500, factor=0.5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('\nLR is set to {}'.format(lr)) 
        print('\n')
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr
    return optimizer

def inner_num_scheduler(epoch, init_num = 5, grow_epoch = 500, factor = 1.2):
    num = round (init_num * (factor ** (epoch // grow_epoch)))
    if epoch % grow_epoch == 0:
        print('max iters for inner loop set to', num, '\n')
    return num

def fit(net,
        num_channels, 
        img_clean_var,
        out_channels=1,
        m = 512, 
        d = 256, 
        net_input = None,
        opt_input = False, 
        find_best = False,
        OPTIMIZER ='adam',
        LR = 0.01, 
        init_inner = 20,
        print_inner = False,
        optim = 'admm',
        num_iter = 5000,
        grow_epoch = 500,
        lr_decay_epoch = 0, 
        w = 0.0,
        rho = 1.0,
        ksi = 0.001,
        weight_decay=0,
        alpha = 0.01,
        img_np1 = None,
        img_np2 = None,
        code = 'gaussian'
       ):
    
    if net_input is not None:
        net_input = net_input
        print(type(net_input))
        print("input provided")

    else:
        print("alpha is:", alpha)
        totalupsample = 2 ** (len(num_channels) - 1)
        width = int(d / (totalupsample))
        height = int(d / (totalupsample))
        shape = [1, num_channels[0], width, height]
        print("shape of latent code Z: ", shape)
        print("initializing latent code Z...")

        net_input = Variable(torch.zeros(shape))
        if code == 'gaussian':
            net_input.data.normal_()
        elif code == 'uniform':
            net_input.data.uniform_()
        net_input.data *= 1. / 10

    net_input_saved = net_input.data.detach()
    p = [t for t in net.decoder.parameters()]  # list of all weights
    if (opt_input == True):  # optimizer over the input as well
        net_input.requires_grad = True
        print('optimizing over latent code Z1')
        p += [net_input]
    else:
        print('not optimizing over latent code Z1')

    mse_wrt_truth = np.zeros(num_iter)
    PSNR1 = np.zeros(num_iter)
    PSNR2 = np.zeros(num_iter)
    best_x = Variable(torch.zeros([1, out_channels, d, d]))
    # inner loop optimizer
    if OPTIMIZER == 'SGD':
        print("optimize decoder with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR, momentum=0.9, weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize decoder with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR, weight_decay=weight_decay) # 优化器
    mse = torch.nn.MSELoss()
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    data = open('PSNR.csv', 'w', encoding='utf8', newline='')
    writer = csv.writer(data)
    writer.writerow(['PSNR', 'loss'])

    if optim == 'Vanilla-TRAD':
        print('optimizing with Vanilla-TRAD...')
        alpha_ = alpha
        I = torch.ones((1, out_channels, m, m)).to(device)
        W = w * I
        b = img_clean_var
        v = Variable(torch.zeros([1, out_channels, m, m]))
        v = ifftn(b).to(device)
        x = Variable(torch.zeros([1, out_channels, d, d]))
        x = x.to(device)
        xx = Variable(torch.zeros([1, out_channels, d, d]))
        xx = xx.to(device)

        for i in range(num_iter):
            if lr_decay_epoch is not 0:
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
            if grow_epoch is not 0:
                numit_inner = inner_num_scheduler(i, init_num=init_inner, grow_epoch=grow_epoch, factor=1.2)
            xx.data = x.data
            x = F.pad(v - W / rho, (0, d - m, 0, d - m), "constant", 0) - alpha_ / rho * tv_grad2(
                F.pad(v - W / rho, (0, d - m, 0, d - m), "constant", 0)).to(device)
            temp2 = fftn(F.pad(x, (0, m - d, 0, m - d), "constant", 0) + W / rho, m)
            temp3 = torch.abs(temp2)
            temp1 = torch.sqrt(b ** 2 + ksi * I) / torch.sqrt(temp3 ** 2 + ksi * I)
            v = ifftn(temp1 * temp2)
            for j in range(numit_inner):
                optimizer.zero_grad()
                out = net(net_input.type(dtype))
                loss_inner = mse(out, v[:, :, 0:d, 0:d])
                loss_inner.backward()
                optimizer.step()
                if print_inner:
                    print('Inner iteration %05d  loss %f' % (j, loss_inner.detach().cpu().numpy()))
                    
            v.data[:, :, 0:d, 0:d] = net(net_input.type(dtype))
            W = W + rho * (F.pad(x, (0, m - d, 0, m - d), "constant", 0) - v)
            output = apply_f(x, m)
            loss_LS = mse(output, img_clean_var)
            mse_wrt_truth[i] = loss_LS.item()
            print('Iteration %05d   loss %f ' % (i, mse_wrt_truth[i]), '\r', end='')

            if i == 500 or i == 1000 or i == 1500:
                img_path = os.path.join('inter/', str(i) + '.png')
                save_fig(x, img_path)

            mse_ = d * d * mse(x, xx)
            x_np = x.data.cpu().numpy()
            psnr1 = psnr(convert(x_np[0]), convert(img_np1))
            psnr2 = psnr(convert(x_np[0]), convert(img_np2))
            psnr_ = max(psnr1, psnr2)
            writer.writerow([psnr_, mse_wrt_truth[i]])

            loss_updated = mse(output, img_clean_var)
            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.00001 * loss_updated.item():
                    best_mse = loss_updated.item()
                    best_net = copy.deepcopy(net)
                    best_x = x

            if mse_.item() <= 0.005:
                break

        if find_best:
            net = best_net

    elif optim == 'Accelerated-TRAD':
        print('optimizing with Accelerated-TRAD...')
        alpha_ = alpha
        I = torch.ones((1, out_channels, m, m)).to(device)
        W = w*I
        b = img_clean_var
        v = Variable(torch.zeros([1, out_channels, m, m]))
        v = ifftn(b).to(device)
        x = Variable(torch.zeros([1, out_channels, d, d]))
        x = x.to(device)
        xx = Variable(torch.zeros([1, out_channels, d, d]))
        xx = xx.to(device)

        for i in range(num_iter):
            xx.data = x.data
            x = F.pad(v - W / rho, (0, d - m, 0, d - m), "constant", 0) - alpha_ / rho * tv_grad2(
                F.pad(v - W / rho, (0, d - m, 0, d - m), "constant", 0)).to(device)
            temp2 = fftn(F.pad(x, (0, m-d, 0, m-d), "constant", 0) + W/rho, m)
            temp3 = torch.abs(temp2)
            temp1 = torch.sqrt(b**2 + ksi * I) / torch.sqrt(temp3**2 + ksi * I)
            v = ifftn(temp1 * temp2)
            mu = math.exp(-(max(i - 1000, 0) / 10) ** 2)
            if mu >= 0.01:
                if lr_decay_epoch is not 0:
                    optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
                if grow_epoch is not 0:
                    numit_inner = inner_num_scheduler(i, init_num=init_inner, grow_epoch=grow_epoch, factor=1.2)
                for j in range(numit_inner):
                    optimizer.zero_grad()
                    out = net(net_input.type(dtype))
                    loss_inner = mse(out, v[:,:,0:d,0:d])
                    loss_inner.backward()
                    optimizer.step()
                    if print_inner:
                        print('Inner iteration %05d  Train loss %f' % (j, loss_inner.detach().cpu().numpy()))

            if mu < 0.01:
                mu = 0
            v.data[:,:,0:d,0:d] = mu * net(net_input.type(dtype)) + (1 - mu) * v.data[:,:,0:d,0:d]

            W = W + rho * (F.pad(x, (0, m - d, 0, m - d), "constant", 0) - v)
            output = apply_f(x, m)
            loss_LS = mse(output, img_clean_var)
            mse_wrt_truth[i] = loss_LS.item()
            print('Iteration %05d   loss %f ' % (i, mse_wrt_truth[i]), '\r', end='')

            if i == 500 or i == 1000 or i == 1500:
                img_path = os.path.join('inter/', str(i) + '.png')
                save_fig(x, img_path)

            mse_ = d * d * mse(x, xx)
            x_np = x.data.cpu().numpy()
            psnr1 = psnr(convert(x_np[0]), convert(img_np1))
            psnr2 = psnr(convert(x_np[0]), convert(img_np2))
            psnr_ = max(psnr1, psnr2)
            writer.writerow([psnr_, mse_wrt_truth[i]])

            loss_updated = mse(output, img_clean_var)
            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.00001 * loss_updated.item():
                    best_mse = loss_updated.item()
                    best_net = copy.deepcopy(net)
                    best_x = x

            if mse_.item() <= 0.005:
                break
        if find_best:
            net = best_net

    print("min of loss:", best_mse)
    data.close()
    return mse_wrt_truth, net_input_saved, net, net_input, best_x, PSNR1, PSNR2