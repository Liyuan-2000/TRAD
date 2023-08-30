from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from include import *
from PIL import Image
import PIL
import pywt
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2ycbcr
from sklearn import linear_model
import torch.fft
import cv2
import csv

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    print("num GPUs",torch.cuda.device_count())
    device = 'cuda'
    if torch.cuda.device_count()==0:
        dtype = torch.FloatTensor
        device = 'cpu'
else:
    dtype = torch.FloatTensor
    device = 'cpu'
import time

# record PSNR and SSIM
data = open('data.csv', 'w', encoding = 'utf8', newline = '')
writer = csv.writer(data)

dataset = 'Cameraman' # 'Cameraman', 'Bridge', 'Yokahoma', 'Tucson', 'ButterflyNebula', 'PillarsofCreation'
filename = 'Cameraman.png'
rr = [2.0, 1.8, 1.6] # sampling ratio
circ_num = 1 # number of runs at each sampling ratio
optim = 'Accelerated-TRAD' # 'Vanilla-TRAD', 'Accelerated-TRAD'
alpha = 1./384 # for TV
num_channels = [128, 128, 128, 128] # number of channels in each layer of network

if os.path.exists('inter') == False:
    os.mkdir('inter')
if os.path.exists('output') == False:
    os.mkdir('output')
if os.path.exists('output/' + optim) == False:
    os.mkdir('output/' + optim)

for i in range(len(rr)):
    max_psnr = 0
    for j in range(circ_num):
        img_path1 = os.path.join('images/' + dataset + '/', filename)
        img_path2 = os.path.join('images/' + dataset + '_/', filename)
        r = rr[i]
        print('sampling ratio: ', r)
        img_pil1 = Image.open(img_path1)
        img_np1 = pil_to_np(img_pil1) # W x H x C [0...255] to C x W x H [0...1]
        img_pil2 = Image.open(img_path2)
        img_np2 = pil_to_np(img_pil2)
        print('Dimensions of input image:', img_np1.shape)

        output_depth = img_np1.shape[0]
        d = img_np1.shape[1]
        img_var = np_to_var(img_np1).type(dtype)
        m = int(r * d)
        print('number of measurement:', m)
        img_var_meas = apply_f(img_var, m)

       # for deep decoder
        net = autoencodernet(num_output_channels=output_depth, num_channels_up=num_channels, need_sigmoid=True,
                             decodetype='upsample').type(dtype)
        print("number of parameters: ", num_param(net))
        # print(net.decoder)
        net_in = copy.deepcopy(net)

        OPTIMIZER = 'adam'  # inner loop optimizer - SGD or adam
        numit = 2000  # number of iterations for SGD or adam
        init_inner = 5  # number of inner loop iterations for projection
        LR = 0.005  # required for inner loop of projection

        lr_decay_epoch = 500  # decay learning rates of outer optimizers
        grow_epoch = 500
        code = 'uniform'

        t0 = time.time()
        mse_t, ni, net, ni_mod, in_np_img, psnr1, psnr2 = fit(
            net=net,
            num_channels=num_channels,
            m=m,
            d=d,
            num_iter=numit,
            init_inner=init_inner,
            LR=LR,
            OPTIMIZER=OPTIMIZER,
            lr_decay_epoch=lr_decay_epoch,
            grow_epoch = grow_epoch,
            img_clean_var=img_var_meas,
            find_best=True,
            optim=optim,
            out_channels=output_depth,
            alpha = alpha,
            img_np1 = img_np1,
            img_np2 = img_np2,
            code = code
        )
        t1 = time.time()
        inter_time = t1 - t0
        print('\ntime elapsed:', inter_time)
        out_img_np = in_np_img.data.cpu().numpy()[0]
        out_img_np = convert(out_img_np)

        if output_depth == 1:
            img_np1 = img_np1[0]
            img_np2 = img_np2[0]
            out_img_np = out_img_np[0]
            SSIM1 = ssim(out_img_np, convert(img_np1))
            SSIM2 = ssim(out_img_np, convert(img_np2))
        else:
            SSIM1 = ssim(out_img_np.transpose(1, 2, 0), convert(img_np1.transpose(1, 2, 0)), multichannel=True)
            SSIM2 = ssim(out_img_np.transpose(1, 2, 0), convert(img_np2.transpose(1, 2, 0)), multichannel=True)
        PSNR1 = psnr(out_img_np, convert(img_np1))
        PSNR2 = psnr(out_img_np, convert(img_np2))
        writer.writerow([filename, str(r), str(max(PSNR1, PSNR2)), str(max(SSIM1, SSIM2)), inter_time])
        print('iter = ', j)
        print('psnr = ', max(PSNR1, PSNR2))
        print('ssim = ', max(SSIM1, SSIM2))
        if max(PSNR1, PSNR2) > max_psnr:
            if os.path.exists('output/' + optim + '/' + filename[0:-4]) == False:
                os.mkdir('output/' + optim + '/' + filename[0:-4])
            out_path = os.path.join('output/' + optim + '/' + filename[0:-4], str(r) + filename)
            if output_depth == 3:
                out_img_np = out_img_np.transpose(1, 2, 0)
            print("shape of output:", out_img_np.shape)
            out_pil = Image.fromarray(out_img_np)
            out_pil.save(out_path)
            max_psnr = max(PSNR1, PSNR2)
data.close()
print("over!")