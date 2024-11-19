import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv

import os
import glob
import warnings
import torch
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data
from natsort import natsorted
import TransModels.LDMMorph as LDMMorph 

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default
from omegaconf import OmegaConf
from torch.autograd import Variable

parser = ArgumentParser()
parser.add_argument("--resume", type=str,
                    dest="resume", default='../stable-diffusion/logs/2024-07-07T23-22-44_casmus-ldm-vq16-64ch/checkpoints/epoch=000682.ckpt',
                    help="pretrained model")
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=24001,
                    help="number of total iterations")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.01, 
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=400,
                    help="frequency of saving models")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../../../data/CAMUS/org/',
                    help="data path for training images") 
parser.add_argument("--beta", type=float,
                    dest="beta", 
                    default=0.6,
                    help="beta loss: range from 0.1 to 1.0")
opt = parser.parse_args()


lr = opt.lr
bs = opt.bs
iteration = opt.iteration
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
beta = opt.beta
t_enc = 1 

opt, unknown = parser.parse_known_args()
ckpt = None
configs = ['./configs/latent-diffusion/casmus-ldm-vq16-64ch.yaml']
opt.ldm = configs
print(opt.resume)

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval() 
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def dice(pred1, truth1):
    if datapath=='acdc':
        VOI_lbls = [2,3]
    else:
        VOI_lbls = [1]
    dice_all=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        
        dice_all[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_all)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    #----------------------------- load ldm model -------------------------------------
    global opt
    print(opt.resume)
    ckpt = opt.resume
    
    configs = [OmegaConf.load(cfg) for cfg in opt.ldm]
    cli = OmegaConf.from_dotlist(unknown)
    configs = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    ldm_model, global_step = load_model(configs, ckpt, gpu, eval_mode)
    #-------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    trainnames = sorted(glob.glob(datapath + '*_2CH.npz'))[:400]
    valnames = sorted(glob.glob(datapath + '*_2CH.npz'))[400:420]
    train_loader = Data.DataLoader(Dataset_epoch_with_name(trainnames), batch_size=1, shuffle=True, num_workers=1)
    val_loader = Data.DataLoader(Dataset_epoch_with_name(valnames), batch_size=1, shuffle=False, num_workers=1)
                                        

    model = LDMMorph.LDMMorph(128*2,192*2,320*2,448*2)
    model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    loss_similarity = MSE().loss
    loss_smooth = smoothloss

    transform = SpatialTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model_dir = './logs/TransScorelm_Smooth_{}_beta_{}/'.format(smooth, beta)
    csv_name = './logs/TransScorelm_Smooth_{}_beta_{}.csv'.format(smooth, beta)

    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice','OrgDice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration))
    
    step = 1
    epoch = 0
    csv_dice = 0
    while step <= iteration:
        for X, Y, segx, segy, _ in train_loader:

            X = X.cuda().float()
            Y = Y.cuda().float()
            
            mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
            fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()

            noise = None
            noise = default(noise, lambda: torch.randn_like(mov_z))
            x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
            y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
        
            outx = ldm_model.apply_model(x_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
            outy = ldm_model.apply_model(y_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

            score0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
            score1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
            score2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
            score3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]),  dim=1)
            
            D_f_xy = model(X, Y, score0, score1, score2, score3)
            _, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1))

            mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X_Y)).detach()
            #tar_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()
            
            loss1 = beta * loss_similarity(Y, X_Y) + (1-beta) * loss_similarity(mov_z, fix_z)
            loss2 = loss_smooth(D_f_xy)
            loss = loss1 + smooth * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(), loss1.item(), loss2.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" - smh "{3:.4f}"'.format(step, loss.item(), loss1.item(), loss2.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    Dices_Validation = [] 
                    Dices_Validation_org = []
                    
                    for xv, yv, xv_seg, yv_seg, _ in val_loader:

                        xv, yv, xv_seg, yv_seg = xv.to(device), yv.to(device), xv_seg.to(device), yv_seg.to(device)
                        
                        model.eval()

                        vmov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(xv)).detach()
                        vfix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(yv)).detach()

                        vx_noisy = ldm_model.q_sample(x_start=vmov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
                        vy_noisy = ldm_model.q_sample(x_start=vfix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
                    
                        voutx = ldm_model.apply_model(vx_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                        vouty = ldm_model.apply_model(vy_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

                        vscore0 = torch.cat((voutx[1][0][0],  voutx[1][0][2], vouty[1][0][0],  vouty[1][0][2]),  dim=1)
                        vscore1 = torch.cat((voutx[1][0][3],  voutx[1][0][5], vouty[1][0][3],  vouty[1][0][5]),  dim=1)
                        vscore2 = torch.cat((voutx[1][0][6],  voutx[1][0][8], vouty[1][0][6],  vouty[1][0][8]),  dim=1)
                        vscore3 = torch.cat((voutx[1][0][9],  voutx[1][0][11], vouty[1][0][9],  vouty[1][0][11]),  dim=1)

                        Dv_f_xy = model(xv, yv, vscore0, vscore1, vscore2, vscore3)
                        _, warped_xv_seg= transform(xv_seg, Dv_f_xy.permute(0, 2, 3, 1), mod = 'nearest')

                        for bs_index in range(bs):
                            dice_bs=dice(warped_xv_seg[bs_index,...].data.cpu().numpy().copy(),yv_seg[bs_index,...].data.cpu().numpy().copy())
                            dice_bs_org=dice(xv_seg[bs_index,...].data.cpu().numpy().copy(),yv_seg[bs_index,...].data.cpu().numpy().copy())
                            Dices_Validation.append(dice_bs)
                            Dices_Validation_org.append(dice_bs_org)
                    modelname = 'DiceVal_{:.4f}_Epoch_{:04d}.pth'.format(np.mean(Dices_Validation), step)
                    csv_dice = np.mean(Dices_Validation)
                    csv_dice_org = np.mean(Dices_Validation_org)
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice, csv_dice_org])

            step += 1

            if step > iteration:
                break
        print("one epoch pass")

    np.save(model_dir + '/Loss.npy', lossall)
    
train()