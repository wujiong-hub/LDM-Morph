import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adam

from natsort import natsorted
import csv
import warnings
import time

from utils.utils import *
import TransModels.LDMMorph as LDMMorph 

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default
from omegaconf import OmegaConf
from torch.autograd import Variable

parser = ArgumentParser()
parser.add_argument("--resume", type=str,
                    dest="resume", default='your/trained/ldm/checkpoint/path',
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
                    default='your/testing/data/path',
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

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def main():
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

    testname = sorted(glob.glob(datapath + '*_2CH.npz'))[400:420]
    test_loader = Data.DataLoader(Dataset_epoch_with_name(testname), batch_size=1, shuffle=False, num_workers=1)
                                        

    transform = SpatialTransform().cuda()
    model_dir = './logs/TransScorelm_Smooth_{}_beta_{}/'.format(smooth, beta)

    model = LDMMorph.LDMMorph(128*2,192*2,320*2,448*2)
    model.cuda()
    total = sum([param.nelement() for param in model.parameters()])

    model_idx=-2
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx],map_location='cuda:0')#['state_dict']
    model.load_state_dict(best_model)
    model.cuda()

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    registTime = []


    with torch.no_grad():
        stdy_idx = 0
        for x, y, x_seg, y_seg, _ in test_loader:

            x, y, x_seg, y_seg = x.to(device), y.to(device), x_seg.to(device), y_seg.to(device)
            grid_img = mk_grid_img(grid_step=6)

            time1 = time.time()
            model.eval()

            mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(x)).detach()
            fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(y)).detach()

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

            D_f_xy = model(x, y, score0, score1, score2, score3)
            time2 = time.time()

            __, def_out= transform(x_seg, D_f_xy.permute(0, 2, 3, 1), mod = 'nearest')
            __, def_grid = transform(grid_img.float().to(device), D_f_xy.permute(0, 2, 3, 1))
            __, def_img = transform(x.float().to(device), D_f_xy.permute(0, 2, 3, 1))
            flow_cpu = D_f_xy.permute(0, 2, 3, 1).detach().cpu().numpy()[0, :, :, :]
            tar = y.detach().cpu().numpy()[0, 0, :, :]
            
            hh, ww = D_f_xy.shape[-2:]
            D_f_xy = D_f_xy.detach().cpu().numpy()
            D_f_xy[:,0,:,:] = D_f_xy[:,0,:,:] * hh / 2
            D_f_xy[:,1,:,:] = D_f_xy[:,1,:,:] * ww / 2
            
            x_cpu = x[0,0,...].cpu().data.numpy()
            y_cpu = y[0,0,...].cpu().data.numpy()
            x_seg_cpu = x_seg[0,0,...].cpu().data.numpy()
            y_seg_cpu = y_seg[0,0,...].cpu().data.numpy()
            def_seg_cpu = def_out[0,0,...].cpu().data.numpy()
            def_img_cpu = def_img[0,0,...].cpu().data.numpy()
            def_grid_cpu = def_grid[0,0,...].cpu().data.numpy()

            '''
            output = './outputs/scorelm/'+os.path.basename(testnames[stdy_idx]).replace('.npz','.nii.gz')
            nib.save(nib.Nifti1Image(x_cpu, np.eye(4)), output.replace('.nii.gz', '_mov.nii.gz'))
            nib.save(nib.Nifti1Image(y_cpu, np.eye(4)), output.replace('.nii.gz', '_fix.nii.gz'))
            nib.save(nib.Nifti1Image(x_seg_cpu, np.eye(4)), output.replace('.nii.gz', '_movseg.nii.gz'))
            nib.save(nib.Nifti1Image(y_seg_cpu, np.eye(4)), output.replace('.nii.gz', '_fixseg.nii.gz'))
            nib.save(nib.Nifti1Image(def_img_cpu, np.eye(4)), output)
            nib.save(nib.Nifti1Image(def_seg_cpu, np.eye(4)), output.replace('.nii.gz','_seg.nii.gz'))
            nib.save(nib.Nifti1Image(def_grid_cpu, np.eye(4)), output.replace('.nii.gz','grid.nii.gz'))
            nib.save(nib.Nifti1Image(flow_cpu, np.eye(4)), output.replace('.nii.gz','Warp.nii.gz'))
            '''
            
            jac_det = jacobian_determinant_vxm(D_f_xy[0, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = dice(def_out.long().data.cpu().numpy().copy(), y_seg.long().data.cpu().numpy().copy())
            dsc_raw = dice(x_seg.long().data.cpu().numpy().copy(), y_seg.long().data.cpu().numpy().copy())

            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

            registTime.append(time2-time1)

    print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                eval_dsc_def.std,
                                                                                eval_dsc_raw.avg,
                                                                                eval_dsc_raw.std))
    print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

    mtime, stime = np.mean(registTime), np.std(registTime)
    print('Deform Time | mean = %.4f, std= %.4f' % (mtime, stime))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    main()
