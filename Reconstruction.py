#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 16:43:22 2025

@author: alberto-razo
"""

import os
import time
import torch
import numpy as np
import Definitions as Def
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

N = 240                         # Number of pixels in the diffuser
ps = 8
psrs = [1] 
WinSize = 1                     # Padding number (Size of the speckle figure N*WinSize)
NPad = 10*WinSize

r_max = 0.9                  
r_min = 0

NHarms = 1                      # Number of Harmonics to find (Look function Harmonics, if Sym=True -> pos and neg harmonics)

LR_init = 5e0
Steps = 50000

TSeeds = 10

thetas = [180]
for theta in thetas:
    alpha = theta*2*np.pi/360
    Nangle = (2*np.pi)/(alpha)
    c1 = -np.abs(np.sinc(1/Nangle))/(Nangle-1)
    
    for psr in psrs:
        print(f'theta={theta:03d}', f'pixelsize={psr:02d}')
        for Seed in range(TSeeds):
            _filename = f'Simulations/Reconstruction_OPixel={ps:02d}_PPixel={psr:02d}_theta={theta:03d}_N={N*WinSize}_LearningRate={LR_init:01.0e}_Steps={Steps:01.0e}_Seed={Seed:02d}.npy'
            print(_filename)
            if not os.path.exists(_filename):
                print(f'seed={Seed}')
                Diffuser = np.exp(1j*Def.PhaseDiffuser(N, ps, Seed))
    
                # Measuring answer preparation
                _Masks = Def.SectionMask(N, 101, theta, r_max, r_min)
                _speckles = []
                for _mask in tqdm(_Masks):
                    _out = Def.GetFarField(_mask*Diffuser, NPad, WinSize)
                    _speckles.append(_out.astype(complex))
    
                S, _, _ = Def.Harmonics(np.abs(_speckles)**2, NHarms, 10, False)
 
                VMasks = Def.VortexMask(N, r_max, r_min, NHarms, True)
                A0 = Def.GetFarField(VMasks[1]*Diffuser, NPad, WinSize)
                A02 = np.abs(A0)**2

                _prop = Def.GetFarField(VMasks[1]*Diffuser, NPad, WinSize)
                RDiffuser = Def.GetFarDiffuser(_prop, NPad, WinSize)

                # Torch preparation    
                A0_target = torch.from_numpy(A0).to(torch.complex64).to(device)
                A02_target = torch.from_numpy(A02).to(torch.complex64).to(device)
                S0_target = torch.from_numpy(S[0]).to(torch.complex64).to(device)
                S1_target = torch.from_numpy(S[1]).to(torch.complex64).to(device)

                A0_target = A0_target/torch.mean(torch.abs(A0_target))
                A02_target = A02_target/torch.mean(torch.abs(A02_target))
                S0_target = S0_target/torch.mean(torch.abs(S0_target))
                S1_target = S1_target/torch.mean(torch.abs(S1_target))

                V0 = torch.from_numpy(VMasks[1]).to(torch.complex64).to(device)
                Vp1 = torch.from_numpy(VMasks[2]).to(torch.complex64).to(device)
                Vm1 = torch.from_numpy(VMasks[0]).to(torch.complex64).to(device)

                Diff_target = torch.from_numpy(Diffuser).to(torch.complex64).to(device)
                RDiff_target = torch.from_numpy(RDiffuser).to(torch.complex64).to(device)
                RDiff_target = RDiff_target/torch.mean(torch.abs(RDiff_target))

                # Optimization parameter
                param_real = torch.ones(N//psr, N//psr, requires_grad=True)
                param_imag = torch.ones(N//psr, N//psr, requires_grad=True)
                optimizer = torch.optim.Adam([param_real, param_imag], lr=LR_init)
    
                # Optimization
                TotLoss = np.zeros([Steps])
                a0Loss = np.zeros([Steps])
                EnergyLoss = np.zeros([Steps])
                DiffLoss = np.zeros([Steps])
                RDiffLoss = np.zeros([Steps])
                ElapTime = np.zeros([Steps])
                _timei = time.time()
                for _step in tqdm(range(Steps)):
                    optimizer.zero_grad()

                    # Combine real and imaginary parts and apply mask
                    diff = torch.complex(param_real, param_imag)
                    diff = diff.repeat_interleave(psr, dim=1)
                    diff = diff.repeat_interleave(psr, dim=0)

                    # Compute the far field
                    pred_a0 = Def.GetFarField_torch(diff*V0, NPad, WinSize)  
                    pred_ap1 = Def.GetFarField_torch(diff*Vp1, NPad, WinSize)  
                    pred_am1 = Def.GetFarField_torch(diff*Vm1, NPad, WinSize)  

                    pred_a0 = pred_a0/torch.mean(torch.abs(pred_a0))
                    pred_ap1 = pred_ap1/torch.mean(torch.abs(pred_ap1))
                    pred_am1 = pred_am1/torch.mean(torch.abs(pred_am1))
                
                    S0_pred = torch.abs(pred_a0)**2 + torch.abs(c1*pred_am1)**2 + torch.abs(c1*pred_ap1)**2
                    S1_pred = -pred_a0*pred_am1.conj() - pred_ap1*pred_a0.conj()
    
                    A02_pred = torch.abs(pred_a0)**2/torch.mean(torch.abs(pred_a0)**2)
                    S0_pred = S0_pred/torch.mean(torch.abs(S0_pred))
                    S1_pred = S1_pred/torch.mean(torch.abs(S1_pred))

                    loss_a0 = Def.L2_norm(A02_pred, A02_target)
                    loss_S0 = Def.L2_norm(S0_pred, S0_target)
                    loss_S1 = Def.L2_norm(S1_pred, S1_target)

                    loss_total = loss_a0 + loss_S0 + loss_S1 

                    loss_total.backward()
                    optimizer.step()

                    _prop = Def.GetFarField_torch(diff*V0, NPad, WinSize)
                    RDiff_pred = Def.GetFarDiffuser_torch(_prop, NPad, WinSize)
                    Diff_pred = diff/torch.mean(torch.abs(diff))
                    RDiff_pred = RDiff_pred/torch.mean(torch.abs(RDiff_pred))

                    EnergyMap = torch.abs(torch.fft.fft2(torch.exp(1j*(torch.angle(A0_target) - torch.angle(pred_a0)))))

                    TotLoss[_step] = loss_total.item()
                    a0Loss[_step] = Def.L2_norm(pred_a0, A0_target).item()
                    DiffLoss[_step] = Def.L2_norm(Diff_pred, Diff_target).item()
                    RDiffLoss[_step] = Def.L2_norm(RDiff_pred, RDiff_target).item()
                    EnergyLoss[_step] = (Def.CropCenter_torch(EnergyMap, 1)/torch.sum(EnergyMap)).item()
                    ElapTime[_step] = time.time() - _timei

                theDict = {}
                theDict['TotLosses'] = TotLoss
                theDict['a0Losses'] = a0Loss
                theDict['DiffLosses'] = DiffLoss
                theDict['RetroDiffLosses'] = RDiffLoss
                theDict['EnergyLosses'] = EnergyLoss
                theDict['ElapsedTime'] = ElapTime
                theDict['Diffuser'] = diff.detach().numpy()

                np.save(_filename, theDict)
