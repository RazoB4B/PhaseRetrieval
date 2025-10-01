#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:21:05 2025

@author: alberto-razo
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.special import jn
from colorsys import hls_to_rgb
from matplotlib import gridspec
from scipy.signal import argrelmax
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap


def ColorMaps(_CLabel):
    if _CLabel == 'BlackRed':
        _colors = [(0, 0, 0), (1, 0, 0)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackGreen':
        _colors = [(0, 0, 0), (0, 1, 0)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackBlue':
        _colors = [(0, 0, 0), (0, 0, 1)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'RedBlue':
        _colors = [(1, 0, 0), (0, 0, 0), (0, 0, 1)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackPink':
        _colors = [(0, 0, 0), (159/255, 43/255, 104/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackBrown':
        _colors = [(0, 0, 0), (193/255, 154/255, 107/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    else:
        print('ColorMap not defined')
    return _cmap


def useTex(_useTex):
    if _useTex:
        plt.rc('text', usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Helvetica']})
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')
    

def colorize(z, theme="dark", saturation=1.0, beta=1.4, transparent=False,
             alpha=1.0, max_threshold=1):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + r**beta) if theme == "white" else 1.0 - 1.0 / (1.0 + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1.0 - np.sum(c**2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c
    
    
def Harmonics(_Imgs, _Nharms=4, _Npad=1, _Sym=True, _axis=0, _Nper=1):
    """
    Takes the array of figures modulated in time and perform the Fourier tranforms
    to find the harmonics

    Imgs = Array of data with time dependence
    Npad = Number of periods after padding
    Nharm = Number of Harmonics that will be extracted from the data
    Nper = Number of periods included in the original signal considering its time dependence
    axis = axis of 'Imgs' that represents the time dependence 
    """
    
    if _Nper !=1:
        _Imgs = np.array_split(_Imgs, _Nper, axis=_axis)[0]
        
    _img = _Imgs[1:]
    for i in range(_Npad-1):
        _Imgs = np.append(_Imgs, _img, axis=_axis)

    del _img
    _harms = np.fft.fft(_Imgs, axis=_axis)
    _harms = np.fft.fftshift(_harms, axes=_axis)

    _amps = np.mean(np.abs(_harms), axis=(1,2))

    _Ntot = len(_amps)
    _frqs = np.linspace(0, _Npad, _Ntot)
    _frqs = _frqs[1] - _frqs[0]
    _frqs = np.linspace(-0.5/_frqs, 0.5/_frqs, _Ntot)

    _inds = []
    for i in range(-_Nharms, _Nharms+1):
        _inds = np.append(_inds, np.where(i==_frqs))

    _inds = _inds.astype(int)
    _harms = np.take(_harms, _inds, axis=_axis)
    if not _Sym:
        _harms = _harms[_Nharms:]
    return _harms, _amps, _frqs


def PhaseDiffuser(_N, _ps, Seed):
    np.random.seed(Seed)
    _phasediffuser = np.random.rand(_N//_ps, _N//_ps)*2*np.pi
    _phasediffuser = np.repeat(np.repeat(_phasediffuser, _ps, axis=0), _ps, axis=1)
    return _phasediffuser


def Corr(_Arr1, _Arr2):
    _Arr1 = _Arr1 - np.mean(_Arr1)
    _Arr2 = _Arr2 - np.mean(_Arr2)
    _corr = np.fft.ifft2(np.conjugate(np.fft.fft2(_Arr1))*np.fft.fft2(_Arr2))
    _corr = np.fft.fftshift(_corr)
    return _corr


def CropCenter(_array, _crop):
    _x, _y = _array.shape
    _startx = _x//2-(_crop//2)
    _starty = _y//2-(_crop//2)
    return _array[_startx:_startx+_crop, _starty:_starty+_crop]


def GetFarField(_array, _Npad=5, _WinSize=5):
    _n = len(_array)
    _pad = (_Npad-1)*_n//2
    _array = np.pad(_array, _pad, mode='constant')
    _array = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(_array)))
    return CropCenter(_array, _n*_WinSize)


def GetFarDiffuser(_array, _Npad=5, _WinSize=5):
    _n = len(_array)
    _pad = (_Npad-1)*_n//2
    _array = np.pad(_array, _pad, mode='constant')
    _array = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(_array)))
    return CropCenter(_array, _n*_WinSize)


def Airy(_r, _a, _b):
    return (_b*jn(1, 2*np.pi*_r*_a)/_r)**2


def GetGrainSize(_IntensArray, _axis=0, _p00=0.1, _p01=1e1):
    _IntensArray = np.abs(Corr(_IntensArray, _IntensArray))
    _IntensArray = _IntensArray/np.max(_IntensArray)
    _len = _IntensArray.shape[_axis]
    _IntensArray = np.take(_IntensArray, indices=_len//2, axis=_axis)
    _dom = np.linspace(-_len//2, _len//2, _len)
    _opt, _cov = curve_fit(Airy, _dom, _IntensArray, p0=[_p00, _p01])
    return 1/(2*_opt[0])


def SectionMask(_size, _nfigs, _angle, _rmax, _rmin=0, _deph=0, _clock=True):
    _ax = np.linspace(-1, 1, _size)
    _axX, _axY = np.meshgrid(_ax, _ax)

    _r = np.sqrt(_axX**2 + _axY**2)
    _theta = np.arctan2(_axY, _axX)

    _dtheta = np.pi*_angle/360
    _deph = np.pi*_deph/180
    _cthetas = np.linspace(0, 2*np.pi, _nfigs) + _deph

    if _clock:
        _c = 1
    else:
        _c = -1

    _Masks = []
    for _ctheta in _cthetas:
        _mask = np.zeros([_size, _size], dtype='complex')
        _mask[_r<=_rmax] = 1
        _mask[_r<=_rmin] = 0
        for _j in range(-3,4):
            _mask[np.abs(_theta-_c*_ctheta+2*_j*np.pi)<=_dtheta] = 0

        _Masks.append(_mask.astype(complex))
    return _Masks


def DoubleSectionMask(_size, _nfigs, _angle1, _rmax1, _rmin1, _deph1,
                      _angle2, _rmax2, _rmin2, _deph2, _clock1=True, _clock2=True):
    _masks1 = SectionMask(_size, _nfigs, _angle1, _rmax1, _rmin1, _deph1, _clock1)
    _masks2 = SectionMask(_size, _nfigs, _angle2, _rmax2, _rmin2, _deph2, _clock2)

    _Masks = []
    for i in range(_nfigs):
        _mask = np.zeros([_size, _size], dtype = complex)
        _mask[(np.abs(_masks1[i])>0.5) | (np.abs(_masks2[i])>0.5)] = 1

        _Masks.append(_mask.astype(complex))
    return _Masks


def VortexMask(_size, _rmax, _rmin, _ordmax=1, _Sym=True):
    _ax = np.linspace(-1, 1, _size)
    _axX, _axY = np.meshgrid(_ax, _ax)

    _r = np.sqrt(_axX**2 + _axY**2)
    _theta = np.arctan2(_axY, _axX)
    
    if _Sym:
        _inds = np.arange(-_ordmax, _ordmax+1)
    else:
        _inds = np.arange(0, _ordmax+1)
    
    _Masks = []
    for i in _inds:
        _phase = (i*_theta+np.pi)%(2*np.pi)-np.pi

        _mask = np.zeros([_size, _size], dtype='complex')
        _indX, _indY = np.where((_r<=_rmax) & (_r>=_rmin))
        _mask[_indX, _indY] = np.exp(-1j*_phase[_indX, _indY])

        _Masks.append(_mask.astype(complex))
    return _Masks


def FindVortex(_PhaseArray):
    _lenX, _lenY = _PhaseArray.shape
    _VP = []
    _VM = []
    for i in range(_lenX-1):
        for j in range(_lenY-1):
            _dp = np.unwrap([_PhaseArray[i,j], _PhaseArray[i,j+1], _PhaseArray[i+1,j+1],
                             _PhaseArray[i+1,j], _PhaseArray[i,j]])
            if np.abs(np.abs(_dp[-1]-_dp[0])-2*np.pi)<1e-2 and (i!=0 and j!=0):
                if _dp[-1]-_dp[0] > 0:
                    _VM.append([j, i])
                else:
                    _VP.append([j, i])
    return _VP, _VM


def FindMax(_IntensArray, _radius=25):
    _len = _IntensArray.shape[0]
    _vec = np.arange(_len)
    _mesh = np.meshgrid(_vec, _vec)

    _Max = []
    for i in _vec:
        _max = argrelmax(_IntensArray[i, :], order=_radius-1)[0]
        for j in _max:
            _r = np.sqrt((_mesh[0]-j)**2 + (_mesh[1]-i)**2)
            _inds = np.where(_r < _radius)
            if np.max(_IntensArray[_inds]) <= _IntensArray[i, j]:
                _Max.append([j, i])
    return _Max


def FindMin(_arr, _radius=25):
    return FindMax(1/_arr, _radius)


def GetTopology(_Maps):
    _GS = []
    _M = []
    _VP = []
    _VM = []

    for _map in _Maps:
        _gs = GetGrainSize(np.abs(_map)**2)
        _vp, _vm = FindVortex(np.angle(_map))
        _m = FindMax(np.abs(_map)**2, np.int(_gs*0.5))
        
        _GS.append(_gs)
        _M.append(_m)
        _VP.append(_vp)
        _VM.append(_vm)
    return _GS, _M, _VP, _VM


def FindDistances(_VortexPos, _MaxPos, _GrainSize=None):
    _dist = []
    for _v in _VortexPos:
        _dist.append(np.inf)
        for _m in _MaxPos:
            _d = np.sqrt((_m[0]-_v[0])**2 + (_m[1]-_v[1])**2)
            if _d < _dist[-1]:
                _dist[-1] = _d

    if _GrainSize != None:
        _dist = _dist/_GrainSize
    return _dist


def RadialHistogram(_dist, _bins):
    _hist, _edge = np.histogram(_dist, bins=_bins, density=True)
    _edge = (_edge[:-1] + _edge[1:])/2
    _hist = _hist/_edge
    _hist = _hist/np.trapezoid(_hist*2*np.pi*_edge, _edge)
    return _hist, _edge


#%% Definitions pyTorch

def L2_norm(_x, _y):
    return torch.sqrt(torch.mean(torch.abs(_x - _y)**2))


def GetFarField_torch(array, Npad=5, WinSize=5):
    """
    Simulate the far-field (FFT) of a complex 2D array with zero padding and cropping.
    """
    n = array.shape[-1]
    pad = (Npad - 1) * n // 2
    # Pad (left, right, top, bottom) — note reversed order in F.pad for 2D
    array_padded = F.pad(array, (pad, pad, pad, pad), mode='constant', value=0)
    # Apply fftshift, fft2, then fftshift again
    shifted_input = torch.fft.fftshift(array_padded, dim=(-2, -1))
    fft_output = torch.fft.fft2(shifted_input)
    shifted_fft = torch.fft.fftshift(fft_output, dim=(-2, -1))
    # Crop center
    return CropCenter_torch(shifted_fft, n * WinSize)


def CropCenter_torch(tensor, size):
    """Crop center of a square 2D tensor to the given size"""
    center = tensor.shape[-1] // 2
    half = size // 2
    return tensor[..., center - half:center + half, center - half:center + half]
