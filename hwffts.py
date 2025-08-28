# -*- coding: utf-8 -*-
"""
Created on Mon May  3 04:05:50 2021

@author: han
"""

import numpy as np
import  os
import matplotlib.pyplot as plt
import glob
import math
import scipy.special as sp
import sys
import numpy.fft as ft

# %%
# BRU

def k_of_x(x):
    """k-vector with leading zero from x-vector
    also x-vector with leading zero from k-vector.
    Can also be used to get x from k """
    dx = x[1] - x[0]
    N = x.size
    dk = 2.*np.pi/(N*dx)
    inull = N//2
    k = dk*(np.linspace(1, N, N)-inull)

    return k



# %%

def hwfft2(x, y, f):
    """ 2D Fast Fourier Transform of f living on (x,y) into Ff. The length of
    x,y  and f must be an even number, preferably a power of two. In hwfft, x=0 is at the center of x, ie. at x[Nx//2-1]/
    The index of the zero mode for k is inull=Nx/2. Simlar requirements hold for y and l"""
    Nx = x.size
    Ny = y.size
    if x[Nx//2-1]!=0:
        raise NameError('hwfft only works if x=0 is at the center of x. Use python default fft etc. instead.')
    if y[Ny//2-1]!=0:
        raise NameError('hwfft only works if y=0 is at the center of y. Use python default fft etc. instead.')
 
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    inull = Nx//2
    # print 'inull = {}'.format(inull)
    jnull = Ny//2
    f=np.fft.ifftshift(f)
    
    Ff = dx * dy * np.roll(np.roll(ft.fft2(f), inull-1, 0), jnull-1, 1)
    # Ff = dx * dy * np.roll(ft.fft2(f),jnull-1,0)
    # Ff = dx * dy * ft.fft2(f)
    # for ii in range(x.size): Ff[:,ii] = Ff[:,ii]*(2**ii)

    return Ff


# %%


def hwifft2(k, l, Ff):
    """ (same as obifft2) 2D inverse FFT of f living on (k,l) into Ff. The length of
    k, l and Ff and f must be an even number, preferably a power of two."""
    x = k_of_x(k)
    y = k_of_x(l)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx = x.size
    Ny = y.size
    inull = Nx//2
    jnull = Ny//2
    Ff=np.roll(np.roll(Ff,-(inull-1),0),-(jnull-1),1)
    f=1./(dx*dy)*ft.ifft2(Ff)
    f=np.roll(np.roll(f,(inull),0),(jnull),1) #Corrected this on 11/16/2022 !!
    #f=np.roll(np.roll(f,(inull-1),0),(jnull-1),1)
    
    #f = 1. / (dx * dy) * ft.ifft2(np.roll(np.roll(Ff, 1-inull, 0), 1-jnull, 1))

    return f



# %%

