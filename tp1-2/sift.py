import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from tools import *


def compute_grad(I):

    Ix = conv_separable(I,np.array([-1,0,1]),np.array([1,2,1]))
    Iy = conv_separable(I,np.array([1,2,1]),np.array([-1,0,1]))
    return Ix, Iy



def compute_grad_mod_ori(I):
    Ix, Iy = compute_grad(I)
    Gn = np.sqrt(np.multiply(Ix,Ix)+np.multiply(Iy,Iy)) #peut etre carré de toute la matrice 
    Go = compute_grad_ori(Ix,Iy,Gn)
    aux = Go[np.where(Go!=0)]
    #print(aux[np.where(aux!=-1)])
    #print(np.max(Go))
    return Gn, Go


def compute_sift_region(Gm, Go, mask=None):
    
    # Note: to apply the mask only when given, do:
    if mask is not None:
        Gmpond=np.multiply(Gm,mask)
    else:
        Gmpond=np.copy(Gm)
        
    k = 0
    ListR=[]
    Go=Go.reshape(4,4,16)
    Gmpond=Gmpond.reshape(4,4,16)
    for i in range(Go.shape[0]):
        for j in range(Go.shape[1]):
            Rtemp = np.zeros(8)
            Rtemp[Go[i][j]]+=Gmpond[i][j]
            """for k in range(Go.shape[2]):
                if Go[i][j][k]!=-1:
                    Rtemp[Go[i][j][k]]+=Gmpond[i][j][k]"""
            ListR.append(Rtemp)  
    Renc=np.stack(ListR).reshape(128)
    #print(Renc.shape)
    normRenc=np.linalg.norm(Renc,2)
    if normRenc<0.5:
        return np.zeros(128)
    Renc/=normRenc
    Renc=np.where(Renc>0.2,0.2,Renc)
    normRenc=np.linalg.norm(Renc,2)
    Renc/=normRenc
    return Renc


def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    #print(im.shape)
    # TODO calculs communs aux patchs
    sifts = np.zeros(len(x)* len(y)* 128).reshape(len(x),len(y),128)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            #print(xi)
            #print(yj)
            #print(im[xi:xi+16,yj:yj+16])
            Gn,Go=compute_grad_mod_ori(im[xi:xi+16,yj:yj+16])
            vec=compute_sift_region(Gn, Go, mask=None)
            sifts[i, j, :] = vec # TODO SIFT du patch de coordonnee (xi, yj)
    return sifts