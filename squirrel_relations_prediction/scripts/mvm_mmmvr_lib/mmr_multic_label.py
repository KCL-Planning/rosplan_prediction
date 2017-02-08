######################
## Version 0.1 #######
## /**********************************************************************
##   Copyright 2015, Sandor Szedmak  
##   email: sandor.szedmak@uibk.ac.at
##          szedmak777@gmail.com
##
##   This file is part of Maximum Margin Multi-valued Regression code(MMMVR).
##
##   MMMVR is free software: you can redistribute it and/or modify
##   it under the terms of the GNU General Public License as published by
##   the Free Software Foundation, either version 3 of the License, or
##   (at your option) any later version. 
##
##   MMMVR is distributed in the hope that it will be useful,
##   but WITHOUT ANY WARRANTY; without even the implied warranty of
##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##   GNU General Public License for more details.
##
##   You should have received a copy of the GNU General Public License
##   along with MMMVR.  If not, see <http://www.gnu.org/licenses/>.
##
## ***********************************************************************/
######################
## import os, string, re, math, pickle, random
## import math, time
## import numpy
## import pylab as lab
from numpy import eye, zeros, where, median, sqrt, ones, linalg, diag
from numpy import dot, argmin

## #################################################################
def mmr_multic_label(ilabmode,Y,X,kk,lpar):
  """
    It labels the outputs for multiclass classification
    Input:

    ilabmode  labeling mode
                =0 indicators
                =1 class mean
                =2 class median
                =3 tetrahedron
                =31 weighted simplex  
    Y         output categories column vector with components =1,...,kk  
    X         corresponding input vectors, the rows contain the input vectors
    kk        number of possible categories
    lpar      optional parameter used by method 31

    Output:

    YL  label vectors in its rows to all sample items
    Y0        all possible labels, it has kk rows and in the rows the
              possible labels  
  """
  if len(Y.shape)==1:
    m=Y.shape[0]
    Y=Y.reshape((m,1))
  else:
    m=Y.shape[0]

  ## number of items and input dimension
  
  if ilabmode==0:
## the indicator case for multiclass learning
    Y0=eye(kk)
    ## setting the label vectors 
    YL=zeros((m,kk))
    for i in range(m):
      YL[i,Y[i]]=1
  elif ilabmode==1:
## class mean
    (m,nx)=X.shape
    Y0=zeros((kk,nx))
    xmm=zeros((kk,nx))
    xnn=zeros(kk)
    for i in range(m):
      iy=Y[i]
      xmm[iy,:]=xmm[iy,:]+X[i,:]
      xnn[iy]+=1
    for k in range(kk):
      if xnn[k]>0:
        Y0[k,:]=xmm[k,:]/xnn[k]
    YL=zeros((m,nx))
    for i in range(m):
      YL[i,:]=Y0[Y[i],:]
  elif ilabmode==2:
## class median
    (m,nx)=X.shape
    Y0=zeros((kk,nx))
    for k in range(kk):
      inx=where(Y==k)[0]
      if len(inx)>0:
        xmm=median(X[inx,:],axis=0)
        Y0[k,:]=xmm/sqrt(sum(xmm**2))
    YL=zeros((m,nx))
    for i in range(m):
      YL[i,:]=Y0[Y[i],:]
  elif ilabmode==3:
## tetrahedron, minimum correlation
    Y0=eye(kk)
    Y0=Y0+Y0/(kk-1)-ones((kk,kk))/(kk-1)
    (S,U)=linalg.eigh(Y0)
    SS=dot(U,diag(sqrt(abs(S))))
    ix=argmin(S)
    Y0=zeros((kk,kk-1))
    j=0
    for k in range(kk):
      if k!=ix:
        Y0[:,j]=SS[:,k]
        j+=1
    YL=zeros((m,kk-1))
    for i in range(m):
      YL[i,:]=Y0[Y[i],:]
  elif ilabmode==31:
    if kk>1:
      lpar=float(1)/(kk-1)
    else:
      lpar=1.0
    Y0=(1+lpar)*eye(kk)-lpar*ones((kk,kk))
    YL=zeros((m,kk))
    for i in range(m):
      YL[i,:]=Y0[Y[i,0],:]
  else:
    pass
  
  return(YL,Y0)

        
        
