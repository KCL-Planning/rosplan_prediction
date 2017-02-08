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
import numpy as np
from scipy import sparse
## ####################
## import mvm_classes
## ####################
def mvm_kernel(X1,X2,params_spec,norm_spec):
##############################################################
# normalize and compute a kernel for training and cross kernel for two sets
# of sources assuming dense input matrixes
##############################################################
# inputs: 
#       X1     training feature matrix
#       X2     second feature matrix for cross kernel
#              in training X1 can be equal to X2
#       params_spec parameters for a kernel
#         ipar1       kernel fist parameter see below
#         ipar2       kernel second parameter
#         inorm       =1 normalization is needed =0 otherwise
#         input_norm  type of input normalization, only l2 norm is
#                   implemented 
#         kernel_type kernel type, linear, polynomial, Gaussian 
#       
# outputs:
#       K         normalized kernel
##############################################################
  
## print('Linear kernel')

  if X2 is None:
    X2=np.array([])

  m1=X1.shape[0]
  m2=X2.shape[0]

  if len(X1.shape)==1:   ## vector
    X1=X1.reshape((m1,1))
    
  if len(X2.shape)==1:
    X2=X2.reshape((m2,1))
  
  if m2>0:    # is the second source given ?
    m2=X2.shape[0]
    d1=np.sum(X1**2,axis=1)
    d2=np.sum(X2**2,axis=1)
    K=np.dot(X1,X2.T)
  else:                        # base case
    m2=m1
    d1=np.sum(X1**2,axis=1)
    d2=d1
    K=np.dot(X1,X1.T)
    
  ## print('Normalization:')

  e1=np.ones(m1)
  e2=np.ones(m2)
  
  if norm_spec.iscale==0:
    # row wise by L2 norm
    if X2.shape[0]>0:
      xnorm1=np.sum(X1**2,axis=1)
      xnorm1=np.sqrt(xnorm1)
      xnorm1=xnorm1+(xnorm1==0)
      xnorm2=np.sum(X2**2,axis=1)
      xnorm2=np.sqrt(xnorm2)
      xnorm2=xnorm2+(xnorm2==0)
      K=K/np.outer(xnorm1,xnorm2)
      d1=e1
      d2=e2
    else:
      d=np.sqrt(np.diag(K))
      d=d+(d==0)       # avoid divison by zero
      K=K/np.outer(d,d)
      d1=e1
      d2=d1

  K=kernel_nlr(K,d1,d2,params_spec)    
  
  return(K)

## ############################################################
def kernel_category_1d(X1,X2,params_spec):

  ncat=max(X1.max(),X2.max())+1
  m1=X1.shape[0]
  (m2,n2)=X2.shape

  vequal=1.0
  vnoteq=-1.0/(ncat-1)

  K=np.zeros((m1,m2))
  for irow in range(m1):
    for icol in range(m2):
      xr=X1[irow]
      xc=X2[icol]
      for i in range(n2):
        if xr[i]==xc[i]:
          K[irow,icol]+=vequal
        else:
          K[irow,icol]+=vnoteq

  if params_spec.iscale>=0:
    ## normalize the kernel
    d1=np.sqrt(np.diag(K))
    d2=d1
    d1=d1+(d1==0)
    d2=d2+(d2==0)
    K=K/np.outer(d1,d2)
    [m1,m2]=K.shape
    d1=np.ones(m1)
    d2=np.ones(m2)
  else:
    d1=np.diag(K)
    d2=d1

  K=kernel_nlr(K,d1,d2,params_spec,isymmetric=1)

  return(K)
## ############################################################
def kernel_category_2d(xdatacls,params_spec,norm_spec):

  xdata=xdatacls.xdata_tra
  ## xdata_2=xdatacls.xdata_tes
  
  nrow=xdatacls.nrow
  ncol=xdatacls.ncol

  vequal=1.0
  ncat=xdatacls.categorymax
  vnoteq=-1.0/(ncat-1)

  ## print('Linear kernel')
  xranges=xdatacls.xranges_rel
  xdata1=xdata[1]
  xdata2=xdata[2]

  K=np.zeros((ncol,ncol))
  for irow1 in range(nrow):
    nlen1=xranges[irow1][1]
    if nlen1>0:
      istart1=xranges[irow1][0]
    else:
      continue
    for i in range(istart1,istart1+nlen1):
      icol1=xdata1[i]
      val1=xdata2[i]
      for j in range(istart1,istart1+nlen1):
        icol2=xdata1[j]
        val2=xdata2[j]
        if val1!=val2:
          K[icol1,icol2]+=vnoteq
        else:
          K[icol1,icol2]+=vequal
    ## print(irow1)
      

  if norm_spec.ilocal>=0:
    ## centralize the kernel
    K=kernel_center(K)
  
  if norm_spec.iscale>=0:
    ## normalize the kernel
    d1=np.sqrt(np.diag(K))
    d2=d1
    d1=d1+(d1==0)
    d2=d2+(d2==0)
    K=K/np.outer(d1,d2)
    [m1,m2]=K.shape
    d1=np.ones(m1)
    d2=np.ones(m2)
  else:
    d1=np.diag(K)
    d2=d1


  K=kernel_nlr(K,d1,d2,params_spec,isymmetric=1)

  return(K)
## ############################################################
def kernel_categoryvec_2d(xdatacls,params_spec,norm_spec):

  xdata=xdatacls.xdata_tra
  ## xdata_2=xdatacls.xdata_tes
  
  nrow=xdatacls.nrow
  ## ncol=xdatacls.ncol

  ## vequal=1.0
  ## ncat=xdatacls.categorymax
  ## vnoteq=-1.0/(ncat-1)

  xranges=xdatacls.xranges_rel
  K=np.zeros((nrow,nrow))

  Kpre=xdatacls.YKernel.Kpre    ## kernel between vector of multiclasses

  xdata1=xdata[1]
  xdata2=xdata[2]
  for irow1 in range(nrow):
    nlen1=xranges[irow1][1]
    if nlen1>0:
      istart1=xranges[irow1][0]
    else:
      continue
    for irow2 in range(irow1,nrow):
      nlen2=xranges[irow2][1]
      if nlen2>0:
        istart2=xranges[irow2][0]

        drow2={}
        for i in range(istart2,istart2+nlen2):
          drow2[xdata1[i]]=xdata2[i]

        for i in range(istart1,istart1+nlen1):
          key1=xdata1[i]
          if key1 in drow2:
            val1=xdata2[i]
            val2=drow2[key1]
            K[irow1,irow2]+=Kpre[val1,val2]
        K[irow2,irow1]=K[irow1,irow2]

  if norm_spec.ilocal>=0:
    ## centralize the kernel
    K=kernel_center(K)
  
  if norm_spec.iscale>=0:
    ## normalize the kernel
    d1=np.sqrt(np.diag(K))
    d2=d1
    d1=d1+(d1==0)
    d2=d2+(d2==0)
    K=K/np.outer(d1,d2)
    [m1,m2]=K.shape
    d1=np.ones(m1)
    d2=np.ones(m2)
  else:
    d1=np.diag(K)
    d2=d1


  K=kernel_nlr(K,d1,d2,params_spec,isymmetric=1)

  return(K)
## #######################################################################
## centralization in the feature space
def kernel_center(K):

  (m1,m2)=K.shape

  K=K-np.outer(np.ones(m1),np.mean(K,axis=0)) \
     -np.outer(np.mean(K,axis=1),np.ones(m2)) \
     +np.ones((m1,m2))*np.mean(K)

  return(K)

## ############################################################
def kernel_nlr(K,d1,d2,params_spec,isymmetric=0):

  (m1,m2)=K.shape
  e1=np.ones(m1)
  e2=np.ones(m2)
  
  ipar1=params_spec.ipar1
  ipar2=params_spec.ipar2

  ## print('Nonlinear kernel:')
  kernel_type=params_spec.kernel_type
  if kernel_type==0:
    pass
  elif kernel_type==1:  # polynomial
    K=np.sign(K+ipar2)*np.abs(K+ipar2)**ipar1   ##in case of fraction power
    ## K=(K+ipar2)**ipar1
    if isymmetric==1:
      d=np.sqrt(np.diag(K))
      d=d+(d==0)       # avoid divison by zero
      K=K/np.outer(d,d)
  elif kernel_type==2:  # sigmoid
    K=np.tanh(ipar1*K+ipar2)
  elif kernel_type==3:  # Gaussian
    K=np.outer(d1,e2)+np.outer(e1,d2)-2*K
    K=K-K*(K<0)
    K=np.exp(-K/(2*ipar1**2))
  elif kernel_type==31:  # PolyGauss
    K=np.outer(d1,e2)+np.outer(e1,d2)-2*K
    K=K-K*(K<0)
    K=np.sqrt(K)
    K=np.exp(-K**ipar2/(2*ipar1**2))
  elif kernel_type==5:   ## Cauchy 
    K=np.outer(d1,e2)+np.outer(e1,d2)-2*K
    K=K-K*(K<0)
    K=2*ipar1/(4*ipar1**2+K)

  return(K)
## ###########################################################
def mvm_kernel_sparse(xdatacls,isymmetric,params_spec,norm_spec):
  """
  normalize and compute a kernel for training and cross kernel for two sets
  of sources assuming sparse input matrixes
  inputs: 
        xdatacls      data class
        isymmetric    =1 training case =0 test, cross kernel case
        params_spec   kernel specific parameters
  outputs:
        K         normalized kernel
  """
  xdata_1=xdatacls.xdata_tra
  xdata_2=xdatacls.xdata_tes

  txdim=xdata_1[2].shape
  if len(txdim)==1:
    nxdim=1
  else:
    nxdim=txdim[1]
  
  nrow=xdatacls.nrow
  ncol=xdatacls.ncol

  ## print('Linear kernel')

  ## xdata_1[1] gives row indeces in the process,
  ## xdata_1[0] gives column indeces in the computation !!!!
  if nxdim==1:
    xdatacol_1=xdata_1[2]
    xdatacol_2=xdata_2[2]
  else:
    xdatacol_1=xdata_1[2][:,0]
    xdatacol_2=xdata_2[2][:,0]
  ## sparse features first step  
  xfeature1=sparse.csr_matrix((xdatacol_1,(xdata_1[1], \
                                      xdata_1[0])), \
                                      shape=(ncol,nrow))
  if isymmetric==1:
    K=xfeature1.dot(xfeature1.T).toarray()
  else:
    xfeature2=sparse.csr_matrix((xdatacol_2,(xdata_2[1], \
                                 xdata_2[0])), \
                                 shape=(ncol,nrow))
    K=xfeature1.dot(xfeature2.T).toarray()
    d1=np.sum(xfeature1**2,axis=1)
    d2=np.sum(xfeature2**2,axis=1)

  ## sparse features next vector items    
  for i in range(1,nxdim):
    xfeature1=sparse.csr_matrix((xdata_1[2][:,i],(xdata_1[1], \
                                      xdata_1[0])), \
                                      shape=(ncol,nrow))
    if isymmetric==1:
      K+=xfeature1.dot(xfeature1.T).toarray()
    else:
      xfeature2=sparse.csr_matrix((xdata_2[2][:,i],(xdata_2[1], \
                                 xdata_2[0])), \
                                 shape=(ncol,nrow))
      K+=xfeature1.dot(xfeature2.T).toarray()
      d1+=np.sum(xfeature1**2,axis=1)
      d2+=np.sum(xfeature2**2,axis=1)

  ## compute norms
  if isymmetric==1:
    d1=np.diag(K)
    d2=np.diag(K)
  else:
    d1=np.sqrt(d1)
    d2=np.sqrt(d2)

  (K,d1,d2)=kernel_centralize_normalize(K,d1,d2,norm_spec)
    
  ## nonlinear transformation
  if params_spec.kernel_type>0:
    K=kernel_nlr(K,d1,d2,params_spec,isymmetric=1)

  return(K)
## ##########################################################33  
def kernel_centralize_normalize(K,d1,d2,norm_spec):
  
  ## centralize
  if norm_spec.ilocal>=0:
    K=kernel_center(K)

  ## normalize
  if norm_spec.iscale>=0:
    d1=d1+(d1==0)
    d2=d2+(d2==0)
    K=K/np.outer(d1,d2)
    [m1,m2]=K.shape
    d1=np.ones(m1)
    d2=np.ones(m2)

  return(K,d1,d2)
  
## ##########################################################33  
def kernel_multiclass_vector(valrange,ndim,classweight):
  """
  compute a kernel on unraveled indexes of a the range of integers
  Input:
            valrange      list of multicategory values
            ndim          number of multiclass labels
            classweight   weights of the classes
  """
  nitem=max(valrange)+1       ## number of classes
  nfullsize=nitem**ndim     ## the length of the range
  tdim=tuple([nitem]*ndim)  
  tfeature=np.unravel_index(np.arange(nfullsize),tdim)
  yfeature=np.array(tfeature).T
  m=yfeature.shape[0]
  K=np.zeros((m,m))
  for i in range(m):
    yi=yfeature[i]
    for j in range(i,m):
      yj=yfeature[j]
      iy=np.where(yi==yj)[0]
      iv=yj[iy]
      idot=np.sum(classweight[iy,iv])
      K[i,j]=idot
      K[j,i]=idot

  ## d=np.diag(K)
  ## d=d+(d==0)
  ## K=K/np.outer(d,d)

  return(K)    
## ##########################################################33  
  

  
  
  
  
  
  
