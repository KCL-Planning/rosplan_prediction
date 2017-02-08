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
## ###########################################################
from mvm_mmmvr_lib.mmr_base_classes import cls_crossval, cls_kernel_params, cls_norm
## from mmr_multic_label import mmr_multic_label
from mvm_mmmvr_lib.mmr_normalization_new import mmr_normalization
## from mmr_kernel_eval import kernel_eval_kernel, kernel_eval_nl, kernel_center

import mvm_mmmvr_lib.mvm_kernel_eval as mvm_kernel_eval

## class definitions

## ##################################################
class cls_feature:

  def __init__(self,ifeature=0):

    self.ifeature=ifeature       ## =0 explicit feature, =1 kernel 
    self.icategory=0      ## =0 vector =number of categories
    self.ncategory=0      ## idf icategory=1 it is the number of categories
    self.cat2vector=0     ## =0 indicator =1 mean 2 =median 3 =tetrahedron
    self.mdata=0
    self.itrain=None
    self.itest=None
    self.dataraw=None
    self.data=None       ## raw input
    self.XTrain=None      ## training features 
    self.XTrainNorm=None  ## normalized features
    self.XTest=None       ## test features 
    self.XTestNorm=None   ## normalized features
    ## self.Y0=None          ## set of distinc feature vectors     
    ## self.Y0Norm=None      ## set of distinc normalizedfeature vectors     

    self.K=None           ## external training kernel
    self.Kcross=None      ## externel test kernel
    self.Kpre=None        ## prekernel can be used to build input and output
    self.d1=None       ## norm of left factor of the kernel
    self.d2=None      ## norm of right factor of the kernel

    ## self.ilocal=2
    ## self.iscale=0

    self.norm=cls_norm()
    self.crossval=cls_crossval()
    self.kernel_params=cls_kernel_params()
    self.prekernel_params=None

    self.title='mvm_y'

    self.ymax=10.0
    self.ymin=-10.0
    self.yrange=20
    self.ystep=(self.ymax-self.ymin)/self.yrange

    self.Y0Tetra=None

    self.ndim=4
    self.valrange=(0,1,2,3)
    self.classweight=np.ones((self.ndim,len(self.valrange)))
    
## -------------------------------------------------------------
  def load_data(self,dataraw):

    self.dataraw=dataraw
    self.mdata=len(self.dataraw)
    self.data=self.dataraw

## -------------------------------------------------------------
  def set_train_test(self,itrain,itest):

    self.itrain=itrain
    self.itest=itest

## -------------------------------------------------------------
  def get_train(self,itrain):

    if self.data is not None:
      return(self.data[itrain,:])
    else:
      return(None)

## -------------------------------------------------------------
  def get_test(self,itest):

    return(self.data[itest,:])
      
## ## -------------------------------------------------------------
##   def get_Y0(self,itrain):

##     if self.Y0 is None:
##       return(self.get_train(itrain))
##     else:
##       return(self.Y0)

## ## -------------------------------------------------------------
##   def get_Y0_norm(self,itrain,itest):

##     if self.Y0Norm is None:
##       if self.XTrainNorm is None:
##         (self.XTrainNorm,self.XTestNorm,opar)= \
##                 mmr_normalization(self.ilocal,self.iscale, \
##                                   self.data[itrain], \
##                                   self.data[itest],0)
##       self.Y0Norm=self.XTrainnorm
##     return(self.Y0Norm)

## --------------------------------------------------------------
  def get_train_norm(self,itrain):

    if self.XTrainNorm is None:
      (self.XTrainNorm,self.XTestNorm)= \
              mmr_normalization(self.norm.ilocal,self.norm.iscale, \
                                self.data[self.itrain], \
                                self.data[self.itest],0)[:2]
    return(self.XTrainNorm)

## --------------------------------------------------------------
  def get_test_norm(self,itest):

    if self.XTestNorm is None:
      (self.XTrainNorm,self.XTestNorm)= \
              mmr_normalization(self.norm.ilocal,self.norm.iscale, \
                                self.data[self.itrain], \
                                self.data[self.itest],0)[:2]

    return(self.XTestNorm)
## ---------------------------------------------------------------
  def compute_kernel(self,xdatacls):

    if xdatacls.category in (0,3):  ## ranks
    ## output kernel
      ymax=self.ymax
      ymin=self.ymin
      ystep=self.ystep
      yinterval=np.arange(ymin,ymax+ystep,ystep)
      save_iscale=self.norm.iscale
      self.norm.iscale=-1
      self.K=mvm_kernel_eval.mvm_kernel(yinterval.T,None,self.kernel_params, \
                                        self.norm)
      self.norm.iscale=save_iscale
    elif xdatacls.category==1:  ## categories
      nmax=xdatacls.categorymax
      self.K=float(nmax)/(nmax-1)*np.eye(nmax) \
                   +np.zeros((nmax,nmax))-float(1)/(nmax-1)
      self.Y0Tetra=np.zeros((nmax,nmax))-1.0/np.sqrt(nmax*(nmax-1))
      self.Y0Tetra+=np.eye(nmax)*float(nmax)/np.sqrt(nmax*(nmax-1))
    elif xdatacls.category==2:  ## \{0,1,2,3\}^n kernel
      ## feature vectors are \{0,1,2,3\}^n
      ## the kernel indexed by numbers 
      ##    via transforming into number of base len(valrange) where the
      ##    first component has has the highest position
      self.K=np.copy(self.Kpre)
      d1=np.diag(self.K)
      d2=d1
      self.K=mvm_kernel_eval.kernel_nlr(self.K,d1,d2,self.kernel_params)

    return

## ---------------------------------------------------------------
  def compute_prekernel(self,xdatacls):
    
    self.Kpre=mvm_kernel_eval.kernel_multiclass_vector(self.valrange, \
                                                       self.ndim,
                                                       self.classweight)

## ---------------------------------------------------------------
  def get_kernel(self,itrain,itest,ioutput=0,itraintest=0,itraindata=1):

    return(self.K,self.d1,self.d2)

## ---------------------------------------------------------------
  def copy(self,data=None):
    
    new_obj=cls_feature(self.ifeature)
    new_obj.title=self.title
    new_obj.kernel_params=cls_kernel_params()
    new_obj.kernel_params.kernel_type=self.kernel_params.kernel_type
    new_obj.kernel_params.ipar1=self.kernel_params.ipar1
    new_obj.kernel_params.ipar2=self.kernel_params.ipar2
    if self.prekernel_params is not None:
      new_obj.prekernel_params=self.prekernel_params
    new_obj.crossval=self.crossval
    new_obj.norm=self.norm
    new_obj.ndim=self.ndim
    new_obj.valrange=self.valrange
    new_obj.classweight=self.classweight

    new_obj.ymax=self.ymax
    new_obj.ymin=self.ymin
    new_obj.ystep=self.ystep
    new_obj.yrange=self.yrange
    
    
    return(new_obj)
  ## #####################################################3 
