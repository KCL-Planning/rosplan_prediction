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

## import numpy as np
## ###########################################################
from mvm_mmmvr_lib.mmr_base_classes import cls_crossval, cls_kernel_params, cls_norm
from mvm_mmmvr_lib.mmr_multic_label import mmr_multic_label
from mvm_mmmvr_lib.mmr_normalization_new import mmr_normalization
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
    self.d1=None       ## norm of left factor of the kernel
    self.d2=None      ## norm of right factor of the kernel

    ## self.ilocal=2
    ## self.iscale=0

    self.norm=cls_norm()
    self.crossval=cls_crossval()
    self.kernel_params=cls_kernel_params()
    self.prekernel_params=None

    self.title='mvm_x'
    self.xbias=0.0

    return
## -------------------------------------------------------------
  def load_data(self,dataraw):

    self.dataraw=dataraw
    self.mdata=len(self.dataraw)
    if self.icategory==0:
      self.data=self.dataraw
    else:
      self.data=mmr_multic_label(self.cat2vector,self.xrawdata,None, \
                                    self.ncategory,None)
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
    """
    the relation kernel, the first one, is computed on the relation data
    the other kernels, if they are given, are loaded 
    """

    if self.ifeature==0:
      if xdatacls.category==0 or xdatacls.category==3:
        ## self.K=mvm_kernel_eval.mvm_kernel_sparse(xdatacls,isymmetric, \
        ##                                          self.kernel_params, \
        ##                                          self.norm)
        isymmetric=1  
        self.K=mvm_kernel_eval.mvm_kernel_sparse(xdatacls,isymmetric, \
                                                 self.kernel_params, \
                                                 self.norm)
      elif xdatacls.category==1:
        self.K=mvm_kernel_eval.kernel_category_2d(xdatacls, \
                                                  self.kernel_params, \
                                                  self.norm)
      elif xdatacls.category==2:
        self.K=mvm_kernel_eval.kernel_categoryvec_2d(xdatacls, \
                                                  self.kernel_params, \
                                                  self.norm)
    elif self.ifeature==1:
      d1=np.diag(self.K)
      d2=d1
      (self.K,d1,d2)=mvm_kernel_eval.kernel_centralize_normalize(K, \
                                                      d1,d2,self.norm)[0]
      if self.kernel_params.kernel_type>0:
        self.K=mvm_kernel_eval.kernel_nlr(self.K,d1,d2,self.kernel_params, \
                                          isymmetric=1)
      
    self.K+=self.xbias
    
    return
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
    new_obj.xbias=self.xbias
    new_obj.K=self.K
        
    return(new_obj)
## #####################################################3 
