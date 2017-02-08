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

## import sys
## import numpy as np
## ###########################################################
## class definitions
## ## global parameters
class cls_empty_class:
  pass

## ###################################################
class cls_norm:

  def __init__(self):

    self.ilocal=-1    ## no centralization or(and) localization
    self.iscale=-1    ## no normalization or(and) scaling

## ------------------------------------------------
  def set(self,dparams):

    if 'ilocal' in dparams:
      self.ilocal=dparams['ilocal']
    if 'iscale' in dparams:
      self.iscale=dparams['iscale']

## ------------------------------------------------
  def get(self):

    dparams={ 'ilocal' : self.ilocal, \
              'iscale' : self.iscale}

    return(dparams)
  
    
## ##################################################
class cls_kernel_params:

  def __init__(self):
    
    self.kernel_type=0   ## external kernel
    self.ipar1=1
    self.ipar2=0

## ------------------------------------------------
  def set(self,dparams):

    if 'kernel_type' in dparams:
      self.kernel_type=dparams['kernel_type']
    if 'ipar1' in dparams:
      self.ipar1=dparams['ipar1']
    if 'ipar2' in dparams:
      self.ipar2=dparams['ipar2']
    
## ------------------------------------------------
  def get(self):

    dparams={ 'kernel_type' : self.kernel_type, \
              'ipar1' : self.ipar1, \
              'ipar2' : self.ipar2}

    return(dparams)
  
## ##################################################
class cls_crossval:

  def __init__(self):
    
    self.par1min=1
    self.par1max=1
    self.par2min=0
    self.par2max=0
    self.par1step=1 
    self.par2step=1 
    self.nrange=1

## ------------------------------------------
  def set(self,dcross):

    if 'par1min' in dcross:
      self.par1min=dcross['par1min']
    if 'par1max' in dcross:
      self.par1max=dcross['par1max']
    if 'par2min' in dcross:
      self.par2min=dcross['par2min']
    if 'par2max' in dcross:
      self.par2max=dcross['par2max']
    if 'par1step' in dcross:
      self.par1step=dcross['par1step']
    if 'par2step' in dcross:
      self.par2step=dcross['par2step']
    if 'nrange' in dcross:
      self.nrange=dcross['nrange']

    if self.par1max<self.par1min:
      self.par1max=self.par1min
    if self.par2max<self.par2min:
      self.par2max=self.par2min
## ----------------------------------------------------
  def get(self):

    dcross={ 'par1min': self.par1min, \
            'par1max': self.par1max, \
            'par2min': self.par2min, \
            'par2max': self.par2max, \
            'par1step': self.par1step, \
            'par2step': self.par2step, \
            'nrange': self.nrange }

    return(dcross)
    
## ##################################################
class cls_data:

  def __init__(self,ninputview):

    self.ninputview=ninputview
    self.mdata=0                       ## number of raw sample items
    self.itrain=None
    self.itest=None
    self.mtrain=0
    self.mtest=0
    
## ##################################################
class cls_perceptron_param:

  def __init__(self,margin=1.0,stepsize=0.01,niter=1):
    self.margin=margin
    self.stepsize=stepsize
    self.niter=niter


## #########################################################
class cls_penalty:

  def __init__(self,C=1.0,D=0.0):

    self.c=C
    self.d=D

    self.crossval=cls_crossval()

## ---------------------------------------------------------------

  def set_crossval(self):

    dcross={ 'par1min' : 1 , 'par1max' : 1,  \
                               'par1step' : 0.1, \
                               'par2min' : 0.0 , 'par2max' : 0.0, \
                               'par2step' : 0.1, \
                               'nrange': 0 }
    self.crossval.set(dcross)
  
## #########################################################
class cls_dual:

  def __init__(self,alpha=None,bias=None):

    self.alpha=alpha             ## optimal dual parameters
    self.bias=bias              ## estimated bias
    self.W=None

## #########################################################
class cls_predict:

  def __init__(self):

    self.ZTest=None       ## prediction matrix, columns - test items,
                          ## rows - categories
                          ## $ dot(Y0,Zw) $
    self.zPremax=None     ## maximum prediction value for each test item
                          ## $ ZTest.max(0)$
    self.iPredCat=None    ## predicted category for each test item
                          ## $ ZTest.argmax(0)$
    self.Zw=None          ## raw prediction transpose of
                          ## $\sum \alpha_i y_i \langel x_i,x \rangle$ 

    self.zPred=None       ## matrix of size ntest*nydim
                          ## in which each row contains the predicted \{-1,+1\}                           ## vector
## #########################################################
class cls_evaluation:
    
  def __init__(self):

    self.accuracy=0.0
    self.precision=0.0
    self.recall=0.0
    self.f1=0.0
    self.confusion=None
    self.classconfusion=None
    self.xresults=None
    self.rmse=None
    self.mae=None

## #########################################################
class cls_mvm_view:

  def __init_(self):

    self.xdata_rel=None
    self.rel_dim=None
    self.xdata_tra=None
    self.xdata_tes=None
    self.xranges_rel=None
    self.xranges_rel_test=None
    self.nrow=None
    self.ncol=None

    self.KXvar=None
    self.glm_model=None
    self.largest_class=None

    self.category=0     ## =0 rank cells =1 category cells =2 {-1,0,+1}^n
    self.categorymax=0
    self.ndata=0

    return
    
## #########################################################
