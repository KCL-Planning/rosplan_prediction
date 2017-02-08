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
import sys
import numpy as np

## ####################
from mvm_mmmvr_lib.mmr_base_classes import cls_empty_class
import mvm_mmmvr_lib.mvm_mvm_cls as mvm_mvm_cls
from mvm_mmmvr_lib.mvm_prepare import mvm_ranges, mvm_ygrid, mvm_largest_category
from mvm_mmmvr_lib.mvm_eval import mvm_eval
import mvm_mmmvr_lib.mvm_glmmodel_cls as mvm_glmmodel_cls
## ####################
class cls_mvm_validation:

  def __init__(self):

    self.vnfold=2    ## number folds(>1) in validation
    self.ivalid=0    ## =0 no validation =1 validation
    self.validation_rkernel='mvm_x'  ## reference kernel to be validated
    self.report=0
    self.best_param=None

  ## -------------------------------
  def mvm_validation(self,xdatacls):
    """

    Input:
    xdatacls      data class
    params        global parameters

    Output:
    best_param    the best kernel parameters found by cross validation
                  on the split training
    """

    if self.validation_rkernel in xdatacls.dkernels:
      kernbest=xdatacls.dkernels[self.validation_rkernel].kernel_params 
    else:
      kernbest=xdatacls.XKernel[0].kernel_params

    if self.ivalid==1:
      best_param=self.mvm_validation_body(xdatacls)
    else:
      best_param=cls_empty_class()
      best_param.c=xdatacls.penalty.c
      best_param.d=xdatacls.penalty.d
      best_param.par1=kernbest.ipar1
      best_param.par2=kernbest.ipar2

    xdatacls.penalty.c=best_param.c
    xdatacls.penalty.d=best_param.d
    kernbest.ipar1=best_param.par1
    kernbest.ipar2=best_param.par2

    self.best_param=best_param

    return(best_param)

  ## -----------------------------------------------
  def mvm_validation_body(self,xdatacls):
    """

    Input:
    xdatacls      data class
    params        global parameters

    Output:
    best_param    the best kernel parameters found by cross validation
                  on the split training
    """

    nrow=xdatacls.nrow

    ## construct the data object out of the training items
    xdatacls_val=mvm_mvm_cls.cls_mvm()
    xdatacls.copy(xdatacls_val)

    xparam=cls_empty_class()

    best_param=cls_empty_class()
    best_param.c=1
    best_param.d=0
    best_param.par1=0
    best_param.par2=0

    if self.validation_rkernel in xdatacls_val.dkernels:
      rkernel=xdatacls_val.dkernels[self.validation_rkernel]
    else:
      rkernel=xdatacls_val.XKernel[0]

    kernel_type=rkernel.kernel_params.kernel_type
    kinput=rkernel.crossval

    if kernel_type==0:
      ip1min=0
      ip1max=0
      ip2min=0
      ip2max=0
      ip1step=1
      ip2step=1
    elif kernel_type in (1,2):
      ip1min=kinput.par1min
      ip1max=kinput.par1max
      ip2min=kinput.par2min
      ip2max=kinput.par2max
      ip1step=kinput.par1step
      ip2step=kinput.par2step
    elif kernel_type in (3,31,32,41,53,5):
      if kinput.nrange>1:
        if kinput.par1max>kinput.par1min:
          dpar= np.power(kinput.par1max/kinput.par1min,1/(kinput.nrange-1))
          ip1max=kinput.nrange
        else:
          dpar=1.0
          ip1max=1.0
      else:
        ip1max=1.0
        dpar=1.0

      ip1min=1
      ip2min=kinput.par2min
      ip2max=kinput.par2max
      ip1step=1
      ip2step=kinput.par2step
    else: 
      ip1min=1
      ip1max=1
      ip2min=1
      ip2max=1
      ip1step=1
      ip2step=1

  #  vnfold=4 # number of validation folds
    mdata=xdatacls_val.xdata_rel[0].shape[0]
    vnfold=self.vnfold # number of validation folds
    vxsel=np.floor(np.random.rand(mdata)*vnfold)
    vxsel=vxsel-(vxsel==vnfold)
  ##  vpredtr=np.zeros(vnfold) # valid
    vpred=np.zeros(vnfold) # train

    print('C,D,par1,par2,traning accuracy,validation test accuracy')    

    # scanning the parameter space

    if xdatacls_val.ieval_type in (0,10,11):
      xxmax=-np.inf
    else:
      xxmax=np.inf

    penalty=xdatacls_val.penalty.crossval
    crange=np.arange(penalty.par1min,penalty.par1max+penalty.par1step/2, \
                     penalty.par1step)
    drange=np.arange(penalty.par2min,penalty.par2max+penalty.par2step/2, \
                     penalty.par2step)

    p1range=np.arange(ip1min,ip1max+ip1step/2,ip1step)
    p2range=np.arange(ip2min,ip2max+ip2step/2,ip2step)

    for iC in crange:
      for iD in drange:
        for ip1 in p1range:
          for ip2 in p2range:
            if kernel_type in (3,31,32,41,53,5): 
              dpar1=kinput.par1min*dpar**(ip1-1)
              dpar2=ip2
            else:
              dpar1=ip1
              dpar2=ip2

            xdatacls_val.penalty.c=iC;
            xdatacls_val.d=iD;
            rkernel.kernel_params.ipar1=dpar1;
            rkernel.kernel_params.ipar2=dpar2;

            for vifold in range(vnfold):

              xdatacls_val.split_train_test(vxsel,vifold)
              xdatacls_val.mvm_datasplit()        
              xdatacls_val.xranges_rel=mvm_ranges(xdatacls_val.xdata_tra, \
                                               xdatacls_val.nrow)
              xdatacls_val.xranges_rel_test=mvm_ranges(xdatacls_val.xdata_tes, \
                                               xdatacls_val.nrow)
              if xdatacls.category==0 or xdatacls.category==3:
                ## pass
                ## self.glm_model=mvm_glmmodel_cls.cls_glmmodel()
                ## xdatacls_val.glm_model.mvm_glm_orig(xdatacls_val)
                ## =================================
                ## data transformation
                xdatacls_val.glm_model=mvm_glmmodel_cls.cls_glmmodel()
                xdatacls_val.glm_model.rfunc=xdatacls.glm_model.rfunc
                xdatacls_val.glm_model.mvm_glm_link(xdatacls_val)
                mvm_ygrid(xdatacls_val)
              else:
                mvm_largest_category(xdatacls_val)

              if self.report==1:
                print('validation training')
              xdatacls_val.mvm_train()

  # validation test
              if self.report==1:
                print('validation test on validation test')
              cPredict=xdatacls_val.mvm_test() 

  # counts the proportion the ones predicted correctly    
  # ##############################################
              cEval=mvm_eval(xdatacls_val.ieval_type,nrow,xdatacls_val, \
                                 cPredict.Zrow)[0]
              if xdatacls_val.ieval_type in (0,10,11):
                if xdatacls_val.ibinary==0:
                  vpred[vifold]=cEval.accuracy
                elif xdatacls_val.ibinary==1:
                  vpred[vifold]=cEval.f1
              else:
                vpred[vifold]=cEval.deval

            print('%9.5g'%iC,'%9.5g'%iD,'%9.5g'%dpar1,'%9.5g'%dpar2, \
                  '%9.5g'%(np.mean(vpred)))
            
            ## print(iC,iD,dpar1,dpar2,np.mean(vpred))
  # searching for the best configuration in validation
            mvpred=np.mean(vpred)
            sys.stdout.flush()

            if xdatacls_val.ieval_type in (0,10,11):
              if mvpred>xxmax:
                xxmax=mvpred
                xparam.c=iC
                xparam.d=iD
                xparam.par1=dpar1
                xparam.par2=dpar2
                print('The best:',xxmax)
            else:
              if mvpred<xxmax:
                xxmax=mvpred
                xparam.c=iC
                xparam.d=iD
                xparam.par1=dpar1
                xparam.par2=dpar2
                print('The best:',xxmax)

            sys.stdout.flush()

    best_param=xparam

    return(best_param)

## ##############################################################3
