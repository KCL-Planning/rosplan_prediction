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
## import time
import numpy as np
## ##########################################
## from mmr_normalization_new import mmr_normalization
## from mmr_kernel_eval import kernel_eval_kernel
## import mmr_subspace
## import objedge_prekernel
## #####################
def mmr_kernel(cMMR,itrain,itest,ioutput=0,itraintest=0,itraindata=0,itensor=0):

  mtra=len(itrain)
  mtes=len(itest)

  d1=None
  d2=None
  if ioutput==0:
    ## input training kernel
    nkernel=len(cMMR.XKernel)
    KK=cMMR.XKernel[0].get_kernel(itrain,itest,itraintest=itraintest, \
                                  itraindata=itraindata)[0]
    for ikernel in range(1,nkernel):
      rkernel=cMMR.XKernel[ikernel]
      if itensor==0:
        KK+=rkernel.get_kernel(itrain,itest,itraintest=itraintest, \
                           itraindata=itraindata)[0]
      else:
        KK*=rkernel.get_kernel(itrain,itest,itraintest=itraintest, \
                           itraindata=itraindata)[0]

    (mtra,mtes)=KK.shape
    return(KK+cMMR.xbias*np.ones((mtra,mtes)),d1,d2)
  elif ioutput==1:
  ## output kernel
    rkernel=cMMR.YKernel
    (KK,d1,d2)=rkernel.get_kernel(itrain,itest,ioutput=ioutput, \
                                  itraintest=itraintest,itraindata=itraindata)
    
  return(KK,d1,d2)
