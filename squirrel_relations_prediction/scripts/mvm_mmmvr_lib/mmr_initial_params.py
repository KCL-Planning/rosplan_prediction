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
## ############################################################
class cls_initial_params:

  def __init__(self):
    """
    parameters for output kernel - ykernel, and input kernels xkernel,
               where
               ykernel['kernel']  xkernel['kernel']            
               ykernel['norm']  xkernel['norm']            
               ykernel['cross']  xkernel['corss']
               refer to a dictionary indexed by eah of the kernels

    Parameters
    kernel parameters:  kernel_type =0 linear
                                    =1 polynomial
                                    =2 sigmoid
                                    =3 Gaussian
                        ipar1, ipar2   kernel parameters, see in
                               mvm_kernel_eval modul in kernel_nlr function
    normlization parameters:
                        ilocal : centralization
                        iscale : scaling parameters,
                                 see in mmr_normalization_new
    cross(validation):
                        validation for ipar1:
                        range (par1min,par1max) in steps par1step
                        validation for ipar2:
                        range (par2min,par2max) in steps par2step
    """


    self.xparams={}
    self.xparams['kernel']={}
    ## self.xparams['kernel'][0]={ 'kernel_type' : 3, 'ipar1' : 1.71,  \
    ##                             'ipar2' : 0.2 }
    ## self.xparams['kernel'][1]={ 'kernel_type' : 3, 'ipar1' : 4.65,  \
    ##                             'ipar2' : 0.0 }
    ## self.xparams['kernel'][2]={ 'kernel_type' : 3, 'ipar1' : 5, \
    ##                             'ipar2' : 0.1 }
    ## self.xparams['kernel'][3]={ 'kernel_type' : 3, 'ipar1' : 4.9, \
    ##                             'ipar2' : 0.1 }
    ## self.xparams['kernel'][4]={ 'kernel_type' : 3, 'ipar1' : 5.0, \
    ##                             'ipar2' : 0.1 }

    self.xparams['kernel'][0]={ 'kernel_type' : 3, 'ipar1' : 0.166,  \
                                'ipar2' : 0.2 }
    self.xparams['kernel'][1]={ 'kernel_type' : 3, 'ipar1' : 4.65,  \
                                'ipar2' : 0.0 }
    self.xparams['kernel'][2]={ 'kernel_type' : 3, 'ipar1' : 5, \
                                'ipar2' : 0.1 }
    self.xparams['kernel'][3]={ 'kernel_type' : 3, 'ipar1' : 4.9, \
                                'ipar2' : 0.1 }
    self.xparams['kernel'][4]={ 'kernel_type' : 3, 'ipar1' : 5.0, \
                                'ipar2' : 0.1 }
    
    self.xparams['kernel'][5]={ 'kernel_type' : 0, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][6]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][7]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][8]={ 'kernel_type' : 0, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][9]={ 'kernel_type' : 0, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][10]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][11]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][12]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][13]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][14]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][15]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }
    self.xparams['kernel'][16]={ 'kernel_type' : 3, 'ipar1' : 0.2, 'ipar2' : 0 }


    self.xparams['norm']={}    
    self.xparams['norm'][0]={ 'ilocal' : 0, 'iscale' : 0}
    self.xparams['norm'][1]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][2]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][3]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][4]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][5]={ 'ilocal' : -1, 'iscale' : -1}
    self.xparams['norm'][6]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][7]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][8]={ 'ilocal' : -1, 'iscale' : -1}
    self.xparams['norm'][9]={ 'ilocal' : -1, 'iscale' : -1}
    self.xparams['norm'][10]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][11]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][12]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][13]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][14]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][15]={ 'ilocal' : 0, 'iscale' : -1}
    self.xparams['norm'][16]={ 'ilocal' : 0, 'iscale' : -1}

    self.xparams['cross']={}
    self.xparams['cross'][0]={ 'par1min' : 0.1 , 'par1max' : 10,  \
                               'par1step' : 1, \
                               'par2min' : 0 , 'par2max' : 0, \
                               'par2step' : 0.1, \
                               'nrange': 10   }
    self.xparams['cross'][1]={ 'par1min' : 4.4 , 'par1max' : 5, \
                               'par1step' : 0.5, \
                               'par2min' : 0.2 , 'par2max' : 0.2, \
                               'par2step' : 0.1, \
                               'nrange': 10   }
    self.xparams['cross'][2]={ 'par1min' : 4.9 , 'par1max' : 5.1, \
                               'par1step' : 0.5, \
                               'par2min' : 0.2 , 'par2max' : 0.2, \
                               'par2step' : 0.1, \
                               'nrange': 10   }
    self.xparams['cross'][3]={ 'par1min' : 4 , 'par1max' : 6, \
                               'par1step' : 1.0, \
                               'par2min' : 0.2 , 'par2max' : 0.2, \
                               'par2step' : 0.1, \
                               'nrange': 10   }
    self.xparams['cross'][4]={ 'par1min' : 4 , 'par1max' : 6, \
                               'par1step' : 1.0, \
                               'par2min' : 0.2 , 'par2max' : 0.2, \
                               'par2step' : 0.1, \
                               'nrange': 10   }
    ## external(default) kernel 
    self.yparams={}
    self.yparams['kernel']={}
    self.yparams['kernel'][0]={ 'kernel_type' : 0, 'ipar1' : 0.3, 'ipar2' : 0 }

    self.yparams['norm']={}
    self.yparams['norm'][0]={ 'ilocal' : -1, 'iscale' : -1}

    self.yparams['cross']={}
    self.yparams['cross'][0]={ 'par1min' : 1.0 , 'par1max' : 1.0,  \
                               'par1step' : 1, \
                               'par2min' : 0.1 , 'par2max' : 1.0, \
                               'par2step' : 0.01, \
                               'nrange': 10   }
    ## internal kernel
    self.yinparams={}
    self.yinparams['kernel']={}
    self.yinparams['kernel'][0]={ 'kernel_type' : 0, 'ipar1' : 0, 'ipar2' : 0 }

    self.yinparams['norm']={}
    self.yinparams['norm'][0]={ 'ilocal' : 2, 'iscale' : 0}

    self.yinparams['cross']={}
    self.yinparams['cross'][0]={ 'par1min' : 1.0 , 'par1max' : 5.0,  \
                               'par1step' : 1, \
                               'par2min' : 0.00 , 'par2max' : 0.00, \
                               'par2step' : 0.02, \
                               'nrange': 10   }
    

    ## internal kernel
    self.xinparams={}
    self.xinparams['kernel']={}
    self.xinparams['kernel'][0]={ 'kernel_type' : 0, 'ipar1' : 0, 'ipar2' : 0 }

    self.xinparams['norm']={}
    self.xinparams['norm'][0]={ 'ilocal' : 2, 'iscale' : -1}

    self.xinparams['cross']={}
    self.xinparams['cross'][0]={ 'par1min' : 1.0 , 'par1max' : 5.0,  \
                               'par1step' : 1, \
                               'par2min' : 0.00 , 'par2max' : 0.00, \
                               'par2step' : 0.02, \
                               'nrange': 10   }

  ## ---------------------------------------------------
  def get_xparams(self,scategory,iindex):

    if iindex in self.xparams[scategory]:
      return(self.xparams[scategory][iindex])
    else:
      return(self.xparams[scategory][0])
  
  ## ---------------------------------------------------
  def get_yparams(self,scategory,iindex):

    if iindex in self.yparams[scategory]:
      return(self.yparams[scategory][iindex])
    else:
      return(self.yparams[scategory][0])
  
  ## ---------------------------------------------------
  def get_yinparams(self,scategory,iindex):

    if iindex in self.yinparams[scategory]:
      return(self.yinparams[scategory][iindex])
    else:
      return(self.yinparams[scategory][0])

  ## ---------------------------------------------------
  def get_xinparams(self,scategory,iindex):

    if iindex in self.xinparams[scategory]:
      return(self.xinparams[scategory][iindex])
    else:
      return(self.xinparams[scategory][0])

