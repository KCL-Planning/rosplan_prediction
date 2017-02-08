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
import time
import numpy as np
## import numexpr
import scipy.sparse as spspar
## ####################
## import mvm_classes
#####################################################
def mvm_test_orig(xdatacls,xalpha):
  """
  Computes the predicted rank for the (row,col) pairs occuring in the test
  Inputs: 
     xdatacls   reference to the data classs 
     xalpha     vector of optimal values of dual variables computed
                in the training  
  Outputs:
     Zrow       dictionary indexed by the rows of the relation table
                each item of the dictionary containts 3 elements:
                    the prediction of each row elements,
                    the raw prediction,
                    the corresponding confidence 
  """
  ## ################################################3
  KXfull=xdatacls.KX
  KYfull=xdatacls.KY
  ## KXvar=xdatacls.KXvar    
  ycategory=xdatacls.category   ## are cells categorical variables?
  Y0Tetra=xdatacls.YKernel.Y0Tetra

  xranges_tra=xdatacls.xranges_rel
  xdata_tra=xdatacls.xdata_tra
  if xdatacls.testontrain==0:  
    xranges_tes=xdatacls.xranges_rel_test
    xdata_tes=xdatacls.xdata_tes
  else:
    xranges_tes=xdatacls.xranges_rel
    xdata_tes=xdatacls.xdata_tra
  
  tdim2=xdata_tra[2].shape
  if len(tdim2)==1:
    nydim=1
  else:
    nydim=tdim2[1]
  
  ## glm_model=xdatacls.glm_model

  if ycategory==1:
    nyrange0=xdatacls.categorymax
  else:
    nyrange0=KYfull.shape[0]

  if ycategory in (1,2):
    row_max_category=xdatacls.largest_class.row_max_category
    ## max_category=xdatacls.largest_class.max_category
  
  nrow=xdatacls.nrow

  # collection of the prediction for each row
  Zrow={}  

  if ycategory==0 or ycategory==3:
    col_mean=xdatacls.glm_model.col_mean      # row averages matrix
    row_mean=xdatacls.glm_model.row_mean      # col averages matrix 
    total_mean=xdatacls.glm_model.total_mean  # total average vector
    product_correction=xdatacls.glm_model.product_correction

    ymax=xdatacls.YKernel.ymax    # parameters to rescale the rank
    ymin=xdatacls.YKernel.ymin
    ystep=xdatacls.YKernel.ystep
  
  nhits=np.zeros(2)

  tfull=0
  tpred=0

  tfull0=time.time()
  
  for irow in range(nrow):              # for each row

    tfull1=time.time()
    tfull+=tfull1-tfull0
    tfull0=tfull1
    if irow%100==0:
      ## print(irow,'%8.2f'%tfull,'%8.2f'%tpred)
      ## sys.stdout.flush()
      tfull=0
      tpred=0
      
    if xranges_tra[irow,1]>0:        # row has training items    
    ## if False:        # row has no training items    

      if xranges_tes[irow,1]>0:     # row has test items
        tpred0=time.time()

        (istart,nlength)=xranges_tra[irow]       
        # training cols seen by the row
        ixrange=xdata_tra[1][istart:istart+nlength]   
        # read the row specific output subkernel
        (istart_tes,nlength_tes)=xranges_tes[irow]
        nhits[0]+=nlength_tes
        ixrange_tes=xdata_tes[1][istart_tes:istart_tes+nlength_tes]
        # read the col specific input kernel
        KXm=KXfull[ixrange.reshape((nlength,1)),ixrange_tes]

        if ycategory==0 or ycategory==3:
          yinterval=np.arange(ymin,ymax,ystep)
          yinterval=np.round((yinterval-ymin)/ystep).astype(int)
          nyrange=len(yinterval)
          if nydim==1:
            iyrange=xdata_tra[2][istart:istart+nlength]   
            iyrange=np.round((iyrange-ymin)/ystep).astype(int) # into indeces
            KY=KYfull[iyrange.reshape((nlength,1)),yinterval]
            tfull0=time.time()
            A1=spspar.csc_matrix(KY.T*np.outer(np.ones(nyrange), \
                         xalpha[istart:istart+nlength]))
            Z=A1.dot(KXm)
            spre=Z.argmax(0)
            zpre=Z.max(0)
            
          else: ## vector valued case 
            tfull0=time.time()
            spre=np.zeros((nlength_tes,nydim))
            zpre=np.zeros((nlength_tes,nydim))
            for i in range(nydim):
              iyrange=xdata_tra[2][istart:istart+nlength][:,i]   
              iyrange=np.round((iyrange-ymin)/ystep).astype(int) # into indeces
              KY=KYfull[iyrange.reshape((nlength,1)),yinterval]
              A1=spspar.csc_matrix(KY.T*np.outer(np.ones(nyrange), \
                         xalpha[istart:istart+nlength]))
              Z=A1.dot(KXm)
              spre[:,i]=Z.argmax(0)
              zpre[:,i]=Z.max(0)
              
        elif ycategory==1:
          yinterval=np.arange(nyrange0)
          iyrange=xdata_tra[2][istart:istart+nlength]   
          nyrange=len(yinterval)
          YY=Y0Tetra[iyrange]
          tfull0=time.time()
          Z0=np.dot((YY.T*np.tile(xalpha[istart:istart+nlength], \
                                     (nyrange,1))),KXm)
          znorm=np.sqrt(np.sum(Z0**2,axis=0))
          znorm=znorm+(znorm==0)
          Z0=Z0/np.outer(np.ones(Y0Tetra.shape[1]),znorm)
          Z=np.dot(Y0Tetra,Z0)
          spre=Z.argmax(0)
          zpre=Z.max(0)
          
        elif ycategory==2:
          yinterval=np.arange(nyrange0)
          iyrange=xdata_tra[2][istart:istart+nlength]   
          nyrange=len(yinterval)
          KY=KYfull[iyrange.reshape((nlength,1)),yinterval]
          tfull0=time.time()
          Z=np.dot((KY.T*np.tile(xalpha[istart:istart+nlength], \
                                     (nyrange,1))),KXm)
          spre=Z.argmax(0)
          zpre=Z.max(0)
         
# compute prediction by the maximum margin principle
        tpred1=time.time()
        tpred+=tpred1-tpred0
        tpred0=tpred1
        
        ## for active learning
        if ycategory in (1,2):
          ## zmin=float(1)/(Y0Tetra.shape[1]-1)
          zconf=zpre
          zpre0=zpre
          zpre=spre
        elif ycategory==0 or ycategory==3:
          # compute the real rank-averages        
          zpre0=spre*ystep+ymin
          zconf=zpre0
          # mean value correction, add to averages to the prediction 
          for i in range(nlength_tes):
            icol=ixrange_tes[i]   # test col index
            zpre[i]=zpre0[i]+col_mean[irow]+row_mean[icol]-total_mean
          if xdatacls.glm_model.rfunc is not None:
            zpre=xdatacls.glm_model.rfunc.rfunc(zpre)
        elif ycategory==2:
          zconf=zpre
          zpre0=zpre
        
        Zrow[irow]=(zpre,zpre0,zconf)    # store the prediction for the row
    else:
# if there is no training item for the row then the averages give the
# prediction
      nlength_tes=xranges_tes[irow,1]
      if nlength_tes>0:
        nhits[1]+=nlength_tes
        if ycategory==0 or ycategory==3:
          (istart_tes,nlength_tes)=xranges_tes[irow]
          ixrange_tes=xdata_tes[1][istart_tes:istart_tes+nlength_tes]
          if nydim==1:
            zpre=np.zeros(nlength_tes)
          else:
            zpre=np.zeros((nlength_tes,nydim))
          zconf=zpre
          # mean value correction, add to averages to the prediction 
          for i in range(nlength_tes):
            icol=ixrange_tes[i]   # test col index
            zpre[i]=col_mean[irow]+row_mean[icol]-total_mean
          if xdatacls.glm_model.rfunc is not None:
            zpre=xdatacls.glm_model.rfunc.rfunc(zpre)
        elif ycategory==1:
          spre=np.zeros(nlength_tes)+row_max_category[irow]
          zpre=spre
          zconf=zpre
        elif ycategory==2:
          spre=np.zeros(nlength_tes)+row_max_category[irow]
          ## spre=np.zeros(nlength_tes)+max_category
          zpre=spre
          zconf=zpre

        if nydim==1:
          Zrow[irow]=(zpre,np.zeros(nlength_tes),zconf)
        else:
          Zrow[irow]=(zpre,np.zeros((nlength_tes,nydim)),zconf)
          

  ## print('nhits:',nhits)

  return(Zrow)

