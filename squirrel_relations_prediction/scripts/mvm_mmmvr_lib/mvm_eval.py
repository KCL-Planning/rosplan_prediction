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
import sys
######################
import numpy as np
import mvm_mmmvr_lib.mmr_base_classes as mmr_base_classes
import mvm_mmmvr_lib.mvm_prepare as mvm_prepare
## ####################
## import mvm_classes
## ####################
# ieval_type      =0 hamming
#                 =1 sqrt(L2)
#                 =2 L1            
def mvm_eval(ieval_type,nrow,xdatacls,ZrowT):
  """
  Compute the gloabal error measures of the predition
  
  Input:
  ieval_type        error measures =0 0/1 loss,=1 RMSE, =2 MAE error
  nrow         number of rows
  datacls           class of features kernels
  ZrowT        predicted values, list indexed by row index,
                    and each list element contains the prediction
                    of all column elements belonging to that row
  Output:
  deval             accuracy in the corresponding error measure
  """
  if xdatacls.testontrain==0:
    xranges_tes=xdatacls.xranges_rel_test
    xdata_tes=xdatacls.xdata_tes
  else:
    xranges_tes=xdatacls.xranges_rel
    xdata_tes=xdatacls.xdata_tra
    
  txdim=xdata_tes[2].shape
  if len(txdim)==1:
    nxdim=1
  else:
    nxdim=txdim[1]

  cEval=mmr_base_classes.cls_evaluation()
  icandidate_w=0
  icandidate_b=0
  
  if ieval_type==0:  ## 0/1 loss
    Y0=xdatacls.Y0
    ncategory=len(Y0)

    nall=0
    nright=0
    tp=0    ## true positive
    tn=0    ## true negative
    fp=0    ## false positive
    fn=0    ## false negative
    xworst=10**3
    xbest=-xworst

    xconfusion=np.zeros((ncategory+1,ncategory+1))
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        xobserved=xdata_tes[2][istart_tes:istart_tes+nlength_tes]
        xpredicted=ZrowT[irow][0]+0.0
        for i in range(nlength_tes):
          if xdatacls.category==0 or xdatacls.category==3:    ## rank
            ipredicted=Y0[np.abs(Y0-xpredicted[i]).argmin()]
          else:
            ipredicted=xpredicted[i]
          xconfusion[xobserved[i],ipredicted]+=1
          if xdatacls.ibinary==0:
            if ipredicted==xobserved[i]:
              nright+=1
          else:  ## Y0=[-1,+1]
            if ipredicted==1:
              if xobserved[i]==1:
                tp+=1
              else:
                fp+=1
            else:
              if xobserved[i]==1:
                fn+=1
              else:
                tn+=1
          
          xconfidence=ZrowT[irow][2][i]
          if xconfidence<xworst:
            xworst=xconfidence
            icandidate_w=istart_tes+i
          if xconfidence>xbest:
            xbest=xconfidence
            icandidate_b=istart_tes+i
            
        nall+=nlength_tes

    if nall==0:
      nall=1
    deval=float(nright)/nall
    if xdatacls.ibinary==0:
      cEval.accuracy=deval
    else:
      cEval.accuracy=(tp+tn)/nall
    cEval.xconfusion=xconfusion
    cEval.deval=cEval.accuracy
    
    if tp+fp>0:
      cEval.precision=float(tp)/(tp+fp)
    else:
      cEval.precision=0.0
    if tp+fn>0:
      cEval.recall=float(tp)/(tp+fn)
    else:
      cEval.recall=0.0

    if cEval.recall+cEval.precision>0:
      cEval.f1=2*cEval.precision*cEval.recall/(cEval.recall+cEval.precision)
    else:
      cEval.f1=0.0
      
    
  elif ieval_type==1:     # RMSE root mean square error
    nall=0
    nright=0
    xworst=10**6
    xbest=0
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        nright+=np.sum((ZrowT[irow][0] \
                        -xdata_tes[2][istart_tes:istart_tes+nlength_tes])**2)
        for i in range(nlength_tes):
          if nxdim==1:
            xconfidence=ZrowT[irow][2][i]**2  ## raw prediction
          else:
            xconfidence=np.mean(ZrowT[irow][2][i])**2  ## raw prediction
            
          if xconfidence<xworst:
            xworst=xconfidence
            icandidate_w=istart_tes+i
          if xconfidence>xbest:
            xbest=xconfidence
            icandidate_b=istart_tes+i
        nall+=nlength_tes*nxdim
        
    if nall==0:
      nall=1
    deval=np.sqrt(float(nright)/nall)
    cEval.rmse=deval
    cEval.deval=deval

  elif ieval_type==2:   # MAE mean absolute error
    nall=0
    nright=0
    xworst=10**6
    xbest=0
    lpredict=[]
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        ## nright+=np.sum(np.abs(np.exp(ZrowT[irow][0]) \
        ##            -np.exp(xdata_tes[2][istart_tes:istart_tes+nlength_tes])))
        nright+=np.sum(np.abs(ZrowT[irow][0] \
                   -xdata_tes[2][istart_tes:istart_tes+nlength_tes]))
        for i in range(nlength_tes):
          if nxdim==1:
            xconfidence=ZrowT[irow][2][i]**2  ## raw prediction
          else:
            xconfidence=np.mean(ZrowT[irow][2][i])**2  ## raw prediction
          if xconfidence<xworst:
            xworst=xconfidence
            icandidate_w=istart_tes+i
          if xconfidence>xbest:
            xbest=xconfidence
            icandidate_b=istart_tes+i
        nall+=nlength_tes*nxdim
        
    if nall==0:
      nall=1
    deval=float(nright)/nall
    cEval.mae=deval
    cEval.deval=deval
    cEval.accuracy=deval

  elif ieval_type==3:   # median absolute error
    nall=0
    nright=0
    xworst=10**3
    xbest=-xworst
    lpredict=[]
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        lpredict.extend(np.abs(ZrowT[irow][0] \
                       -xdata_tes[2][istart_tes:istart_tes+nlength_tes]))
        for i in range(nlength_tes):
          xconfidence=np.abs(ZrowT[irow][2][i]) ## raw prediction
          if xconfidence<xworst:
            xworst=xconfidence
            icandidate_w=istart_tes+i
          if xconfidence>xbest:
            xbest=xconfidence
            icandidate_b=istart_tes+i
        nall+=nlength_tes
        
    if nall==0:
      nall=1
    deval=np.median(np.array(lpredict))
    cEval.mae=deval
    cEval.deval=deval
    cEval.accuracy=deval
    # cEval.xpredict=np.array(lpredict)

  if ieval_type==10:  ## \{0,1,2,3\}^n
    nall=0
    nright=0
    tp=0    ## true positive
    tn=0    ## true negative
    fp=0    ## false positive
    fn=0    ## false negative
    xworst=10**3
    xbest=-xworst

    ndim=xdatacls.YKernel.ndim
    valrange=xdatacls.YKernel.valrange
    nval=max(valrange)+1
    tdim=[nval]*ndim
    xconfusion=np.zeros((ndim,nval,nval))
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        xobserved=xdata_tes[2][istart_tes:istart_tes+nlength_tes]
        xpredicted=ZrowT[irow][0].astype(int)
        ixobserved=np.unravel_index(xobserved,tdim)
        ixpredicted=np.unravel_index(xpredicted,tdim)
        for i in range(nlength_tes):
          for j in range(ndim):
            xconfusion[j,ixobserved[j][i],ixpredicted[j][i]]+=1
          ## !!!!!! should be changed 
          xconfidence=ZrowT[irow][2][i]
          if xconfidence<xworst:
            xworst=xconfidence
            icandidate_w=istart_tes+i
          if xconfidence>xbest:
            xbest=xconfidence
            icandidate_b=istart_tes+i
            
        nall+=nlength_tes

    cEval.accuracy=0
    cEval.xconfusion3=xconfusion

    ndim=xconfusion.shape[0]
    (accuracy_full,accuracy_no0)=confusion_toys(xconfusion)    
    cEval.accuracy=accuracy_no0[ndim]
    cEval.accuracy_full=accuracy_full
    cEval.accuracy_no0=accuracy_no0
    
    cEval.deval=cEval.accuracy  
    if tp+fp>0:
      cEval.precision=float(tp)/(tp+fp)
    else:
      cEval.precision=0.0
    if tp+fn>0:
      cEval.recall=float(tp)/(tp+fn)
    else:
      cEval.recall=0.0

    if cEval.recall+cEval.precision>0:
      cEval.f1=2*cEval.precision*cEval.recall/(cEval.recall+cEval.precision)
    else:
      cEval.f1=0.0

  ## sign comparison of the residues of test and prediction
  elif ieval_type==11:  ## 0/1 loss
    Y0=xdatacls.Y0
    ncategory=len(Y0)

    nall=0
    nright=0
    tp=0    ## true positive
    tn=0    ## true negative
    fp=0    ## false positive
    fn=0    ## false negative
    xworst=10**3
    xbest=-xworst

    xconfusion=np.zeros((ncategory+1,ncategory+1))
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        xobserved=xdata_tes[2][istart_tes:istart_tes+nlength_tes]
        ## xpredicted=ZrowT[irow][0]+0.0
        xobserved=np.sign(xobserved-ZrowT[irow][0]+ZrowT[irow][1])
        xpredicted=np.sign(ZrowT[irow][1]+0.0)
        
        for i in range(nlength_tes):
          if xdatacls.category==0 or xdatacls.category==3:    ## rank
            ipredicted=Y0[np.abs(Y0-xpredicted[i]).argmin()]
          else:
            ipredicted=xpredicted[i]
          ## we have values -1,0,+1 to make them be correct index 1 is added
          xconfusion[xobserved[i]+1,ipredicted+1]+=1
          if xdatacls.ibinary==0:
            if ipredicted==xobserved[i]:
              nright+=1
          else:  ## Y0=[-1,+1]
            if ipredicted==1:
              if xobserved[i]==1:
                tp+=1
              else:
                fp+=1
            else:
              if xobserved[i]==1:
                fn+=1
              else:
                tn+=1
          
          xconfidence=ZrowT[irow][2][i]
          if xconfidence<xworst:
            xworst=xconfidence
            icandidate_w=istart_tes+i
          if xconfidence>xbest:
            xbest=xconfidence
            icandidate_b=istart_tes+i
            
        nall+=nlength_tes

    if nall==0:
      nall=1
    deval=float(nright)/nall
    if xdatacls.ibinary==0:
      cEval.accuracy=deval
    else:
      cEval.accuracy=(tp+tn)/nall
    cEval.xconfusion=xconfusion
    cEval.deval=cEval.accuracy
    
    if tp+fp>0:
      cEval.precision=float(tp)/(tp+fp)
    else:
      cEval.precision=0.0
    if tp+fn>0:
      cEval.recall=float(tp)/(tp+fn)
    else:
      cEval.recall=0.0

    if cEval.recall+cEval.precision>0:
      cEval.f1=2*cEval.precision*cEval.recall/(cEval.recall+cEval.precision)
    else:
      cEval.f1=0.0
      
    
      
  return(cEval,icandidate_w,icandidate_b)

## #############################################################3
def confusion_toys(xconfusion):

  ndim=xconfusion.shape[0]
  accuracy_full=np.zeros(ndim+1)
  accuracy_no0=np.zeros(ndim+1)
  xnsample_full=np.zeros(ndim+1)
  xnsample_no0=np.zeros(ndim+1)
  ## xnsample_0=np.zeros(ndim+1)
 
  for i in range(ndim):
    xslice=xconfusion[i]
    xnsample_full[i]=np.sum(xslice)
    xnsample_no0[i]=np.sum(xslice[1:,1:])
    ## xnsample_0=np.sum(xnsample[0,1:])
    xdiag=np.diag(xslice)
    accuracy_full[i]=np.sum(xdiag)
    accuracy_no0[i]=np.sum(xdiag[1:])

  accuracy_full[ndim]=np.sum(accuracy_full[:ndim])
  accuracy_full[ndim]/=np.sum(xnsample_full[:ndim])
   
  accuracy_no0[ndim]=np.sum(accuracy_no0[:ndim])
  accuracy_no0[ndim]/=np.sum(xnsample_no0[:ndim])

  for i in range(ndim):
    accuracy_full[i]/=xnsample_full[i]
    if xnsample_no0[i]>0:
      accuracy_no0[i]/=xnsample_no0[i]

  return(accuracy_full,accuracy_no0)
## #############################################################3
def confusion_latex(xconfusion,lfiles):

  ##averaging on repeations and folds
  xconfusion_mean=np.mean(np.mean(xconfusion,axis=0),axis=0)

  accuracy_full,accuracy_no0=confusion_toys(xconfusion_mean)
  ndim=xconfusion_mean.shape[0]

  ## xsum=np.sum(xconfusion_mean)
  for i in range(ndim):
    xtable=xconfusion_mean[i]
    xsumt=np.sum(xtable)
    (mt,nt)=xtable.shape
    for jr in range(mt):
      for jc in range(nt):
        v=100*xtable[jr,jc]/xsumt
        #print('%6.2f'%v,sep=' ',end='')
      print()
    print()
      

  print('\\begin{tabular}{l|rr}')
  print('Table & Full & Only known \\\\ \\hline')
  for i in range(ndim):
    print(lfiles[i][0]+' & '+'%5.3f'%accuracy_full[i]+' & ' \
          +'%5.3f'%accuracy_no0[i]+' \\\\') 
  print('\\hline')
  print('Total'+' & '+'%5.3f'%accuracy_full[ndim]+' & ' \
          +'%5.3f'%accuracy_no0[ndim]+' \\\\') 
  print('\\end{tabular}')

  ## print()
  ## print('\\begin{tabular}{l|rrr}')
  ## print('Table & -1 & 0 & +1 \\\\ \\hline')
  ## for i in range(ndim):
  ##   pred0=xconfusion_mean[i,0,1:ndim]
  ##   pred0=100*pred0/np.sum(pred0)
    
  ##   print(lfiles[i][0]+' & '+'%5.3f'%pred0[0] \
  ##         +' & ' +'%5.3f'%pred0[1] \
  ##         +' & ' +'%5.3f'%pred0[2]+' \\\\') 
  ## print('\\end{tabular}')

  print()
 
  
  sys.stdout.flush()
  
## #############################################################3
def makearray(xdatacls,ZrowT):

  if xdatacls.testontrain==0:
    xranges_tes=xdatacls.xranges_rel_test
    xdata_tes=xdatacls.xdata_tes
  else:
    xranges_tes=xdatacls.xranges_rel
    xdata_tes=xdatacls.xdata_tra
    
  nrow=xdatacls.nrow
  ncol=xdatacls.ncol

  txdim=xdata_tes[2].shape
  if len(txdim)==1:
    nxdim=1
  else:
    nxdim=txdim[1]

  ytest=np.zeros((nrow,ncol,nxdim))
  ypred=np.zeros((nrow,ncol,nxdim))
  ypred0=np.zeros((nrow,ncol,nxdim))
  
  for irow in range(nrow):
    if xranges_tes[irow,1]>0:
      istart_tes=xranges_tes[irow,0]
      nlength_tes=xranges_tes[irow,1]
      for i in range(nlength_tes):
        ii=istart_tes+i
        i1=xdata_tes[0][ii]
        i2=xdata_tes[1][ii]
        ytest[i1,i2]=xdata_tes[2][ii]
        ypred[i1,i2]=ZrowT[irow][0][i]
        ypred0[i1,i2]=ZrowT[irow][1][i]

  ytest=np.squeeze(ytest)
  ypred=np.squeeze(ypred)
  ypred0=np.squeeze(ypred0)
  
  return(ytest,ypred,ypred0)
## ###############################################################
def full_test_orig(xdatacls):

  col_mean=xdatacls.glm_model.col_mean      # row averages matrix
  row_mean=xdatacls.glm_model.row_mean      # col averages matrix 
  total_mean=xdatacls.glm_model.total_mean  # total average vector

  nrow=xdatacls.nrow
  ncol=xdatacls.ncol

  ## xranges_full=mvm_prepare.mvm_ranges(xdatacls.xdata_rel,xdatacls.nrow)

  ndata=nrow*ncol
  npart=3
  xdata_tes=[None]*3
  xdata_tes[0]=np.zeros(ndata,dtype=int)
  xdata_tes[1]=np.zeros(ndata,dtype=int)
  xdata_tes[2]=np.zeros(ndata)

  k=0
  for i in range(nrow):
    for j in range(ncol):
      xdata_tes[0][k]=i
      xdata_tes[1][k]=j
      if xdatacls.glmmean==1:
        xdata_tes[2][k]=col_mean[i]*row_mean[j]/total_mean
      else:
        xdata_tes[2][k]=col_mean[i]+row_mean[j]-total_mean
      k+=1

  for idata in range(len(xdatacls.xdata_rel[0])):
    i=xdatacls.xdata_rel[0][idata]
    j=xdatacls.xdata_rel[1][idata]
    xdata_tes[2][i*ncol+j]=xdatacls.xdata_rel[2][idata]

  xdatacls.xranges_rel_test=mvm_prepare.mvm_ranges(xdata_tes,nrow)

  return(xdata_tes)
## ###############################################################
def full_test_link(xdatacls):

  col_mean=xdatacls.glm_model.col_mean      # row averages matrix
  row_mean=xdatacls.glm_model.row_mean      # col averages matrix 
  total_mean=xdatacls.glm_model.total_mean  # total average vector

  nrow=xdatacls.nrow
  ncol=xdatacls.ncol

  ## xranges_full=mvm_prepare.mvm_ranges(xdatacls.xdata_rel,xdatacls.nrow)

  ndata=nrow*ncol
  npart=3
  xdata_tes=[None]*3
  xdata_tes[0]=np.zeros(ndata,dtype=int)
  xdata_tes[1]=np.zeros(ndata,dtype=int)
  xdata_tes[2]=np.zeros(ndata)

  k=0
  for i in range(nrow):
    for j in range(ncol):
      xdata_tes[0][k]=i
      xdata_tes[1][k]=j
      xdata_tes[2][k]=col_mean[i]+row_mean[j]-total_mean
      k+=1
  if xdatacls.glm_model.rfunc is not None:
    xdata_tes[2]=xdatacls.glm_model.rfunc.rfunc(xdata_tes[2])

  for idata in range(len(xdatacls.xdata_rel[0])):
    i=xdatacls.xdata_rel[0][idata]
    j=xdatacls.xdata_rel[1][idata]
    xdata_tes[2][i*ncol+j]=xdatacls.xdata_rel[2][idata]

  xdatacls.xranges_rel_test=mvm_prepare.mvm_ranges(xdata_tes,nrow)

  return(xdata_tes)
## ###############################################################

          
  
