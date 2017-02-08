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
## ####################
## ######################################################################
class cls_mvm_solver:

  def __init__(self):

    self.niter=20   ## maximum iteration
    self.normx1=1  ## normalization within the kernel by this power
    self.normy1=1  ## normalization within the kernel by this power
    self.normx2=1  ## normalization of duals bound
    self.normy2=1  ## normalization of duals bound
    self.ilabel=0  ## 1 explicit labels, 0 implicit labels 
    self.ibias=0   ## 0 no bias considered, 1 bias is computed by solver 
    self.ibias_estim=0  ## estimated bias =0 no =1 computed 
    self.report=0
    self.isteptype=1 ## =0 exact line serach, =1 diminishing step size
  
  ## ------------------------------------------------------  
  def mvm_solver(self,xdatacls):
    """
    It solves the maximum margin solution to the relation learning problem

    Input:
      xdatacls      data class

    Output:
      xalpha        optimal dual variables
    """
    ## load data, kernels, parameters of sparse representation
    KXfull=xdatacls.KX
    KYfull=xdatacls.KY
    KXvar=xdatacls.KXvar
    xranges=xdatacls.xranges_rel
    xdata=xdatacls.xdata_tra
    xdata1=xdata[1]
    xdata2=xdata[2]

    txdim=xdata2.shape
    if len(txdim)==1:
      nxdim=1
    else:
      nxdim=txdim[1]

    nrow=xdatacls.nrow
    ncol=xdatacls.ncol
    ndata=xdata[0].shape[0]

    ycategory=xdatacls.category   ## cells are categorical variables

    ## ymax=xdatacls.YKernel.ymax
    ymin=xdatacls.YKernel.ymin
    ystep=xdatacls.YKernel.ystep

    C=xdatacls.penalty.c

    ## xnrow=zeros(nrow)

    niter=self.niter

    ## generate column->row references from row->column
    lcols={}           ## dictionary for ncol
    if self.report==1:  
      print('Preparing the solver input')
    xncols=np.zeros(ncol)
    for idata in range(ndata):
      icol=xdata1[idata]
      xncols[icol]+=1

    for icol in range(ncol):
      lcols[icol]=np.zeros(xncols[icol],dtype=int)

    xpcols=np.zeros(ncol)
    for idata in range(ndata):
      icol=xdata1[idata]
      ## irow=xdata[0][idata]
      ## istart=xranges[irow,0]
      lcols[icol][xpcols[icol]]=idata
      xpcols[icol]+=1

    xweight_row=np.ones(nrow)
    ## xweight_col=np.ones(ncol)*C

    ## initialize the dual variables, xalpha, and gradient ,xnabla0_prev
    xalpha=np.zeros(ndata)
    tau=0
    ixalpha_star=np.zeros((ncol,3),dtype=int)-1
    xnabla0_prev=np.zeros(ndata)
    for irow in range(nrow):
      istart=xranges[irow,0]
      nlength=xranges[irow,1]
      if nlength>0:
        xnabla0_prev[istart:istart+nlength]=(-1)*xweight_row[irow]

    xtime=np.zeros(5)    

  ##  print('Solving optimization problem') 
  ## conditional gradient iteration
    for iiter in range(niter):
      t0=time.clock()
  ## current gradient
      xnabla0=np.zeros(ndata)       ## initialize current gradient
      for irow in range(nrow):
        ## get the row related subsets of columsn
        istart=xranges[irow,0]
        nlength=xranges[irow,1] 
        if nlength>0:
          ## previous gradient slice belonging one row 
          xnabla0_prev_s=xnabla0_prev[istart:istart+nlength]
          ## correpsonding optimal values of the subproblem
          icol_index=np.where(ixalpha_star[:,1]==irow)[0]
          if len(icol_index)>0:
            icols=ixalpha_star[icol_index,2]
            ixrange=xdata1[istart:istart+nlength]
            ixsubrange=ixrange[icols]

            if ycategory in (0,3):
              if nxdim==1:  ## row,column -> R^{1}
                iyrange=xdata2[istart:istart+nlength]
              else: ## row,column -> R^{nxdim}
                iyrange=xdata2[istart:istart+nlength][:,0]
              ## create the slice related output values
              iyrange=np.round((iyrange-ymin)/ystep).astype(int)
              iysubrange=iyrange[icols]
              ly=len(iyrange)
              ## slice related output kernel
              KKY=KYfull[iyrange.reshape((ly,1)),iysubrange]
              for i in range(1,nxdim): ## vector valued relation
                iyrange=xdata2[istart:istart+nlength][:,i]
                iyrange=np.round((iyrange-ymin)/ystep).astype(int)
                iysubrange=iyrange[icols]
                ly=len(iyrange)
                KKY+=KYfull[iyrange.reshape((ly,1)),iysubrange]
            elif ycategory in (1,2):  ## we have categorical relations
              iyrange=xdata2[istart:istart+nlength]
              iysubrange=iyrange[icols]
              ly=len(iyrange)
              KKY=KYfull[iyrange.reshape((ly,1)),iysubrange]

            lx=len(ixrange)
            ## input kernel to the slice
            KKX=KXfull[ixrange.reshape((lx,1)),ixsubrange]
            ## input-output joint kernel
            KKZ=KKX*KKY
            if KXvar is not None:
              KKZ*=KXvar[irow,irow] ## including row scale

            ## xnabla_star=C*np.dot(KKZ,xweight_col[icols])
            xnabla_star=C*np.dot(KKZ,np.ones(len(icols))*C)

            ## the change of the gradient on the slice
            xnabla0[istart:istart+nlength]+=tau*xnabla_star
          ## new gradient on the slice  
          xnabla0[istart:istart+nlength]+=(1-tau)*xnabla0_prev_s-tau

      xnabla0_prev=xnabla0  ## save the current gradient

      t1=time.clock()
      xtime[0]+=t1-t0
  ## optimum solution of subproblem    
      ixalpha_star=np.zeros((ncol,3),dtype=int)-1
      for icol in range(ncol):
        irows=lcols[icol]
        ## find the smallest value of the gradient to each column
        if len(irows)>0:
            vm=np.min(xnabla0[irows])
            imall=np.where(xnabla0[irows]==vm)[0]
            try:
              imp=np.random.randint(0,len(imall),1)[0]
            except:
              print(imall)
            im=imall[imp]
            if vm<0:
              iglob=irows[im]
              ixalpha_star[icol,0]=iglob
      ## binary search for row index of the smallest gradient
      ## corresponding to the column
              irow=nrow>>1
              ibegin=0
              iend=nrow-1
              istat=1
              while istat==1:
                istart=xranges[irow,0]
                nlength=xranges[irow,1]
                if iglob>=istart:
                  if iglob<istart+nlength:
                    ixalpha_star[icol,1]=irow
                    ixalpha_star[icol,2]=iglob-istart
                    istat=0
                  else:
                    ibegin=irow+1
                    irow=(ibegin+iend)>>1
                    ## if irow==ibegin:
                    ##   irow=iend
                else:
                  iend=irow-1
                  irow=(ibegin+iend)>>1
                  ## if irow==iend:
                  ##   irow=ibegin

  ## find best convex combination (tau) of old and new
      t2=time.clock()
      xtime[1]+=t2-t1

      xdelta=-xalpha
      for icol in range(ncol):
        if ixalpha_star[icol,0]>=0: 
          ## xdelta[ixalpha_star[icol,0]]=xdelta[ixalpha_star[icol,0]] \
          ##                                  +C*xweight_col[icol]

          ## the column wise change of the duals 
          xdelta[ixalpha_star[icol,0]]=xdelta[ixalpha_star[icol,0]] \
                                           +C*C
      ## xnumerator=-sum(xnabla0*xdelta)+sum(xdelta)

      ## -----------------------------------------
      ## self.isteptype  =0 optimal line search
      ##            =1 diminisheng stepsize s_{t+!}=s_{t}-s_{t}^2, s_0=0.5
      ## on small data optimal line search is more precise,
      ##    but on large data set is significantly, magnitude, slower
      ##    and requires more memory  
      ##------------------------------------------
      if self.isteptype==0:
        ## optimal line search
        xnumerator=-np.sum(xnabla0*xdelta)
        xdenominator=0

        for irow in range(nrow):
          istart=xranges[irow,0]
          nlength=xranges[irow,1]
          if nlength>0:
            xdeltas=xdelta[istart:istart+nlength]
            inzero=np.where(xdeltas!=0)[0]
            if len(inzero)>0:
              ixsubrange=xdata1[istart+inzero]
              if ycategory in (0,3):
                if nxdim==1:
                  iysubrange=xdata2[istart+inzero]
                else:
                  iysubrange=xdata2[istart+inzero][:,0]
                iysubrange=np.round((iysubrange-ymin)/ystep).astype(int)
                ly=len(iysubrange)
                KKY=KYfull[iysubrange.reshape((ly,1)),iysubrange]
                for i in range(1,nxdim):
                  iysubrange=xdata2[istart+inzero][:,i]
                  iysubrange=np.round((iysubrange-ymin)/ystep).astype(int)
                  ly=len(iysubrange)
                  KKY+=KYfull[iysubrange.reshape((ly,1)),iysubrange]
              elif ycategory in (1,2):
                iysubrange=xdata2[istart+inzero]
                ly=len(iysubrange)
                KKY=KYfull[iysubrange.reshape((ly,1)),iysubrange]

              lx=len(ixsubrange)
              KKX=KXfull[ixsubrange.reshape((lx,1)),ixsubrange]
              KKZ=KKX*KKY

              if KXvar is not None:
                KKZ*=KXvar[irow,irow] ## including row scale

              xdeltas=xdeltas[inzero]
              xdenominator=xdenominator+np.dot(xdeltas,np.dot(KKZ,xdeltas))

        if (irow+1) % 1000==0:
  ##        print(iiter,irow)
          pass

        if xdenominator!=0.0:
          tau=xnumerator/xdenominator
        else:
          tau=1.0

        if tau<0:
          tau=0
        if tau>1:
          tau=1

      elif self.isteptype==1:
        ## independent tau sequence
        ## tau_{k+1} = tau_{k}-tau^2_{k}/2, tau_{0}=0.5
        ## \sum_k tau_k \rightarrow \infty
        ## \sum_k tau^2_k < \infty

        if iiter==0:
          tau=0.5
        else:
          tau-=tau**2/2
      ## update the vector of dual variables 
      xalpha=xalpha+tau*xdelta

      t3=time.clock()
      xtime[2]+=t3-t2

      if (iiter+1)%10==0:
        xerr=np.sqrt(np.sum((tau*xdelta)**2))
        if self.report==1:  
          print('%4d'%iiter, '%6.4f'%tau, '%12.6f'%xerr,'%12.6f'%(-xnumerator))
          ## print(xtime)

    if self.report==1:  
      print(xtime)

    return(xalpha)
                 
## ####################################################################
