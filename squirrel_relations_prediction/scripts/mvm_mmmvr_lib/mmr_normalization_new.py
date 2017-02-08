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
from numpy import mean, concatenate, median, std, tile, dot, diag
from numpy import ones, zeros, where, outer, sqrt, array, abs, copy
from numpy import sum as np_sum
from numpy import max as np_max
from numpy import linalg
## ###########################################################3
def mmr_normalization(ilocal,iscale,XTrain,XTest,ipar):
## function to normalize the input and the output data
## !!!! the localization happens before normalization if both given !!! 
## input
##      ilocal centralization   
##                  =-1 no localization
##                  =0 mean
##                  =1 median
##                  =2 geometric median
##                  =3 shift by ipar
##                  =4 smallest enclosing ball
##                  =5 row mean row wise 
##      icenter
##                  =-1 no scaling
##                  =0 scale item wise by L2 norm
##                  =1 scale item wise by L1 norm
##                  =2 scale item wise by L_infty norm
##                  =3 scale items by stereographic projection relative to zero
##                  =4 scale variables by STD(standard deviation)
##                  =5 scale variables by MAD(median absolute deviation)
##                  =6 scale variables by absolute deviation
##                  =7 scale all variables by average STD 
##                  =8 scale all variables by maximum STD 
##                  =9 scale all variables by median MAD 
##                  =10 scale item wise by Minkowski norm, power given by ipar
##                  =11 \sum_i||u-x_i||/m where u=0
##                  =12 scale all variables by overall max
##                  =13 Mahalonobis scaling  
##      XTrain       Data matrix which will be normalized. It assumed the
##                   rows are the sample vetors and the columns are variables 
##      XTest        Data matrix which will be normalized. It assumed the
##                   rows are the sample vetors and the columns are
##                   variables.
##                   It herites the center and the scale in the variable wise ca##                   se from the XTrain,
##                   otherwise it is normalized independently  
##      ipar         additional parameter   
##  output
##      XTrain       Data matrix which is the result of the normalization
##                   of input XTrain. It assumed the rows are the sample
##                   vetors and the columns are variables  
##      XTest        Data matrix which is the result of the normalization
##                   of input XTest. It assumed the rows are the sample
##                   vetors and the columns are variables.
##      opar         the radius in case of ixnorm=2.  
##  
  if XTest is None:
    XTest=array([])
    
  opar=0;
  (mtrain,n)=XTrain.shape
  if len(XTest.shape)>=2:
    mtest=XTest.shape[0]
  elif len(XTest.shape)==1:
    mtest=XTest.shape[0]
    XTest=XTest.reshape((mtest,1))
  else:
    mtest=0
    XTest=array([])

  if ilocal==-1:
    pass
  elif ilocal==0:   ##  mean
    xcenter=mean(XTrain,axis=0)
  elif ilocal==1:   ##  median
    xcenter=median(XTrain,axis=0)
  elif ilocal==2:    ##  geometric median
    xcenter=mmr_geometricmedian(XTrain)[0]
  elif ilocal==3:    ##  shift by ipar
    xcenter=ipar
  elif ilocal==4:   ##  smallest comprising ball
    xalpha=mmr_outerball(0,XTrain)
    xcenter=dot(XTrain.T,xalpha)
  elif ilocal==5:   ## row mean row wise
    xcenter=mean(XTrain,axis=1)

  if ilocal in (0,1,2,3,4):
    XTrain=XTrain-tile(xcenter,(mtrain,1))
    if mtest>0:
      XTest=XTest-tile(xcenter,(mtest,1))
  elif ilocal==5:
    XTrain=XTrain-outer(xcenter,ones(n))
    if mtest>0:
      xcenter=mean(XTest,axis=1)
      XTest=XTest-outer(xcenter,ones(n))

## itemwise normalizations
  if iscale==-1:
    pass
  elif iscale==0:     ## scale items by L2 norm
    xscale_tra=sqrt(np_sum(XTrain**2,axis=1))
    if mtest>0:
      xscale_tes=sqrt(np_sum(XTest**2,axis=1))
  elif iscale==1:     ## scale items by L1 norm
    xscale_tra=np_sum(abs(XTrain),axis=1)
    if mtest>0:
      xscale_tes=np_sum(abs(XTest),axis=1)
  elif iscale==2:     ## scale items by L_infty norm
    xscale_tra=np_max(abs(XTrain),axis=1)
    if mtest>0:
      xscale_tes=np_max(abs(XTest),axis=1)
  elif iscale==10:     ## scale items by Minowski with ipar
    xscale_tra=np_sum(abs(XTrain)**ipar,axis=1)**(1/ipar)
    if mtest>0:
      xscale_tes=np_sum(abs(XTest)**ipar,axis=1)**(1/ipar)

  if iscale in (0,1,2,10):    
    xscale_tra=xscale_tra+(xscale_tra==0)
    XTrain=XTrain/tile(xscale_tra.reshape(mtrain,1),(1,n))
    if mtest>0:
      xscale_tes=xscale_tes+(xscale_tes==0)
      XTest=XTest/tile(xscale_tes.reshape(mtest,1),(1,n))
          
  if iscale==3:   ## scale items by stereographic projection relative to zero
    xnorm2=np_sum(XTrain**2,axis=1)
    R=ipar
    xhom=ones(mtrain)/(xnorm2+R**2)
    xhom2=xnorm2-R**2
    XTrain=concatenate((2*R**2*XTrain*outer(xhom,ones(n)),R*xhom2*xhom), \
                       axis=1)
    if mtest>0:
      xnorm2=np_sum(XTest**2,axis=1)
      xhom=ones(mtest)/(xnorm2+R**2)
      xhom2=xnorm2-R**2
      XTest=concatenate((2*R**2*XTest*outer(xhom,ones(n)),R*xhom2*xhom), \
                        axis=1)

## variable wise normalization relative to zero
## test has to use of the training scale 

  if iscale==-1:
    pass
  elif iscale==4:     ## scale vars by std to zeros center
    xscale=std(XTrain,axis=0)
##    xscale=sqrt(mean(XTrain**2,axis=0)) 
  elif iscale==5:     ## scale vars by mad
    xscale=median(abs(XTrain),axis=0)
  elif iscale==6:     ## scale vars by absolut deviation
    xscale=mean(abs(XTrain),axis=0)

  if iscale in (4,5,6):
    xscale=xscale+(xscale==0)
    XTrain=XTrain/tile(xscale,(mtrain,1))
    if mtest>0:
      XTest=XTest/tile(xscale,(mtest,1))

  if iscale==-1:
    pass
  if iscale==7:     ## scale vars by average std to zero center
##    xscale=mean(std(XTrain,axis=0))
    xscale=mean(sqrt(mean(XTrain**2,axis=0)))
  elif iscale==8:     ## scale vars by max std to zero center
##    xscale=np_max(std(XTrain,axis=0))
    xscale=np_max(sqrt(mean(XTrain**2,axis=0)))
  elif iscale==9:     ## scale vars by median mad
    xscale=median(median(abs(XTrain),axis=0))
  elif iscale==11:    ## \sum_i||u-x_i||/m where u=0
    xscale=mean(sqrt(np_sum(XTrain**2,axis=1)))
  elif iscale==12:    ## \sum_i||u-x_i||/m where u=0
    xscale=XTrain.max()

##  print(xscale)
  if iscale in (7,8,9,11,12):
    xscale=xscale+(xscale==0)
    XTrain=XTrain/xscale
    if mtest>0:
      XTest=XTest/xscale

  if iscale==13:     ## scale by Mahalonobis
    xsigma=dot(XTrain.T,XTrain) ## covariance
    [w,v]=linalg.eigh(xsigma)
    iw=where(w<=10**(-10))[0]
    w[iw]=0.0
    iw=where(w>0.0)[0]
    w_sqinv=zeros(XTrain.shape[1])
    w_sqinv[iw]=1/sqrt(w[iw])
    XTrain=dot(XTrain,v)*outer(ones(mtrain),w_sqinv)
    if mtest>0:
      XTest=dot(XTest,v)*outer(ones(mtest),w_sqinv)
    
  return(XTrain,XTest,opar)
    
## ******************************************************
def mmr_outerball(iker,xdata):
  """ solves the minimum enclosing ball problem via
      one-class maximum margin dual
      the algorithm is based on the conditional gradient method
      
      Base problem

      \min_u \max_{\alpha} \sum_i^m \alpha_i \| u - \phi(x_i)\|^2
      
      if $u=\sum_j^m \beta_j\phi(x_j)  which is equivalent to

      Primal problem
      \min_{\beta,r} r
      \text{s.t.}  \|\sum_j \beta_j\phi(x_j) - \phi(x_i)\|^2 \le r, i=1,\dots\m

      which is equivalent to
      
      \min_{\beta,r} r
      \text{s.t.}  K_{ii} + \beta^{T}K\beta - 2 \beta^T K_i  \le r, i=1,\dots\m

      where K=[\langle \phi(x_i),\phi(x_j) \rangle], and K_i is column i of K

      then the dual problem can be written as 

      \min_{\alpha} \alpha^{T} K \alpha - \alpha^T diag(K)
      \text{s.t.} \sum_i \alpha_i =1, \alpha_i\ge 0 i=1,\dots,m 
     
  """
  (m,n)=xdata.shape
  if iker==0 and n<m:
    kdiag=np_sum(xdata**2,1)
  else:
    if iker==1:
      K=xdata
    else:
      K=dot(xdata,xdata.T)
    kdiag=diag(K)
    
  niter=1000
  l1eps=0.02/m
  alpha=ones(m)/m
##  alpha=rand(m,1)
##  alpha=alpha/np_sum(abs(alpha))
  for iiter in range(niter):
## compute the gradient
##    fgrad=2*K*alpha-kdiag
    if iker==0 and n<m:   ## to reduce computation
      fgrad=2*dot(xdata,dot(xdata.T,alpha))-kdiag
    else:
      fgrad=2*dot(K,alpha)-kdiag
      
## solve subproblem:   min_u fgrad'*u, s.t. 1'*u=1, u>=0
    u=zeros(m)
    vm=min(fgrad)
    imm=where(fgrad==vm)[0]
    limm=len(imm)
    if limm==0:
      limm=1
##      u(imm(randi(limm,1)))=1
    u[imm]=float(1)/limm
    xdelta=u-alpha

    if iker==0 and n<m:
      xbeta=dot(xdata.T,xdelta)
      xdenom=2*dot(xbeta,xbeta)
    else:
      xdenom=2*dot(dot(xdelta,K),xdelta)

    if xdenom==0:
      break
    tau=-dot(fgrad,xdelta)/xdenom
    if tau<=0:
      tau=0
      break
    if tau>1:
      tau=1
##    print(iiter,tau)
    alphanew=tau*u+(1-tau)*alpha
##    f=(alphanew'*K*alphanew-kdiag'*alphanew)/2
    alphadiff=tau*xdelta
    xerr=np_sum(abs(alphadiff))
##    disp([iiter,tau,xerr,f])
    if xerr/m<l1eps: 
      alpha=alphanew
      break
    alpha=alphanew
  
##  xcenter=dot(xdata.T,alpha)

  return(alpha)
## ####################################################
""" Weszfeld's algorithm:
Weiszfeld, E. (1937). "Sur le pour lequel la somme des distances de n points donnes est minimum" Tohou Math. Journal 43:355-386
"""
def mmr_geometricmedian(X):
  (m,n)=X.shape
  u=mean(X,axis=0)
  niter=1000
  xeps=sqrt(np_sum(u**2))/1000
  if xeps==0:
    xeps=10**(-6)
  xerr=2*xeps
  for i in range(niter):
    d2u=sqrt(np_sum((X-tile(u,(m,1)))**2,axis=1))
    inul=where(d2u<xeps)[0]
    d2u[inul]=xeps
    unext=np_sum(X/tile(d2u.reshape((m,1)),(1,n)),axis=0)/np_sum(ones(m)/d2u)
    if np_max(unext-u)<xerr:
      break
    u=copy(unext)
  return(unext,i,np_max(unext-u))

## ####################################################
""" Weszfeld's algorithm for kernel representation:
Weiszfeld, E. (1937). "Sur le pour lequel la somme des distances de n points donnes est minimum" Tohou Math. Journal 43:355-386

\sum_{i} \frac{K_i}{da_i} = Ka \sum_i \frac{1}{da_i}
(da_i)^2 = K_{ii}+a'Ka - 2\braket{K_i,a}

Ka= \frac{\sum_{i} \frac{K_i}{da_i}}{\sum_i \frac{1}{da_i}}
a'Ka= \frac{\sum_{i} \frac{\braket{K_i,a}}{da_i}}{\sum_i \frac{1}{da_i}}

"""
def mmr_geometricmedian_ker(K):
  m=K.shape[0]
  Ka=mean(K,axis=1)
  aKa=np_sum(Ka)/m

  niter=1000
  xeps=sqrt(np_sum(Ka**2))/100
  xerr=2*xeps

  e1=ones(m)

  for iiter in range(niter):
    ## d2u=sqrt((zeros(m)+aKa)+diag(K)-2*Ka)
    d2u_2=aKa+diag(K)-2*Ka
    ineg=where(d2u_2<0)[0]
    d2u_2[ineg]=0.0
    d2u=sqrt(d2u_2)

    inul=where(d2u<xeps)[0]
    d2u[inul]=xeps
    xdenom=np_sum(e1/d2u)
    Kanext=np_sum(K/outer(d2u,e1),axis=0)/xdenom 
    aKanext=np_sum(Ka/d2u)/xdenom
    if np_max(Kanext-Ka)<xerr:
      Ka=copy(Kanext)
      aKa=aKanext
      break
    Ka=copy(Kanext)
    aKa=aKanext
    
  return(Ka,aKa)
## ###############################################################
""" Residual features of a matrix

  x_residual[i,j]=x[i,j]-rowmean[i]-columnmean[j]+total

"""
def mmr_residual(X):

  (m,n)=X.shape
  rowmean=mean(X,axis=1)
  colmean=mean(X,axis=0)
  totalmean=mean(rowmean)
  Xresidual=X-outer(rowmean,ones(n))-outer(ones(m),colmean)+totalmean

  return(Xresidual,rowmean,colmean,totalmean)
## ###############################################################
    
    
