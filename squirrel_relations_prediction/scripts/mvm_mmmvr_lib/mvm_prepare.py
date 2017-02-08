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

from scipy import sparse
## ####################
import mvm_mmmvr_lib.mmr_base_classes as mmr_base_classes
## ####################
def mvm_datasplit(xdatacls,itrain,itest):
  """
  splitting the full data into training and test
  xdata_rel -> xdata_train, xdata_test
  
  Input:
  xdatacls        data class
  itrain       indexs of training items in xdata_rel 
  itest        indexs of test items in xdata_rel 
  """
  xdata_rel=xdatacls.xdata_rel
  nitem=len(xdata_rel)
  xdatacls.xdata_tra=[None]*nitem
  xdatacls.xdata_tes=[None]*nitem
  for i in range(nitem):
    xdatacls.xdata_tra[i]=xdata_rel[i][itrain]
    xdatacls.xdata_tes[i]=xdata_rel[i][itest]

  return
## ###########################################################
def mvm_datasplit_subset(xdatacls,itrain,itest):
  """
  splitting the full data into training and test
  xdata_rel -> xdata_train, xdata_test
  
  Input:
  xdatacls        data class
  itrain       indexs of training items in xdata_rel 
  itest        indexs of test items in xdata_rel 
  """
  xdata_rel=xdatacls.xdata_rel
  nitem=len(xdata_rel)
  xdatacls.xdata_tra=[None]*nitem
  xdatacls.xdata_tes=[None]*nitem
  for i in range(nitem):
    xdatacls.xdata_tra[i]=xdata_rel[i][xdatacls.iobjects_data[itrain]]
    xdatacls.xdata_tes[i]=xdata_rel[i][xdatacls.iobjects_data[itest]]

  return
## ###########################################################
def mvm_ranges(xdata,nitem):
  """
  Creates the ranges of the data to each column index

  Input:
  xdata       data, for example xdatacls.xdata_tra
  nitem       number of rows, for example  xdatacls.nrow

  Output:
  xranges     the range matrix with two coluns (starting point, length)
  """
  
  xranges=np.zeros((nitem+1,2),dtype=int)

  mdata=xdata[0].shape[0]
  for idata in range(mdata):
    iitem=xdata[0][idata]
    xranges[iitem,1]+=1       ## counts the row items to one column
                              ## to get the length
  
  xranges[:,0]=np.cumsum(xranges[:,1])-xranges[:,1]  ## compute starting points
    
  return(xranges)
## ###########################################################
def mvm_loadmatrix(xdatacls):
  """
  It can load the sparse relation matrix of the training data
  Not used in the current version
  
  Input:
  xdatacls          data class
  """
  
  nrow=xdatacls.nrow
  ncol=xdatacls.ncol
  
  ydata=xdatacls.xdata_tra[2]
  xdatacls.xrelations=sparse.csr_matrix((ydata,(xdatacls.xdata_tra[1], \
                                              xdatacls.xdata_tra[0])), \
                                      shape=(ncol,nrow))
  
  return

## ###########################################################
def mvm_ygrid(xdatacls):
  """
  It digitalizes the values to be predicted to constract probability density function features, e.g. Gaussian densities centralized on the observed value, of the output items  
  
  Input:
  xdatacls      data class  
  """

##  nrow=xdatacls.nrow
##  ncol=xdatacls.ncol

  ydata=xdatacls.xdata_tra[2]

## compute grid  
  ymax=np.max(ydata)
  ymin=np.min(ydata)
  
  ## ystep=xdatacls.YKernel.ystep
  yrange=xdatacls.YKernel.yrange
  ## ymax=np.ceil((ymax+ystep)/ystep)*ystep
  ## ymin=np.floor((ymin-ystep)/ystep)*ystep
  ymax=np.ceil(ymax+0.01*(ymax-ymin))
  ymin=np.floor(ymin-0.01*(ymax-ymin))
  ## if ymax>xdatacls.YKernel.ymax:
  xdatacls.YKernel.ymax=ymax
  ## if ymin<xdatacls.YKernel.ymin:
  xdatacls.YKernel.ymin=ymin
    
  ## xdatacls.YKernel.yrange=(ymax-ymin)/ystep
  ystep=xdatacls.YKernel.ystep
  ## ystep=(ymax-ymin)/yrange  
  ## xdatacls.YKernel.ystep=ystep
  xdatacls.YKernel.yrange=np.ceil((ymax-ymin)/ystep)

  ydata=xdatacls.xdata_tra[2]
  xdatacls.xdata_tra[2]=np.round(ydata/ystep)*ystep
    
  return
## #######################################################
def mvm_largest_category(xdatacls):
  """
  find the largest category in each training row as default prediction
  Input:
  xdatacls      data class  

  Output:
  xdatacls.largest_class.row_max_category is filled
                                  with the largest category label 
  """
  xdata=xdatacls.xdata_tra
  xranges=xdatacls.xranges_rel
  mdata=xdata[0].shape[0]
  nrow=xdatacls.nrow
  ncol=xdatacls.ncol
  
  xdatacls.largest_class=mmr_base_classes.cls_empty_class()

  row_max_category=np.zeros(nrow)
  col_max_category=np.zeros(ncol)

  if xdatacls.category==1:
    nyrange0=xdatacls.categorymax
    for irow in range(nrow):
      (istart,nlength)=xranges[irow,:]
      xcat=np.zeros(nyrange0)
      for i in range(nlength):
        icat=xdata[2][istart+i]
        xcat[icat]+=1
      row_max_category[irow]=xcat.argmax()  

    xcat=np.zeros(nyrange0)
    for i in range(mdata):
      icat=xdata[2][i]
      xcat[icat]+=1

    xdatacls.largest_class.row_max_category=row_max_category
    xdatacls.largest_class.max_category=xcat/np.sum(xcat)
    
  elif xdatacls.category==2:
    ndim=xdatacls.YKernel.ndim
    valrange=xdatacls.YKernel.valrange
    nval=max(valrange)+1
    tdim=[nval]*ndim

    xtotalmax=np.zeros((ndim,nval))
    xmaxrow=np.zeros((nrow,ndim,nval))
    xmaxcol=np.zeros((ncol,ndim,nval))
    
    for irow in range(nrow):
      (istart,nlength)=xranges[irow,:]
      icat=xdata[2][istart:istart+nlength]
      xcat=np.array(np.unravel_index(icat,tdim)).T
      for i in range(nlength):
        for j in range(ndim):
          if xcat[i,j]!=0:
            xmaxrow[irow,j,xcat[i,j]]+=1
            xmaxcol[i,j,xcat[i,j]]+=1
            xtotalmax[j,xcat[i,j]]+=1

    for irow in range(nrow):  
      row_max_category[irow]=np.ravel_multi_index(xmaxrow[irow].argmax(1),tdim)
    xdatacls.largest_class.row_max_category=row_max_category
    for icol in range(ncol):  
      col_max_category[icol]=np.ravel_multi_index(xmaxcol[icol].argmax(1),tdim)
    xdatacls.largest_class.col_max_category=col_max_category
    
    xdatacls.largest_class.max_category= \
             np.ravel_multi_index(xtotalmax.argmax(1),tdim)

  return

## #######################################################
def sort_table(xdata0,ifloat=1,idata=1):

  nitem=len(xdata0)
  ndata=len(xdata0[0])
  ldata=[ [xdata0[0][i],xdata0[1][i],i] for i in range(ndata)]
  ldata.sort()
  xdata=[None]*nitem 
  xdata[0]=np.array([ xitem[0] for xitem in ldata]).astype(int)
  xdata[1]=np.array([ xitem[1] for xitem in ldata]).astype(int)
  if idata==1:
    xdata[2]=np.array([ xdata0[2][xitem[2]] for xitem in ldata]).astype(float)
    if ifloat==0:
      xdata[2]=xdata[2].astype(int)
  else:
    ## only data indexes preserved in the original xdata0
    xdata[2]=np.array([ xitem[2] for xitem in ldata]).astype(int)
 
  return(xdata)
## #######################################################

