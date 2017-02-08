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
## ###########################################################
## from mmr_normalization_new import mmr_normalization
import mvm_mmmvr_lib.mmr_base_classes as base
from mvm_mmmvr_lib.mmr_kernel import mmr_kernel 
import mvm_mmmvr_lib.mmr_kernel_mvm_y as mmr_kernel_mvm_y
import mvm_mmmvr_lib.mmr_kernel_mvm_x as mmr_kernel_mvm_x
import mvm_mmmvr_lib.mvm_test_orig as mvm_test_orig
import mvm_mmmvr_lib.mvm_solver_cls as mvm_solver_cls
import mvm_mmmvr_lib.mvm_prepare as mvm_prepare
import mvm_mmmvr_lib.mvm_glmmodel_cls as mvm_glmmodel_cls
## import mvm_prepare
from mvm_mmmvr_lib.mmr_initial_params import cls_initial_params

## class definitions
## ##################################################
class cls_mvm(base.cls_data):

  def __init__(self,ninputview=1):
    base.cls_data.__init__(self,ninputview)

    self.XKernel=[ None ]*ninputview  ## list of input kernel objects
    self.YKernel=None                 ## ouput kernel object
    self.KX=None                      ## compound input kernel
    self.KY=None                      ## output kernel
    ## mvm specific

    self.xdata_rel=None   ## all data tuples (irow,icol,value)
    self.xdata_tra=None   ## training tuples
    self.xdata_tes=None   ## test tuples
    self.xranges_rel=None     ## training ranges, start position, and length 
                              ## of the known items in each row in
                              ## the sparse representation
    self.xranges_rel_test=None    ## ranges for the test
    self.KXvar=None             
    self.glm_model=None       ## the parameters, means of the GLM model:
                              ## total mean, row means, column means
    self.glmmean=0            ## =0 additive GLM
                              ## =1 multiplicative GLM

    self.largest_class=None   ## row wise largest classes if values are class
                              ## indexes

    self.penalty=base.cls_penalty() ## setting C,D penelty term paramters
    ## other classes
    self.dual=None    ## the vector of the dual variables computed
                      ## by the solver

    self.xbias=0.0    ## penalty term for projective bias, it can be =0
    self.kmode=0    ## =0 additive (feature concatenation)
                    ## =1 multiplicative (fetaure tensor product)

    self.ifixtrain=None   ## xdata_rel relative indexes of the fix training
    self.ifixtest=None    ## xdata_rel relative indexes of the fix test  

    self.crossval_mode=0  ## =0 random cross folds =1 fixtraining
    self.itestmode=3      ## 0 active learning 1 cross validation
                          ## 2 random subsets, 3 fix training test
    self.prandsubset=0.5  ## if itestmode=2, the probability of
                          ## selecting training examples 
    self.ibootstrap=2     ## if itestmode=0
                          ## =0 random =1 worst case =2 best case
                          ## =3 alternate between worst case and random  

    self.nrepeat=1    ## number of repetation of the folding
    self.nfold=5      ## number of folds
    self.nrepeat0=1   ## number of effective repetation of the folding
    self.nfold0=5     ## number of effective folds
    
    self.ieval_type=0     ## =0 category, =1 RMSE , =2 MAE 
    self.ibinary=0        ## =1 Y0=[-1,+1], =0 [0,1,...,categorymax-1]

    ## mvm specific
    self.category=1     ## =0 rank cells =1 category cells =2 {-1,0,+1}^n
                        ## =3 joint table on all categories

    self.categorymax=0
    self.ndata=0        ## all non-missing example in the relation table  
    self.ncol=0         ## number of rows in the relation table
    self.nrow=0         ## number of column in the relation table

    ## test
    self.verbose=0

    ## row-column exchange
    self.rowcol=0       ## =0 row-col order =1 col-row order

    ## for n-fold cross validation
    self.xselector=None   ## a 1d array randomly loaded with
                          ## the indexes of the folds to select the training
                          ## and test in the cross-validation

    ## active learning pointers
    self.icandidate_w=-1    ## the test relative index of
                            ## the worst case in prediction by confidence
    self.icandidate_b=-1    ## the test relative index of
                            ## the best case in prediction by confidence
    self.nmintest=1000         ## minimum test size in active learning
    self.nmaxtrain=100000  ## maximum training size in active learning

    ## special test conditions
    self.testontrain=0      ## =0 test on test, =1 test on train
    self.knowntest=1        ## =0 test items are unknown, =1 known
    self.ifulltest=1        ## =1 test on all possible relations, =0 not
    self.confidence_scale=2   ## scale paramter, e.g. standard deviation,
                              ## of the distribution
                              ## used in the confidence estimation
    self.confidence_local=0   ## localization paramter, e.g. mean,
                              ## of the distribution
                              ## used in the confidence estimation

  ## ---------------------------------------------------------
  def load_data(self,xreldata,lxdatarow,ncategory,nrow,ncol,Y0):
    """
    load the sparse row,column,value format into the mvm object

    Input:
          xreldata    list of arrays of row indexes, column indexes, values
          lxdatarow   list of additional kernels to rows, can be empty
          ## lxdatacol   list of additional kernels to columns, can be empty
          ncategory   number of categories of the values,
                        otherwise =0, or None
          nrow        number of rows
          ncol        number of columns
          Y0          array of possible values, or None              
    """

    nldata=len(xreldata)
    self.xdata_rel=[None]*nldata
    self.xdata_rel[0]=xreldata[0]
    self.xdata_rel[1]=xreldata[1]
    self.ncol=ncol
    self.nrow=nrow
    self.xdata_rel[2]=xreldata[2]
    
    ## self.YKernel=mmr_kernel_mvm_y.cls_feature()
    self.XKernel[0]=mmr_kernel_mvm_x.cls_feature()
    self.categorymax=ncategory
    self.ndata=len(xreldata[0])
    cparams=cls_initial_params()
    
    self.YKernel=mmr_kernel_mvm_y.cls_feature(ifeature=0)
    self.YKernel.kernel_params.set(cparams.get_yparams('kernel',0))
    self.YKernel.crossval.set(cparams.get_yparams('cross',0))
    self.YKernel.norm.set(cparams.get_yparams('norm',0))

    ## setting input parameters
    iview=0   ## we have only one kernel
    self.XKernel[iview]=mmr_kernel_mvm_x.cls_feature(ifeature=0)
    self.XKernel[iview].kernel_params.set(cparams.get_xparams('kernel',iview))
    self.XKernel[iview].crossval.set(cparams.get_xparams('cross',iview))
    self.XKernel[iview].norm.set(cparams.get_xparams('norm',iview))
    iview+=1
    for i in range(len(lxdatarow)):
      self.XKernel[iview]=mmr_kernel_mvm_x.cls_feature(ifeature=1)
      self.XKernel[iview].K=lxdatarow[i]
      self.XKernel[iview].kernel_params.set(cparams.get_xparams('kernel', \
                                                                iview))
      self.XKernel[iview].crossval.set(cparams.get_xparams('cross',iview))
      self.XKernel[iview].norm.set(cparams.get_xparams('norm',iview))
      iview+=1
    
    ## self.Y0=np.arange(self.categorymax)
    self.Y0=Y0
    
    self.penalty.set_crossval()
    
  ## ---------------------------------------------------------
  def set_validation(self):
    """
    Collects the kernel identifiers for choosing kernel to crossvalidate
    """

    self.dkernels={}
    self.dkernels[self.YKernel.title]=self.YKernel
    nkernel=len(self.XKernel)
    for ikernel in range(nkernel):
      self.dkernels[self.XKernel[ikernel].title]=self.XKernel[ikernel]  

  ## ---------------------------------------------------------
  def split_train_test(self,xselector,ifold):
    """
    Selects the training and text indexes to xdata
    Inputs:
            xselector   vector of fold indexes
            ifold       the index of current fold to process 
    """

    if self.itestmode==0:   # active learning
      self.itest=np.where(xselector==ifold)[0]
      self.itrain=np.where(xselector!=ifold)[0]
    elif self.itestmode==1:   # random subset of rank data
      self.itest=np.where(xselector==ifold)[0]
      self.itrain=np.where(xselector!=ifold)[0]
    elif self.itestmode==2:   # random subset of rank data
      self.itest=np.where(xselector==0)[0]
      self.itrain=np.where(xselector!=0)[0]
    elif self.itestmode==3:   # fix training and test
      self.itest=np.where(xselector==0)[0]
      self.itrain=np.where(xselector!=0)[0]

    self.mtrain=len(self.itrain)
    self.mtest=len(self.itest)

  ## ---------------------------------------------------------
  def mvm_datasplit(self):
    """
    splitting the full data into training and test
    xdata_rel -> xdata_train, xdata_test

    Used:
          xdatacls        data class
          itrain       indexs of training items in xdata_rel 
          itest        indexs of test items in xdata_rel 
    """
    xdata_rel=self.xdata_rel
    nitem=len(xdata_rel)
    self.xdata_tra=[None]*nitem
    self.xdata_tes=[None]*nitem
    for i in range(nitem):
      self.xdata_tra[i]=xdata_rel[i][self.itrain]
      self.xdata_tes[i]=xdata_rel[i][self.itest]

    return
  ## ---------------------------------------------------
  def compute_kernels(self):
    """
    Compute output and all input kernels
    """
    ## print('Kernel computation')
    if self.category==2:
      self.YKernel.compute_prekernel(self)
    self.YKernel.compute_kernel(self)
    nkernel=len(self.XKernel)
    for ikernel in range(nkernel):
      self.XKernel[ikernel].compute_kernel(self)

  ## ---------------------------------------------------------
  def mvm_train(self):
    """
    execute the trianing procedure
    Inputs:
    """

    ## print('Generate kernels')
    time0=time.time()
    self.compute_kernels()
    if self.verbose==1:
      print('Kernel computation:',time.time()-time0)
    ## print('Solve optimization problem')
    time0=time.time()
    self.KX=mmr_kernel(self,self.itrain,self.itrain,ioutput=0, \
                            itraintest=0, itensor=self.kmode)[0]
    self.KY=mmr_kernel(self,self.itrain,self.itrain,ioutput=1)[0]
    if self.verbose==1:
      print('Kernel merge computation:',time.time()-time0)
    ## self.solvertime=time.time()-time0

    ## t0=time.clock()
    cOptDual=base.cls_dual(None,None)
    self.dual=cOptDual
    time0=time.time()

    cmvm_solver=mvm_solver_cls.cls_mvm_solver()
    self.dual.alpha=cmvm_solver.mvm_solver(self)
    self.solvertime=time.time()-time0

    return(cOptDual)

  ## ------------------------------------------------------------------
  def mvm_test(self):
    """
    select the potential test method

    wrapper around the possible test evaluations of different losses
    inputs:
          cOptDual      object containing the optimal values of the
                        dual variables    
    outputs:
          cPredict      object containing the prediction results      
    """
    cPredict=base.cls_predict()
    itest_method=0   # =0 orig 
    if itest_method==0:
      # matrix completition test
      cPredict.Zrow=mvm_test_orig.mvm_test_orig(self,self.dual.alpha)
    
    return(cPredict)
  ## ------------------------------------------------------------------
  def glm_norm_in(self,X):
    """
    Compute the residue values of table elements:
    Xres_{ij}=X_ij-row_mean_i-col_mean_j+total_mean
    Input:
            X     array of raw data with 2 dimension
    Output:
            Xres  array of residues with the same shape as X

    Store:
            colmeans. rowmeans and totalmean locally in the object
    """

    (m,n)=X.shape
    self.colmean=np.mean(X,axis=0)
    self.rowmean=np.mean(X,axis=1)
    self.totalmean=np.mean(self.colmean)
    
    Xres=X-np.outer(np.ones(m),self.colmean) \
          -np.outer(self.rowmean,np.ones(n))+self.totalmean

    return(Xres)
  ## ------------------------------------------------------------------
  def glm_norm_out(self,Xres):

    (m,n)=Xres.shape
    X=Xres+np.outer(np.ones(m),self.colmean) \
          +np.outer(self.rowmean,np.ones(n))-self.totalmean

    return(X)
  ## ------------------------------------------------------------------
  def prepare_repetition_folding(self,init_train_size=100):

    if self.itestmode==0:
      ## initialize the active learning seeds
      self.xselector=np.zeros(self.ndata)
      nprime=4999
      ip=0
      for i in range(init_train_size):
        ip+=nprime
        if ip>self.ndata:
          ip=ip%self.ndata
        self.xselector[ip]=1  

      ndatainit=int(np.sum(self.xselector))
      mtest=self.ndata-ndatainit
      self.itest=np.where(self.xselector==0)[0]
      self.icandidate_w=-1
      self.icandidate_b=-1
      ## !!!!!! test size 
      self.nrepeat0=min(self.nmaxtrain,self.ndata-ndatainit-self.nmintest)  
      ## self.nrepeat0=1000
      self.nfold0=1
    elif self.itestmode==1:   ## n-fold cross validation
      self.nrepeat0=self.nrepeat
    elif self.itestmode==2:   ## random subsets based cross validation
      self.nrepeat0=self.nrepeat
      self.nfold0=1
    elif self.itestmode==3:
      self.nrepeat0=self.nrepeat
      self.nfold0=1

  ## ------------------------------------------------------------------
  def prepare_repetition_training(self):

    if self.itestmode==0:
      if self.ibootstrap==0:
        if self.icandidate_w>=0:
          self.icandidate_w=np.random.randint(self.mtest,size=1)
          self.icandidate_w=self.itest[self.icandidate_w]
          self.xselector[self.icandidate_w]=1
          ## xselector[self.icandidate_b]=0     ## delete the best 
      elif self.ibootstrap==1:  ## worst confidence
        if self.icandidate_w>=0:
          self.xselector[self.icandidate_w]=1
          ## xselector[self.icandidate_b]=0     ## delete the best 
      elif self.ibootstrap==2:  ## best confidence
        if self.icandidate_b>=0:
          self.xselector[self.icandidate_b]=1
      elif self.ibootstrap==3:  ## worst+random
        if self.icandidate_w>=0:
          pselect=np.random.rand()
          if pselect<0.5:
            self.icandidate_w=np.random.randint(self.mtest)
            self.icandidate_w=self.itest[self.icandidate_w]
          self.xselector[self.icandidate_w]=1
          ## xselector[self.icandidate_b]=0     ## delete the best
    elif self.itestmode==1:   ## n-fold cross-validation
      self.xselector=np.floor(np.random.random(self.ndata)*self.nfold0)
      self.xselector=self.xselector-(self.xselector==self.nfold0)
    elif self.itestmode==2:   ## n-fold cross-validation
      self.xselector=1*(np.random.random(self.ndata)<self.prandsubset)
    elif self.itestmode==3:   ## fixtraining and fixtest
      self.xselector=np.zeros(self.ndata,dtype=int)
      if self.ifixtrain is not None:
        self.xselector[self.ifixtrain]=1

    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ## for test only
    elif self.itestmode==-1:
      for i in range(self.ndata):
        self.xselector[i]=i%self.nfold0
    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          
    
  ## ------------------------------------------------------------------
  def prepare_fold_training(self,ifold):

    self.split_train_test(self.xselector,ifold)
    mtest=len(self.itest)
    ## if mtest<=0:
    ##   print('!!!!!!!')
    ##   break

    #print('mtest:',mtest,'mtrain:',len(self.itrain))

    self.mvm_datasplit()        

    # sparse matrices of ranks-row_avarage-col_average+total_avarege  
    self.xranges_rel=mvm_prepare.mvm_ranges(self.xdata_tra,self.nrow)
    self.xranges_rel_test=mvm_prepare.mvm_ranges(self.xdata_tes,self.nrow)
    if self.category==0 or self.category==3:
      self.glm_model=mvm_glmmodel_cls.cls_glmmodel()
      ## self.glm_model.mvm_glm_orig(self)
      self.glm_model.rfunc=None
      ## =================================
      ## data transformation
      ## self.glm_model.rfunc=mvm_glmmodel_cls.rfunc_exp_cls()
      self.glm_model.mvm_glm_link(self)
      
      mvm_prepare.mvm_ygrid(self)
    elif self.category==1:
      mvm_prepare.mvm_largest_category(self)
    elif self.category==2:
      mvm_prepare.mvm_largest_category(self)

  ## ------------------------------------------------------------------
  def copy(self,new_obj):

    nkernel=len(self.XKernel)
    ndata=len(self.xdata_rel)
    new_obj.xdata_rel=[None]*ndata
    for i in range(ndata):
      new_obj.xdata_rel[i]=self.xdata_rel[i][self.itrain]

    for ikernel in range(nkernel):
      new_obj.XKernel[ikernel]=self.XKernel[ikernel].copy() 
    new_obj.YKernel=self.YKernel.copy()
      
    new_obj.set_validation()

    new_obj.penalty=base.cls_penalty()
    new_obj.penalty.c=self.penalty.c
    new_obj.penalty.d=self.penalty.d
    new_obj.penalty.crossval=self.penalty.crossval

    ## new_obj.glm_model=self.glm_model
    new_obj.glmmean=self.glmmean
    
    new_obj.nrow=self.nrow
    new_obj.ncol=self.ncol
    new_obj.itestmode=self.itestmode
    new_obj.kmode=self.kmode
    new_obj.xbias=self.xbias
    new_obj.ieval_type=self.ieval_type
    new_obj.ibinary=self.ibinary
    new_obj.categorymax=self.categorymax
    new_obj.Y0=self.Y0
    new_obj.rowcol=self.rowcol
    
## #######################################################################
  

    
