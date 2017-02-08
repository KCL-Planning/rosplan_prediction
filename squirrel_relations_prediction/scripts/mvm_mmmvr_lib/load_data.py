## ##################################################
import csv
import numpy as np
## import scipy.io
## import pickle
import scipy.stats
import os

## import scipy.linalg as sp_linalg
import mvm_mmmvr_lib.mvm_prepare as mvm_prepare
## import tensor_decomp
## ###################################################

## ##################################################
class cls_label_files:
  """
  Loads a image label files and gives back the sparse mvm form:
                (row_index,column_index,value), in list of arrays

  """

  def __init__(self, data_path, input_file, output_file, number_of_columns):

    self.sbasedir=data_path
    self.listfull=input_file
    self.listknown=input_file
    self.listout=output_file
    self.fileext='.csv'
    self.csv_delimiter=','
    self.headerline=1
    self.listrel=['can_fit_inside','can_pickup','can_push','can_stack_on','inside','object_at','on','push_location','can_fit_inside','can_pickup']
 
    self.test_predicted_data='test_prediction.txt'
    self.full_predicted_data='full_prediction.txt'

    self.nfeature=number_of_columns ## number of features assigned to each pair of objects
    self.feature_orig=2

    self.dobject={}   ## object name -> object index
    self.dobject_inv={}   ## object index -> object name

    self.ddata_tra={} ## collected relations in training
    self.ddata_tes={} ## collected relations in test

    self.nobject=0  ## number of objects
    self.nrow=0     ## number of internal rows
    self.ncol=0     ## number of internal columns

    self.irowcol=0  ## =0 3D array mapped into 2D array
                    ##      (object,object * feature)
                    ## =1 3D array mapped into 2D array
                    ##  (object * feature,object)

    
  ## --------------------------------------------------
  def load_onefile(self,iknown,ifile):
    """
    - load collected known entities (object, action,feature)
    into the relation table.
     - one file is loaded and cut into training and test
    - it is to find the best parameters for the general case
    """

    ## first phase collect object labels
    if iknown==0:
      sfile=self.listfull
    else:
      sfile=self.listknown
      
    self.collect_objects(sfile)
    self.collect_relations(sfile,self.ddata_tra)

    print('<<<'+sfile)
    ## second phase load the relation table of known examples
    xdata=[[],[],[]]    ## row index, column index , value

    for iobject1,dobject2 in self.ddata_tra.items():
      for iobject2,dvalues in dobject2.items():
        for ifeature in range(self.nfeature):
          if ifeature in dvalues:
            svalue=dvalues[ifeature]
          else:
            svalue=''
          if len(svalue)>0:   ## if value is not given it is assumed as missing
            if self.irowcol==0: 
              xdata[0].append(iobject1)
              xdata[1].append(ifeature*self.nobject+iobject2)
              xdata[2].append(float(svalue))
            elif self.irowcol==1:
              xdata[0].append(ifeature*self.nobject+iobject1)
              xdata[1].append(iobject2)
              xdata[2].append(float(svalue))
        
    if self.irowcol==0:
      self.nrow=self.nobject
      self.ncol=self.nobject*self.nfeature
    elif self.irowcol==1:
      self.nrow=self.nobject*self.nfeature
      self.ncol=self.nobject

    xdata=mvm_prepare.sort_table(xdata,ifloat=1)

    return(xdata,self.nrow,self.ncol)

  ## --------------------------------------------------
  def load_twofiles(self):
    """
    load collected known entities (object, action,feature)
    + the unknown object, action pairs without value into the relation table.

    !!! here we assume that all files contains the same set of object pairs.
    !!! it can be changed later!!!
    iknown1,ifile1   give the training
    iknown2,ifile2   give the test
    """
    ## ['full','full_20','full_40','full_60', \
    ##  'known','known_20','known_40','known_60']
    ifile1=0   ## file index in list known
    ifile2=0   ## file index in list full
    iknown1=1  ## known 
    iknown2=0  ## full
    ## first phase collect object labels
    sfile1=self.listknown
    sfile2=self.listknown

   
    self.collect_objects(sfile1)
    self.collect_relations(sfile1,self.ddata_tra)
    self.collect_relations(sfile2,self.ddata_tes)
    
    ## second phase load the relation table of known examples
    xdata=[[],[],[]]    ## row index, column index , value

    ## ndata=self.nobject**2
    idata=0
    ifixtrain=[]
    ifixtest=[]
    for iobject1,dobject2 in self.ddata_tra.items():
      for iobject2,dvalues in dobject2.items():
        for ifeature,svalue in dvalues.items():
          if len(svalue)>0:
            if self.irowcol==0: 
              xdata[0].append(iobject1)
              ## xdata[1].append(ifeature*self.nobject+iobject2)
              xdata[1].append(ifeature+self.nfeature*iobject2)
              xdata[2].append(float(svalue))
            elif self.irowcol==1:
              ## xdata[0].append(ifeature*self.nobject+iobject1)
              xdata[0].append(ifeature+self.nfeature*iobject1)
              xdata[1].append(iobject2)
              xdata[2].append(float(svalue))
            ifixtrain.append(idata)
            idata+=1
          else:
            svalue=self.ddata_tes[iobject1][iobject2][ifeature]
            if len(svalue)>0:
              fvalue=float(svalue)
            else:
              fvalue=0
            if self.irowcol==0: 
              xdata[0].append(iobject1)
              ## xdata[1].append(ifeature*self.nobject+iobject2)
              xdata[1].append(ifeature+self.nfeature*iobject2)
              xdata[2].append(fvalue)
            elif self.irowcol==1:
              ## xdata[0].append(ifeature*self.nobject+iobject1)
              xdata[0].append(ifeature+self.nfeature*iobject1)
              xdata[1].append(iobject2)
              xdata[2].append(fvalue)
            ifixtest.append(idata)
            idata+=1

    if self.irowcol==0:
      self.nrow=self.nobject
      self.ncol=self.nobject*self.nfeature
    elif self.irowcol==1:
      self.nrow=self.nobject*self.nfeature
      self.ncol=self.nobject

    xdata=mvm_prepare.sort_table(xdata,ifloat=0)

    ifixtrain=np.array(ifixtrain)
    ifixtest=np.array(ifixtest)

    return(xdata,self.nrow,self.ncol,ifixtrain,ifixtest)

  ## -----------------------------------------------
  def collect_objects(self,sfile):
    """
    collect objects from text file and stored in the dictionary self.dobject
    object string -> object index

    In the same time the inverse dictionary is created 
    object index -> object string
    to recover the object names
    """
    
    with open(self.sbasedir+sfile) as infile:
      csv_reader = csv.reader(infile, delimiter=self.csv_delimiter)
      ifirst=self.headerline
      iobject=0
      for line in csv_reader:
        if ifirst>0:
          ifirst-=1
          continue
        if len(line)==0:
          continue
        sobject=line[0]
        sobject=sobject[1:-1] ## remove '', of the ascii string delimiters
        if sobject not in self.dobject:
          self.dobject[sobject]=iobject
          self.dobject_inv[iobject]=sobject
          iobject+=1
    infile.close()

    self.nobject=iobject

    return
  ## -----------------------------------------------
  def collect_relations(self,sfile,ddata):
    """
    collect feature values
    the tuples (object1,object2,list_features) is loaded into a dictionary:
      ddata[object1 index][object2 index]=[feture values] as string
    """
    ## second phase load the relation table
    with open(self.sbasedir+sfile) as infile:
      csv_reader = csv.reader(infile, delimiter=self.csv_delimiter)
      ifirst=self.headerline
      for line in csv_reader:
        if ifirst>0:
          ifirst-=1
          continue
        if len(line)==0:
          continue
        sobject=line[0]
        sobject=sobject[1:-1]
        iobject1=self.dobject[sobject]
        sobject=line[1]
        sobject=sobject[1:-1]
        iobject2=self.dobject[sobject]
        if iobject1 not in ddata:
          ddata[iobject1]={}
        if iobject2 not in ddata[iobject1]:
          ddata[iobject1][iobject2]={}
        for ifeature in range(self.nfeature):
          iposition=ifeature+self.feature_orig
          svalue=line[iposition]
          if ifeature not in ddata[iobject1][iobject2]:
            ddata[iobject1][iobject2][ifeature]=svalue
          else:
            print('Repeated object1,object2 pair!!!!')
          
    infile.close()
        
    return
  ## ----------------------------------------------
  def export_full_prediction(self,Zrow):
    """
    Zrow is loaded with continous values of the relation array
    """

    if self.irowcol==0:
      nrow=self.nobject
      ncol=self.naction
    elif self.irowcol==1:
      ncol=self.naction
      nrow=self.nobject
    
    fout=open(self.sbasedir+self.full_predicted_data, 'w')
    for irow in range(nrow):
      for icol in range(ncol):
        if self.irowcol==0:
          sline='"'+self.dobject_inv[irow]+'"'+',' \
                 +'"'+self.daction_inv[icol]+'"'
        elif self.irowcol==1:
          sline='"'+self.dobject_inv[icol]+'"'+',' \
                 +'"'+self.daction_inv[irow]+'"'
        vpred=Zrow[irow][0][icol]
        sline+=','+'"'+str('%7.5f'%vpred)+'"'
        
        fout.write(sline+'\n')
    fout.close()

    return
  ## ----------------------------------------------
  def export_test_prediction(self,filename,xdatacls,Zrow):
    """
    the prediction is exported into "filename"
    The predicted values is extended with confidence too
    """

    Y0=xdatacls.Y0
    if xdatacls.testontrain==0:  ## predicting test   
      xranges_tes=xdatacls.xranges_rel_test
      xdata_tes=xdatacls.xdata_tes
      xranges_tra=xdatacls.xranges_rel
      xdata_tra=xdatacls.xdata_tra
    else:   ## predicting training
      xranges_tes=xdatacls.xranges_rel
      xdata_tes=xdatacls.xdata_tra
      xranges_tra=xdatacls.xranges_rel
      xdata_tra=xdatacls.xdata_tra

    xdata=np.zeros((self.nrow,self.ncol,2))

    xdata_rel=xdatacls.xdata_rel
    ntrain=xdatacls.itrain.shape[0]
    for i in range(ntrain):
      idata=xdatacls.itrain[i]
      irow=xdata_rel[0][idata]
      icol=xdata_rel[1][idata]
      vval=xdata_rel[2][idata]
      xdata[irow,icol,0]=vval
      xdata[irow,icol,1]=1.0
          
    ## load test items as predictions
    for irow in range(self.nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        ## the indexes of test items 
        xcol=xdata_tes[1][istart_tes:istart_tes+nlength_tes]
        for i in range(nlength_tes):
          icol=xcol[i]
          vpred=Zrow[irow][0][i]
          vconf=Zrow[irow][2][i]
          vpred=Y0[np.abs(Y0-vpred).argmin()]
          ## confidence predicted via exponential distribution
          vconf=1-np.exp(-xdatacls.confidence_scale*np.abs(vconf-0.5))
          xdata[irow,icol,0]=vpred
          xdata[irow,icol,1]=vconf

    ## export xdata into test file
    #fout=open(self.sbasedir+filename, 'w')
    fout=open(self.sbasedir+self.listout, 'w')
    #print('<<<   writiting results in file')
    fout.write("'object1','object2',"+'\n')
    if self.irowcol==0:
      for irow in range(self.nrow):
        iobject1=irow
        for iobject2 in range(self.nobject):
          sobject1=self.dobject_inv[iobject1]
          sobject2=self.dobject_inv[iobject2]
          sline="'"+sobject1+"'"+','+"'"+sobject2+"'"
          ## predictions
          for ifeature in range(self.nfeature):
            icol=ifeature+self.nfeature*iobject2
            vpred=xdata[irow,icol,0]
            sline+=','+str(int(vpred))
          ## confidences
          for ifeature in range(self.nfeature):
            icol=ifeature+self.nfeature*iobject2
            vconf=xdata[irow,icol,1]
            sline+=','+str('%6.4f'%vconf)
          fout.write(sline+'\n')
    elif self.irowcol==1:
      for iobject1 in range(self.nobject):
        for iobject2 in range(self.nobject):
          sobject1=self.dobject_inv[iobject1]
          sobject2=self.dobject_inv[iobject2]
          sline="'"+sobject1+"'"+','+"'"+sobject2+"'"
          ## predictions
          for ifeature in range(self.nfeature):
            irow=ifeature+self.nfeature*iobject1
            icol=iobject2
            vpred=xdata[irow,icol,0]
            sline+=','+str(int(vpred))
          ## confidences
          for ifeature in range(self.nfeature):
            icol=ifeature+self.nfeature*iobject1
            icol=iobject2
            vconf=xdata[irow,icol,1]
            sline+=','+str('%6.4f'%vconf)
          fout.write(sline+'\n')
       
    fout.close()
       
    fout.close()
        
    return
  ## ----------------------------------------------
  def full_test(self):

    ndata=self.nrow*self.ncol
    npart=3
    xdata_tes=[ np.zeros(ndata,dtype=int) for i in range(npart)]
    k=0
    for i in range(self.nrow):
      for j in range(self.ncol):
        xdata_tes[0][k]=i
        xdata_tes[1][k]=j
        k+=1
        
    return(xdata_tes)

  ## --------------------------------------------------
  def order_dict(self,ddictin):

    lkeys=list(ddictin.keys())
    lkeys.sort()

    ddictout={}
    iindex=0
    for key in lkeys:
      ddictout[key]=iindex
      iindex+=1

    return(ddictout)
  ## --------------------------------------------------
  def invert_dict(self,ddictin):

    ddictout={}
    for key,val in ddictin.items():
      if val not in ddictout:
        ddictout[val]=key

    return(ddictout)
  ## --------------------------------------------------
  def export_prediction(self,filename,xdatacls,ZrowT):

    Y0=xdatacls.Y0
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


    pred_list=[]
    value_list=[]
    for irow in range(nrow):
      if xranges_tes[irow,1]>0:
        istart_tes=xranges_tes[irow,0]
        nlength_tes=xranges_tes[irow,1]
        for i in range(nlength_tes):
          ii=istart_tes+i
          vpred=ZrowT[irow][0][i]
          vconf=ZrowT[irow][2][i]
          if self.irowcol==0:
            iobject1=irow
            icol=xdata_tes[1][ii]
            ifeature=icol//self.nobject
            iobject2=icol % self.nobject
          else:
            ifeature=irow//self.nobject
            iobject2=irow % self.nobject
            iobject2=xdata_tes[1][ii]
            
          vpred=Y0[np.abs(Y0-vpred).argmin()]
          vconf=1-np.exp(-xdatacls.confidence_scale*np.abs(vconf-0.5))

          sobject1=self.dobject_inv[iobject1]
          sobject2=self.dobject_inv[iobject2]
          ## confidence assumes normal_distribution(0,1)
          ## vconf=scipy.stats.norm.cdf(vconf, \
          ##                            loc=xdatacls.confidence_local, \
          ##                            scale=xdatacls.confidence_scale)
          pred_list.append([sobject1,sobject2,ifeature,vpred,vconf])

    xvalue=np.array(value_list)
    pred_list.sort()
    
    fout=open(filename,'w')
    fout.write('# Predicted relations\n')
    fout.write('# Columns: object, object, feature, score, confidence \n')
    fout.write('\n')
    
    for (sobject1,sobject2,ifeature,vpred,vconf) in pred_list:
      sline='"'+sobject1+'"'+','
      sline+='"'+sobject2+'"'+','
      sline+=str(ifeature)+','
      sline+=str(vpred)+','
      sline+=str(vconf)
      fout.write(sline+'\n')
    fout.close()

    return
## ##################################################


  
