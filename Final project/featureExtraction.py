#!/usr/bin/env python3
from __future__ import print_function
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import nibabel as nib
#get_ipython().magic('matplotlib inline')
import csv
from tqdm import tqdm_notebook as tqdm
import shutil
import gzip
import pandas as pd  
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from numpy import nan
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from tools.lasotools import feature_extraction
from tools.lasotools import report
from tools.lasotools import load_nii
from tools.feature_selection import DT
from tools.feature_selection import RFECV
from tools.feature_selection import RF
from tools.feature_selection import LR
from tools.feature_selection import PCA


report('#####################################################################')
report('#####################################################################')
report('#####################################################################')
report('libraries imported')
report('#####################################################################')
report('#####################################################################')


# ### Reading .CSV file

try:
	ROOT= os.getcwd()
	findings= []
	findings_seq_num=[]
	with open(os.path.join(ROOT,'PROSTATEx_Classes.csv'), newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in spamreader:
			findings.append(', '.join(row))
	for row in range(len(findings)):
		findings[row]= findings[row].split(',')


	print(len(findings))
	findings.pop(0) # remove first row (header)
	print(len(findings))
	print(findings[-1]) # print to check
	report(str(len(findings))+' findings!')
	findings=findings[:] 

	# In[165]:


	df = pd.read_csv(os.path.join(ROOT,'PROSTATEx_Classes.csv'))
	df.dtypes


	# In[166]:


	obj_df = df.select_dtypes(include=['boolean']).copy()
	obj_df.head()


	# In[167]:


	cleanup_nums = {"Clinically Significant":     {True: 1, False: 0}}
	obj_df.replace(cleanup_nums, inplace=True)
	obj_df.head()


	# In[174]:


	id_finding=[]
	label_finding=[]
	gleason_finding=[]

	for row in range(len(findings)):
		id_finding=list(df['ID'])
	id_finding=id_finding
	#print(df['ID'].value_counts())
	for row in range(len(findings)):
		label_finding=list(obj_df['Clinically Significant'])
	label_finding=label_finding[:]
	print(df['Clinically Significant'].value_counts(),'\n')
	for row in range(len(findings)):
		gleason_finding=list(df['Gleason Grade Group'])
	gleason_finding=gleason_finding
	print(df['Gleason Grade Group'].value_counts())



	# ### Radiomics
	'''
	report('#####################################################################')
	report('########################### RADIOMCS ################################')
	
	feature_extraction(masks_path= os.path.join(ROOT,'data','Masks','T2'),
		nii_path= os.path.join(ROOT,'data','nii','T2'),
		labels= label_finding,
		params_file= 'params.yaml',
		output_filename= 'Features.csv')
		
	report('#####################################################################')
	report('#####################################################################')
	report('#####################################################################')
	'''

	# ### SVM classifier

	# In[177]:

	
	report('#####################################################################')
	report('#####################################################################')
	labels=np.array(label_finding)
	report('#labels: '+str(np.shape(labels)))

	# In[178]:


	df_features=pd.read_csv('Features.csv', sep=',',header=0) # remove first line: 1>> header
	df_features=df_features[:]
	report('#training data: '+str(np.shape(df_features.values)))
	


	train_data, test_data, train_labels, test_labels = train_test_split(df_features, labels, test_size = 0.3, random_state=2)

	# Correcting NaN values: 
	values = train_data
	imputer = SimpleImputer(missing_values=nan, strategy='mean')
	# transform the dataset
	train_data = imputer.fit_transform(values)
	
	
	# Correcting NaN values: 
	values = test_data
	imputer = SimpleImputer(missing_values=nan, strategy='mean')
	# transform the dataset
	test_data = imputer.fit_transform(values)



	clf = svm.SVC()
	clf.fit(train_data, train_labels)
	report('dimensions fit')
	report('#####################################################################')
	report('#####################################################################')
	report('#####################################################################')


	# linear SVM kernel
	'''
	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}  
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(train_data, train_labels)
	
	report("The best all_features_in_linear-SVM parameters are %s with a score of %0.2f"
		  % (grid.best_params_, grid.best_score_))

	clf = svm.SVC(**grid.best_params_, cache_size=200, class_weight=None, coef0=0.0, degree=3,
	  max_iter=-1, probability=True, random_state=None, shrinking=True,
	  tol=0.001, verbose=False)
	clf.fit(train_data, train_labels)
	'''
	
	# rbf SVM kernel
	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(train_data, train_labels)

	report("The best all_features_in_rbf-SVM parameters are %s with a score of %0.2f"
		  % (grid.best_params_, grid.best_score_))
		  
	clf = svm.SVC(**grid.best_params_, cache_size=200, class_weight=None, coef0=0.0, degree=3,
	  max_iter=-1, probability=True, random_state=None, shrinking=True,
	  tol=0.001, verbose=False)
	clf.fit(train_data, train_labels)
		
	
	# poly SVM kernel
	'''
	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['poly']}  
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(train_data, train_labels)

	report("The best all_features_in_poly-SVM parameters are %s with a score of %0.2f"
		  % (grid.best_params_, grid.best_score_))

	clf = svm.SVC(**grid.best_params_, cache_size=200, class_weight=None, coef0=0.0, degree=3,
	  max_iter=-1, probability=True, random_state=None, shrinking=True,
	  tol=0.001, verbose=False)
	clf.fit(train_data, train_labels)
	'''
	
	report('all_features_in-SVM classifier(s) trained!')
	report('#####################################################################')





	# ### Feature Selection
	
	report('#####################################################################')
	report('Atempting FEATURE SELECTION...')
	report('#####################################################################')
	
	
	X= train_data
	y= train_labels
	
	
	# image folder
	try:
		shutil.rmtree(os.path.join(ROOT, 'plots'))
	except Exception:
		pass
	os.mkdir(os.path.join(ROOT, 'plots'))




	#### RFECV rbf-kernel SVM
	
	report('#####################################################################')
	RFECV_valid_indices= RFECV(X,y)
	report('#####################################################################')




	# #### Random Forest
	
	X= train_data
	y= train_labels
	
	report('########################### RF ######################################')
	RF_valid_indices= [0, 5, 11, 12, 13, 15, 21]#RF(X,y)
	report('#####################################################################')
	
	
	
	
	# ### Decion Tree
	
	X= train_data
	y= train_labels
	
	report('########################### DT ######################################')
	DT_valid_indices= DT(X,y)
	report('#####################################################################')
	
	
	

	# ### Logistic Regression
	
	X= train_data
	y= train_labels

	report('########################### LR ######################################')
	LR_valid_indices= LR(X,y)
	report('#####################################################################')






	# ### Validation

	report('#####################################################################')
	report('#####################################################################')
	report('Attempting Validation...')
	report('#####################################################################')



	# RFECV
	
	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X[:,RFECV_valid_indices], y)

	report("The best RFECV SVM parameters are %s with a score of %0.2f"
		  % (grid.best_params_, grid.best_score_))

	mean_results=cross_val_score(estimator=svm.SVC(**grid.best_params_),X=X[:,RFECV_valid_indices],y=y,scoring='accuracy')
	report('RFECV results: >mean: '+str(mean_results.mean())+', >std: '+str(mean_results.std()))
	
	model= svm.SVC(**grid.best_params_)
	model.fit(X[:,RFECV_valid_indices],y)
	RFECV_predictions = model.predict(test_data[:,RFECV_valid_indices])
	report('RFECV accuracy: '+str(accuracy_score(RFECV_predictions, test_labels)))
	


	# RF

	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X[:,RF_valid_indices], y)

	report("The best RF SVM parameters are %s with a score of %0.2f"
		  % (grid.best_params_, grid.best_score_))

	mean_results=cross_val_score(estimator=svm.SVC(**grid.best_params_),X=X[:,RF_valid_indices],y=y,scoring='accuracy')
	report('RF results: >mean: '+str(mean_results.mean())+', >std: '+str(mean_results.std()))

	model= svm.SVC(**grid.best_params_)
	print(RF_valid_indices)
	model.fit(X[:,RF_valid_indices],y)
	RF_predictions = model.predict(test_data[:,RF_valid_indices])
	report('accuracy: '+str(accuracy_score(RF_predictions, test_labels)))
	

	

	# DT

	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X[:,DT_valid_indices], y)

	report("The best DT SVM parameters are %s with a score of %0.2f"
		  % (grid.best_params_, grid.best_score_))

	mean_results=cross_val_score(estimator=svm.SVC(**grid.best_params_),X=X[:,DT_valid_indices],y=y,scoring='accuracy')
	report('DT results: >mean: '+str(mean_results.mean())+', >std: '+str(mean_results.std()))

	model= svm.SVC(**grid.best_params_)
	model.fit(X[:,DT_valid_indices],y)
	DT_predictions = model.predict(test_data[:,DT_valid_indices])
	report('DT accuracy: '+str(accuracy_score(DT_predictions, test_labels)))

	
	
	try:
		# LR
		param_grid = {'C': [0.1, 1, 10, 100, 1000],  
				  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
				  'kernel': ['rbf']}  
		cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
		grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
		grid.fit(X[:,LR_valid_indices], y)

		report("The best LR SVM parameters are %s with a score of %0.2f"
			  % (grid.best_params_, grid.best_score_))

		mean_results=cross_val_score(estimator=svm.SVC(**grid.best_params_),X=X[:,LR_valid_indices],y=y,scoring='accuracy')
		report('LR: >mean: '+str(mean_results.mean())+', >std: '+str(mean_results.std()))
		
		model= svm.SVC(**grid.best_params_)
		model.fit(X[:,LR_valid_indices],y)
		LR_predictions = model.predict(test_data[:,LR_valid_indices])
		report('LR accuracy: '+str(accuracy_score(LR_predictions, test_labels)))
	except Exception:
		report('LR validation skipped')


	
	
	PCA(df_features,labels)
	

	#    
	# ---
	# ##### HRL
	# ---
	report('#####################################################################')
	report('#####################################################################')
	report('all run sucessfully!')
	report('#####################################################################')
	report('#####################################################################')
	report('#####################################################################')

except Exception as e:
	report('#####################################################################')
	report('#####################################################################')
	report('FATAL ERROR OCURRED while running the HRLscript!!!!!!\n'+str(e))
	report('#####################################################################')
	report('#####################################################################')
	report('#####################################################################')