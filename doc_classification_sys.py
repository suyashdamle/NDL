from itertools import izip
import array
import json
import math
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer

class DdcClassifier():


	def __init__(self,characteristics='title'):

		if('title' not in characteristics) and ('abstract' not in characteristics) and ('table_of_contents' not in characteristics):
			raise ValueError("Invalid sequence passed as 'characteristics' parameter")
		if 'title' in characteristics and 'abstract' in characteristics and 'table_of_contents' in characteristics:
			self.param='title_contents_abs'
		elif 'title' in characteristics and 'abstract' in characteristics and 'table_of_contents'  not in characteristics:
			self.param='title_abs'
		elif 'title' in characteristics and 'abstract' not in characteristics and 'table_of_contents' in characteristics:
			self.param='title_contents'
		else:
			self.param='title'

		self.vocab_org,self.vocab_final=self.__get_vocab(self.param)
		self.clfs=self.__get_classifiers(self.param)                                               #list of classifiers


	def __check_vectorize(self,data_initial,characteristics):

		for entry in data_initial:
			if len(entry)!=3:
				raise ValueError("The provided data must consist of 3-tuples : (title,abstract,table of contents). Some of them could be None or empty string if unavailable")
		#if ('title' in characteristics and title is None) or ('abstract' in characteristics and abstract is None) or ('table_of_contents' in characteristics and table_of_contents is None):
		#raise ValueError("A characteristic mentioned in 'characteristics' parameter must be passed as a non - None string")
		
		data=[]
		for entry in data_initial:
			title,abstract,table_of_contents=entry
			data_here=''
			if title is not None and 'title' in characteristics:
				title_new=''
				for word in title.strip().split(' '):
					title_new+=(' t__'+word)
				title=title_new
				data_here=data_here+' '+title

			if abstract is not None and 'abstract' in characteristics:
				abstract_new=''
				for word in abstract.strip().split(' '):
					abstract_new+=(' a__'+word)
				abstract=abstract_new
				data_here=data_here+' '+abstract

			if table_of_contents is not None and 'table_of_contents' in characteristics:
				contents_new=''
				for word in table_of_contents.strip().split(' '):
					contents_new+=(' c__'+word)
				table_of_contents=contents_new
				data_here=data_here+' '+table_of_contents
			data.append(data_here)

		vec=CountVectorizer()
		counts=vec.fit_transform(data)
		features=vec.get_feature_names()
		return counts,features


	def __get_vocab(self,param):
		f=open('vocab_'+param+'.txt')
		vocab_org=json.load(f)
		vocab_final=joblib.load('vocab_final_'+param+'.pkl')
		return vocab_org,vocab_final

	def __get_classifiers(self,param):
		clfs=[]
		for level in range(1,4):
			clf="trained_clf_"+param+"_"+str(level)+".pkl"
			clfs.append(joblib.load(clf))              #'trained_clf/'+
		return clfs

	def __get_classification(self,counts,features,vocab_org,vocab_final,clfs,param, number_of_predictions, defined_minimum_confidence):
		if number_of_predictions is None and defined_minimum_confidence is None:       #defining defaults for successive levels
			if param =='title':
				number_of_predictions=[2,2,2]
				defined_minimum_confidence=[0,0,-2]
			if param == 'title_abs':
				number_of_predictions=[2,2,2]
				defined_minimum_confidence=[1,0.05,-2]
			if param == 'title_contents_abs':
				number_of_predictions=[2,2,2]
				defined_minimum_confidence=[0.05,0.05,-2]
			if param == 'title_contents':
				number_of_predictions=[2,2,2]
				defined_minimum_confidence=[0.05,0.05,-2]
				
		if (number_of_predictions is None and defined_minimum_confidence is not None) or (number_of_predictions is not None and defined_minimum_confidence is None) :
			raise ValueError("Either both 'number_of_predictions' and 'defined_minimum_confidence' should be specified or neither of them")

		coo_count=counts.tocoo()                 #convering to COO matrix
		row_temp=[]
		col_temp=[]
		data_temp=[]
		for row,col,countdata in izip(coo_count.row,coo_count.col,coo_count.data):
			feature=features[col]
			if feature in vocab_org:
				row_temp.append(row)
				col_temp.append(vocab_org[feature])
				data_temp.append(float(countdata)*vocab_final[vocab_org[feature]][0])

		row_temp=np.array(row_temp,dtype=np.int)
		col_temp=np.array(col_temp,dtype=np.int)
		data_temp=np.array(data_temp,dtype=float)
		X=csr_matrix((data_temp, (row_temp, col_temp)), shape=(counts.shape[0], len(vocab_org)))
		mat0=clfs[0].decision_function(X)
		mat1=clfs[1].decision_function(X)
		mat2=clfs[2].decision_function(X)
		if (mat0.shape[0]!=mat1.shape[0] or mat0.shape[0]!=mat2.shape[0]):
			print "Error: Internal inconsistency spotted"
			exit()
		first_level=[]
		sec_level=[]
		third_level=[]
		#for level 1
		mat_indices=np.argsort(mat0,axis=1)        #sorting the indices along the classification axis and taking top few
		prediction_level=(mat0>defined_minimum_confidence[0]).sum(axis=1)
		for i in range(mat0.shape[0]):
			pred_level=prediction_level[i]
			if number_of_predictions[0]>pred_level:
				temp_array=mat_indices[i][::-1][:pred_level]
			else:
				temp_array=mat_indices[i][::-1][:number_of_predictions[0]]
			first_level.append(temp_array)


		#for level 2
		mat_indices=np.argsort(mat1,axis=1)        #sorting the indices along the classification axis and taking top few
		prediction_level=(mat1>defined_minimum_confidence[1]).sum(axis=1)
		for i in range(mat1.shape[0]):
			pred_level=prediction_level[i]
			if number_of_predictions[1]>pred_level:
				temp_array=mat_indices[i][::-1][:pred_level]
			else:
				temp_array=mat_indices[i][::-1][:number_of_predictions[1]]
			sec_level.append(temp_array)

		#for level 3
		mat_indices=np.argsort(mat2,axis=1)        #sorting the indices along the classification axis and taking top few
		prediction_level=(mat2>defined_minimum_confidence[2]).sum(axis=1)
		for i in range(mat2.shape[0]):
			pred_level=prediction_level[i]
			if number_of_predictions[2]>pred_level:
				temp_array=mat_indices[i][::-1][:pred_level]
			else:
				temp_array=mat_indices[i][::-1][:number_of_predictions[2]]
			third_level.append(temp_array)


		final_classification=[]
		for i in range(mat0.shape[0]):
			temp_classification=[]
			level1=first_level[i].tolist()
			level2=sec_level[i].tolist()
			level3=third_level[i].tolist()
			for pred in level3:
				if (pred/100) in level1 and (pred/10) in level2:
					temp_classification.append((pred/100,pred/10,pred))
					del level1[level1.index(pred/100)]
					del level2[level2.index(pred/10)]
			for pred in level2:
				if (pred/10) in level1:
					temp_classification.append((pred/10,pred,None))
					del level1[level1.index(pred/10)]
			for pred in level1:
					temp_classification.append((pred,None,None))
			final_classification.append(temp_classification)
		return final_classification

	'''Takes a LIST of tuple of 3 characteristics - (title,abstract,table_of_contents) IN THAT ORDER. One of these characterisitics that is not available or not required can be replaced by None or empty string '''
	def get_ddc(self,data_initial, number_of_predictions=None, defined_min_confidence=None):
		counts,features=self.__check_vectorize(data_initial, self.param)
		classifications=self.__get_classification(counts,features,self.vocab_org,self.vocab_final,self.clfs,self.param,number_of_predictions,defined_min_confidence) # list of 3 lists for each level of classes
		return classifications
