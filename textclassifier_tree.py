#param to change - classifier type; ddc col header name and related operations to obtain last digit; db name; maxrows_train and test in get_data function; total no of training docs; file names of to-be-saved files; get_classification function


from itertools import izip
import sqlite3
import array
import json
import math
import numpy as np
#from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.externals import joblib
import pickle
import csv
#from sklearn.naive_bayes import MultinomialNB


param='title'
details_file=open('details_'+param+'.txt')
details=details_file.next().strip().split(' ')
total_train=int(details[2])
details_output_file=None
writer=None

conn = sqlite3.connect('database_'+param+'.db')
cursor=conn.cursor()
train_target=None
train_level=None
level_to_train=3
number_of_predictions=3
defined_min_confidence=5


def get_classification(classification):
    if level_to_train==1:
        return int(classification)/100
    elif level_to_train==2:
        return int(classification/10)         #%10
    elif level_to_train==3:
        return int(classification)



def calc_idf(vocab,idf_idx, docs=None, data=None, features=None):#expects a dictionary with an array corresponding to each key;  NO defaultdict here in any case
    if(docs and features):
        for doc,feature in izip(docs, features):
            if data>0:
                vocab[feature][idf_idx]+=1
    else:
        for doc,feature,data in get_data('train_data'):
            if data>0:
                vocab[feature][idf_idx]+=1
    return vocab
        
def calc_idf_final(vocab,idf_idx, total_docs):
    to_delete=[]
    for item in vocab.iterkeys():
        if(float(vocab[item][idf_idx])==0.0):
            to_delete.append(item)                   #these words must have been in test set alone
            continue
        vocab[item][idf_idx]=math.log(float(total_docs),10)-math.log(float(vocab[item][idf_idx]), 10)

   
    return vocab



def get_data(parameter, limit=7000000000):          #change when needed
    global train_target
    global train_level

    offset=0
    maxrows_train_data=int(details[0])     #define value
    maxrows_test_data=  int(details[1])    #define
    

    if (train_target is None):          #creating at first call
        train_target=array.array('i')
        train_level=array.array('i')
        temp=[]
        temp_level=[]
        dbtable='targetvector'
        command="SELECT * from "+dbtable
        cursor.execute(command)
        rows=cursor.fetchall()
        for row in rows:
            classification=int(row[1])
            level=int(row[2])
            temp.append(get_classification(classification))
            temp_level.append(level)
            if len(temp)>=1000000:
                train_target.extend(temp)
                train_level.extend(temp_level)
                if not(len(train_target)-1==row[0]):
                    print "exiting get_data"
                    print len(train_target), row[0]
                    exit()
                temp=[]
        if len(temp)>0:
            train_target.extend(temp)
            train_level.extend(temp_level)

    
    if(parameter=='train_data'or parameter=='train_data_target'):
        dbtable='table1'
        count=0
        limit_temp=limit
        while count< maxrows_train_data:
            command="SELECT * from "+dbtable+" limit "+str(limit)+" offset "+str(offset)
            cursor.execute(command)
            offset+=limit_temp
            #count+=limit_temp
            #limit*=2
            rows=cursor.fetchall()
            if (parameter=='train_data'):
                for row in rows:
                    count+=1
                    yield [int(row[1]),int(row[2]),int(row[3])]
            else:
                for row in rows:
                    count+=1
                    doc_id=int(row[1])
                    classification=train_target[doc_id]
                    yield [int(row[1]), int(row[2]),classification,train_level[doc_id] ,int(row[3])]
                    
    elif(parameter=='train_target'):
        for i,j,k in enumerate(izip(train_target,train_level)):
            yield (i,j,k)
        
    elif(parameter=='test_data'):
        dbtable='table1_test'
        count=0
        limit_temp=limit
        while count<maxrows_test_data:
            command="SELECT * from "+dbtable+" limit "+str(limit)+" offset "+str(offset)
            cursor.execute(command)
            offset+=limit_temp
            count+=limit_temp
            #limit*=2
            rows=cursor.fetchall()
            for row in rows:
                yield[int(row[1]),int(row[2]),int(row[3])]
    elif(parameter=='test_target'):
        dbtable='targetvector_test'
        command="SELECT * from "+dbtable
        cursor.execute(command)
        rows=cursor.fetchall()
        for row in rows:
            classification=str(row[1]).strip().split(' ')        #int() removed to cater for ambiguous ddc case
            classification_new=[get_classification(int(i)) for i in classification] 
            yield [row[0],classification_new,row[2]]

    else:
        print "invlaid parameter:", parameter
        exit()

def textclassifier(vocab,idf_idx):                    #edited for one-time run
    classes=[]
    clf = linear_model.SGDClassifier()                         # MultinomialNB()        #change as needed
    features=[]
    docs=[]
    data=[]
    target=[]
    prev_doc=None
    row_count=0
    count=0                        # no. of rows of the db
    limit=700000000               #change as needed
    last_class=None
    last_doc=None
    to_proceed=False
    for doc,feature,classification,classification_level,feature_count in get_data('train_data_target',limit=limit):    #,limit=limit
        if number_of_predictions > classification_level:           # not training on this data point if it is not classified at the training level
            continue
        if classification not in classes:
            classes.append(classification)
        last_doc=doc
        row_count+=1
        last_class=classification           #needs to be added at the end
        if feature_count==0:
            continue
        if prev_doc is None or prev_doc != doc:              #end of a document details detected
            if prev_doc is not None:
                count+=1                        
            prev_doc=doc
            target.append(classification)
            if row_count>=0.7*limit:        #hoping that the last doc isnt more than 70% of the total number of rows obtained in one batch from the db
                to_proceed=True
        
        docs.append(count)
        last_doc=count
        features.append(feature)
        data.append(float(feature_count)*vocab[feature][idf_idx])               #tf-idf
        if to_proceed:

            docs=np.array(docs,dtype=np.int)
            features=np.array(features,dtype=np.int)
            data=np.array(data,dtype=float)
            target=np.array(target,dtype=np.int)
            X=csr_matrix((data, (docs, features)), shape=(count+1, len(vocab)))           #complete this
            clf.partial_fit(X,target,classes=classes)
            print 'clf called'
            prev_doc=None
            features=[]
            docs=[]
            data=[]
            target=[]
            count=0    # no. of rows of the db
            row_count=0
            to_proceed=False
            
    if docs[-1]!=last_doc:
        target.append(last_class)
    docs=np.array(docs,dtype=np.int)
    features=np.array(features,dtype=np.int)
    data=np.array(data,dtype=float)
    target=np.array(target,dtype=np.int)
    X=csr_matrix((data, (docs, features)), shape=(count+1, len(vocab))) 
    clf.partial_fit(X,target,classes=classes)      #  
    print 'clf called here'
    return clf 

def get_predictions(vocab,idf_idx,clf):
    limit=700000000
    features=[]
    docs=[]
    data=[]
    prev_doc=None
    count=0                        # no. of rows of the db
    row_count=0
    mat=None
    for doc,feature,feature_count in get_data('test_data',limit=limit):           #
        row_count+=1
        if prev_doc is None or prev_doc != doc:              #end of a document details detected
            if prev_doc is not None:
                count+=1
            prev_doc=doc
            if row_count>=0.7*limit:        #         hoping that the last doc isnt more than 70% of the total number of rows obtained in one batch from the db
                docs=np.array(docs,dtype=np.int)
                features=np.array(features,dtype=np.int)
                data=np.array(data,dtype=float)
                X=csr_matrix((data, (docs, features)), shape=(count+1, len(vocab)))
                if mat is None:
                    mat=clf.decision_function(X)
                else:
                    mat2=clf.decision_function(X)
                    mat=np.concatenate(mat,mat2)
                #predictions.extend(clf.predict(X))
                
                features=[]
                docs=[]
                data=[]
                count=0                        # no. of rows of the db
                row_count=0
                prev_doc=None

        docs.append(count)
        features.append(feature)
        if feature in vocab:
            data.append(float(feature_count)*vocab[feature][idf_idx])               #tf-idf
        else:
            data.append(0.0)

 
    docs=np.array(docs,dtype=np.int)
    features=np.array(features,dtype=np.int)
    data=np.array(data,dtype=float)

    
    X=csr_matrix((data, (docs, features)), shape=(count+1, len(vocab)))           #complete this
    #predictions.extend(clf.predict(X))
    if mat is None:
        mat=clf.decision_function(X)
    else:
        mat2=clf.decision_function(X)
        mat=np.concatenate(mat,mat2)

    print float(np.sort(mat,axis=1)[-1].sum(axis=0))/mat.shape[0]
    #joblib.dump(mat,"confidence_title.pkl")
    # now, finding multiple predictions for each observation based on confidence levels
    prediction_level=(mat>defined_min_confidence).sum(axis=1)
    mat_indices=np.argsort(mat,axis=1)        #sorting the indices along the classification axis and taking top few
    return mat_indices,prediction_level


        

    
def controller(first_time):
    idf_idx=0
    if first_time:
        f=open('vocab_'+param+'.txt')
        vocab_org=json.load(f)
        print "Dict loaded"
        feature_count=len(vocab_org)
    
        vocab={}
        for word in vocab_org.iterkeys():
            vocab[vocab_org[word]]=array.array('f')
            vocab[vocab_org[word]].append(0.0)                       #for idf

        del vocab_org
        vocab= calc_idf(vocab,idf_idx)
        vocab= calc_idf_final(vocab,idf_idx,total_train)
        print "idf calculated"
        joblib.dump(vocab,"vocab_final_"+param+".pkl")
    else:
        vocab=joblib.load("vocab_final_"+param+".pkl")
    
    trained_clf=textclassifier(vocab, idf_idx)
    predictions,prediction_level=get_predictions(vocab, idf_idx,trained_clf)
    total_test_cases=predictions.shape[0]

    # calculating errors and confusion matrix
    confusion_matrix=np.zeros(((10**level_to_train),(10**level_to_train)))
    targets=[]
    corr=0
    count=0
    flag=0
    not_classified=0
    rightly_left_unclassified=0
    for idx, item, level_of_classification in get_data('test_target'):
        to_count=True
        if level_to_train > level_of_classification:
            to_count=False

        if to_count:
            count+=1                                                   #score thus corresponds to "correct" out of ALL TESTS DOCS THAT ARE SUPPOSED TO BE CLASSIFIED AT THIS LEVEL
        doc_prediction,doc_prediction_level=predictions[idx],prediction_level[idx]
        if number_of_predictions>= doc_prediction_level:
            doc_prediction=doc_prediction[-doc_prediction_level:][::-1]
        else:
            doc_prediction=doc_prediction[-number_of_predictions:][::-1]

        if doc_prediction_level==0:
            not_classified+=1
            if not (to_count):                                        #it is not meant to be classified
                rightly_left_unclassified+=1
            continue

        if not to_count:
            continue

        #added to account for ambiguous ddc
        values=[int(j) for j in  item]
        for i in values:                                           # 'for all true classifications in the test set...'
            for j in doc_prediction:
                confusion_matrix[i][j]+=1.0/float(len(doc_prediction))
            if (i in doc_prediction):          #'....if the classification is also indicated by the classifier,...' 
                corr+=1.0/len(values)         #'...accuracy score increases in proportion to number of actual classifications- if real classes=2,3;classifier o/p= 2,5,6- score increases by 0.5'


    score=float(corr)/count                       # Denominator=
    print count
    joblib.dump(confusion_matrix,"confusion_matrix_tree_"+param+"_"+str(level_to_train)+"_"+str(defined_min_confidence)+"_"+str(number_of_predictions) +".pkl")
    print "score=", score
    print "precision", float(corr)/(count-not_classified)
    print "not_classified",not_classified,(not_classified*1.0)/(count+not_classified)  
    print "correctly left unclassified",(rightly_left_unclassified*1.0)/total_test_cases
    writer.writerow([param,level_to_train, number_of_predictions,defined_min_confidence,(not_classified*1.0)/(count+not_classified),(rightly_left_unclassified*1.0)/total_test_cases,score,float(corr)/(count-not_classified)])



    
def main():
    
    global defined_min_confidence
    global number_of_predictions
    global level_to_train
    global details_output_file
    global writer

    min_confidence_list=[-10,-5,-.05,0,0.5,1,5,10]
    train_levels=range(1,4)
    no_of_predictions=range(1,4)
    i=0
    
    details_output_file=open("details_exh_"+param+".csv",'w')
    writer=csv.writer(details_output_file)
    writer.writerow(['parameter','ddc_level','number of predictions','minimum confidence','percent left unclassified','correctly left unclassified','score','precision'])
    for train_level in train_levels:
        level_to_train=train_level
        for predictions in no_of_predictions:
            number_of_predictions=predictions
            for confidence in min_confidence_list:
                defined_min_confidence=confidence
                i+=1
                if i==1:
                    controller(True)
                else:
                    controller(False)

    
main()