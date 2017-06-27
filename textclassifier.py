#param to change -classifier; ddc col header name and related operations to obtain last digit; db name; maxrows_train and test in get_data function; total no of training docs


from itertools import izip
import sqlite3
import array
import json
import math
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix

conn = sqlite3.connect('database_title_both.db')
cursor=conn.cursor()
train_target=None
total_train=60894

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



def get_data(parameter, limit=5000000):          #change when needed
    global train_target


    offset=0
    maxrows_train_data=603055   #define value
    maxrows_test_data=178612    #define
    

    if (train_target==None):          #creating at first call
        train_target=array.array('i')
        temp=[]
        dbtable='targetvector'
        command="SELECT * from "+dbtable
        cursor.execute(command)
        rows=cursor.fetchall()
        for row in rows:
            classification=int(row[1])
            temp.append(classification)
            if len(temp)>=1000000:
                train_target.extend(temp)
                if not(len(train_target)-1==row[0]):
                    print "exiting get_data"
                    print len(train_target), row[0]
                    exit()
                temp=[]
        if len(temp)>0:
            train_target.extend(temp)

    
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
                    yield [int(row[1]), int(row[2]),classification, int(row[3])]
                    
    elif(parameter=='train_target'):
        for i,j in enumerate(train_target):
            yield (i,j)
        
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
            classification=(row[1])        #int() removed to cater for ambiguous ddc case
            yield [row[0],classification]

    else:
        print "invlaid parameter:", parameter
        exit()

def textclassifier(vocab,idf_idx):
    #classes=array.array('i')
    classes=range(10)
    '''
    for i in range(10):
        classes.append(i)
    '''
    clf = MultinomialNB()            #svm.LinearSVC()
    features=[]
    docs=[]
    data=[]
    target=[]
    prev_doc=None
    row_count=0
    count=0                        # no. of rows of the db
    limit=500000              #change as needed
    last_class=None
    last_doc=None
    to_proceed=False
    i=0
    for doc,feature,classification,feature_count in get_data('train_data_target',limit=limit):
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
            if row_count>=0.5*limit:        #hoping that the last doc isnt more than 50% of the total number of rows obtained in one batch from the db
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
    print len(vocab)
    X=csr_matrix((data, (docs, features)), shape=(count+1, len(vocab)))           #complete this
    clf.partial_fit(X,target,classes=classes)      #
    return clf

def get_predictions(vocab,idf_idx,clf):
    limit=5000000
    predictions=array.array('i') 
    features=[]
    docs=[]
    data=[]
    prev_doc=None
    count=0                        # no. of rows of the db
    row_count=0

    for doc,feature,feature_count in get_data('test_data'):
        row_count+=1
        if prev_doc is None or prev_doc != doc:              #end of a document details detected
            if prev_doc is not None:
                count+=1
            prev_doc=doc
            if row_count>=0.5*limit:        #hoping that the last doc isnt more than 50% of the total number of rows obtained in one batch from the db
                docs=np.array(docs,dtype=np.int)
                features=np.array(features,dtype=np.int)
                data=np.array(data,dtype=float)
                X=csr_matrix((data, (docs, features)), shape=(count+1, len(vocab)))           #complete this
                predictions.extend(clf.predict(X))
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
    predictions.extend(clf.predict(X))
    return predictions


        

    
def main():
    idf_idx=0
    f=open('vocab_abs_both.txt')
    vocab_org=json.load(f)
    print "Dict loaded"
    feature_count=len(vocab_org)
    
    vocab={}
    max_vocab=0
    for word in vocab_org.iterkeys():
        vocab[vocab_org[word]]=array.array('f')
        if vocab_org[word]>max_vocab:
            max_vocab=vocab_org[word]
        vocab[vocab_org[word]].append(0.0)                       #for idf

    del vocab_org
    #class_data=[0.0]*10
    #class_data, total_train=class_priors(class_data)
    #total_train+=len(class_data)               #initializing by ones
    
    #class_data=class_priors_final(class_data,total_train)
    #print "class_data obtained"
    vocab= calc_idf(vocab,idf_idx)
    vocab= calc_idf_final(vocab,idf_idx,total_train)
    print "idf calculated"
    
    from sklearn.externals import joblib
    import pickle

    #vocab=joblib.load("vocab_final.pkl")
    joblib.dump(vocab,"vocab_final.pkl")
    trained_clf=textclassifier(vocab, idf_idx)
    predictions=get_predictions(vocab, idf_idx,trained_clf)
    
    
    targets=[]
    corr=0
    count=0
    flag=0
    correctly_done=[]
    #load previous correctely classified docs here and check, for each test set, whether it is needed to be checked
    for idx, item in get_data('test_target'):
        '''added to account for ambiguous ddc'''
        i=predictions[count]
        count+=1
        values=[int(j) for j in  str(item).strip().split(' ')]
        if i in values:
            corr+=1
            correctly_done.append(idx)
    score=float(corr)/count
    #item=int(item)
    #temp.append(item)
    '''
    if len(temp)>=1000000:
    test_targets.extend(temp)
    temp=[]
    if len(temp)>0:
    test_targets.extend(temp)
    '''
    #from sklearn.metrics import confusion_matrix, accuracy_score
    #from pandas import crosstab

    #function to map class labels to numerical values
    #test_targets=np.frombuffer(test_targets,dtype=np.intc)
    
    #score = custom_accuracy_score(targets, predictions)
    joblib.dump(correctly_done,"correctly_done.pkl")
    print "score=", score
    #print crosstab(test_targets, predictions, rownames=['True'], colnames=['Predicted'], margins=False)




def custom_accuracy_score(targets,predictions):
    corr=0
    count=0
    for i,j in izip(predictions,targets):
        print j,type(j)
        count+=1
        values=[int(j) for j in j.strip().split(' ')]
        if i in j:
          corr+=1  
    score=float(corr)/count
    return score

    
main()

