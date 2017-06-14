from itertools import izip
import sqlite3
import array
import json
import math
import numpy as np

conn = sqlite3.connect('database_title_both.db')
cursor=conn.cursor()
train_target=None


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
        if(vocab[item][idf_idx]==0):
            to_delete.append(item)                   #these words must have been in test set alone
            continue
        vocab[item][idf_idx]=math.log(float(total_docs)/vocab[item][idf_idx], 10)
        
    for item in to_delete:
        del(vocab[item])
    return vocab

def class_priors(class_data, total_docs=0, doc_ids=None, classifications=None):#NO defaultdict here in any case
    if(doc_ids and classifications):
        for doc_id, classification in izip(doc_ids,classifications):
            class_data[int(classification)]+=1
            total_docs+=1
    else:
        for doc_id, classification in get_data('train_target'): 
            class_data[int(classification)]+=1
            total_docs+=1
    return class_data, total_docs
            
def class_priors_final(class_data, total_docs):
    for idx,item  in enumerate(class_data):
        class_data[idx]=float(item)/total_docs
    return class_data


# (pass-2) functions:
# initialize counts with 1
           
        
def train_model(vocab,denm, idf_idx, features=None, counts=None, classifications=None):
    if ( features and classifications and count):
        for feature, classification, data in izip(features, classifications,counts):
            vocab[feature][int(classification)]+=vocab[feature][idf_idx]*data
            denm[int(classification)]+=vocab[feature][idf_idx]*data

    else:
        for feature, classification, data in get_data('train_data_target'):            #complete this
            vocab[feature][int(classification)]+=vocab[feature][idf_idx]*data
            denm[int(classification)]+=vocab[feature][idf_idx]*data
    return vocab, denm

def train_model_final(vocab,denm,vocab_length):
    for item in vocab.iterkeys():
        for i in range(0,10):
            vocab[item][i]=vocab[item][i]/(denm[i]+vocab_length)  #***IMP:create float  arrays as values of the dict
    return vocab

def predict(class_data, idf_idx, vocab):
    predicted=array.array('i')
    prev_doc_id=None
    doc_feature_count={}
    for doc_id, feature, count in get_data('test_data'):         #compelete this
        if(doc_id==prev_doc_id or prev_doc_id==None):
            doc_feature_count[feature]=count    # depends on the same-doc_id entries coming in succession
            prev_doc_id=doc_id
        else:
            predicted.append(predict_class(doc_feature_count,vocab, class_data, idf_idx))            #storing as characters [doc_id]
            doc_feature_count={}
            doc_feature_count[feature]=count
            if not(len(predicted)==prev_doc_id+1):
                print "Inconsistency spotted...Exiting---function predict()"
                print "doc_id=", doc_id
                print "length of predictions=", len(predicted)
                exit()
            prev_doc_id=doc_id
                    
    if not (len(doc_feature_count)==0):        
        predicted.append(predict_class(doc_feature_count,vocab, class_data, idf_idx))
    if not(len(predicted)==doc_id+1):
        print "Inconsistency spotted...Exiting---function predict() at end of function"
        print "doc_id=", doc_id
        print "length of predictions=", len(predicted)
        exit()

    return predicted

def predict_class(doc_feature_count, vocab, class_data, idf_idx):
    class_posteriors=[0.0]*10
    for i in range (0, 10):
        if class_data[i]==0:
            class_posteriors[i]=float("-inf")
            continue
        class_posteriors[i]=math.log(class_data[i])
    for feature in doc_feature_count.iterkeys():
        if feature in vocab:
            for i in  range(0,10):
                class_posteriors[i]+=math.log(vocab[feature][i])*doc_feature_count[feature]*vocab[feature][idf_idx]      # prior prob x tf x idf
    max_value=max(class_posteriors)
    return class_posteriors.index(max_value)

def get_data(parameter):
    global train_target

    limit=6500000
    offset=0
    maxrows_train_data=58453354   #define value
    maxrows_test_data=  19492452    #define
    

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
                    yield [int(row[2]),classification, int(row[3])]
                    
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
            yield classification

    else:
        print "invlaid parameter:", parameter
        exit()

    
def main():
    idf_idx=10
    f=open('vocab_abs_both.txt')
    vocab_org=json.load(f)
    print "Dict loaded"
    vocab={}
    for word in vocab_org.iterkeys():
        vocab[vocab_org[word]]=array.array('f')
        for i in range (0, 10):
            vocab[vocab_org[word]].append(1.0)
        vocab[vocab_org[word]].append(0.0)                       #for idf
    del vocab_org
    class_data=[0.0]*10
    class_data, total_train=class_priors(class_data)
    total_train+=len(class_data)               #initializing by ones
    
    class_data=class_priors_final(class_data,total_train)
    print "class_data obtained"
    vocab= calc_idf(vocab,idf_idx)
    vocab= calc_idf_final(vocab,idf_idx,total_train)
    print "idf calculated"
    
    denm=[0.0]*10
    vocab_length=len(vocab)
    vocab,denm=train_model(vocab, denm,idf_idx)
    vocab=train_model_final(vocab, denm,vocab_length)

    from sklearn.externals import joblib
    import pickle

    #vocab=joblib.load("vocab_final.pkl")
    joblib.dump(vocab,"vocab_final.pkl")
    print "Training complete"
    #testing phase

    predictions=predict(class_data, idf_idx, vocab)
    #predictions=np.frombuffer(predictions,dtype=np.intc)
    #test_targets=array.array('s')
    #temp=[]
    targets=[]
    corr=0
    count=0
    flag=0
    for item in get_data('test_target'):
        '''added to account for ambiguous ddc'''
        i=predictions[count]
        count+=1
        values=[int(j) for j in  str(item).strip().split(' ')]
        if i in values:
            corr+=1
                
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
    from sklearn.metrics import confusion_matrix, accuracy_score
    from pandas import crosstab

    #function to map class labels to numerical values
    #test_targets=np.frombuffer(test_targets,dtype=np.intc)
    
    #score = custom_accuracy_score(targets, predictions)
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

