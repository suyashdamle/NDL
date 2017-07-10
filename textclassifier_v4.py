#param to change - classifier type; ddc col header name and related operations to obtain last digit; db name; maxrows_train and test in get_data function; total no of training docs; file names of to-be-saved files; get_classification function


from itertools import izip,combinations
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


param=None
details=None
writer=None

conn = None
cursor=None
train_target=None
train_level=None
level_to_train=1
number_of_predictions=3
defined_min_confidence=-20.0
rows_of_test_target=None
class_freq_matrix=None
mat=None

def get_classification(classification):
    if level_to_train==1:
        return int(classification)/100
    elif level_to_train==2:
        return int(classification/10)%10
    elif level_to_train==3:
        return int(classification)%10



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



def get_data(parameter, limit=7000000000,train_new_clf=True):          #change when needed
    global train_target
    global train_level
    global rows_of_test_target
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
            temp.append(classification)
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
                    classification=get_classification(train_target[doc_id])
                    yield [int(row[1]), int(row[2]),classification,train_level[doc_id] ,int(row[3])]
                    
    elif(parameter=='train_target'):
        for i,j,k in enumerate(izip(train_target,train_level)):
            yield (i,get_classification(j),k)
        
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
        if not train_new_clf:
            for row in rows:
                classification=str(row[1]).strip().split(' ')        #int() removed to cater for ambiguous ddc case
                classification_new=[int(i) for i in classification]
                class_level= str(row[2]).strip().split(' ')
                yield [row[0],classification_new,class_level]


        if train_new_clf:
            dbtable='targetvector_test'
            command="SELECT * from "+dbtable
            cursor.execute(command)
            rows=cursor.fetchall()
            rows_of_test_target=rows
            for row in rows:
                classification=str(row[1]).strip().split(' ')        #int() removed to cater for ambiguous ddc case
                classification_new=[int(i) for i in classification]
                class_level= str(row[2]).strip().split(' ')
                yield [row[0],classification_new,class_level]

    else:
        print "invlaid parameter:", parameter
        exit()

def textclassifier(vocab,idf_idx):
    classes=range(10)
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

def get_predictions(vocab,idf_idx,clf,train_new_clf):
    global mat
    if not (train_new_clf):                       # if new classifier is not being trained
        prediction_level=(mat>defined_min_confidence).sum(axis=1)
        mat_indices=np.argsort(mat,axis=1)        #sorting the indices along the classification axis and taking top few
        mat_confidence=np.sort(mat,axis=1)
        return mat_indices,mat_confidence,prediction_level

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
    #joblib.dump(mat,"confidence_"+param+".pkl")
    # now, finding multiple predictions for each observation based on confidence levels
    prediction_level=(mat>defined_min_confidence).sum(axis=1)
    mat_indices=np.argsort(mat,axis=1)        #sorting the indices along the classification axis and taking top few
    mat_confidence=np.sort(mat,axis=1)
    return mat_indices,mat_confidence,prediction_level


        

    
def controller(first_time,train_new_clf):
    global trained_clf
    idf_idx=0
    total_train=int(details[2])

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

    trained_clf=None
    if train_new_clf:    
        trained_clf=textclassifier(vocab, idf_idx)

    predictions,prediction_confidence,prediction_level=get_predictions(vocab, idf_idx,trained_clf,train_new_clf )
    print predictions.shape
    previous_prediction=None
    if level_to_train>1:
        previous_prediction=joblib.load("prediction_"+param+"_"+str(level_to_train-1)+"_"+str(defined_min_confidence)+"_"+str(number_of_predictions) +".pkl")

    present_prediction=np.full((predictions.shape[0],number_of_predictions,2),-1)

    # calculating errors and confusion matrix
    confusion_matrix=np.zeros(((10**level_to_train),(10**level_to_train)),dtype=np.float)

    targets=[]
    corr=0
    count=1
    flag=0
    not_classified=0
    correctly_done=set()
    if level_to_train>1:
        flag=1
    if flag==1:
        pre=joblib.load("correctly_done_"+param+"_"+str(level_to_train-1)+"_"+str(defined_min_confidence)+"_"+str(number_of_predictions) +".pkl")

    for idx, item, level_of_classification in get_data('test_target'):
        original_classification=item
        values=[get_classification(int(j)) for j in  item]
        class_level=[int(j) for j in  level_of_classification]
        item=values
        level_of_classification=class_level

        item_new=[]
        level_of_classification_new=[]
        for i,j in zip(item,level_of_classification):
            if j>=level_to_train:
                item_new.append(i)
                level_of_classification_new.append(j)
        item=item_new
        level_of_classification=level_of_classification_new
        if len(item)==0:
            continue

        count+=1                                                    #score thus corresponds to "correct" out of ALL TESTS DOCS THAT ARE SUPPOSED TO BE CLASSIFIED AT THIS LEVEL
        doc_prediction,doc_prediction_level,doc_prediction_confidence=predictions[idx],prediction_level[idx],prediction_confidence[idx]
        if number_of_predictions>= doc_prediction_level:
            doc_prediction=doc_prediction[-doc_prediction_level:][::-1]
            doc_prediction_confidence=doc_prediction_confidence[-doc_prediction_level:][::-1]
        else:
            doc_prediction=doc_prediction[-number_of_predictions:][::-1]
            doc_prediction_confidence=doc_prediction_confidence[-number_of_predictions:][::-1]        

        if doc_prediction_level==0:
            not_classified+=1
            continue

        if len(doc_prediction)<number_of_predictions:
            doc_prediction=np.pad(doc_prediction,(0,number_of_predictions-len(doc_prediction)),mode='constant',constant_values=(-1))
            doc_prediction_confidence=np.pad(doc_prediction_confidence,(0,number_of_predictions-len(doc_prediction_confidence)),mode='constant',constant_values=(-999999))
        present_prediction[idx][:,0]=doc_prediction
        present_prediction[idx][:,1]=doc_prediction_confidence
        #added to account for ambiguous ddc
        for i in item:                                                    # 'for all true classifications in the test set...'
            if (i in doc_prediction and (flag==0 or idx in pre)):          #'....if the classification is also indicated by the classifier,...' 
                corr+=1.0/len(item)  
                correctly_done.add(idx)                            #'...accuracy score increases in proportion to number of actual classifications- if real classes=2,3;classifier o/p= 2,5,6- score increases by 0.5'

 
        if level_to_train==1 or len((previous_prediction[idx][:,0])[previous_prediction[idx][:,0]>-1])!=0:   # ruling out those that weren't classified at earlier level
            if level_to_train>1:
                confusion_matrix,present_prediction[idx]=edit_confusion_matrix(confusion_matrix,previous_prediction[idx],present_prediction[idx],original_classification) 
            else:
                confusion_matrix,present_prediction[idx]=edit_confusion_matrix(confusion_matrix,None,present_prediction[idx],original_classification) 


    if count>1:
        count-=1
    score=float(corr*100.0)/count                       # Denominator=
    
    print "param,level,min_confidene,no of predictions:",param,level_to_train,defined_min_confidence,number_of_predictions
    print count

    joblib.dump(present_prediction,"prediction_"+param+"_"+str(level_to_train)+"_"+str(defined_min_confidence)+"_"+str(number_of_predictions) +".pkl")
    joblib.dump(correctly_done,"correctly_done_"+param+"_"+str(level_to_train)+"_"+str(defined_min_confidence)+"_"+str(number_of_predictions) +".pkl")
    joblib.dump(confusion_matrix,"confusion_matrix_"+param+"_"+str(level_to_train)+"_"+str(defined_min_confidence)+"_"+str(number_of_predictions) +".pkl")
    print "score=", score
    print "precision", float(corr)/(count-not_classified)
    print "not_classified",not_classified,(not_classified*1.0)/(count)       
    writer.writerow([param,level_to_train, number_of_predictions,defined_min_confidence,(not_classified*100.0)/(count),score,float(corr*100)/(count-not_classified)])

def edit_confusion_matrix(confusion_matrix,previous_prediction,present_prediction,ground_truth):
    global class_freq_matrix

    if class_freq_matrix is None and level_to_train>1:
        class_freq_matrix=joblib.load("class_freq_mat_"+str(level_to_train)+"_"+param+".pkl")

    if level_to_train==1:
        ground_truth=[int(i/100) for i in ground_truth]
    if level_to_train==2:
        ground_truth=[int(i/10) for i in ground_truth]

    all_present_predictions=present_prediction[:,0]
    pre_pred=all_present_predictions[all_present_predictions>-1]
    if previous_prediction is None or level_to_train==1:       # must be level 1
        for i in ground_truth:
            for j in pre_pred:
                j=int(round(j))
                confusion_matrix[i][j]=1.0/len(pre_pred)
        return confusion_matrix,present_prediction
    
    all_previous_predictions=previous_prediction[:,0]
    prev_pred=all_previous_predictions[all_previous_predictions>-1]
    

    #case 1
    if len(prev_pred)==1:
        prev_pred=prev_pred[0]
        pred_class=[prev_pred*10+i for i in pre_pred] #predicted values

        for i in ground_truth:
            for j in pred_class:
                j=int(round(j))
                confusion_matrix[i][j]+=1.0/len(pred_class)
        present_prediction[:,0][:len(all_present_predictions[all_present_predictions>-1])]=all_present_predictions[all_present_predictions>-1]+prev_pred*10
        present_prediction[:,1]=present_prediction[:,1]+previous_prediction[np.where(previous_prediction==prev_pred)[0][0]][1]
        return confusion_matrix,present_prediction


    new_present_prediction=[]
    for a,b in combinations(prev_pred,2):           #the case of just one prediction handled earlier
        a,b=int(round(a)),int(round(b))
        matrix_of_interest=class_freq_matrix[a*10:(a+1)*10,b*10:(b+1)*10]
        possibilities=np.unravel_index(np.argsort(matrix_of_interest,axis=None),(10,10))
        for i,j in zip(possibilities[0][::-1],possibilities[1][::-1]):        #in descending order
            if i in pre_pred and j in pre_pred:
                new_present_prediction.append([10*a+i,previous_prediction[np.where(all_previous_predictions==a)[0][0]][1]+present_prediction[np.where(all_present_predictions==i)[0][0]][1]])
                new_present_prediction.append([10*b+j,previous_prediction[np.where(all_previous_predictions==b)[0][0]][1]+present_prediction[np.where(all_present_predictions==j)[0][0]][1]])

    list_1=[]                          #retaining unique elements from the list
    list_2=[]
    for i,j in new_present_prediction:
        if int(round(i)) not in list_1:
            list_1.append(i)
            list_2.append(j)

    new_present_prediction=np.column_stack((np.asarray(list_1),np.asarray(list_2)))
    #adjusting size of the array
    if len(new_present_prediction)<number_of_predictions:
        new_present_prediction=np.pad(new_present_prediction,[(0,number_of_predictions-len(new_present_prediction)),(0,0)],mode='constant',constant_values=(-1))

    elif len(new_present_prediction)>number_of_predictions:
        to_take=(np.argsort(new_present_prediction[:,1])[::-1])[:number_of_predictions]
        new_present_prediction=new_present_prediction[to_take]

    for i in ground_truth:
        for j in new_present_prediction[:,0]:
            j=int(round(j))
            confusion_matrix[i][j]+=1.0/len(new_present_prediction)

    return confusion_matrix,new_present_prediction

def main():
    
    global defined_min_confidence
    global number_of_predictions
    global level_to_train
    global writer
    global param
    global conn
    global cursor
    global details
    global rows_of_test_target
    global train_target
    global train_level
    global class_freq_matrix

    min_confidence_list=[-2,0,0.05,1,3]
    train_levels=range(1,4)
    no_of_predictions=range(1,4)
    param_list=['title','title_abs','title_contents_abs','title_contents']
    details_output_file=open("details_exh.csv",'w')
    writer=csv.writer(details_output_file)
    writer.writerow(['parameter','ddc_level','number of predictions','minimum confidence','percent left unclassified','score','precision'])
    
    for params in param_list:
        param=params
        conn=sqlite3.connect('database_'+param+'.db')
        cursor=conn.cursor()
        details_file=open('details_'+param+'.txt')
        details=details_file.next().strip().split(' ')
        i=0
        for train_level_here in train_levels:
            level_to_train=train_level_here
            j=0
            for predictions in no_of_predictions:
                number_of_predictions=predictions
                for confidence in min_confidence_list:
                    defined_min_confidence=confidence
                    i+=1
                    j+=1

                    if i==1:
                        first_time=True
                        class_freq_matrix=None
                        train_target=None
                        train_level=None
                    else:
                        first_time=False
                    if j==1:
                        train_new_clf=True
                        rows_of_test_target=None

                    else:
                        train_new_clf=False

                    controller(first_time,train_new_clf)

    
main()