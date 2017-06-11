import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib
#import pickle
#from pandas import DataFrame
import json
import numpy
from collections import defaultdict
import gc
import sqlite3
from itertools import izip
import random

conn = sqlite3.connect('database_test.db')
cursor=conn.cursor()

path= 'title_primary_1'            #"abs_title_primary"

test_to_train=0.25

totalentry= 8930992
feature_count={}
field1='title'
field2='abstract'
vocabulary = defaultdict()
vocabulary.default_factory = vocabulary.__len__
'''
row_ind=array.array('i')
col_ind=array.array('i')
sparse_data=array.array('i')
'''
csv.field_size_limit(1024*1024*512)

def decision(probability):
    return random.random() < probability


def get_size(filename):
    statinfo=os.stat(path+'/'+filename)
    return float(statinfo.st_size)/(1024*1024)

def is_empty(d):
    if d:
        return False
    else:return True

def get_next_batch():
    setsize=200               #size of each chunk in MB
    fileleft={}
    filequeue=[]
    chunksize=0.0
    
    for filename in os.listdir(path):
        if not(filename.endswith('.csv')):
               continue
        fileleft[filename]=get_size(filename)   #creating a dictionary of file sizes

    while not is_empty(fileleft):
        for filename in fileleft.keys():
            chunksize+=fileleft[filename]
            del(fileleft[filename])
            filequeue.append(filename)
            if(chunksize>=setsize or len(fileleft)==0):
                if(chunksize>=1.5*setsize):
                    filequeue.sort(key=get_size)
                    while (len(filequeue)>=2 and chunksize>=1.5*setsize):
                        filename_rem=filequeue.pop(0)
                        deducted_size=get_size(filename_rem)
                        chunksize-=deducted_size
                        fileleft[filename_rem]=deducted_size

                queue=filequeue
                print "Passing chunksize=",chunksize,"MB"
                print filequeue
                chunksize=0.0
                filequeue=[]
                yield queue
               
def main():
    global vocabulary

    cursor.execute("CREATE TABLE table1(idx int,row_ind int ,col_ind int, sparse_data int)")
    cursor.execute("CREATE TABLE targetvector(idx int, target char )")

    cursor.execute("CREATE TABLE table1_test(idx int,row_ind int ,col_ind int, sparse_data int)")
    cursor.execute("CREATE TABLE targetvector_test(idx int, target char )")

    
    filecount=0                              #present no of entries completed
    global totalentry
    if(totalentry==-1):
       totalentry=count_data(path)               
    print totalentry

    rows_table1=0         #no of training data rows in the db table.NOT equal to no of docs
    rows_test=0

    entrycount=0         #total no of entries- equal to no of targets
    testcount=0
    for filequeue in get_next_batch():
        entrycount,testcount,rows_table1, rows_test=vectorizer(filequeue,entrycount,testcount,rows_table1, rows_test)

    print entrycount,testcount
    print rows_table1, rows_test       
    '''
    row_ind = numpy.frombuffer(row_ind, dtype=numpy.intc)
    col_ind = numpy.frombuffer(col_ind, dtype=numpy.intc)
    sparse_data = numpy.frombuffer(sparse_data, dtype=numpy.intc)



    import scipy.sparse as sp
    X = sp.csr_matrix((sparse_data, (row_ind, col_ind)),shape=(totalentry,len(vocabulary)))
    X.sort_indices()
    
    print X
    print row_ind
    print col_ind
    print sparse_data
    print vocabulary
    
    joblib.dump(X,'x.pkl')
    joblib.dump(targetvector,'targetvector.pkl')
    print X
    

    cursor.execute("SELECT * FROM table1")
    conn.commit()
    rows=cursor.fetchall()
    print "Printing training_data:"
    for row in rows:
        print row
    cursor.execute("SELECT * FROM targetvector")
    conn.commit()
    rows=cursor.fetchall()
    print "Printing training_target:"
    for row in rows:
        print row
    


    print "\nTest:"
    cursor.execute("SELECT * FROM table1_test")
    conn.commit()
    rows=cursor.fetchall()
    print "Printing test_data:"
    for row in rows:
        print row
    cursor.execute("SELECT * FROM targetvector_test")
    conn.commit()
    rows=cursor.fetchall()
    print "Printing test_target:"
    for row in rows:
        print row
    

    print vocabulary
    '''    

    with open("vocab.txt",'w') as f:
        json.dump(vocabulary,f)
    s=str(rows_table1)+" "+str(rows_test)+ " "+str(entrycount)+" "+str(testcount)
    with open('details.txt','w') as f:
        f.write(s)

        
    
def vectorizer(filequeue,entrycount,testcount,rows_table1, rows_test): 

    vec=CountVectorizer()
    row_train_temp=[]
    col_train_temp=[]
    data_train_temp=[]
    row_test_temp=[]
    col_test_temp=[]
    data_test_temp=[]
    
    #index=[]
    ddc_train=[]
    ddc_test=[]

    data,ddc= dataframeCreator(filequeue)       #pre_entrycount is count at the end

    dbtable='table1'
    new_testcount=testcount
    new_entrycount=entrycount
    new_rows_table1=rows_table1
    new_rows_test=rows_test

    
    count=vec.fit_transform(data)
    features=vec.get_feature_names()
    del data
    del vec
    gc.collect()
    print "Vectorization done. Adding data to original vectors"

    coo_count=count.tocoo()
    prev=None
    isTestCase=False

    
    for row,col,countdata in izip(coo_count.row,coo_count.col,coo_count.data):  # the occurrence of the same-row entries consecutively is critical for later coputations as no conversion to sparse matrix has been done

        feature=features[col]
        feature_idx=vocabulary[feature]
        col=feature_idx
        if(row!=prev):  #new entry begins here
            if (decision(test_to_train)):                   #to be put in test data
                ddc_test.append(ddc[row])
                isTestCase=True
                rowNo=new_testcount
                new_testcount+=1
            else:
                ddc_train.append(ddc[row])
                isTestCase=False
                rowNo=new_entrycount
                new_entrycount+=1
        if(isTestCase):
            new_rows_test+=1
            row_test_temp.append(rowNo)
            col_test_temp.append(col)
            data_test_temp.append(countdata)
            if len(row_test_temp)>=3000000:             #inserting 5 million entries into test table in one go

                cursor.executemany('''INSERT INTO table1_test (idx,row_ind, col_ind, sparse_data) VALUES(?,?,?,?)''',izip(xrange(rows_test,new_rows_test),row_test_temp,col_test_temp,data_test_temp))
                rows_test=new_rows_test
                conn.commit()
                row_test_temp=[]
                col_test_temp=[]
                data_test_temp=[]

        else:                #is train set
            
            new_rows_table1+=1
            row_train_temp.append(rowNo)
            col_train_temp.append(col)
            data_train_temp.append(countdata)
            if len(row_train_temp)>=3000000:             #inserting 5 million entries into test table in one go

                cursor.executemany('''INSERT INTO table1 (idx,row_ind, col_ind, sparse_data) VALUES(?,?,?,?)''',izip(xrange(rows_table1,new_rows_table1),row_train_temp,col_train_temp,data_train_temp))
                rows_table1=new_rows_table1
                conn.commit()
                row_train_temp=[]
                col_train_temp=[]
                data_train_temp=[]


        prev=row

        '''-------------------------------------------------------------------------------------------------------------'''

    if len(row_test_temp)>0:
        cursor.executemany('''INSERT INTO table1_test (idx,row_ind, col_ind, sparse_data) VALUES(?,?,?,?)''',izip(xrange(rows_test,new_rows_test),row_test_temp,col_test_temp,data_test_temp))
    if len(row_train_temp)>0:
        cursor.executemany('''INSERT INTO table1 (idx,row_ind, col_ind, sparse_data) VALUES(?,?,?,?)''',izip(xrange(rows_table1,new_rows_table1),row_train_temp,col_train_temp,data_train_temp))
    conn.commit()

        
    '''Inserting target vectors in db'''
 
    cursor.executemany('''INSERT INTO targetvector (idx,target) VALUES(?,?)''',izip(xrange(entrycount,new_entrycount),ddc_train))
    cursor.executemany('''INSERT INTO targetvector_test (idx,target) VALUES(?,?)''',izip(xrange(testcount,new_testcount),ddc_test))
    conn.commit()
    
        
            
    print "batch done"
    del count
    del features

    return new_entrycount, new_testcount, new_rows_table1, new_rows_test
                
def dataframeCreator(filequeue):
    rows=[]
    ddc=[]
    for files in filequeue:
        filepath=path+'/'+files
        for text, classification in read_files(filepath):
            '''
            if not (decision(test_to_train)):
                ddc.append(classification)
                rows.append(text)
                index.append(simpleIdx)
                simpleIdx+=1
            else:
                ddc_test.append(classification)
                rows_test.append(text)
                index_test.append(testIdx)
                testIdx+=1
            '''
            ddc.append(classification)
            rows.append(text)
    return rows,ddc
        

def read_files(filepath):
    print "Reading file:"+filepath
    preF=open(filepath)
    reader=csv.reader(preF)
    header=next(reader)
    for idx in range (0,len(header)):
        #if item=='id':idIdx=idx
        if header[idx]=='ddc':ddcIdx=idx
        elif header[idx]=='title': titleIdx=idx
        elif header[idx]=='abstract':absIdx=idx
    for row in reader:
        yield row[titleIdx], row[ddcIdx]         # This is now a generator               +" "+row[absIdx]
    preF.close()




    
main()

